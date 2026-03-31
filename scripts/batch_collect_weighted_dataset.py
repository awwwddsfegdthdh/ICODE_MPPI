#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
import random
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_OBSTACLE_GEOM_NAMES = (
    "obs_front_geom",
    "obs_left_geom",
    "obs_right_geom",
    "obs_mid_left_geom",
    "obs_mid_right_geom",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run weighted batch collection for ICODE training. "
            "Pipeline: collect -> convert -> prepare bundle."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "e1_weighted_sampling_60_25_15.json",
        help="JSON config path that defines weighted scenario profiles.",
    )
    parser.add_argument("--total-episodes", type=int, default=600, help="Total episodes across all profiles.")
    parser.add_argument("--base-seed", type=int, default=7000, help="Base seed for per-shard collection runs.")
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable for invoking collection/convert/prepare scripts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root directory. Default: datasets/<plan_name>_<timestamp>/",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip final bundle generation step.")
    parser.add_argument(
        "--delete-raw-after-convert",
        action="store_true",
        help="Delete per-shard raw .npz after successful convert to reduce disk usage.",
    )
    parser.add_argument(
        "--delete-converted-after-prepare",
        action="store_true",
        help="Delete converted shard .npz after successful final bundle generation.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def allocate_counts(total: int, weights: Sequence[float]) -> List[int]:
    if total <= 0:
        raise ValueError("total episodes must be > 0")
    if len(weights) == 0:
        raise ValueError("profiles must not be empty")
    if any(w <= 0 for w in weights):
        raise ValueError("all profile weights must be > 0")

    w_sum = float(sum(weights))
    exact = [total * (w / w_sum) for w in weights]
    base = [int(math.floor(x)) for x in exact]
    remainder = total - sum(base)
    frac_order = sorted(range(len(exact)), key=lambda i: (exact[i] - base[i]), reverse=True)
    for i in range(remainder):
        base[frac_order[i]] += 1
    return base


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    rendered = shlex.join(cmd)
    print(f"[cmd] {rendered}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def validate_config(cfg: Dict[str, object]) -> None:
    if "profiles" not in cfg or not isinstance(cfg["profiles"], list) or not cfg["profiles"]:
        raise ValueError("config must include non-empty list key 'profiles'")
    for idx, p in enumerate(cfg["profiles"]):
        if not isinstance(p, dict):
            raise ValueError(f"profile[{idx}] must be dict")
        for k in ("name", "weight"):
            if k not in p:
                raise ValueError(f"profile[{idx}] missing key '{k}'")


def as_path(path_like: str, repo_root: Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def parse_shape_pattern(pattern: str, expected_len: int) -> Tuple[str, ...]:
    pattern = pattern.strip().upper()
    if len(pattern) != expected_len:
        raise ValueError(f"shape pattern '{pattern}' length must be {expected_len}")
    out: List[str] = []
    for ch in pattern:
        if ch == "B":
            out.append("box")
        elif ch == "C":
            out.append("cylinder")
        else:
            raise ValueError(f"invalid shape pattern char '{ch}' in '{pattern}', only B/C allowed")
    return tuple(out)


def parse_size(text: str) -> List[float]:
    parts = [float(x) for x in text.strip().split()]
    if len(parts) not in (2, 3):
        raise ValueError(f"unexpected geom size dims: {text}")
    return parts


def format_size(vals: Sequence[float]) -> str:
    return " ".join(f"{float(v):.6f}".rstrip("0").rstrip(".") for v in vals)


def convert_size_for_shape(size_vals: Sequence[float], target_shape: str) -> List[float]:
    if target_shape == "box":
        if len(size_vals) == 3:
            return [float(size_vals[0]), float(size_vals[1]), float(size_vals[2])]
        radius = float(size_vals[0])
        half_h = float(size_vals[1])
        return [radius, radius, half_h]
    if target_shape == "cylinder":
        if len(size_vals) == 2:
            return [float(size_vals[0]), float(size_vals[1])]
        radius = 0.5 * (float(size_vals[0]) + float(size_vals[1]))
        half_h = float(size_vals[2])
        return [radius, half_h]
    raise ValueError(f"unsupported target shape: {target_shape}")


def generate_xml_variants_from_patterns(
    base_xml: Path,
    output_dir: Path,
    obstacle_geom_names: Sequence[str],
    patterns: Sequence[str],
) -> List[Path]:
    if not base_xml.exists():
        raise FileNotFoundError(f"base xml not found: {base_xml}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_tree = ET.parse(str(base_xml))
    base_root = base_tree.getroot()
    base_geom_info: Dict[str, Tuple[str, List[float]]] = {}
    for gname in obstacle_geom_names:
        geom = base_root.find(f".//geom[@name='{gname}']")
        if geom is None:
            raise KeyError(f"geom '{gname}' not found in {base_xml}")
        gtype = geom.attrib.get("type")
        if gtype not in ("box", "cylinder"):
            raise ValueError(f"geom '{gname}' type must be box/cylinder, got {gtype}")
        gsize = parse_size(geom.attrib.get("size", ""))
        base_geom_info[gname] = (gtype, gsize)

    out_xmls: List[Path] = []
    for pat in patterns:
        shapes = parse_shape_pattern(pat, expected_len=len(obstacle_geom_names))
        tree = ET.parse(str(base_xml))
        root = tree.getroot()
        for i, gname in enumerate(obstacle_geom_names):
            geom = root.find(f".//geom[@name='{gname}']")
            if geom is None:
                raise KeyError(f"geom '{gname}' not found in variant generation")
            target_shape = shapes[i]
            _, src_size = base_geom_info[gname]
            geom.attrib["type"] = target_shape
            geom.attrib["size"] = format_size(convert_size_for_shape(src_size, target_shape))

        out_name = f"{base_xml.stem}_shape_{pat.upper()}.xml"
        out_path = output_dir / out_name
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        out_xmls.append(out_path)

    return out_xmls


def prepare_xml_rotation_pool(
    cfg: Mapping[str, object],
    repo_root: Path,
    output_root: Path,
) -> Optional[Dict[str, object]]:
    rotation = cfg.get("xml_rotation")
    if not isinstance(rotation, Mapping):
        return None
    enabled = bool(rotation.get("enabled", False))
    if not enabled:
        return None

    granularity = str(rotation.get("granularity", "episode"))
    if granularity != "episode":
        raise ValueError(f"only xml_rotation.granularity='episode' is supported, got '{granularity}'")

    pool: List[Path] = []
    if "xml_pool" in rotation and isinstance(rotation["xml_pool"], list) and rotation["xml_pool"]:
        pool = [as_path(str(x), repo_root) for x in rotation["xml_pool"]]
    else:
        generator = rotation.get("generator")
        if not isinstance(generator, Mapping):
            raise ValueError("xml_rotation requires either xml_pool or generator config")

        base_xml = as_path(
            str(
                generator.get(
                    "base_xml",
                    "/home/wmh/ICODE/E1_Robot/simulation/models/mjcf/E1_SimpleSensor.xml",
                )
            ),
            repo_root,
        )
        obstacle_geom_names = generator.get("obstacle_geom_names", list(DEFAULT_OBSTACLE_GEOM_NAMES))
        if not isinstance(obstacle_geom_names, list) or not obstacle_geom_names:
            raise ValueError("xml_rotation.generator.obstacle_geom_names must be non-empty list")
        patterns = generator.get("shape_patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("xml_rotation.generator.shape_patterns must be non-empty list")

        # Keep generated variants in the same directory as base XML so
        # relative mesh paths (for example ../meshes/*.STL) remain valid.
        variants_dir = base_xml.parent
        pool = generate_xml_variants_from_patterns(
            base_xml=base_xml,
            output_dir=variants_dir,
            obstacle_geom_names=[str(x) for x in obstacle_geom_names],
            patterns=[str(x) for x in patterns],
        )

    if not pool:
        raise ValueError("xml rotation pool resolved empty")
    for p in pool:
        if not p.exists():
            raise FileNotFoundError(f"xml in rotation pool not found: {p}")

    seed = int(rotation.get("seed", 12345))
    return {
        "seed": seed,
        "pool": pool,
        "pool_labels": [p.name for p in pool],
        "granularity": granularity,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    validate_config(cfg)

    plan_name = str(cfg.get("plan_name", "weighted_sampling"))
    horizon = int(cfg.get("horizon", 200))
    chunk_episodes = int(cfg.get("chunk_episodes", 40))
    global_collect_args = [str(x) for x in cfg.get("global_collect_args", [])]
    global_convert_args = [str(x) for x in cfg.get("global_convert_args", [])]
    prepare_args = cfg.get("prepare_args", {})
    if not isinstance(prepare_args, dict):
        raise ValueError("prepare_args must be a dict")

    profiles = cfg["profiles"]
    weights = [float(p["weight"]) for p in profiles]
    profile_counts = allocate_counts(args.total_episodes, weights)

    if args.output_root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = repo_root / "datasets" / f"{plan_name}_{stamp}"
    else:
        output_root = args.output_root

    raw_dir = output_root / "raw"
    converted_dir = output_root / "converted"
    bundle_dir = output_root / "bundle"
    manifest_path = output_root / "manifest.json"
    for d in (raw_dir, converted_dir, bundle_dir):
        d.mkdir(parents=True, exist_ok=True)

    xml_rotation = prepare_xml_rotation_pool(cfg=cfg, repo_root=repo_root, output_root=output_root)
    xml_rng = None
    xml_pool: Optional[List[Path]] = None
    xml_pool_labels: List[str] = []
    if xml_rotation is not None:
        xml_rng = random.Random(int(xml_rotation["seed"]) + int(args.base_seed))
        xml_pool = list(xml_rotation["pool"])
        xml_pool_labels = [str(x) for x in xml_rotation["pool_labels"]]
        print(f"[plan] xml rotation enabled ({len(xml_pool)} variants, per-episode random)")

    collect_script = repo_root / "collect_mujoco_multilayer_data.py"
    convert_script = repo_root / "convert_multilayer_dataset.py"
    prepare_script = repo_root / "prepare_icode_training_data.py"

    if not collect_script.exists() or not convert_script.exists() or not prepare_script.exists():
        raise FileNotFoundError("Expected collect/convert/prepare scripts under repository root.")

    converted_files: List[Path] = []
    manifest_profiles: List[Dict[str, object]] = []
    seed_cursor = int(args.base_seed)
    xml_usage: Dict[str, int] = {}

    for profile, ep_total in zip(profiles, profile_counts):
        name = str(profile["name"])
        collect_args = [str(x) for x in profile.get("collect_args", [])]
        shards: List[Dict[str, object]] = []
        if ep_total <= 0:
            manifest_profiles.append({"name": name, "episodes": 0, "shards": shards})
            continue

        remaining = ep_total
        shard_idx = 0
        while remaining > 0:
            if xml_pool is not None:
                ep_this = 1
            else:
                ep_this = min(chunk_episodes, remaining)

            tag = f"{name}_s{shard_idx:04d}"
            raw_out = raw_dir / f"{tag}.npz"
            converted_out = converted_dir / f"{tag}.npz"
            xml_selected = None
            if xml_pool is not None:
                if xml_rng is None:
                    raise RuntimeError("xml rng not initialized")
                xml_selected = xml_rng.choice(xml_pool)
                xml_usage[str(xml_selected)] = xml_usage.get(str(xml_selected), 0) + 1

            collect_cmd = [
                args.python_bin,
                str(collect_script),
                "--output",
                str(raw_out),
                "--episodes",
                str(ep_this),
                "--horizon",
                str(horizon),
                "--seed",
                str(seed_cursor),
            ]
            if xml_selected is not None:
                collect_cmd.extend(["--xml", str(xml_selected)])
            collect_cmd.extend(global_collect_args)
            collect_cmd.extend(collect_args)

            convert_cmd = [
                args.python_bin,
                str(convert_script),
                "--input",
                str(raw_out),
                "--output",
                str(converted_out),
            ]
            convert_cmd.extend(global_convert_args)

            print(
                f"[plan] profile={name} shard={shard_idx} episodes={ep_this} "
                f"seed={seed_cursor} raw={raw_out.name}"
                + (f" xml={xml_selected.name}" if xml_selected is not None else "")
            )
            run_cmd(collect_cmd, dry_run=args.dry_run)
            run_cmd(convert_cmd, dry_run=args.dry_run)
            if args.delete_raw_after_convert and (not args.dry_run):
                try:
                    raw_out.unlink(missing_ok=True)
                except Exception as exc:
                    print(f"[warn] failed to delete raw shard {raw_out}: {exc}")

            converted_files.append(converted_out)
            shards.append(
                {
                    "shard_index": shard_idx,
                    "episodes": ep_this,
                    "seed": seed_cursor,
                    "raw_output": str(raw_out),
                    "converted_output": str(converted_out),
                    "xml": str(xml_selected) if xml_selected is not None else None,
                }
            )

            remaining -= ep_this
            shard_idx += 1
            seed_cursor += 1

        manifest_profiles.append({"name": name, "episodes": ep_total, "shards": shards})

    bundle_output = bundle_dir / f"{plan_name}_bundle.npz"
    prepare_cmd = [
        args.python_bin,
        str(prepare_script),
        "--inputs",
        *[str(p) for p in converted_files],
        "--output",
        str(bundle_output),
        "--seed",
        str(int(prepare_args.get("seed", 42))),
        "--train-ratio",
        str(float(prepare_args.get("train_ratio", 0.8))),
        "--val-ratio",
        str(float(prepare_args.get("val_ratio", 0.1))),
    ]

    if not args.skip_prepare:
        print(f"[plan] build bundle from {len(converted_files)} converted shards")
        run_cmd(prepare_cmd, dry_run=args.dry_run)
        if args.delete_converted_after_prepare and (not args.dry_run):
            for p in converted_files:
                try:
                    p.unlink(missing_ok=True)
                except Exception as exc:
                    print(f"[warn] failed to delete converted shard {p}: {exc}")

    manifest = {
        "plan_name": plan_name,
        "config_path": str(args.config),
        "generated_at": dt.datetime.now().isoformat(),
        "repo_root": str(repo_root),
        "total_episodes_requested": int(args.total_episodes),
        "horizon": horizon,
        "chunk_episodes": chunk_episodes,
        "base_seed": int(args.base_seed),
        "python_bin": args.python_bin,
        "dry_run": bool(args.dry_run),
        "skip_prepare": bool(args.skip_prepare),
        "delete_raw_after_convert": bool(args.delete_raw_after_convert),
        "delete_converted_after_prepare": bool(args.delete_converted_after_prepare),
        "output_root": str(output_root),
        "global_collect_args": global_collect_args,
        "global_convert_args": global_convert_args,
        "xml_rotation": {
            "enabled": xml_pool is not None,
            "granularity": "episode" if xml_pool is not None else None,
            "pool": [str(p) for p in xml_pool] if xml_pool is not None else [],
            "pool_labels": xml_pool_labels,
            "usage": xml_usage,
        },
        "prepare_args": {
            "seed": int(prepare_args.get("seed", 42)),
            "train_ratio": float(prepare_args.get("train_ratio", 0.8)),
            "val_ratio": float(prepare_args.get("val_ratio", 0.1)),
        },
        "profiles": manifest_profiles,
        "converted_files": [str(p) for p in converted_files],
        "bundle_output": str(bundle_output),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] manifest: {manifest_path}")
    if not args.skip_prepare:
        print(f"[done] bundle: {bundle_output}")


if __name__ == "__main__":
    main()
