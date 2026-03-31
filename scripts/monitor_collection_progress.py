#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


PLAN_RE = re.compile(
    r"\[plan\]\s+profile=(?P<profile>\S+)\s+shard=(?P<shard>\d+)\s+episodes=(?P<episodes>\d+)\s+seed=\d+\s+raw=(?P<raw>\S+)"
)
CONVERT_DONE_RE = re.compile(
    r"Saved converted multilayer dataset to:\s+(?P<path>.+?/converted/(?P<tag>[^/\s]+)\.npz)"
)
DONE_BUNDLE_RE = re.compile(r"\[done\]\s+bundle:\s+")
DEPTH_TAG_RE = re.compile(r"depth(?P<episodes>\d+)")
STAMP_RE = re.compile(r"_(?P<stamp>\d{8}_\d{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show collection progress: completed episodes, ETA, and current disk usage."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Output root directory of batch_collect_weighted_dataset.py",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path. Default: <run-root>.log",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help="Override total episodes. If omitted, infer from manifest or run-root name.",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval (seconds) in watch mode.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    return parser.parse_args()


def infer_log_path(run_root: Path, explicit_log: Optional[Path]) -> Path:
    if explicit_log is not None:
        return explicit_log
    return run_root.parent / f"{run_root.name}.log"


def parse_start_time(run_root: Path, log_path: Path) -> float:
    m = STAMP_RE.search(run_root.name)
    if m:
        try:
            t = dt.datetime.strptime(m.group("stamp"), "%Y%m%d_%H%M%S")
            return t.timestamp()
        except ValueError:
            pass
    if log_path.exists():
        return float(log_path.stat().st_mtime)
    return time.time()


def infer_total_episodes(run_root: Path, explicit_total: Optional[int]) -> Optional[int]:
    if explicit_total is not None and explicit_total > 0:
        return int(explicit_total)

    manifest = run_root / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            total = int(data.get("total_episodes_requested", 0))
            if total > 0:
                return total
        except Exception:
            pass

    m = DEPTH_TAG_RE.search(run_root.name)
    if m:
        try:
            return int(m.group("episodes"))
        except ValueError:
            return None
    return None


def read_log_progress(log_path: Path) -> Tuple[Dict[str, int], Set[str], bool, bool]:
    shard_episodes: Dict[str, int] = {}
    converted_tags: Set[str] = set()
    saw_traceback = False
    saw_bundle_done = False

    if not log_path.exists():
        return shard_episodes, converted_tags, saw_traceback, saw_bundle_done

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            pm = PLAN_RE.search(line)
            if pm:
                raw_name = Path(pm.group("raw")).name
                tag = raw_name[:-4] if raw_name.endswith(".npz") else raw_name
                shard_episodes[tag] = int(pm.group("episodes"))
                continue

            cm = CONVERT_DONE_RE.search(line)
            if cm:
                converted_tags.add(cm.group("tag"))
                continue

            if "Traceback (most recent call last):" in line:
                saw_traceback = True

            if DONE_BUNDLE_RE.search(line):
                saw_bundle_done = True

    return shard_episodes, converted_tags, saw_traceback, saw_bundle_done


def bytes_to_str(nbytes: int) -> str:
    if nbytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(min(len(units) - 1, math.floor(math.log(nbytes, 1024))))
    value = nbytes / (1024 ** i)
    return f"{value:.2f} {units[i]}"


def sec_to_str(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        return "N/A"
    s = int(round(sec))
    h = s // 3600
    m = (s % 3600) // 60
    t = s % 60
    return f"{h:02d}:{m:02d}:{t:02d}"


def calc_dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


def build_report(
    run_root: Path,
    log_path: Path,
    total_episodes: Optional[int],
    start_ts: float,
) -> str:
    shard_eps, converted_tags, saw_traceback, saw_bundle_done = read_log_progress(log_path)
    done_eps = 0
    for tag in converted_tags:
        done_eps += int(shard_eps.get(tag, 1))
    done_shards = len(converted_tags)
    known_shards = len(shard_eps)

    now = time.time()
    elapsed = max(0.0, now - start_ts)
    rate_eps_per_h = (done_eps / elapsed * 3600.0) if elapsed > 0 and done_eps > 0 else 0.0

    eta = float("nan")
    pct = float("nan")
    if total_episodes is not None and total_episodes > 0:
        pct = 100.0 * min(max(done_eps, 0), total_episodes) / total_episodes
        if done_eps > 0 and done_eps < total_episodes and rate_eps_per_h > 1e-6:
            eta = (total_episodes - done_eps) / (rate_eps_per_h / 3600.0)

    if saw_bundle_done:
        status = "COMPLETED"
    elif saw_traceback:
        status = "FAILED"
    else:
        status = "RUNNING"

    run_size = calc_dir_size(run_root)
    disk = shutil.disk_usage(run_root.parent if run_root.parent.exists() else Path("/"))
    log_mtime = dt.datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if log_path.exists() else "N/A"

    lines = []
    lines.append(f"Run Root: {run_root}")
    lines.append(f"Log File: {log_path}")
    lines.append(f"Status: {status}")
    if total_episodes is None:
        lines.append(f"Progress: {done_eps} episodes done (total unknown)")
    else:
        lines.append(f"Progress: {done_eps}/{total_episodes} episodes ({pct:.2f}%)")
    lines.append(f"Shards: {done_shards} converted / {known_shards} planned-seen")
    lines.append(f"Elapsed: {sec_to_str(elapsed)}")
    lines.append(f"Speed: {rate_eps_per_h:.2f} ep/hour")
    lines.append(f"ETA: {sec_to_str(eta)}")
    lines.append(
        f"Storage: run={bytes_to_str(run_size)} | disk_used={bytes_to_str(disk.used)} | "
        f"disk_free={bytes_to_str(disk.free)}"
    )
    lines.append(f"Last Log Update: {log_mtime}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    log_path = infer_log_path(run_root=run_root, explicit_log=args.log)
    total_episodes = infer_total_episodes(run_root=run_root, explicit_total=args.total_episodes)
    start_ts = parse_start_time(run_root=run_root, log_path=log_path)

    while True:
        report = build_report(
            run_root=run_root,
            log_path=log_path,
            total_episodes=total_episodes,
            start_ts=start_ts,
        )
        if not args.once:
            sys.stdout.write("\033[2J\033[H")
        print(report, flush=True)
        if args.once:
            break
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    main()
