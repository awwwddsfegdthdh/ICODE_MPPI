import argparse
from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
from state_convention import assert_meta_contract, merge_and_validate_meta, read_npz_meta


def quat_wxyz_to_yaw_batch(quat_wxyz: np.ndarray) -> np.ndarray:
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)


def world_to_body_x(v_world_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return (c * v_world_xy[:, 0] + s * v_world_xy[:, 1]).astype(np.float32)


def load_transitions_from_converted(
    path: Path,
    label_source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    data = np.load(path, allow_pickle=True)
    meta = read_npz_meta(data)
    assert_meta_contract(meta)

    if "icode__x_t" not in data.files or "icode__u_t" not in data.files:
        raise KeyError(f"{path} missing icode__x_t or icode__u_t")
    if "raw__episode" not in data.files:
        raise KeyError(f"{path} missing raw__episode")

    x = data["icode__x_t"].astype(np.float32)
    u = data["icode__u_t"].astype(np.float32)
    ep = data["raw__episode"].astype(np.int32)
    if label_source == "gt_state":
        if "icode__x_label_t" in data.files:
            x_label = data["icode__x_label_t"].astype(np.float32)
            if "icode__x_label_valid" in data.files:
                x_label_valid = data["icode__x_label_valid"].astype(np.uint8)
            else:
                x_label_valid = np.ones((x_label.shape[0],), dtype=np.uint8)
        else:
            required_gt = ("gt__base_pos_gt", "gt__base_quat_gt", "gt__base_linvel_gt", "gt__base_angvel_gt")
            if not all(k in data.files for k in required_gt):
                raise KeyError(
                    f"{path} missing icode__x_label_t and required GT keys {required_gt} "
                    "for label_source=gt_state"
                )
            base_pos = data["gt__base_pos_gt"].astype(np.float32)
            base_quat = data["gt__base_quat_gt"].astype(np.float32)
            base_linvel = data["gt__base_linvel_gt"].astype(np.float32)
            base_angvel = data["gt__base_angvel_gt"].astype(np.float32)
            yaw_gt = quat_wxyz_to_yaw_batch(base_quat)
            v_body_gt = world_to_body_x(base_linvel[:, :2], yaw_gt)
            wz_gt = base_angvel[:, 2].astype(np.float32)
            if x.shape[1] >= 7:
                dq_l = x[:, 5].astype(np.float32)
                dq_r = x[:, 6].astype(np.float32)
            else:
                dq_l = np.zeros((x.shape[0],), dtype=np.float32)
                dq_r = np.zeros((x.shape[0],), dtype=np.float32)
            x_label = np.stack(
                [base_pos[:, 0], base_pos[:, 1], yaw_gt, v_body_gt, wz_gt, dq_l, dq_r],
                axis=1,
            ).astype(np.float32)
            x_label_valid = np.all(np.isfinite(x_label), axis=1).astype(np.uint8)
    else:
        x_label = x
        x_label_valid = np.ones((x.shape[0],), dtype=np.uint8)

    if x.shape[0] < 2:
        return (
            np.zeros((0, x.shape[1]), dtype=np.float32),
            np.zeros((0, u.shape[1]), dtype=np.float32),
            np.zeros((0, x.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            meta,
        )

    same_episode = ep[1:] == ep[:-1]

    if "raw__step" in data.files:
        step = data["raw__step"].astype(np.int32)
        contiguous_step = step[1:] == (step[:-1] + 1)
        valid = same_episode & contiguous_step
    else:
        valid = same_episode

    valid = valid & (x_label_valid[:-1] > 0) & (x_label_valid[1:] > 0)

    x_t = x[:-1][valid]
    u_t = u[:-1][valid]
    x_tp1 = x_label[1:][valid]

    if "icode__obs_dist_t" in data.files:
        obs_dist = data["icode__obs_dist_t"].astype(np.float32)
        obs_dist_tp1 = obs_dist[1:][valid]
        obs_dist_valid = np.ones((obs_dist_tp1.shape[0],), dtype=np.uint8)
    else:
        obs_dist_tp1 = np.zeros((x_t.shape[0],), dtype=np.float32)
        obs_dist_valid = np.zeros((x_t.shape[0],), dtype=np.uint8)

    if "raw__sim_time" in data.files:
        sim_time = data["raw__sim_time"].astype(np.float32)
        dt = (sim_time[1:] - sim_time[:-1])[valid]
        dt = np.where(np.isfinite(dt) & (dt >= 0.0), dt, 0.0).astype(np.float32)
    else:
        dt = np.zeros((x_t.shape[0],), dtype=np.float32)

    return x_t, u_t, x_tp1, dt, obs_dist_tp1, obs_dist_valid, meta


def safe_std(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = a.std(axis=0)
    s[s < eps] = eps
    return s


def build_dataset(args: argparse.Namespace) -> None:
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    dts: List[np.ndarray] = []
    obs_dists: List[np.ndarray] = []
    obs_valids: List[np.ndarray] = []
    metas: List[Mapping[str, object]] = []

    for p in args.inputs:
        x_t, u_t, x_tp1, dt, obs_dist_tp1, obs_dist_valid, meta = load_transitions_from_converted(
            p,
            label_source=str(args.label_source),
        )
        print(f"Loaded {p}: transitions={x_t.shape[0]}")
        metas.append(meta)
        if x_t.shape[0] == 0:
            continue
        xs.append(x_t)
        us.append(u_t)
        ys.append(x_tp1)
        dts.append(dt)
        obs_dists.append(obs_dist_tp1)
        obs_valids.append(obs_dist_valid)

    if not xs:
        raise RuntimeError("No valid transitions loaded from inputs.")
    merged_meta = merge_and_validate_meta(metas)

    x_all = np.concatenate(xs, axis=0)
    u_all = np.concatenate(us, axis=0)
    y_all = np.concatenate(ys, axis=0)
    dt_all = np.concatenate(dts, axis=0)
    obs_dist_all = np.concatenate(obs_dists, axis=0)
    obs_valid_all = np.concatenate(obs_valids, axis=0)
    dx_all = y_all - x_all

    n = x_all.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    train_n = int(n * args.train_ratio)
    val_n = int(n * args.val_ratio)
    test_n = n - train_n - val_n

    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    test_idx = perm[train_n + val_n :]

    x_train = x_all[train_idx]
    u_train = u_all[train_idx]
    y_train = y_all[train_idx]
    dx_train = dx_all[train_idx]
    dt_train = dt_all[train_idx]
    obs_dist_train = obs_dist_all[train_idx]
    obs_valid_train = obs_valid_all[train_idx]

    x_val = x_all[val_idx]
    u_val = u_all[val_idx]
    y_val = y_all[val_idx]
    dx_val = dx_all[val_idx]
    dt_val = dt_all[val_idx]
    obs_dist_val = obs_dist_all[val_idx]
    obs_valid_val = obs_valid_all[val_idx]

    x_test = x_all[test_idx]
    u_test = u_all[test_idx]
    y_test = y_all[test_idx]
    dx_test = dx_all[test_idx]
    dt_test = dt_all[test_idx]
    obs_dist_test = obs_dist_all[test_idx]
    obs_valid_test = obs_valid_all[test_idx]

    x_mean = x_train.mean(axis=0)
    x_std = safe_std(x_train)
    u_mean = u_train.mean(axis=0)
    u_std = safe_std(u_train)
    y_mean = y_train.mean(axis=0)
    y_std = safe_std(y_train)
    dx_mean = dx_train.mean(axis=0)
    dx_std = safe_std(dx_train)

    output_parent = args.output.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "meta__inputs": np.array([str(p) for p in args.inputs], dtype=object),
        "meta__seed": np.array([args.seed], dtype=np.int32),
        "meta__split_ratio": np.array([args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio], dtype=np.float32),
        "meta__num_total": np.array([n], dtype=np.int32),
        "meta__num_train": np.array([train_n], dtype=np.int32),
        "meta__num_val": np.array([val_n], dtype=np.int32),
        "meta__num_test": np.array([test_n], dtype=np.int32),
        "meta__x_fields": np.array(["x_odom", "y_odom", "psi_odom", "v_body", "wz_body", "dqL", "dqR"], dtype=object),
        "meta__u_fields": np.array(["u_0", "u_1"], dtype=object),
        "meta__state_convention_version": np.array([str(merged_meta["meta__state_convention_version"].reshape(-1)[0])], dtype=object),
        "meta__drive_sign": np.array([float(merged_meta["meta__drive_sign"].reshape(-1)[0])], dtype=np.float32),
        "meta__pose_source": np.array([str(merged_meta["meta__pose_source"].reshape(-1)[0])], dtype=object),
        "meta__heading_source": np.array([str(merged_meta["meta__heading_source"].reshape(-1)[0])], dtype=object),
        "meta__control_definition": np.array([str(merged_meta["meta__control_definition"].reshape(-1)[0])], dtype=object),
        "meta__label_source": np.array([str(args.label_source)], dtype=object),
        "train__x_t": x_train,
        "train__u_t": u_train,
        "train__x_tp1": y_train,
        "train__dx_t": dx_train,
        "train__dt": dt_train,
        "train__obs_dist_tp1": obs_dist_train,
        "train__obs_dist_valid": obs_valid_train,
        "val__x_t": x_val,
        "val__u_t": u_val,
        "val__x_tp1": y_val,
        "val__dx_t": dx_val,
        "val__dt": dt_val,
        "val__obs_dist_tp1": obs_dist_val,
        "val__obs_dist_valid": obs_valid_val,
        "test__x_t": x_test,
        "test__u_t": u_test,
        "test__x_tp1": y_test,
        "test__dx_t": dx_test,
        "test__dt": dt_test,
        "test__obs_dist_tp1": obs_dist_test,
        "test__obs_dist_valid": obs_valid_test,
        "stats__x_mean": x_mean.astype(np.float32),
        "stats__x_std": x_std.astype(np.float32),
        "stats__u_mean": u_mean.astype(np.float32),
        "stats__u_std": u_std.astype(np.float32),
        "stats__y_mean": y_mean.astype(np.float32),
        "stats__y_std": y_std.astype(np.float32),
        "stats__dx_mean": dx_mean.astype(np.float32),
        "stats__dx_std": dx_std.astype(np.float32),
    }

    np.savez_compressed(args.output, **save_dict)
    print(f"Saved ICODE training bundle to: {args.output}")
    print(f"Total transitions: {n} (train={train_n}, val={val_n}, test={test_n})")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge converted datasets, split train/val/test, and compute normalization stats.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Paths to converted .npz files.")
    parser.add_argument("--output", type=Path, required=True, help="Output bundled .npz path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--label-source",
        type=str,
        default="state_est",
        choices=("state_est", "gt_state"),
        help="Supervision target source for x(t+1): state_est (legacy) or gt_state.",
    )
    return parser


if __name__ == "__main__":
    build_dataset(build_argparser().parse_args())
