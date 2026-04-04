import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

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


def safe_std(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = a.std(axis=0)
    s[s < eps] = eps
    return s


def compute_train_sample_weights(
    near_clearance: np.ndarray,
    recover_flag: np.ndarray,
    wz_abs: np.ndarray,
    near_thr: float,
    high_wz_thr: float,
    w_near: float,
    w_recover: float,
    w_high_wz: float,
    w_max: float,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    near_mask = np.isfinite(near_clearance) & (near_clearance <= float(near_thr))
    recover_mask = recover_flag > 0.5
    high_wz_mask = wz_abs >= float(high_wz_thr)

    w = np.ones((near_clearance.shape[0],), dtype=np.float32)
    w *= np.where(near_mask, float(max(1.0, w_near)), 1.0).astype(np.float32)
    w *= np.where(recover_mask, float(max(1.0, w_recover)), 1.0).astype(np.float32)
    w *= np.where(high_wz_mask, float(max(1.0, w_high_wz)), 1.0).astype(np.float32)
    w = np.clip(w, 1.0, float(max(1.0, w_max))).astype(np.float32)

    summary = {
        "near_ratio": float(np.mean(near_mask.astype(np.float32))) if near_mask.size > 0 else 0.0,
        "recover_ratio": float(np.mean(recover_mask.astype(np.float32))) if recover_mask.size > 0 else 0.0,
        "high_wz_ratio": float(np.mean(high_wz_mask.astype(np.float32))) if high_wz_mask.size > 0 else 0.0,
        "weight_mean": float(np.mean(w)) if w.size > 0 else 1.0,
        "weight_max": float(np.max(w)) if w.size > 0 else 1.0,
    }
    tags = {
        "near_mask": near_mask.astype(np.uint8),
        "recover_mask": recover_mask.astype(np.uint8),
        "high_wz_mask": high_wz_mask.astype(np.uint8),
    }
    return w, summary, tags


def build_context_from_converted(data: np.lib.npyio.NpzFile, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if "icode__ctx_t" in data.files:
        ctx = data["icode__ctx_t"].astype(np.float32)
        if "meta__icode_context_fields" in data.files:
            fields = data["meta__icode_context_fields"].astype(object)
        else:
            fields = np.array([f"ctx_{i}" for i in range(ctx.shape[1])], dtype=object)
        return ctx, fields

    if "mppi__cost_context" in data.files:
        mctx = data["mppi__cost_context"].astype(np.float32)
        # [goal_rel(2), goal_dist, heading_err, depth(3), corridor, collision, touch, v, wz]
        if mctx.shape[1] >= 10:
            ctx = mctx[:, :10].astype(np.float32)
            fields = np.array(
                [
                    "goal_rel_body_x",
                    "goal_rel_body_y",
                    "goal_dist",
                    "goal_heading_err",
                    "depth_lf_min",
                    "depth_f_min",
                    "depth_rf_min",
                    "free_corridor_width",
                    "depth_collision_flag",
                    "touch_flag",
                ],
                dtype=object,
            )
            return ctx, fields

    if "derived__obs_gt_sector_min" in data.files:
        goal_rel = (
            data["derived__goal_rel_body_from_odom_yaw"].astype(np.float32)
            if "derived__goal_rel_body_from_odom_yaw" in data.files
            else np.zeros((n, 2), dtype=np.float32)
        )
        obs_sector = data["derived__obs_gt_sector_min"].astype(np.float32)
        obs_clear_min = (
            data["derived__obs_gt_clear_min"].astype(np.float32).reshape(n, 1)
            if "derived__obs_gt_clear_min" in data.files
            else np.min(obs_sector, axis=1, keepdims=True).astype(np.float32)
        )
        obs_bearing_sc = (
            data["derived__obs_gt_nearest_bearing_sin_cos"].astype(np.float32)
            if "derived__obs_gt_nearest_bearing_sin_cos" in data.files
            else np.zeros((n, 2), dtype=np.float32)
        )

        def get1(key: str) -> np.ndarray:
            if key in data.files:
                return data[key].astype(np.float32).reshape(n, 1)
            return np.zeros((n, 1), dtype=np.float32)

        ctx = np.concatenate(
            [
                goal_rel,
                get1("derived__goal_dist"),
                get1("derived__goal_heading_err_from_odom_yaw"),
                obs_sector,
                obs_clear_min,
                obs_bearing_sc,
                get1("derived__free_corridor_width"),
                get1("derived__depth_collision_flag"),
                get1("derived__touch_flag"),
            ],
            axis=1,
        ).astype(np.float32)
        fields = np.array(
            [
                "goal_rel_body_x",
                "goal_rel_body_y",
                "goal_dist",
                "goal_heading_err",
                "obs_sector_min_0",
                "obs_sector_min_1",
                "obs_sector_min_2",
                "obs_sector_min_3",
                "obs_sector_min_4",
                "obs_sector_min_5",
                "obs_sector_min_6",
                "obs_sector_min_7",
                "obs_clear_min_gt",
                "obs_nearest_bearing_sin",
                "obs_nearest_bearing_cos",
                "free_corridor_width",
                "depth_collision_flag",
                "touch_flag",
            ],
            dtype=object,
        )
        return ctx, fields

    # Last fallback: reconstruct from derived keys if present, else zeros.
    def get1(key: str) -> np.ndarray:
        if key in data.files:
            return data[key].astype(np.float32).reshape(n, 1)
        return np.zeros((n, 1), dtype=np.float32)

    if "derived__goal_rel_body_from_odom_yaw" in data.files:
        goal_rel = data["derived__goal_rel_body_from_odom_yaw"].astype(np.float32)
    else:
        goal_rel = np.zeros((n, 2), dtype=np.float32)
    if "derived__depth_sector_min" in data.files:
        depth3 = data["derived__depth_sector_min"].astype(np.float32)
    else:
        depth3 = np.zeros((n, 3), dtype=np.float32)

    ctx = np.concatenate(
        [
            goal_rel,
            get1("derived__goal_dist"),
            get1("derived__goal_heading_err_from_odom_yaw"),
            depth3,
            get1("derived__free_corridor_width"),
            get1("derived__depth_collision_flag"),
            get1("derived__touch_flag"),
        ],
        axis=1,
    ).astype(np.float32)
    fields = np.array(
        [
            "goal_rel_body_x",
            "goal_rel_body_y",
            "goal_dist",
            "goal_heading_err",
            "depth_lf_min",
            "depth_f_min",
            "depth_rf_min",
            "free_corridor_width",
            "depth_collision_flag",
            "touch_flag",
        ],
        dtype=object,
    )
    return ctx, fields


def load_transitions_from_converted(
    path: Path,
    label_source: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[str, object],
]:
    data = np.load(path, allow_pickle=True)
    meta = read_npz_meta(data)
    assert_meta_contract(meta)

    if "icode__x_t" not in data.files or "icode__u_t" not in data.files:
        raise KeyError(f"{path} missing icode__x_t or icode__u_t")
    if "raw__episode" not in data.files:
        raise KeyError(f"{path} missing raw__episode")

    x = data["icode__x_t"].astype(np.float32)
    u = data["icode__u_t"].astype(np.float32)
    ctx, ctx_fields = build_context_from_converted(data=data, n=x.shape[0])
    ep = data["raw__episode"].astype(np.int32)
    if "raw__recover_trigger" in data.files:
        recover_full = data["raw__recover_trigger"].astype(np.float32).reshape(-1)
    else:
        recover_full = np.zeros((x.shape[0],), dtype=np.float32)
    if "raw__post_collision_random_applied" in data.files:
        post_random_full = data["raw__post_collision_random_applied"].astype(np.float32).reshape(-1)
        recover_full = np.maximum(recover_full, post_random_full)

    if "derived__obs_gt_clear_min" in data.files:
        clear_full = data["derived__obs_gt_clear_min"].astype(np.float32).reshape(-1)
    elif "derived__depth_sector_min" in data.files:
        clear_full = np.min(data["derived__depth_sector_min"].astype(np.float32), axis=1).astype(np.float32)
    elif "icode__obs_dist_t" in data.files:
        clear_full = data["icode__obs_dist_t"].astype(np.float32).reshape(-1)
    else:
        clear_full = np.full((x.shape[0],), 10.0, dtype=np.float32)
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
            np.zeros((0, ctx.shape[1]), dtype=np.float32),
            np.zeros((0, x.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            ctx_fields,
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
    ctx_t = ctx[:-1][valid]
    x_tp1 = x_label[1:][valid]
    ep_t = ep[:-1][valid].astype(np.int64)
    recover_t = recover_full[:-1][valid].astype(np.float32)
    clear_t = clear_full[:-1][valid].astype(np.float32)
    wz_abs_t = np.abs(x_t[:, 4]).astype(np.float32) if x_t.shape[1] > 4 else np.zeros((x_t.shape[0],), dtype=np.float32)

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

    return x_t, u_t, ctx_t, x_tp1, dt, obs_dist_tp1, obs_dist_valid, ep_t, recover_t, wz_abs_t, clear_t, ctx_fields, meta


def build_dataset(args: argparse.Namespace) -> None:
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    ctxs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    dts: List[np.ndarray] = []
    obs_dists: List[np.ndarray] = []
    obs_valids: List[np.ndarray] = []
    recovers: List[np.ndarray] = []
    wz_abses: List[np.ndarray] = []
    clearances: List[np.ndarray] = []
    episode_ids: List[np.ndarray] = []
    metas: List[Mapping[str, object]] = []
    ctx_fields_ref: np.ndarray | None = None

    for file_idx, p in enumerate(args.inputs):
        x_t, u_t, ctx_t, x_tp1, dt, obs_dist_tp1, obs_dist_valid, ep_t, recover_t, wz_abs_t, clear_t, ctx_fields, meta = load_transitions_from_converted(
            p,
            label_source=str(args.label_source),
        )
        print(f"Loaded {p}: transitions={x_t.shape[0]}")
        metas.append(meta)
        if ctx_fields_ref is None:
            ctx_fields_ref = ctx_fields
        else:
            if ctx_fields_ref.shape != ctx_fields.shape or np.any(ctx_fields_ref != ctx_fields):
                raise RuntimeError(f"Context fields mismatch in {p}")
        if x_t.shape[0] == 0:
            continue
        xs.append(x_t)
        us.append(u_t)
        ctxs.append(ctx_t)
        ys.append(x_tp1)
        dts.append(dt)
        obs_dists.append(obs_dist_tp1)
        obs_valids.append(obs_dist_valid)
        recovers.append(recover_t)
        wz_abses.append(wz_abs_t)
        clearances.append(clear_t)
        episode_ids.append(ep_t + np.int64(file_idx) * np.int64(10_000_000))

    if not xs:
        raise RuntimeError("No valid transitions loaded from inputs.")
    merged_meta = merge_and_validate_meta(metas)
    if ctx_fields_ref is None:
        raise RuntimeError("No context fields inferred from inputs.")

    x_all = np.concatenate(xs, axis=0)
    u_all = np.concatenate(us, axis=0)
    ctx_all = np.concatenate(ctxs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    dt_all = np.concatenate(dts, axis=0)
    obs_dist_all = np.concatenate(obs_dists, axis=0)
    obs_valid_all = np.concatenate(obs_valids, axis=0)
    recover_all = np.concatenate(recovers, axis=0)
    wz_abs_all = np.concatenate(wz_abses, axis=0)
    clear_all = np.concatenate(clearances, axis=0)
    ep_all = np.concatenate(episode_ids, axis=0)
    dx_all = y_all - x_all

    # Automatic data cleaning (default on).
    fail_non_finite = (
        (~np.all(np.isfinite(x_all), axis=1))
        | (~np.all(np.isfinite(u_all), axis=1))
        | (~np.all(np.isfinite(ctx_all), axis=1))
        | (~np.all(np.isfinite(y_all), axis=1))
        | (~np.isfinite(dt_all))
    )
    fail_state_range = (
        (np.abs(x_all[:, 3]) > float(args.clean_max_abs_v))
        | (np.abs(y_all[:, 3]) > float(args.clean_max_abs_v))
        | (np.abs(x_all[:, 4]) > float(args.clean_max_abs_wz))
        | (np.abs(y_all[:, 4]) > float(args.clean_max_abs_wz))
        | (np.abs(x_all[:, 5]) > float(args.clean_max_abs_wheel))
        | (np.abs(x_all[:, 6]) > float(args.clean_max_abs_wheel))
        | (np.abs(y_all[:, 5]) > float(args.clean_max_abs_wheel))
        | (np.abs(y_all[:, 6]) > float(args.clean_max_abs_wheel))
    )
    fail_dt = np.zeros((dt_all.shape[0],), dtype=bool)
    if bool(args.clean_enforce_dt_positive):
        fail_dt = dt_all <= float(args.clean_min_dt)
    else:
        positive_dt = dt_all > 0.0
        fail_dt = positive_dt & ((dt_all < float(args.clean_min_dt)) | (dt_all > float(args.clean_max_dt)))
    fail_clear = ~np.isfinite(clear_all) | (clear_all < 0.0)

    clean_keep = ~(fail_non_finite | fail_state_range | fail_dt | fail_clear)
    n_raw = int(x_all.shape[0])
    n_kept = int(np.sum(clean_keep))
    if n_kept <= 0:
        raise RuntimeError("Data cleaning removed all transitions; please relax clean thresholds.")
    x_all = x_all[clean_keep]
    u_all = u_all[clean_keep]
    ctx_all = ctx_all[clean_keep]
    y_all = y_all[clean_keep]
    dt_all = dt_all[clean_keep]
    obs_dist_all = obs_dist_all[clean_keep]
    obs_valid_all = obs_valid_all[clean_keep]
    recover_all = recover_all[clean_keep]
    wz_abs_all = wz_abs_all[clean_keep]
    clear_all = clear_all[clean_keep]
    ep_all = ep_all[clean_keep]
    dx_all = dx_all[clean_keep]

    n = x_all.shape[0]
    rng = np.random.default_rng(args.seed)
    split_by = str(args.split_by)
    if split_by == "episode":
        eps = np.unique(ep_all)
        eps_perm = rng.permutation(eps)
        ep_n = eps_perm.shape[0]
        train_ep_n = int(ep_n * args.train_ratio)
        val_ep_n = int(ep_n * args.val_ratio)
        test_ep_n = ep_n - train_ep_n - val_ep_n

        train_eps = eps_perm[:train_ep_n]
        val_eps = eps_perm[train_ep_n : train_ep_n + val_ep_n]
        test_eps = eps_perm[train_ep_n + val_ep_n :]

        train_idx = np.where(np.isin(ep_all, train_eps))[0]
        val_idx = np.where(np.isin(ep_all, val_eps))[0]
        test_idx = np.where(np.isin(ep_all, test_eps))[0]
        split_episodes = {
            "train": int(train_ep_n),
            "val": int(val_ep_n),
            "test": int(test_ep_n),
            "total": int(ep_n),
        }
    else:
        perm = rng.permutation(n)
        train_n = int(n * args.train_ratio)
        val_n = int(n * args.val_ratio)
        test_n = n - train_n - val_n
        train_idx = perm[:train_n]
        val_idx = perm[train_n : train_n + val_n]
        test_idx = perm[train_n + val_n :]
        split_episodes = {
            "train": int(-1),
            "val": int(-1),
            "test": int(-1),
            "total": int(-1),
        }

    x_train = x_all[train_idx]
    u_train = u_all[train_idx]
    ctx_train = ctx_all[train_idx]
    y_train = y_all[train_idx]
    dx_train = dx_all[train_idx]
    dt_train = dt_all[train_idx]
    obs_dist_train = obs_dist_all[train_idx]
    obs_valid_train = obs_valid_all[train_idx]
    recover_train = recover_all[train_idx]
    wz_abs_train = wz_abs_all[train_idx]
    clear_train = clear_all[train_idx]

    x_val = x_all[val_idx]
    u_val = u_all[val_idx]
    ctx_val = ctx_all[val_idx]
    y_val = y_all[val_idx]
    dx_val = dx_all[val_idx]
    dt_val = dt_all[val_idx]
    obs_dist_val = obs_dist_all[val_idx]
    obs_valid_val = obs_valid_all[val_idx]
    recover_val = recover_all[val_idx]
    wz_abs_val = wz_abs_all[val_idx]
    clear_val = clear_all[val_idx]

    x_test = x_all[test_idx]
    u_test = u_all[test_idx]
    ctx_test = ctx_all[test_idx]
    y_test = y_all[test_idx]
    dx_test = dx_all[test_idx]
    dt_test = dt_all[test_idx]
    obs_dist_test = obs_dist_all[test_idx]
    obs_valid_test = obs_valid_all[test_idx]
    recover_test = recover_all[test_idx]
    wz_abs_test = wz_abs_all[test_idx]
    clear_test = clear_all[test_idx]

    sample_w_train, sample_w_summary, sample_w_tags = compute_train_sample_weights(
        near_clearance=clear_train,
        recover_flag=recover_train,
        wz_abs=wz_abs_train,
        near_thr=float(args.weight_near_clearance_thr),
        high_wz_thr=float(args.weight_high_wz_thr),
        w_near=float(args.weight_near_factor),
        w_recover=float(args.weight_recover_factor),
        w_high_wz=float(args.weight_high_wz_factor),
        w_max=float(args.weight_max),
    )

    x_mean = x_train.mean(axis=0)
    x_std = safe_std(x_train)
    u_mean = u_train.mean(axis=0)
    u_std = safe_std(u_train)
    ctx_mean = ctx_train.mean(axis=0)
    ctx_std = safe_std(ctx_train)
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
        "meta__num_total_before_clean": np.array([n_raw], dtype=np.int32),
        "meta__num_removed_by_clean": np.array([n_raw - n_kept], dtype=np.int32),
        "meta__clean_non_finite_removed": np.array([int(np.sum(fail_non_finite))], dtype=np.int32),
        "meta__clean_state_range_removed": np.array([int(np.sum(fail_state_range))], dtype=np.int32),
        "meta__clean_dt_removed": np.array([int(np.sum(fail_dt))], dtype=np.int32),
        "meta__clean_clearance_removed": np.array([int(np.sum(fail_clear))], dtype=np.int32),
        "meta__num_train": np.array([int(train_idx.shape[0])], dtype=np.int32),
        "meta__num_val": np.array([int(val_idx.shape[0])], dtype=np.int32),
        "meta__num_test": np.array([int(test_idx.shape[0])], dtype=np.int32),
        "meta__split_by": np.array([split_by], dtype=object),
        "meta__num_train_episodes": np.array([split_episodes["train"]], dtype=np.int32),
        "meta__num_val_episodes": np.array([split_episodes["val"]], dtype=np.int32),
        "meta__num_test_episodes": np.array([split_episodes["test"]], dtype=np.int32),
        "meta__num_total_episodes": np.array([split_episodes["total"]], dtype=np.int32),
        "meta__x_fields": np.array(["x_odom", "y_odom", "psi_odom", "v_body", "wz_body", "dqL", "dqR"], dtype=object),
        "meta__u_fields": np.array(["u_0", "u_1"], dtype=object),
        "meta__ctx_fields": np.array(ctx_fields_ref, dtype=object),
        "meta__state_convention_version": np.array([str(merged_meta["meta__state_convention_version"].reshape(-1)[0])], dtype=object),
        "meta__drive_sign": np.array([float(merged_meta["meta__drive_sign"].reshape(-1)[0])], dtype=np.float32),
        "meta__pose_source": np.array([str(merged_meta["meta__pose_source"].reshape(-1)[0])], dtype=object),
        "meta__heading_source": np.array([str(merged_meta["meta__heading_source"].reshape(-1)[0])], dtype=object),
        "meta__control_definition": np.array([str(merged_meta["meta__control_definition"].reshape(-1)[0])], dtype=object),
        "meta__label_source": np.array([str(args.label_source)], dtype=object),
        "meta__weight_near_clearance_thr": np.array([float(args.weight_near_clearance_thr)], dtype=np.float32),
        "meta__weight_high_wz_thr": np.array([float(args.weight_high_wz_thr)], dtype=np.float32),
        "meta__weight_near_factor": np.array([float(args.weight_near_factor)], dtype=np.float32),
        "meta__weight_recover_factor": np.array([float(args.weight_recover_factor)], dtype=np.float32),
        "meta__weight_high_wz_factor": np.array([float(args.weight_high_wz_factor)], dtype=np.float32),
        "meta__weight_max": np.array([float(args.weight_max)], dtype=np.float32),
        "meta__train_near_ratio": np.array([sample_w_summary["near_ratio"]], dtype=np.float32),
        "meta__train_recover_ratio": np.array([sample_w_summary["recover_ratio"]], dtype=np.float32),
        "meta__train_high_wz_ratio": np.array([sample_w_summary["high_wz_ratio"]], dtype=np.float32),
        "meta__train_sample_weight_mean": np.array([sample_w_summary["weight_mean"]], dtype=np.float32),
        "meta__train_sample_weight_max": np.array([sample_w_summary["weight_max"]], dtype=np.float32),
        "train__x_t": x_train,
        "train__u_t": u_train,
        "train__ctx_t": ctx_train,
        "train__x_tp1": y_train,
        "train__dx_t": dx_train,
        "train__dt": dt_train,
        "train__obs_dist_tp1": obs_dist_train,
        "train__obs_dist_valid": obs_valid_train,
        "train__sample_weight": sample_w_train.astype(np.float32),
        "train__tag_near": sample_w_tags["near_mask"],
        "train__tag_recover": sample_w_tags["recover_mask"],
        "train__tag_high_wz": sample_w_tags["high_wz_mask"],
        "train__recover_flag": recover_train.astype(np.float32),
        "train__wz_abs": wz_abs_train.astype(np.float32),
        "train__clearance_t": clear_train.astype(np.float32),
        "val__x_t": x_val,
        "val__u_t": u_val,
        "val__ctx_t": ctx_val,
        "val__x_tp1": y_val,
        "val__dx_t": dx_val,
        "val__dt": dt_val,
        "val__obs_dist_tp1": obs_dist_val,
        "val__obs_dist_valid": obs_valid_val,
        "val__recover_flag": recover_val.astype(np.float32),
        "val__wz_abs": wz_abs_val.astype(np.float32),
        "val__clearance_t": clear_val.astype(np.float32),
        "test__x_t": x_test,
        "test__u_t": u_test,
        "test__ctx_t": ctx_test,
        "test__x_tp1": y_test,
        "test__dx_t": dx_test,
        "test__dt": dt_test,
        "test__obs_dist_tp1": obs_dist_test,
        "test__obs_dist_valid": obs_valid_test,
        "test__recover_flag": recover_test.astype(np.float32),
        "test__wz_abs": wz_abs_test.astype(np.float32),
        "test__clearance_t": clear_test.astype(np.float32),
        "stats__x_mean": x_mean.astype(np.float32),
        "stats__x_std": x_std.astype(np.float32),
        "stats__u_mean": u_mean.astype(np.float32),
        "stats__u_std": u_std.astype(np.float32),
        "stats__ctx_mean": ctx_mean.astype(np.float32),
        "stats__ctx_std": ctx_std.astype(np.float32),
        "stats__y_mean": y_mean.astype(np.float32),
        "stats__y_std": y_std.astype(np.float32),
        "stats__dx_mean": dx_mean.astype(np.float32),
        "stats__dx_std": dx_std.astype(np.float32),
    }

    np.savez_compressed(args.output, **save_dict)
    print(f"Saved ICODE training bundle to: {args.output}")
    print(
        f"Total transitions: {n} (raw={n_raw}, removed={n_raw - n_kept}) "
        f"(train={train_idx.shape[0]}, val={val_idx.shape[0]}, test={test_idx.shape[0]}) "
        f"| split_by={split_by}"
    )
    print(
        "Train weighting: "
        f"near_ratio={sample_w_summary['near_ratio']:.3f}, "
        f"recover_ratio={sample_w_summary['recover_ratio']:.3f}, "
        f"high_wz_ratio={sample_w_summary['high_wz_ratio']:.3f}, "
        f"w_mean={sample_w_summary['weight_mean']:.3f}, "
        f"w_max={sample_w_summary['weight_max']:.3f}"
    )


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
    parser.add_argument(
        "--split-by",
        type=str,
        default="episode",
        choices=("episode", "transition"),
        help="Dataset split granularity.",
    )
    parser.add_argument("--clean-max-abs-v", type=float, default=3.5, help="Drop transitions with |v_body| above this.")
    parser.add_argument("--clean-max-abs-wz", type=float, default=8.0, help="Drop transitions with |wz_body| above this.")
    parser.add_argument("--clean-max-abs-wheel", type=float, default=55.0, help="Drop transitions with wheel speed magnitude above this.")
    parser.add_argument("--clean-min-dt", type=float, default=1e-4, help="Minimum valid positive dt.")
    parser.add_argument("--clean-max-dt", type=float, default=0.08, help="Maximum valid positive dt.")
    parser.add_argument(
        "--clean-enforce-dt-positive",
        action="store_true",
        help="If set, drop all dt<=clean-min-dt transitions (strict mode).",
    )
    parser.add_argument("--weight-near-clearance-thr", type=float, default=0.35)
    parser.add_argument("--weight-high-wz-thr", type=float, default=1.2)
    parser.add_argument("--weight-near-factor", type=float, default=2.0)
    parser.add_argument("--weight-recover-factor", type=float, default=2.5)
    parser.add_argument("--weight-high-wz-factor", type=float, default=1.8)
    parser.add_argument("--weight-max", type=float, default=4.0)
    return parser


if __name__ == "__main__":
    build_dataset(build_argparser().parse_args())
