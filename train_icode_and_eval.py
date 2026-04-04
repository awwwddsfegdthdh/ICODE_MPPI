import argparse
import copy
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from icode_dynamics import ICODEDynamics
from state_convention import assert_meta_contract, read_npz_meta


class TransitionDataset(Dataset):
    def __init__(
        self,
        x_t: np.ndarray,
        u_t: np.ndarray,
        x_tp1: np.ndarray,
        ctx_t: Optional[np.ndarray] = None,
        obs_dist_tp1: Optional[np.ndarray] = None,
        obs_dist_valid: Optional[np.ndarray] = None,
    ):
        self.x_t = torch.from_numpy(x_t.astype(np.float32))
        self.u_t = torch.from_numpy(u_t.astype(np.float32))
        self.x_tp1 = torch.from_numpy(x_tp1.astype(np.float32))
        self.ctx_t = None if ctx_t is None else torch.from_numpy(ctx_t.astype(np.float32))
        self.obs_dist_tp1 = None if obs_dist_tp1 is None else torch.from_numpy(obs_dist_tp1.astype(np.float32))
        self.obs_dist_valid = None if obs_dist_valid is None else torch.from_numpy(obs_dist_valid.astype(np.uint8))

    def __len__(self) -> int:
        return self.x_t.shape[0]

    def __getitem__(self, idx: int):
        ctx_i = self.ctx_t[idx] if self.ctx_t is not None else torch.zeros((0,), dtype=torch.float32)
        if self.obs_dist_tp1 is not None and self.obs_dist_valid is not None:
            return self.x_t[idx], self.u_t[idx], self.x_tp1[idx], ctx_i, self.obs_dist_tp1[idx], self.obs_dist_valid[idx]
        return self.x_t[idx], self.u_t[idx], self.x_tp1[idx], ctx_i


class RolloutWindowDataset(Dataset):
    def __init__(self, x0: np.ndarray, u_windows: np.ndarray, y_windows: np.ndarray, c_windows: np.ndarray):
        # x0: [N, D], u_windows: [N, K, U], y_windows: [N, K, D], c_windows: [N, K, C]
        self.x0 = torch.from_numpy(x0.astype(np.float32))
        self.u_windows = torch.from_numpy(u_windows.astype(np.float32))
        self.y_windows = torch.from_numpy(y_windows.astype(np.float32))
        self.c_windows = torch.from_numpy(c_windows.astype(np.float32))

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int):
        return self.x0[idx], self.u_windows[idx], self.y_windows[idx], self.c_windows[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def pick_amp_dtype(amp_mode: str):
    if amp_mode == "fp16":
        return torch.float16
    if amp_mode == "bf16":
        return torch.bfloat16
    return None


def build_amp_context(device: torch.device, amp_dtype):
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def build_state_loss_weights(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    w = torch.ones((int(args.state_dim),), dtype=torch.float32, device=device)
    if w.shape[0] > 2:
        w[2] = float(args.yaw_loss_weight)
    if w.shape[0] > 4:
        w[4] = float(args.wz_loss_weight)
    return w


def wrap_angle_torch(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def normalized_state_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_std: torch.Tensor,
    yaw_index: int = 2,
) -> torch.Tensor:
    err = pred - target
    if err.ndim == 2 and err.shape[1] > yaw_index:
        yaw_err = wrap_angle_torch(err[:, yaw_index : yaw_index + 1])
        left = err[:, :yaw_index] if yaw_index > 0 else None
        right = err[:, yaw_index + 1 :] if (yaw_index + 1) < err.shape[1] else None
        parts = []
        if left is not None and left.shape[1] > 0:
            parts.append(left)
        parts.append(yaw_err)
        if right is not None and right.shape[1] > 0:
            parts.append(right)
        err = torch.cat(parts, dim=1)
    return err / y_std


def obstacle_distance_loss(
    pred_obs: torch.Tensor,
    target_obs: torch.Tensor,
    near_weight: float,
    near_scale: float,
    use_log_target: bool,
) -> torch.Tensor:
    pred_c = torch.clamp(pred_obs, min=0.0)
    tgt_c = torch.clamp(target_obs, min=0.0)
    if use_log_target:
        pred_e = torch.log1p(pred_c)
        tgt_e = torch.log1p(tgt_c)
    else:
        pred_e = pred_c
        tgt_e = tgt_c
    base = F.smooth_l1_loss(pred_e, tgt_e, reduction="none")
    scale = max(float(near_scale), 1e-6)
    w = 1.0 + float(max(0.0, near_weight)) * torch.exp(-tgt_c / scale)
    return torch.sum(base * w) / torch.sum(w)


def effective_rollout_steps_for_epoch(args: argparse.Namespace, epoch: int, start_epoch: int) -> int:
    max_steps = int(max(1, args.rollout_steps))
    min_steps = int(np.clip(args.rollout_curriculum_min_steps, 1, max_steps))
    warmup = int(max(0, args.rollout_curriculum_warmup_epochs))
    if warmup <= 0 or min_steps >= max_steps:
        return max_steps
    rel = int(max(0, epoch - start_epoch))
    frac = float(np.clip(rel / max(warmup, 1), 0.0, 1.0))
    out = int(round(min_steps + frac * float(max_steps - min_steps)))
    return int(np.clip(out, min_steps, max_steps))


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


def build_gt_state_labels_or_none(d: np.lib.npyio.NpzFile, x_fallback: np.ndarray) -> Optional[np.ndarray]:
    required = ("gt__base_pos_gt", "gt__base_quat_gt", "gt__base_linvel_gt", "gt__base_angvel_gt")
    if not all(k in d.files for k in required):
        return None

    base_pos = d["gt__base_pos_gt"].astype(np.float32)
    base_quat = d["gt__base_quat_gt"].astype(np.float32)
    base_linvel = d["gt__base_linvel_gt"].astype(np.float32)
    base_angvel = d["gt__base_angvel_gt"].astype(np.float32)
    yaw_gt = quat_wxyz_to_yaw_batch(base_quat)
    v_body_gt = world_to_body_x(base_linvel[:, :2], yaw_gt)
    wz_gt = base_angvel[:, 2].astype(np.float32)
    if x_fallback.shape[1] >= 7:
        dq_l = x_fallback[:, 5].astype(np.float32)
        dq_r = x_fallback[:, 6].astype(np.float32)
    else:
        dq_l = np.zeros((x_fallback.shape[0],), dtype=np.float32)
        dq_r = np.zeros((x_fallback.shape[0],), dtype=np.float32)
    out = np.stack([base_pos[:, 0], base_pos[:, 1], yaw_gt, v_body_gt, wz_gt, dq_l, dq_r], axis=1).astype(np.float32)
    return out


def context_from_npz_or_none(d: np.lib.npyio.NpzFile, n_rows: int) -> Optional[np.ndarray]:
    if "icode__ctx_t" in d.files:
        return d["icode__ctx_t"].astype(np.float32)
    if "mppi__cost_context" in d.files:
        mctx = d["mppi__cost_context"].astype(np.float32)
        if mctx.shape[1] >= 10:
            return mctx[:, :10].astype(np.float32)

    if "derived__obs_gt_sector_min" in d.files:
        def get1(key: str) -> np.ndarray:
            if key in d.files:
                return d[key].astype(np.float32).reshape(n_rows, 1)
            return np.zeros((n_rows, 1), dtype=np.float32)

        goal_rel = d["derived__goal_rel_body_from_odom_yaw"].astype(np.float32) if "derived__goal_rel_body_from_odom_yaw" in d.files else np.zeros((n_rows, 2), dtype=np.float32)
        obs_sector = d["derived__obs_gt_sector_min"].astype(np.float32)
        obs_clear_min = (
            d["derived__obs_gt_clear_min"].astype(np.float32).reshape(n_rows, 1)
            if "derived__obs_gt_clear_min" in d.files
            else np.min(obs_sector, axis=1, keepdims=True).astype(np.float32)
        )
        obs_bearing_sc = (
            d["derived__obs_gt_nearest_bearing_sin_cos"].astype(np.float32)
            if "derived__obs_gt_nearest_bearing_sin_cos" in d.files
            else np.zeros((n_rows, 2), dtype=np.float32)
        )
        return np.concatenate(
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

    def get1(key: str) -> np.ndarray:
        if key in d.files:
            return d[key].astype(np.float32).reshape(n_rows, 1)
        return np.zeros((n_rows, 1), dtype=np.float32)

    if "derived__goal_rel_body_from_odom_yaw" in d.files or "derived__depth_sector_min" in d.files:
        goal_rel = d["derived__goal_rel_body_from_odom_yaw"].astype(np.float32) if "derived__goal_rel_body_from_odom_yaw" in d.files else np.zeros((n_rows, 2), dtype=np.float32)
        depth3 = d["derived__depth_sector_min"].astype(np.float32) if "derived__depth_sector_min" in d.files else np.zeros((n_rows, 3), dtype=np.float32)
        return np.concatenate(
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
    return None


def align_context_dim(ctx: Optional[np.ndarray], target_dim: int, n_rows: int) -> np.ndarray:
    if target_dim <= 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    if ctx is None:
        return np.zeros((n_rows, target_dim), dtype=np.float32)
    if ctx.shape[1] == target_dim:
        return ctx.astype(np.float32)
    if ctx.shape[1] > target_dim:
        return ctx[:, :target_dim].astype(np.float32)
    pad = np.zeros((ctx.shape[0], target_dim - ctx.shape[1]), dtype=np.float32)
    return np.concatenate([ctx.astype(np.float32), pad], axis=1)


def bundle_context_or_zeros(bundle: np.lib.npyio.NpzFile, split: str) -> np.ndarray:
    key = f"{split}__ctx_t"
    n = bundle[f"{split}__x_t"].shape[0]
    if key in bundle.files:
        return bundle[key].astype(np.float32)
    return np.zeros((n, 0), dtype=np.float32)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_std: torch.Tensor,
    amp_dtype,
    obs_model: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_norm_mse = 0.0
    total_n = 0
    total_obs_mse = 0.0
    total_obs_mae = 0.0
    total_obs_n = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                x_t, u_t, y_t, c_t, obs_t, obs_valid = batch
                obs_t = obs_t.to(device, non_blocking=True)
                obs_valid = obs_valid.to(device, non_blocking=True)
            else:
                x_t, u_t, y_t, c_t = batch
                obs_t = None
                obs_valid = None
            x_t = x_t.to(device, non_blocking=True)
            u_t = u_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)
            c_t = c_t.to(device, non_blocking=True)

            with build_amp_context(device, amp_dtype):
                pred = model(x_t, u_t, c_t)
                norm_err = normalized_state_error(pred=pred, target=y_t, y_std=y_std)
                norm_mse = torch.mean(norm_err * norm_err)

            mse = F.mse_loss(pred, y_t, reduction="sum")
            mae = F.l1_loss(pred, y_t, reduction="sum")

            bsz_numel = y_t.numel()
            total_norm_mse += float(norm_mse.item()) * bsz_numel
            total_mse += float(mse.item())
            total_mae += float(mae.item())
            total_n += bsz_numel

            if obs_model is not None and hasattr(obs_model, "predict_obstacle_distance") and obs_t is not None and obs_valid is not None:
                mask = obs_valid > 0
                if torch.any(mask):
                    pred_obs = obs_model.predict_obstacle_distance(x_t, u_t, c_t)
                    err = pred_obs[mask] - obs_t[mask]
                    total_obs_mse += float(torch.sum(err * err).item())
                    total_obs_mae += float(torch.sum(torch.abs(err)).item())
                    total_obs_n += int(mask.sum().item())

    avg_norm_mse = total_norm_mse / max(total_n, 1)
    avg_mse = total_mse / max(total_n, 1)
    out = {
        "norm_mse": avg_norm_mse,
        "norm_rmse": math.sqrt(avg_norm_mse),
        "mse": avg_mse,
        "rmse": math.sqrt(avg_mse),
        "mae": total_mae / max(total_n, 1),
    }
    if total_obs_n > 0:
        out["obs_dist_rmse"] = math.sqrt(total_obs_mse / total_obs_n)
        out["obs_dist_mae"] = total_obs_mae / total_obs_n
        out["obs_dist_n"] = float(total_obs_n)
    else:
        out["obs_dist_rmse"] = float("nan")
        out["obs_dist_mae"] = float("nan")
        out["obs_dist_n"] = 0.0
    return out


def load_gt_transition_eval(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    required = [
        "icode__x_t",
        "icode__u_t",
        "raw__episode",
        "gt__base_pos_gt",
        "gt__base_angvel_gt",
    ]
    for key in required:
        if key not in d.files:
            raise KeyError(f"{path} missing {key}")

    x = d["icode__x_t"].astype(np.float32)
    u = d["icode__u_t"].astype(np.float32)
    ctx = context_from_npz_or_none(d=d, n_rows=x.shape[0])
    ep = d["raw__episode"].astype(np.int32)
    if "icode__x_label_t" in d.files:
        x_label = d["icode__x_label_t"].astype(np.float32)
        if "icode__x_label_valid" in d.files:
            x_label_valid = d["icode__x_label_valid"].astype(np.uint8)
        else:
            x_label_valid = np.ones((x_label.shape[0],), dtype=np.uint8)
    else:
        x_label_from_gt = build_gt_state_labels_or_none(d=d, x_fallback=x)
        if x_label_from_gt is not None:
            x_label = x_label_from_gt
            x_label_valid = np.all(np.isfinite(x_label), axis=1).astype(np.uint8)
        else:
            x_label = x
            x_label_valid = np.ones((x.shape[0],), dtype=np.uint8)
    gt_pos = d["gt__base_pos_gt"].astype(np.float32)
    gt_ang = d["gt__base_angvel_gt"].astype(np.float32)

    if "raw__step" in d.files:
        step = d["raw__step"].astype(np.int32)
        valid = (ep[1:] == ep[:-1]) & (step[1:] == step[:-1] + 1)
    else:
        valid = ep[1:] == ep[:-1]
    valid = valid & (x_label_valid[:-1] > 0) & (x_label_valid[1:] > 0)

    x_t = x[:-1][valid]
    u_t = u[:-1][valid]
    c_t = ctx[:-1][valid] if ctx is not None else np.zeros((x_t.shape[0], 0), dtype=np.float32)
    x_tp1 = x_label[1:][valid]
    gt_xy_tp1 = gt_pos[1:, :2][valid]
    gt_wz_tp1 = gt_ang[1:, 2][valid]
    ep_tp1 = ep[1:][valid]
    return x_t, u_t, c_t, x_tp1, gt_xy_tp1, gt_wz_tp1, ep_tp1


def evaluate_against_groundtruth(
    model: torch.nn.Module,
    eval_paths: List[Path],
    device: torch.device,
    batch_size: int,
    rollout_horizon: int,
    context_dim: int,
    amp_dtype,
) -> Dict[str, float]:
    if len(eval_paths) == 0:
        return {
            "one_step_label_rmse": float("nan"),
            "one_step_gt_xy_rmse": float("nan"),
            "one_step_gt_xy_mae": float("nan"),
            "one_step_gt_wz_rmse": float("nan"),
            "one_step_gt_wz_mae": float("nan"),
            "rollout_ade": float("nan"),
            "rollout_fde": float("nan"),
            "rollout_episodes": 0,
        }

    x_list, u_list, c_list, y_list, gt_xy_list, gt_wz_list = [], [], [], [], [], []
    for p in eval_paths:
        x_t, u_t, c_t, x_tp1, gt_xy_tp1, gt_wz_tp1, _ = load_gt_transition_eval(p)
        x_list.append(x_t)
        u_list.append(u_t)
        c_list.append(align_context_dim(c_t, target_dim=context_dim, n_rows=x_t.shape[0]))
        y_list.append(x_tp1)
        gt_xy_list.append(gt_xy_tp1)
        gt_wz_list.append(gt_wz_tp1)

    x_all = np.concatenate(x_list, axis=0)
    u_all = np.concatenate(u_list, axis=0)
    c_all = np.concatenate(c_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    gt_xy_all = np.concatenate(gt_xy_list, axis=0)
    gt_wz_all = np.concatenate(gt_wz_list, axis=0)
    if x_all.shape[0] == 0:
        return {
            "one_step_label_rmse": float("nan"),
            "one_step_gt_xy_rmse": float("nan"),
            "one_step_gt_xy_mae": float("nan"),
            "one_step_gt_wz_rmse": float("nan"),
            "one_step_gt_wz_mae": float("nan"),
            "rollout_ade": float("nan"),
            "rollout_fde": float("nan"),
            "rollout_episodes": 0,
        }

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb = torch.from_numpy(x_all[i : i + batch_size]).to(device, non_blocking=True)
            ub = torch.from_numpy(u_all[i : i + batch_size]).to(device, non_blocking=True)
            cb = torch.from_numpy(c_all[i : i + batch_size]).to(device, non_blocking=True)
            with build_amp_context(device, amp_dtype):
                pb = model(xb, ub, cb).float().cpu().numpy()
            preds.append(pb)
    pred_all = np.concatenate(preds, axis=0)

    one_step_label_rmse = float(np.sqrt(np.mean((pred_all - y_all) ** 2)))
    one_step_xy_rmse = float(np.sqrt(np.mean((pred_all[:, :2] - gt_xy_all) ** 2)))
    one_step_xy_mae = float(np.mean(np.abs(pred_all[:, :2] - gt_xy_all)))
    one_step_wz_rmse = float(np.sqrt(np.mean((pred_all[:, 4] - gt_wz_all) ** 2)))
    one_step_wz_mae = float(np.mean(np.abs(pred_all[:, 4] - gt_wz_all)))

    ade_list = []
    fde_list = []
    with torch.no_grad():
        for p in eval_paths:
            d = np.load(p, allow_pickle=True)
            x = d["icode__x_t"].astype(np.float32)
            u = d["icode__u_t"].astype(np.float32)
            c = align_context_dim(
                context_from_npz_or_none(d=d, n_rows=x.shape[0]),
                target_dim=context_dim,
                n_rows=x.shape[0],
            )
            ep = d["raw__episode"].astype(np.int32)
            gt_pos = d["gt__base_pos_gt"].astype(np.float32)

            for ep_id in np.unique(ep):
                idx = np.where(ep == ep_id)[0]
                if idx.shape[0] < 3:
                    continue
                h = min(rollout_horizon, idx.shape[0] - 1)
                if h < 2:
                    continue

                x_pred = torch.from_numpy(x[idx[0] : idx[0] + 1]).to(device, non_blocking=True)
                errs = []
                for k in range(h):
                    u_k = torch.from_numpy(u[idx[k] : idx[k] + 1]).to(device, non_blocking=True)
                    c_k = torch.from_numpy(c[idx[k] : idx[k] + 1]).to(device, non_blocking=True)
                    with build_amp_context(device, amp_dtype):
                        x_pred = model(x_pred, u_k, c_k)
                    gt_xy = gt_pos[idx[k + 1], :2]
                    pred_xy = x_pred[0, :2].float().cpu().numpy()
                    errs.append(float(np.linalg.norm(pred_xy - gt_xy)))

                ade_list.append(float(np.mean(errs)))
                fde_list.append(float(errs[-1]))

    return {
        "one_step_label_rmse": one_step_label_rmse,
        "one_step_gt_xy_rmse": one_step_xy_rmse,
        "one_step_gt_xy_mae": one_step_xy_mae,
        "one_step_gt_wz_rmse": one_step_wz_rmse,
        "one_step_gt_wz_mae": one_step_wz_mae,
        "rollout_ade": float(np.mean(ade_list)) if ade_list else float("nan"),
        "rollout_fde": float(np.mean(fde_list)) if fde_list else float("nan"),
        "rollout_episodes": int(len(ade_list)),
    }


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, device: torch.device):
    opt_name = args.optimizer.lower()
    use_fused = bool(args.fused_optimizer and device.type == "cuda")

    if opt_name == "adamw":
        kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": (0.9, 0.95),
        }
        if use_fused:
            kwargs["fused"] = True
        return torch.optim.AdamW(model.parameters(), **kwargs)

    if opt_name == "adam":
        kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": (0.9, 0.95),
        }
        if use_fused:
            kwargs["fused"] = True
        return torch.optim.Adam(model.parameters(), **kwargs)

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_rollout_windows_from_paths(
    paths: Sequence[Path],
    rollout_steps: int,
    max_windows: int,
    seed: int,
    supervision_source: str,
    context_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0_windows = []
    u_windows = []
    y_windows = []
    c_windows = []

    for p in paths:
        d = np.load(p, allow_pickle=True)
        required = ["icode__x_t", "icode__u_t", "raw__episode"]
        for key in required:
            if key not in d.files:
                raise KeyError(f"{p} missing {key}")

        x = d["icode__x_t"].astype(np.float32)
        c = align_context_dim(
            ctx=context_from_npz_or_none(d=d, n_rows=x.shape[0]),
            target_dim=context_dim,
            n_rows=x.shape[0],
        )
        if supervision_source == "gt_state":
            if "icode__x_label_t" in d.files:
                y_ref = d["icode__x_label_t"].astype(np.float32)
                if "icode__x_label_valid" in d.files:
                    y_valid = d["icode__x_label_valid"].astype(np.uint8)
                else:
                    y_valid = np.ones((x.shape[0],), dtype=np.uint8)
            else:
                y_ref_from_gt = build_gt_state_labels_or_none(d=d, x_fallback=x)
                if y_ref_from_gt is None:
                    raise KeyError(
                        f"{p} missing icode__x_label_t and GT keys needed for rollout supervision gt_state"
                    )
                y_ref = y_ref_from_gt
                y_valid = np.all(np.isfinite(y_ref), axis=1).astype(np.uint8)
        else:
            y_ref = x
            y_valid = np.ones((x.shape[0],), dtype=np.uint8)
        u = d["icode__u_t"].astype(np.float32)
        ep = d["raw__episode"].astype(np.int32)
        step = d["raw__step"].astype(np.int32) if "raw__step" in d.files else None

        for ep_id in np.unique(ep):
            idx = np.where(ep == ep_id)[0]
            if idx.shape[0] <= rollout_steps:
                continue

            if step is not None:
                seq_idx = [idx[0]]
                for ii in idx[1:]:
                    if step[ii] == step[seq_idx[-1]] + 1:
                        seq_idx.append(ii)
                    else:
                        if len(seq_idx) > rollout_steps:
                            seq_arr = np.array(seq_idx, dtype=np.int32)
                            for s in range(0, len(seq_arr) - rollout_steps):
                                w = seq_arr[s : s + rollout_steps + 1]
                                if np.all(y_valid[w] > 0):
                                    x0_windows.append(x[w[0]])
                                    u_windows.append(u[w[:-1]])
                                    y_windows.append(y_ref[w[1:]])
                                    c_windows.append(c[w[:-1]])
                        seq_idx = [ii]
                if len(seq_idx) > rollout_steps:
                    seq_arr = np.array(seq_idx, dtype=np.int32)
                    for s in range(0, len(seq_arr) - rollout_steps):
                        w = seq_arr[s : s + rollout_steps + 1]
                        if np.all(y_valid[w] > 0):
                            x0_windows.append(x[w[0]])
                            u_windows.append(u[w[:-1]])
                            y_windows.append(y_ref[w[1:]])
                            c_windows.append(c[w[:-1]])
            else:
                for s in range(0, idx.shape[0] - rollout_steps):
                    w = idx[s : s + rollout_steps + 1]
                    if np.all(y_valid[w] > 0):
                        x0_windows.append(x[w[0]])
                        u_windows.append(u[w[:-1]])
                        y_windows.append(y_ref[w[1:]])
                        c_windows.append(c[w[:-1]])

    if not x0_windows:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0, rollout_steps, 2), dtype=np.float32),
            np.zeros((0, rollout_steps, 7), dtype=np.float32),
            np.zeros((0, rollout_steps, 0), dtype=np.float32),
        )

    x0_arr = np.stack(x0_windows, axis=0)
    u_arr = np.stack(u_windows, axis=0)
    y_arr = np.stack(y_windows, axis=0)
    c_arr = np.stack(c_windows, axis=0)

    if max_windows > 0 and x0_arr.shape[0] > max_windows:
        rng = np.random.default_rng(seed)
        choose = rng.choice(x0_arr.shape[0], size=max_windows, replace=False)
        x0_arr = x0_arr[choose]
        u_arr = u_arr[choose]
        y_arr = y_arr[choose]
        c_arr = c_arr[choose]

    return x0_arr, u_arr, y_arr, c_arr


def cycle_loader(loader: DataLoader) -> Iterator:
    while True:
        for item in loader:
            yield item


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, p in model_params.items():
        ema_params[name].mul_(decay).add_(p.detach(), alpha=(1.0 - decay))

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, b in model_buffers.items():
        ema_buffers[name].copy_(b.detach())


def load_checkpoint_if_requested(
    args: argparse.Namespace,
    save_dir: Path,
    raw_model: torch.nn.Module,
    ema_model: Optional[torch.nn.Module],
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> Tuple[int, float, Optional[Path]]:
    if not args.resume:
        return 1, float("inf"), None

    ckpt_path: Optional[Path]
    if args.resume_path:
        ckpt_path = Path(args.resume_path)
    else:
        ckpt_path = save_dir / "icode_best.pt"

    if ckpt_path is None or not ckpt_path.exists():
        return 1, float("inf"), None

    ckpt = torch.load(ckpt_path, map_location=device)
    allow_aux_mismatch = bool(getattr(raw_model, "predict_obs_distance_enabled", False))
    try:
        load_res = raw_model.load_state_dict(ckpt["model_state_dict"], strict=(not allow_aux_mismatch))
        if allow_aux_mismatch and hasattr(load_res, "missing_keys"):
            if load_res.missing_keys or load_res.unexpected_keys:
                print(
                    "Resume with non-strict loading due to auxiliary obstacle-distance head. "
                    f"missing={list(load_res.missing_keys)} unexpected={list(load_res.unexpected_keys)}"
                )
    except RuntimeError as exc:
        print(f"Resume skipped due to checkpoint/model mismatch: {exc}")
        return 1, float("inf"), None

    if ema_model is not None and "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        try:
            ema_model.load_state_dict(ckpt["ema_state_dict"], strict=True)
        except RuntimeError as exc:
            print(f"EMA resume skipped due to mismatch: {exc}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_norm_mse = float(ckpt.get("best_val_norm_mse", float("inf")))

    if args.resume_optimizer:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    print(f"Resumed from checkpoint: {ckpt_path} (start_epoch={start_epoch}, best_val_norm_mse={best_val_norm_mse:.6f})")
    return start_epoch, best_val_norm_mse, ckpt_path


def compute_rollout_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    u_window: torch.Tensor,
    y_window: torch.Tensor,
    c_window: torch.Tensor,
    y_std: torch.Tensor,
    state_loss_weights: torch.Tensor,
    late_bias: float,
    terminal_weight: float,
    effective_steps: int,
) -> torch.Tensor:
    # x0: [B, D], u_window: [B, K, U], y_window: [B, K, D], c_window: [B, K, C]
    x_pred = x0
    losses = []
    steps = int(max(1, min(u_window.shape[1], effective_steps)))
    for t in range(steps):
        x_pred = model(x_pred, u_window[:, t, :], c_window[:, t, :])
        target = y_window[:, t, :]
        norm_err = normalized_state_error(pred=x_pred, target=target, y_std=y_std)
        weighted = (norm_err * norm_err) * state_loss_weights
        per_sample = torch.sum(weighted, dim=1) / torch.sum(state_loss_weights)
        losses.append(torch.mean(per_sample))
    losses_t = torch.stack(losses)
    weights = torch.linspace(1.0, max(late_bias, 1.0), steps=steps, device=losses_t.device, dtype=losses_t.dtype)
    if steps > 0 and terminal_weight != 1.0:
        weights[-1] = weights[-1] * terminal_weight
    return torch.sum(losses_t * weights) / torch.sum(weights)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    bundle = np.load(args.bundle, allow_pickle=True)
    bundle_meta = read_npz_meta(bundle)
    assert_meta_contract(bundle_meta)
    has_obs_dist_labels = all(
        k in bundle.files
        for k in (
            "train__obs_dist_tp1",
            "train__obs_dist_valid",
            "val__obs_dist_tp1",
            "val__obs_dist_valid",
            "test__obs_dist_tp1",
            "test__obs_dist_valid",
        )
    )
    predict_obs_distance_active = bool(args.predict_obs_distance and has_obs_dist_labels)
    if args.predict_obs_distance and (not has_obs_dist_labels):
        print(
            "Obstacle-distance prediction requested but bundle has no obs distance labels. "
            "Fallback to dynamics-only training."
        )

    train_ctx = bundle_context_or_zeros(bundle=bundle, split="train")
    val_ctx = bundle_context_or_zeros(bundle=bundle, split="val")
    test_ctx = bundle_context_or_zeros(bundle=bundle, split="test")
    context_dim = int(train_ctx.shape[1])
    if val_ctx.shape[1] != context_dim or test_ctx.shape[1] != context_dim:
        raise RuntimeError(
            f"Context dim mismatch across splits: train={context_dim}, val={val_ctx.shape[1]}, test={test_ctx.shape[1]}"
        )

    train_ds = TransitionDataset(
        bundle["train__x_t"],
        bundle["train__u_t"],
        bundle["train__x_tp1"],
        ctx_t=train_ctx,
        obs_dist_tp1=(bundle["train__obs_dist_tp1"] if predict_obs_distance_active else None),
        obs_dist_valid=(bundle["train__obs_dist_valid"] if predict_obs_distance_active else None),
    )
    val_ds = TransitionDataset(
        bundle["val__x_t"],
        bundle["val__u_t"],
        bundle["val__x_tp1"],
        ctx_t=val_ctx,
        obs_dist_tp1=(bundle["val__obs_dist_tp1"] if predict_obs_distance_active else None),
        obs_dist_valid=(bundle["val__obs_dist_valid"] if predict_obs_distance_active else None),
    )
    test_ds = TransitionDataset(
        bundle["test__x_t"],
        bundle["test__u_t"],
        bundle["test__x_tp1"],
        ctx_t=test_ctx,
        obs_dist_tp1=(bundle["test__obs_dist_tp1"] if predict_obs_distance_active else None),
        obs_dist_valid=(bundle["test__obs_dist_valid"] if predict_obs_distance_active else None),
    )

    device = pick_device(args.device)
    amp_dtype = pick_amp_dtype(args.amp)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_sampler = None
    train_shuffle = True
    if bool(args.train_weighted_sampling) and ("train__sample_weight" in bundle.files):
        w_np = bundle["train__sample_weight"].astype(np.float64).reshape(-1)
        w_np = np.where(np.isfinite(w_np) & (w_np > 0.0), w_np, 1.0)
        if w_np.shape[0] == len(train_ds):
            train_sampler = WeightedRandomSampler(
                weights=torch.from_numpy(w_np),
                num_samples=int(w_np.shape[0]),
                replacement=True,
            )
            train_shuffle = False
            print(
                "Train weighted sampler enabled: "
                f"n={w_np.shape[0]}, mean={float(np.mean(w_np)):.3f}, max={float(np.max(w_np)):.3f}"
            )
        else:
            print(
                "Train weighted sampler requested but weight length mismatch: "
                f"weights={w_np.shape[0]} vs dataset={len(train_ds)}. Fallback to shuffle."
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    rollout_loader = None
    rollout_iter = None
    if args.rollout_loss_weight > 0 and args.rollout_steps >= 2:
        rollout_paths = [Path(p) for p in args.rollout_train_inputs]
        x0w, uw, yw, cw = build_rollout_windows_from_paths(
            paths=rollout_paths,
            rollout_steps=args.rollout_steps,
            max_windows=args.rollout_max_windows,
            seed=args.seed,
            supervision_source=str(args.rollout_supervision_source),
            context_dim=context_dim,
        )
        if x0w.shape[0] > 0:
            rollout_ds = RolloutWindowDataset(x0w, uw, yw, cw)
            rollout_loader = DataLoader(
                rollout_ds,
                batch_size=args.rollout_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=False,
            )
            rollout_iter = cycle_loader(rollout_loader)
            print(
                f"Rollout loss enabled: windows={x0w.shape[0]}, steps={args.rollout_steps}, "
                f"batch={args.rollout_batch_size}, weight={args.rollout_loss_weight}, "
                f"supervision={args.rollout_supervision_source}"
            )
        else:
            print(
                "Rollout loss requested but no valid windows found "
                f"(source={args.rollout_supervision_source}). Fallback to one-step only."
            )

    raw_model = ICODEDynamics(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        context_dim=context_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dt=args.dt,
        predict_obs_distance=predict_obs_distance_active,
        obs_head_hidden_dim=args.obs_head_hidden_dim,
    ).to(device)
    x_mean_np = bundle["stats__x_mean"].astype(np.float32) if "stats__x_mean" in bundle.files else np.zeros((args.state_dim,), dtype=np.float32)
    x_std_np = bundle["stats__x_std"].astype(np.float32) if "stats__x_std" in bundle.files else np.ones((args.state_dim,), dtype=np.float32)
    u_mean_np = bundle["stats__u_mean"].astype(np.float32) if "stats__u_mean" in bundle.files else np.zeros((args.action_dim,), dtype=np.float32)
    u_std_np = bundle["stats__u_std"].astype(np.float32) if "stats__u_std" in bundle.files else np.ones((args.action_dim,), dtype=np.float32)
    ctx_mean_np = bundle["stats__ctx_mean"].astype(np.float32) if "stats__ctx_mean" in bundle.files else np.zeros((context_dim,), dtype=np.float32)
    ctx_std_np = bundle["stats__ctx_std"].astype(np.float32) if "stats__ctx_std" in bundle.files else np.ones((context_dim,), dtype=np.float32)
    raw_model.set_input_normalization(
        x_mean=x_mean_np,
        x_std=x_std_np,
        u_mean=u_mean_np,
        u_std=u_std_np,
        ctx_mean=ctx_mean_np,
        ctx_std=ctx_std_np,
        enabled=bool(args.input_normalize),
    )

    model = raw_model
    if args.compile_mode != "none":
        model = torch.compile(raw_model, mode=args.compile_mode)

    ema_model = None
    if 0.0 < args.ema_decay < 1.0:
        ema_model = copy.deepcopy(raw_model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    optimizer = build_optimizer(args, raw_model, device)

    scheduler = None
    scheduler_step_per_batch = False
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.lr * 0.05,
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=max(args.epochs, 1),
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=0.1,
            anneal_strategy="cos",
        )
        scheduler_step_per_batch = True

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    y_std_np = bundle["stats__y_std"].astype(np.float32)
    y_std = torch.from_numpy(y_std_np).to(device)
    state_loss_weights = build_state_loss_weights(args=args, device=device)

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val_norm_mse, resumed_from = load_checkpoint_if_requested(
        args=args,
        save_dir=save_dir,
        raw_model=raw_model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    history = {
        "epoch": [],
        "train_norm_mse": [],
        "train_rollout_mse": [],
        "train_obs_dist_loss": [],
        "val_norm_mse": [],
        "val_rmse": [],
        "val_obs_dist_rmse": [],
        "train_time_s": [],
        "samples_per_s": [],
        "lr": [],
    }

    best_path = save_dir / "icode_best.pt"
    last_path = save_dir / "icode_last.pt"

    def make_checkpoint_payload(epoch_idx: int) -> Dict[str, object]:
        return {
            "epoch": int(epoch_idx),
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "config": {
                "state_dim": args.state_dim,
                "action_dim": args.action_dim,
                "context_dim": int(context_dim),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dt": args.dt,
                "yaw_loss_weight": float(args.yaw_loss_weight),
                "wz_loss_weight": float(args.wz_loss_weight),
                "predict_obs_distance": bool(predict_obs_distance_active),
                "obs_head_hidden_dim": int(args.obs_head_hidden_dim),
                "input_norm_enabled": bool(args.input_normalize),
                "x_mean": x_mean_np.astype(np.float32).tolist(),
                "x_std": x_std_np.astype(np.float32).tolist(),
                "u_mean": u_mean_np.astype(np.float32).tolist(),
                "u_std": u_std_np.astype(np.float32).tolist(),
                "ctx_mean": ctx_mean_np.astype(np.float32).tolist(),
                "ctx_std": ctx_std_np.astype(np.float32).tolist(),
                "state_convention_version": str(bundle_meta["meta__state_convention_version"].reshape(-1)[0]),
                "drive_sign": float(bundle_meta["meta__drive_sign"].reshape(-1)[0]),
                "pose_source": str(bundle_meta["meta__pose_source"].reshape(-1)[0]),
                "heading_source": str(bundle_meta["meta__heading_source"].reshape(-1)[0]),
                "control_definition": str(bundle_meta["meta__control_definition"].reshape(-1)[0]),
            },
            "best_val_norm_mse": best_val_norm_mse,
            "device": str(device),
        }

    if resumed_from is not None and not best_path.exists():
        torch.save(make_checkpoint_payload(max(start_epoch - 1, 0)), best_path)
        print(f"Bootstrapped local best checkpoint from resume source: {best_path}")

    use_amp = amp_dtype is not None and device.type == "cuda"
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if args.max_train_seconds > 0 and (time.time() - start_time) >= args.max_train_seconds:
            print(f"Reached time budget: {args.max_train_seconds:.1f}s at epoch {epoch - 1}.")
            break

        epoch_t0 = time.time()
        model.train()
        total_one_step_loss_sum = 0.0
        total_rollout_loss_sum = 0.0
        total_obs_loss_sum = 0.0
        total_obs_count = 0
        total_samples = 0
        epoch_rollout_steps = effective_rollout_steps_for_epoch(args=args, epoch=epoch, start_epoch=start_epoch)

        for batch in train_loader:
            if len(batch) == 6:
                x_t, u_t, y_t, c_t, obs_t, obs_valid = batch
                obs_t = obs_t.to(device, non_blocking=True)
                obs_valid = obs_valid.to(device, non_blocking=True)
            else:
                x_t, u_t, y_t, c_t = batch
                obs_t = None
                obs_valid = None
            x_t = x_t.to(device, non_blocking=True)
            u_t = u_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)
            c_t = c_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with build_amp_context(device, amp_dtype) if use_amp else nullcontext():
                pred = model(x_t, u_t, c_t)
                norm_err = normalized_state_error(pred=pred, target=y_t, y_std=y_std)
                one_step_weighted = (norm_err * norm_err) * state_loss_weights
                one_step_loss = torch.mean(torch.sum(one_step_weighted, dim=1) / torch.sum(state_loss_weights))

                rollout_loss = torch.tensor(0.0, device=device)
                if rollout_iter is not None:
                    x0w, uw, yw, cw = next(rollout_iter)
                    x0w = x0w.to(device, non_blocking=True)
                    uw = uw.to(device, non_blocking=True)
                    yw = yw.to(device, non_blocking=True)
                    cw = cw.to(device, non_blocking=True)
                    rollout_loss = compute_rollout_loss(
                        model,
                        x0w,
                        uw,
                        yw,
                        cw,
                        y_std=y_std,
                        state_loss_weights=state_loss_weights,
                        late_bias=args.rollout_late_bias,
                        terminal_weight=args.rollout_terminal_weight,
                        effective_steps=epoch_rollout_steps,
                    )

                obs_dist_loss = torch.tensor(0.0, device=device)
                obs_count = 0
                if predict_obs_distance_active and obs_t is not None and obs_valid is not None:
                    mask = obs_valid > 0
                    if torch.any(mask):
                        pred_obs = raw_model.predict_obstacle_distance(x_t, u_t, c_t)
                        obs_dist_loss = obstacle_distance_loss(
                            pred_obs=pred_obs[mask],
                            target_obs=obs_t[mask],
                            near_weight=float(args.obs_dist_near_weight),
                            near_scale=float(args.obs_dist_near_scale),
                            use_log_target=bool(args.obs_dist_use_log_target),
                        )
                        obs_count = int(mask.sum().item())

                loss = (
                    one_step_loss
                    + args.rollout_loss_weight * rollout_loss
                    + args.obs_dist_loss_weight * obs_dist_loss
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            if ema_model is not None:
                ema_update(ema_model=ema_model, model=raw_model, decay=args.ema_decay)

            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

            bsz = x_t.shape[0]
            total_one_step_loss_sum += float(one_step_loss.item()) * bsz
            total_rollout_loss_sum += float(rollout_loss.item()) * bsz
            if obs_count > 0:
                total_obs_loss_sum += float(obs_dist_loss.item()) * obs_count
                total_obs_count += obs_count
            total_samples += bsz

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        train_norm_mse = total_one_step_loss_sum / max(total_samples, 1)
        train_rollout_mse = total_rollout_loss_sum / max(total_samples, 1)
        train_obs_dist_loss = total_obs_loss_sum / max(total_obs_count, 1)
        eval_model = ema_model if (ema_model is not None and args.ema_eval) else model
        eval_obs_model = ema_model if (ema_model is not None and args.ema_eval) else raw_model
        val_metrics = evaluate_loader(
            eval_model,
            val_loader,
            device,
            y_std=y_std,
            amp_dtype=amp_dtype,
            obs_model=(eval_obs_model if predict_obs_distance_active else None),
        )

        epoch_time = time.time() - epoch_t0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_norm_mse"].append(train_norm_mse)
        history["train_rollout_mse"].append(train_rollout_mse)
        history["train_obs_dist_loss"].append(train_obs_dist_loss)
        history["val_norm_mse"].append(val_metrics["norm_mse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_obs_dist_rmse"].append(float(val_metrics.get("obs_dist_rmse", float("nan"))))
        history["train_time_s"].append(epoch_time)
        history["samples_per_s"].append(total_samples / max(epoch_time, 1e-6))
        history["lr"].append(cur_lr)

        if epoch % args.log_interval == 0 or epoch == start_epoch:
            print(
                f"Epoch {epoch:04d} | train_norm_mse={train_norm_mse:.6f} "
                f"| train_rollout_mse={train_rollout_mse:.6f} "
                f"| train_obs={train_obs_dist_loss:.6f} "
                f"| rollout_steps_eff={int(epoch_rollout_steps)} "
                f"| val_norm_mse={val_metrics['norm_mse']:.6f} | val_rmse={val_metrics['rmse']:.6f} "
                f"| val_obs_rmse={val_metrics.get('obs_dist_rmse', float('nan')):.6f} "
                f"| samples/s={history['samples_per_s'][-1]:.1f} | lr={cur_lr:.2e}"
            )

        if val_metrics["norm_mse"] < best_val_norm_mse:
            best_val_norm_mse = val_metrics["norm_mse"]
            torch.save(make_checkpoint_payload(epoch), best_path)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Best val_norm_mse={best_val_norm_mse:.6f}")

    last_epoch = history["epoch"][-1] if history["epoch"] else max(start_epoch - 1, 0)
    torch.save(make_checkpoint_payload(last_epoch), last_path)

    eval_ckpt_path: Optional[Path] = None
    if best_path.exists():
        eval_ckpt_path = best_path
    elif resumed_from is not None and Path(resumed_from).exists():
        eval_ckpt_path = Path(resumed_from)
        print(f"No local best checkpoint saved; fallback to resumed checkpoint for evaluation: {eval_ckpt_path}")
    elif last_path.exists():
        eval_ckpt_path = last_path
        print(f"No best checkpoint available; fallback to last checkpoint for evaluation: {eval_ckpt_path}")

    if eval_ckpt_path is not None:
        ckpt = torch.load(eval_ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if ema_model is not None and "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
            ema_model.load_state_dict(ckpt["ema_state_dict"])

    eval_model = ema_model if (ema_model is not None and args.ema_eval) else model
    eval_obs_model = ema_model if (ema_model is not None and args.ema_eval) else raw_model
    test_metrics = evaluate_loader(
        eval_model,
        test_loader,
        device,
        y_std=y_std,
        amp_dtype=amp_dtype,
        obs_model=(eval_obs_model if predict_obs_distance_active else None),
    )

    gt_eval_paths = [Path(p) for p in args.gt_eval_inputs]
    gt_metrics = evaluate_against_groundtruth(
        model=eval_model,
        eval_paths=gt_eval_paths,
        device=device,
        batch_size=args.batch_size,
        rollout_horizon=args.rollout_horizon,
        context_dim=context_dim,
        amp_dtype=amp_dtype,
    )

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_weighted_sampling": bool(args.train_weighted_sampling),
        "context_dim": int(context_dim),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "amp": args.amp,
        "compile_mode": args.compile_mode,
        "input_normalize": bool(args.input_normalize),
        "rollout_loss_weight": args.rollout_loss_weight,
        "rollout_steps": args.rollout_steps,
        "rollout_curriculum_min_steps": int(args.rollout_curriculum_min_steps),
        "rollout_curriculum_warmup_epochs": int(args.rollout_curriculum_warmup_epochs),
        "rollout_late_bias": args.rollout_late_bias,
        "rollout_terminal_weight": args.rollout_terminal_weight,
        "rollout_supervision_source": args.rollout_supervision_source,
        "yaw_loss_weight": float(args.yaw_loss_weight),
        "wz_loss_weight": float(args.wz_loss_weight),
        "predict_obs_distance": bool(predict_obs_distance_active),
        "obs_dist_loss_weight": float(args.obs_dist_loss_weight),
        "obs_dist_near_weight": float(args.obs_dist_near_weight),
        "obs_dist_near_scale": float(args.obs_dist_near_scale),
        "obs_dist_use_log_target": bool(args.obs_dist_use_log_target),
        "ema_decay": args.ema_decay,
        "ema_eval": args.ema_eval,
        "best_val_norm_mse": best_val_norm_mse,
        "resumed_from": str(resumed_from) if resumed_from is not None else "",
        "eval_checkpoint": str(eval_ckpt_path) if eval_ckpt_path is not None else "",
        "test": test_metrics,
        "groundtruth_eval": gt_metrics,
        "history": history,
        "bundle": str(args.bundle),
        "gt_eval_inputs": [str(p) for p in gt_eval_paths],
    }

    metrics_path = save_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        save_dir / "history.npz",
        epoch=np.array(history["epoch"], dtype=np.int32),
        train_norm_mse=np.array(history["train_norm_mse"], dtype=np.float32),
        train_rollout_mse=np.array(history["train_rollout_mse"], dtype=np.float32),
        train_obs_dist_loss=np.array(history["train_obs_dist_loss"], dtype=np.float32),
        val_norm_mse=np.array(history["val_norm_mse"], dtype=np.float32),
        val_rmse=np.array(history["val_rmse"], dtype=np.float32),
        val_obs_dist_rmse=np.array(history["val_obs_dist_rmse"], dtype=np.float32),
        train_time_s=np.array(history["train_time_s"], dtype=np.float32),
        samples_per_s=np.array(history["samples_per_s"], dtype=np.float32),
        lr=np.array(history["lr"], dtype=np.float32),
    )

    print("Final test metrics:", test_metrics)
    print("Groundtruth metrics:", gt_metrics)
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved last checkpoint: {last_path}")
    print(f"Saved metrics: {metrics_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ICODE dynamics and evaluate against collected groundtruth.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_icode_train_bundle_304k.npz"),
    )
    parser.add_argument(
        "--gt-eval-inputs",
        type=Path,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--rollout-train-inputs",
        type=Path,
        nargs="*",
        default=[],
    )

    parser.add_argument("--save-dir", type=Path, default=Path("/home/wmh/ICODE/domo/ICODE_MPPI/runs/icode_e1_opt"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--batch-size", type=int, default=98304)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--train-weighted-sampling", dest="train_weighted_sampling", action="store_true")
    parser.add_argument("--no-train-weighted-sampling", dest="train_weighted_sampling", action="store_false")
    parser.set_defaults(train_weighted_sampling=True)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=("adamw", "adam"))
    parser.add_argument("--fused-optimizer", dest="fused_optimizer", action="store_true")
    parser.add_argument("--no-fused-optimizer", dest="fused_optimizer", action="store_false")
    parser.set_defaults(fused_optimizer=True)

    parser.add_argument("--scheduler", type=str, default="onecycle", choices=("none", "cosine", "onecycle"))
    parser.add_argument("--amp", type=str, default="bf16", choices=("off", "fp16", "bf16"))
    parser.add_argument("--compile-mode", type=str, default="default", choices=("none", "default", "reduce-overhead", "max-autotune"))

    parser.add_argument("--hidden-dim", type=int, default=640)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--predict-obs-distance", dest="predict_obs_distance", action="store_true")
    parser.add_argument("--no-predict-obs-distance", dest="predict_obs_distance", action="store_false")
    parser.set_defaults(predict_obs_distance=True)
    parser.add_argument("--input-normalize", dest="input_normalize", action="store_true")
    parser.add_argument("--no-input-normalize", dest="input_normalize", action="store_false")
    parser.set_defaults(input_normalize=True)
    parser.add_argument("--obs-head-hidden-dim", type=int, default=128)
    parser.add_argument("--obs-dist-loss-weight", type=float, default=0.20)
    parser.add_argument("--obs-dist-near-weight", type=float, default=2.0)
    parser.add_argument("--obs-dist-near-scale", type=float, default=0.35)
    parser.add_argument("--obs-dist-use-log-target", dest="obs_dist_use_log_target", action="store_true")
    parser.add_argument("--no-obs-dist-use-log-target", dest="obs_dist_use_log_target", action="store_false")
    parser.set_defaults(obs_dist_use_log_target=True)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--yaw-loss-weight", type=float, default=1.0)
    parser.add_argument("--wz-loss-weight", type=float, default=1.0)

    parser.add_argument("--rollout-loss-weight", type=float, default=0.15)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=8192)
    parser.add_argument("--rollout-max-windows", type=int, default=120000)
    parser.add_argument("--rollout-curriculum-min-steps", type=int, default=3)
    parser.add_argument("--rollout-curriculum-warmup-epochs", type=int, default=80)
    parser.add_argument(
        "--rollout-supervision-source",
        type=str,
        default="state_est",
        choices=("state_est", "gt_state"),
        help="Target source for rollout supervision windows.",
    )
    parser.add_argument("--rollout-late-bias", type=float, default=1.5)
    parser.add_argument("--rollout-terminal-weight", type=float, default=2.0)

    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--resume-path", type=str, default="")
    parser.add_argument("--resume-optimizer", dest="resume_optimizer", action="store_true")
    parser.add_argument("--no-resume-optimizer", dest="resume_optimizer", action="store_false")
    parser.set_defaults(resume_optimizer=True)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260312)
    parser.add_argument("--rollout-horizon", type=int, default=50)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-eval", dest="ema_eval", action="store_true")
    parser.add_argument("--no-ema-eval", dest="ema_eval", action="store_false")
    parser.set_defaults(ema_eval=True)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
