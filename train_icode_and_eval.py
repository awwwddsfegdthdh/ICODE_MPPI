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
from torch.utils.data import DataLoader, Dataset

from icode_dynamics import ICODEDynamics
from state_convention import assert_meta_contract, read_npz_meta


class TransitionDataset(Dataset):
    def __init__(self, x_t: np.ndarray, u_t: np.ndarray, x_tp1: np.ndarray):
        self.x_t = torch.from_numpy(x_t.astype(np.float32))
        self.u_t = torch.from_numpy(u_t.astype(np.float32))
        self.x_tp1 = torch.from_numpy(x_tp1.astype(np.float32))

    def __len__(self) -> int:
        return self.x_t.shape[0]

    def __getitem__(self, idx: int):
        return self.x_t[idx], self.u_t[idx], self.x_tp1[idx]


class RolloutWindowDataset(Dataset):
    def __init__(self, x_windows: np.ndarray, u_windows: np.ndarray):
        # x_windows: [N, K+1, D], u_windows: [N, K, U]
        self.x_windows = torch.from_numpy(x_windows.astype(np.float32))
        self.u_windows = torch.from_numpy(u_windows.astype(np.float32))

    def __len__(self) -> int:
        return self.x_windows.shape[0]

    def __getitem__(self, idx: int):
        return self.x_windows[idx], self.u_windows[idx]


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


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_std: torch.Tensor,
    amp_dtype,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_norm_mse = 0.0
    total_n = 0

    with torch.no_grad():
        for x_t, u_t, y_t in loader:
            x_t = x_t.to(device, non_blocking=True)
            u_t = u_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)

            with build_amp_context(device, amp_dtype):
                pred = model(x_t, u_t)
                norm_err = (pred - y_t) / y_std
                norm_mse = torch.mean(norm_err * norm_err)

            mse = F.mse_loss(pred, y_t, reduction="sum")
            mae = F.l1_loss(pred, y_t, reduction="sum")

            bsz_numel = y_t.numel()
            total_norm_mse += float(norm_mse.item()) * bsz_numel
            total_mse += float(mse.item())
            total_mae += float(mae.item())
            total_n += bsz_numel

    avg_norm_mse = total_norm_mse / max(total_n, 1)
    avg_mse = total_mse / max(total_n, 1)
    return {
        "norm_mse": avg_norm_mse,
        "norm_rmse": math.sqrt(avg_norm_mse),
        "mse": avg_mse,
        "rmse": math.sqrt(avg_mse),
        "mae": total_mae / max(total_n, 1),
    }


def load_gt_transition_eval(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    ep = d["raw__episode"].astype(np.int32)
    gt_pos = d["gt__base_pos_gt"].astype(np.float32)
    gt_ang = d["gt__base_angvel_gt"].astype(np.float32)

    if "raw__step" in d.files:
        step = d["raw__step"].astype(np.int32)
        valid = (ep[1:] == ep[:-1]) & (step[1:] == step[:-1] + 1)
    else:
        valid = ep[1:] == ep[:-1]

    x_t = x[:-1][valid]
    u_t = u[:-1][valid]
    x_tp1 = x[1:][valid]
    gt_xy_tp1 = gt_pos[1:, :2][valid]
    gt_wz_tp1 = gt_ang[1:, 2][valid]
    ep_tp1 = ep[1:][valid]
    return x_t, u_t, x_tp1, gt_xy_tp1, gt_wz_tp1, ep_tp1


def evaluate_against_groundtruth(
    model: torch.nn.Module,
    eval_paths: List[Path],
    device: torch.device,
    batch_size: int,
    rollout_horizon: int,
    amp_dtype,
) -> Dict[str, float]:
    x_list, u_list, y_list, gt_xy_list, gt_wz_list = [], [], [], [], []
    for p in eval_paths:
        x_t, u_t, x_tp1, gt_xy_tp1, gt_wz_tp1, _ = load_gt_transition_eval(p)
        x_list.append(x_t)
        u_list.append(u_t)
        y_list.append(x_tp1)
        gt_xy_list.append(gt_xy_tp1)
        gt_wz_list.append(gt_wz_tp1)

    x_all = np.concatenate(x_list, axis=0)
    u_all = np.concatenate(u_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    gt_xy_all = np.concatenate(gt_xy_list, axis=0)
    gt_wz_all = np.concatenate(gt_wz_list, axis=0)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb = torch.from_numpy(x_all[i : i + batch_size]).to(device, non_blocking=True)
            ub = torch.from_numpy(u_all[i : i + batch_size]).to(device, non_blocking=True)
            with build_amp_context(device, amp_dtype):
                pb = model(xb, ub).float().cpu().numpy()
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
                    with build_amp_context(device, amp_dtype):
                        x_pred = model(x_pred, u_k)
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
) -> Tuple[np.ndarray, np.ndarray]:
    x_windows = []
    u_windows = []

    for p in paths:
        d = np.load(p, allow_pickle=True)
        required = ["icode__x_t", "icode__u_t", "raw__episode"]
        for key in required:
            if key not in d.files:
                raise KeyError(f"{p} missing {key}")

        x = d["icode__x_t"].astype(np.float32)
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
                                x_windows.append(x[w])
                                u_windows.append(u[w[:-1]])
                        seq_idx = [ii]
                if len(seq_idx) > rollout_steps:
                    seq_arr = np.array(seq_idx, dtype=np.int32)
                    for s in range(0, len(seq_arr) - rollout_steps):
                        w = seq_arr[s : s + rollout_steps + 1]
                        x_windows.append(x[w])
                        u_windows.append(u[w[:-1]])
            else:
                for s in range(0, idx.shape[0] - rollout_steps):
                    w = idx[s : s + rollout_steps + 1]
                    x_windows.append(x[w])
                    u_windows.append(u[w[:-1]])

    if not x_windows:
        return np.zeros((0, rollout_steps + 1, 7), dtype=np.float32), np.zeros((0, rollout_steps, 2), dtype=np.float32)

    x_arr = np.stack(x_windows, axis=0)
    u_arr = np.stack(u_windows, axis=0)

    if max_windows > 0 and x_arr.shape[0] > max_windows:
        rng = np.random.default_rng(seed)
        choose = rng.choice(x_arr.shape[0], size=max_windows, replace=False)
        x_arr = x_arr[choose]
        u_arr = u_arr[choose]

    return x_arr, u_arr


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
    try:
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
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
    x_window: torch.Tensor,
    u_window: torch.Tensor,
    y_std: torch.Tensor,
    late_bias: float,
    terminal_weight: float,
) -> torch.Tensor:
    # x_window: [B, K+1, D], u_window: [B, K, U]
    x_pred = x_window[:, 0, :]
    losses = []
    steps = u_window.shape[1]
    for t in range(steps):
        x_pred = model(x_pred, u_window[:, t, :])
        target = x_window[:, t + 1, :]
        norm_err = (x_pred - target) / y_std
        losses.append(torch.mean(norm_err * norm_err))
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
    train_ds = TransitionDataset(bundle["train__x_t"], bundle["train__u_t"], bundle["train__x_tp1"])
    val_ds = TransitionDataset(bundle["val__x_t"], bundle["val__u_t"], bundle["val__x_tp1"])
    test_ds = TransitionDataset(bundle["test__x_t"], bundle["test__u_t"], bundle["test__x_tp1"])

    device = pick_device(args.device)
    amp_dtype = pick_amp_dtype(args.amp)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
        xw, uw = build_rollout_windows_from_paths(
            paths=rollout_paths,
            rollout_steps=args.rollout_steps,
            max_windows=args.rollout_max_windows,
            seed=args.seed,
        )
        if xw.shape[0] > 0:
            rollout_ds = RolloutWindowDataset(xw, uw)
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
                f"Rollout loss enabled: windows={xw.shape[0]}, steps={args.rollout_steps}, "
                f"batch={args.rollout_batch_size}, weight={args.rollout_loss_weight}"
            )
        else:
            print("Rollout loss requested but no valid windows found. Fallback to one-step only.")

    raw_model = ICODEDynamics(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dt=args.dt,
    ).to(device)

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
        "val_norm_mse": [],
        "val_rmse": [],
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
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dt": args.dt,
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
        total_samples = 0

        for x_t, u_t, y_t in train_loader:
            x_t = x_t.to(device, non_blocking=True)
            u_t = u_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with build_amp_context(device, amp_dtype) if use_amp else nullcontext():
                pred = model(x_t, u_t)
                norm_err = (pred - y_t) / y_std
                one_step_loss = torch.mean(norm_err * norm_err)

                rollout_loss = torch.tensor(0.0, device=device)
                if rollout_iter is not None:
                    xw, uw = next(rollout_iter)
                    xw = xw.to(device, non_blocking=True)
                    uw = uw.to(device, non_blocking=True)
                    rollout_loss = compute_rollout_loss(
                        model,
                        xw,
                        uw,
                        y_std=y_std,
                        late_bias=args.rollout_late_bias,
                        terminal_weight=args.rollout_terminal_weight,
                    )

                loss = one_step_loss + args.rollout_loss_weight * rollout_loss

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
            total_samples += bsz

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        train_norm_mse = total_one_step_loss_sum / max(total_samples, 1)
        train_rollout_mse = total_rollout_loss_sum / max(total_samples, 1)
        eval_model = ema_model if (ema_model is not None and args.ema_eval) else model
        val_metrics = evaluate_loader(eval_model, val_loader, device, y_std=y_std, amp_dtype=amp_dtype)

        epoch_time = time.time() - epoch_t0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_norm_mse"].append(train_norm_mse)
        history["train_rollout_mse"].append(train_rollout_mse)
        history["val_norm_mse"].append(val_metrics["norm_mse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["train_time_s"].append(epoch_time)
        history["samples_per_s"].append(total_samples / max(epoch_time, 1e-6))
        history["lr"].append(cur_lr)

        if epoch % args.log_interval == 0 or epoch == start_epoch:
            print(
                f"Epoch {epoch:04d} | train_norm_mse={train_norm_mse:.6f} "
                f"| train_rollout_mse={train_rollout_mse:.6f} "
                f"| val_norm_mse={val_metrics['norm_mse']:.6f} | val_rmse={val_metrics['rmse']:.6f} "
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
    test_metrics = evaluate_loader(eval_model, test_loader, device, y_std=y_std, amp_dtype=amp_dtype)

    gt_eval_paths = [Path(p) for p in args.gt_eval_inputs]
    gt_metrics = evaluate_against_groundtruth(
        model=eval_model,
        eval_paths=gt_eval_paths,
        device=device,
        batch_size=args.batch_size,
        rollout_horizon=args.rollout_horizon,
        amp_dtype=amp_dtype,
    )

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "amp": args.amp,
        "compile_mode": args.compile_mode,
        "rollout_loss_weight": args.rollout_loss_weight,
        "rollout_steps": args.rollout_steps,
        "rollout_late_bias": args.rollout_late_bias,
        "rollout_terminal_weight": args.rollout_terminal_weight,
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
        val_norm_mse=np.array(history["val_norm_mse"], dtype=np.float32),
        val_rmse=np.array(history["val_rmse"], dtype=np.float32),
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
        nargs="+",
        default=[
            Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_train_converted_60k.npz"),
            Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_train_converted_240k.npz"),
            Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_depth_converted_5k.npz"),
        ],
    )
    parser.add_argument(
        "--rollout-train-inputs",
        type=Path,
        nargs="+",
        default=[
            Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_train_converted_60k.npz"),
            Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_train_converted_240k.npz"),
            Path("/home/wmh/ICODE/domo/ICODE_MPPI/datasets/e1_depth_converted_5k.npz"),
        ],
    )

    parser.add_argument("--save-dir", type=Path, default=Path("/home/wmh/ICODE/domo/ICODE_MPPI/runs/icode_e1_opt"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--batch-size", type=int, default=98304)
    parser.add_argument("--num-workers", type=int, default=6)
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
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--rollout-loss-weight", type=float, default=0.15)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=8192)
    parser.add_argument("--rollout-max-windows", type=int, default=120000)
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
