from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class DepthGateReport:
    allow_depth: bool
    reason: str
    valid_ratio: float
    sector_delta_mean: float
    front_delta: float


class DepthConsistencyGate:
    def __init__(
        self,
        window: int = 20,
        min_valid_ratio: float = 0.015,
        max_sector_delta_mean: float = 0.70,
        max_front_delta: float = 1.10,
        max_fail_ratio: float = 0.55,
    ):
        self.window = max(3, int(window))
        self.min_valid_ratio = float(max(0.0, min_valid_ratio))
        self.max_sector_delta_mean = float(max(0.05, max_sector_delta_mean))
        self.max_front_delta = float(max(0.05, max_front_delta))
        self.max_fail_ratio = float(np.clip(max_fail_ratio, 0.0, 1.0))
        self._hist = deque(maxlen=self.window)

    @staticmethod
    def _sectors_from_depth(depth_image: np.ndarray, default_far: float) -> np.ndarray:
        d = np.asarray(depth_image, dtype=np.float32)
        if d.ndim != 2 or d.size <= 0:
            return np.array([default_far, default_far, default_far], dtype=np.float32)
        h, w = d.shape
        r0 = int(0.45 * (h - 1))
        r1 = int(0.65 * (h - 1))
        band = d[r0 : r1 + 1, :]
        if band.size <= 0:
            return np.array([default_far, default_far, default_far], dtype=np.float32)
        cols = band.shape[1]
        c0 = 0
        c1 = max(1, cols // 3)
        c2 = max(c1 + 1, (2 * cols) // 3)
        c3 = cols
        out = np.array([default_far, default_far, default_far], dtype=np.float32)
        for i, seg in enumerate((band[:, c0:c1], band[:, c1:c2], band[:, c2:c3])):
            valid = np.isfinite(seg) & (seg > 0.0)
            if np.any(valid):
                out[i] = float(np.nanmin(seg[valid]))
        return out

    def update(
        self,
        depth_image: np.ndarray | None,
        lidar_triplet: np.ndarray,
        default_far: float,
        min_valid_depth: float,
    ) -> DepthGateReport:
        if depth_image is None:
            return DepthGateReport(
                allow_depth=False,
                reason="depth_none",
                valid_ratio=0.0,
                sector_delta_mean=float("inf"),
                front_delta=float("inf"),
            )

        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim != 2 or depth.size <= 0:
            return DepthGateReport(
                allow_depth=False,
                reason="depth_invalid_shape",
                valid_ratio=0.0,
                sector_delta_mean=float("inf"),
                front_delta=float("inf"),
            )

        valid = np.isfinite(depth) & (depth >= float(min_valid_depth))
        valid_ratio = float(np.mean(valid)) if depth.size > 0 else 0.0

        lidar = np.asarray(lidar_triplet, dtype=np.float32).reshape(-1)
        if lidar.shape[0] != 3:
            lidar = np.array([default_far, default_far, default_far], dtype=np.float32)
        bad = (~np.isfinite(lidar)) | (lidar <= 0.0)
        lidar[bad] = float(default_far)

        sec = self._sectors_from_depth(depth, default_far=float(default_far))
        sec = np.clip(sec, 0.0, float(default_far))
        delta = np.abs(sec - lidar)
        sector_delta_mean = float(np.mean(delta))
        front_delta = float(delta[1])

        fail = (
            valid_ratio < self.min_valid_ratio
            or sector_delta_mean > self.max_sector_delta_mean
            or front_delta > self.max_front_delta
        )
        self._hist.append(1 if fail else 0)
        fail_ratio = float(np.mean(self._hist)) if len(self._hist) > 0 else 0.0

        allow_depth = bool((not fail) and (fail_ratio <= self.max_fail_ratio))
        if valid_ratio < self.min_valid_ratio:
            reason = "valid_ratio_low"
        elif sector_delta_mean > self.max_sector_delta_mean:
            reason = "sector_delta_too_large"
        elif front_delta > self.max_front_delta:
            reason = "front_delta_too_large"
        elif fail_ratio > self.max_fail_ratio:
            reason = "historical_fail_ratio_high"
        else:
            reason = "ok"

        return DepthGateReport(
            allow_depth=allow_depth,
            reason=reason,
            valid_ratio=valid_ratio,
            sector_delta_mean=sector_delta_mean,
            front_delta=front_delta,
        )

    def as_dict(self, report: DepthGateReport) -> Dict[str, float | str | bool]:
        return {
            "allow_depth": bool(report.allow_depth),
            "reason": str(report.reason),
            "valid_ratio": float(report.valid_ratio),
            "sector_delta_mean": float(report.sector_delta_mean),
            "front_delta": float(report.front_delta),
        }
