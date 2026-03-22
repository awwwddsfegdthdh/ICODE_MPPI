from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from state_convention import body_to_world


@dataclass
class RayObservation:
    angle_rad: float
    distance: float
    max_range: float
    hit: bool
    source: str


class LidarRayModel:
    def __init__(self, angles_deg: Sequence[float] = (-30.0, 0.0, 30.0)):
        self.angles_deg = tuple(float(a) for a in angles_deg)

    def build_rays(
        self,
        ranges: np.ndarray,
        default_far: float,
        min_range: float,
        max_range: float,
    ) -> List[RayObservation]:
        rr = np.asarray(ranges, dtype=np.float32).reshape(-1)
        if rr.shape[0] != len(self.angles_deg):
            rr = np.full((len(self.angles_deg),), float(default_far), dtype=np.float32)
        rays: List[RayObservation] = []
        for i, a_deg in enumerate(self.angles_deg):
            d = float(rr[i])
            if not np.isfinite(d) or d <= 0.0:
                d = float(default_far)
            d = float(np.clip(d, min_range, max_range))
            hit = d < (max_range - 1e-4)
            rays.append(
                RayObservation(
                    angle_rad=float(np.deg2rad(a_deg)),
                    distance=d,
                    max_range=float(max_range),
                    hit=hit,
                    source="lidar",
                )
            )
        return rays


class DepthRayModel:
    def __init__(
        self,
        hfov_deg: float = 86.0,
        sample_count: int = 21,
        row_lo: float = 0.45,
        row_hi: float = 0.65,
    ):
        self.hfov_deg = float(hfov_deg)
        self.sample_count = max(3, int(sample_count))
        self.row_lo = float(np.clip(row_lo, 0.0, 1.0))
        self.row_hi = float(np.clip(row_hi, 0.0, 1.0))

    def build_rays(
        self,
        depth_image: np.ndarray | None,
        min_range: float,
        max_range: float,
    ) -> List[RayObservation]:
        if depth_image is None:
            return []
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim != 2 or depth.size <= 0:
            return []

        h, w = depth.shape
        r0 = int(np.clip(round(self.row_lo * (h - 1)), 0, h - 1))
        r1 = int(np.clip(round(self.row_hi * (h - 1)), 0, h - 1))
        if r1 < r0:
            r0, r1 = r1, r0
        band = depth[r0 : r1 + 1, :]
        if band.size <= 0:
            return []

        d_row = np.nanmedian(band, axis=0)
        idx = np.linspace(0, w - 1, num=self.sample_count, dtype=np.int32)
        hfov = np.deg2rad(self.hfov_deg)
        rays: List[RayObservation] = []
        for cidx in idx:
            d = float(d_row[cidx])
            if not np.isfinite(d) or d <= min_range:
                continue
            d = float(np.clip(d, min_range, max_range))
            u = (float(cidx) / max(1.0, float(w - 1))) * 2.0 - 1.0
            ang = 0.5 * hfov * u
            hit = d < (max_range - 1e-4)
            rays.append(
                RayObservation(
                    angle_rad=float(ang),
                    distance=d,
                    max_range=float(max_range),
                    hit=hit,
                    source="depth",
                )
            )
        return rays


def fuse_rays_to_hits_free(
    lidar_rays: Iterable[RayObservation],
    depth_rays: Iterable[RayObservation] | None = None,
    prefer_nearer: bool = True,
    angle_bin_deg: float = 2.0,
) -> List[RayObservation]:
    all_rays = list(lidar_rays)
    if depth_rays is not None:
        all_rays.extend(list(depth_rays))
    if len(all_rays) <= 0:
        return []

    if not prefer_nearer:
        return all_rays

    bin_rad = np.deg2rad(max(0.2, float(angle_bin_deg)))
    bins: dict[int, RayObservation] = {}
    for r in all_rays:
        bi = int(np.round(float(r.angle_rad) / bin_rad))
        old = bins.get(bi)
        if old is None or float(r.distance) < float(old.distance):
            bins[bi] = r
    return [bins[k] for k in sorted(bins.keys())]


def rays_to_obstacles_world(
    base_xy: np.ndarray,
    base_yaw: float,
    rays: Iterable[RayObservation],
    obs_radius: float,
    min_range: float,
    max_range: float,
) -> np.ndarray:
    base = np.asarray(base_xy, dtype=np.float32).reshape(2)
    pts: List[np.ndarray] = []
    for r in rays:
        d = float(np.clip(r.distance, min_range, max_range))
        if d <= min_range or d >= max_range:
            continue
        if not bool(r.hit):
            continue
        p_body = np.array([d * np.cos(r.angle_rad), d * np.sin(r.angle_rad)], dtype=np.float32)
        p_world = base + body_to_world(p_body, float(base_yaw))
        pts.append(p_world)

    if len(pts) <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)
    q = max(1e-3, float(obs_radius))
    bins = np.round(arr / q, decimals=0).astype(np.int32)
    _, uniq_idx = np.unique(bins, axis=0, return_index=True)
    arr = arr[np.sort(uniq_idx)]
    rr = np.full((arr.shape[0], 1), float(obs_radius), dtype=np.float32)
    return np.concatenate([arr, rr], axis=1)
