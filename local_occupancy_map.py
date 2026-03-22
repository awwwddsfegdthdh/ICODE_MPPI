from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from obs_geometry import RayObservation
from state_convention import body_to_world


@dataclass
class OccupancyMeta:
    min_xy: np.ndarray
    resolution: float


class LocalOccupancyMap:
    def __init__(
        self,
        x_range: Tuple[float, float] = (-1.0, 4.0),
        y_range: Tuple[float, float] = (-2.0, 2.0),
        resolution: float = 0.05,
        logodds_hit: float = 0.9,
        logodds_free: float = -0.35,
        logodds_decay: float = 0.98,
        logodds_min: float = -4.0,
        logodds_max: float = 4.0,
    ):
        self.x_min = float(min(x_range[0], x_range[1]))
        self.x_max = float(max(x_range[0], x_range[1]))
        self.y_min = float(min(y_range[0], y_range[1]))
        self.y_max = float(max(y_range[0], y_range[1]))
        self.resolution = float(max(1e-3, resolution))
        self.logodds_hit = float(logodds_hit)
        self.logodds_free = float(logodds_free)
        self.logodds_decay = float(np.clip(logodds_decay, 0.8, 1.0))
        self.logodds_min = float(logodds_min)
        self.logodds_max = float(logodds_max)

        self.nx = int(np.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.ny = int(np.ceil((self.y_max - self.y_min) / self.resolution)) + 1
        self.logodds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.observed = np.zeros((self.ny, self.nx), dtype=np.float32)

    def reset(self) -> None:
        self.logodds.fill(0.0)
        self.observed.fill(0.0)

    def decay(self) -> None:
        self.logodds *= self.logodds_decay
        self.observed *= self.logodds_decay

    def as_meta(self) -> OccupancyMeta:
        return OccupancyMeta(
            min_xy=np.array([self.x_min, self.y_min], dtype=np.float32),
            resolution=float(self.resolution),
        )

    def world_to_rc(self, xy: np.ndarray) -> tuple[int, int] | None:
        x = float(xy[0])
        y = float(xy[1])
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return None
        c = int(np.round((x - self.x_min) / self.resolution))
        r = int(np.round((y - self.y_min) / self.resolution))
        c = int(np.clip(c, 0, self.nx - 1))
        r = int(np.clip(r, 0, self.ny - 1))
        return r, c

    def rc_to_world(self, r: int, c: int) -> np.ndarray:
        x = self.x_min + float(c) * self.resolution
        y = self.y_min + float(r) * self.resolution
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        r = r0
        c = c0
        if dc >= dr:
            err = dc // 2
            while c != c1:
                out.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr // 2
            while r != r1:
                out.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        out.append((r1, c1))
        return out

    def update_from_rays(
        self,
        base_xy: np.ndarray,
        base_yaw: float,
        rays: Iterable[RayObservation],
        min_range: float,
        max_range: float,
    ) -> None:
        self.decay()
        base = np.asarray(base_xy, dtype=np.float32).reshape(2)
        rc0 = self.world_to_rc(base)
        if rc0 is None:
            return

        for ray in rays:
            d = float(np.clip(ray.distance, min_range, max_range))
            p_body = np.array([d * np.cos(ray.angle_rad), d * np.sin(ray.angle_rad)], dtype=np.float32)
            p_world = base + body_to_world(p_body, float(base_yaw))
            rc1 = self.world_to_rc(p_world)
            if rc1 is None:
                continue
            cells = self._bresenham(rc0[0], rc0[1], rc1[0], rc1[1])
            if len(cells) <= 0:
                continue
            for rr, cc in cells[:-1]:
                self.logodds[rr, cc] = np.clip(self.logodds[rr, cc] + self.logodds_free, self.logodds_min, self.logodds_max)
                self.observed[rr, cc] = np.clip(self.observed[rr, cc] + 1.0, 0.0, 8.0)
            if ray.hit and d < (max_range - 1e-4):
                rr, cc = cells[-1]
                self.logodds[rr, cc] = np.clip(self.logodds[rr, cc] + self.logodds_hit, self.logodds_min, self.logodds_max)
                self.observed[rr, cc] = np.clip(self.observed[rr, cc] + 1.0, 0.0, 8.0)
            else:
                rr, cc = cells[-1]
                self.logodds[rr, cc] = np.clip(self.logodds[rr, cc] + self.logodds_free, self.logodds_min, self.logodds_max)
                self.observed[rr, cc] = np.clip(self.observed[rr, cc] + 1.0, 0.0, 8.0)

    def occupancy(self, threshold: float = 0.0) -> np.ndarray:
        return self.logodds >= float(threshold)

    def observed_mask(self, threshold: float = 0.15) -> np.ndarray:
        return self.observed >= float(threshold)

    def occupancy_and_observed(
        self,
        occ_threshold: float = 0.0,
        observed_threshold: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.occupancy(threshold=occ_threshold),
            self.observed_mask(threshold=observed_threshold),
        )

    def extract_obstacles_as_circles(
        self,
        threshold: float = 0.0,
        min_cluster_cells: int = 2,
        max_obstacles: int = 96,
    ) -> np.ndarray:
        occ = self.occupancy(threshold=threshold)
        if not np.any(occ):
            return np.zeros((0, 3), dtype=np.float32)

        visited = np.zeros_like(occ, dtype=np.uint8)
        h, w = occ.shape
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        clusters: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not occ[r, c] or visited[r, c]:
                    continue
                q = deque([(r, c)])
                visited[r, c] = 1
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in nbrs:
                        r2 = rr + dr
                        c2 = cc + dc
                        if r2 < 0 or r2 >= h or c2 < 0 or c2 >= w:
                            continue
                        if visited[r2, c2] or (not occ[r2, c2]):
                            continue
                        visited[r2, c2] = 1
                        q.append((r2, c2))
                if len(cells) >= int(max(1, min_cluster_cells)):
                    clusters.append(np.asarray(cells, dtype=np.int32))

        if len(clusters) <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        rows = sorted(clusters, key=lambda c: c.shape[0], reverse=True)
        rows = rows[: max(1, int(max_obstacles))]
        out = []
        for cells in rows:
            xy = np.stack([self.rc_to_world(int(rc[0]), int(rc[1])) for rc in cells], axis=0)
            center = np.mean(xy, axis=0)
            radius = max(
                0.5 * self.resolution,
                float(np.sqrt(float(cells.shape[0]) / np.pi) * self.resolution),
            )
            out.append(np.array([center[0], center[1], radius], dtype=np.float32))
        return np.stack(out, axis=0).astype(np.float32)

    def line_of_sight_blocked(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        robot_radius: float,
        margin: float = 0.0,
        threshold: float = 0.0,
    ) -> bool:
        occ = self.occupancy(threshold=threshold)
        a = np.asarray(start_xy, dtype=np.float32)
        b = np.asarray(goal_xy, dtype=np.float32)
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-6:
            return False
        step = max(0.5 * self.resolution, 0.02)
        n = max(2, int(np.ceil(L / step)))
        rr = max(0.0, float(robot_radius) + float(margin))
        r_cells = int(np.ceil(rr / max(self.resolution, 1e-6)))

        for i in range(1, n):
            t = float(i) / float(n)
            p = a + t * ab
            rc = self.world_to_rc(p)
            if rc is None:
                continue
            r0, c0 = rc
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if (dr * dr + dc * dc) > (r_cells * r_cells):
                        continue
                    r1 = r0 + dr
                    c1 = c0 + dc
                    if r1 < 0 or r1 >= occ.shape[0] or c1 < 0 or c1 >= occ.shape[1]:
                        continue
                    if occ[r1, c1]:
                        return True
        return False
