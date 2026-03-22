from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from collections import deque


def point_segment_distance_and_t(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den < 1e-12:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / den)
    t_clamped = float(np.clip(t, 0.0, 1.0))
    q = a + t_clamped * ab
    return float(np.linalg.norm(p - q)), t_clamped


def line_signed_lateral(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    point_xy: np.ndarray,
) -> Tuple[float, float]:
    a = np.asarray(start_xy, dtype=np.float32)
    b = np.asarray(goal_xy, dtype=np.float32)
    p = np.asarray(point_xy, dtype=np.float32)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return 0.0, 0.0
    dn = d / L
    perp = np.array([-dn[1], dn[0]], dtype=np.float32)
    rel = p - a
    t = float(np.dot(rel, dn) / max(L, 1e-9))
    lat = float(np.dot(rel, perp))
    return t, lat


def line_of_sight_blocked(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.12,
    occ_grid: Optional[np.ndarray] = None,
    occ_min_xy: Optional[np.ndarray] = None,
    occ_resolution: Optional[float] = None,
) -> bool:
    if occ_grid is not None and occ_min_xy is not None and occ_resolution is not None:
        occ = np.asarray(occ_grid, dtype=bool)
        min_xy = np.asarray(occ_min_xy, dtype=np.float32).reshape(2)
        res = float(max(1e-6, occ_resolution))
        h, w = occ.shape
        a = np.asarray(start_xy, dtype=np.float32)
        b = np.asarray(goal_xy, dtype=np.float32)
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-6:
            return False
        step = max(0.5 * res, 0.02)
        n = max(2, int(np.ceil(L / step)))
        r_eff = max(0.0, float(robot_radius) + float(margin))
        r_cells = int(np.ceil(r_eff / res))
        for i in range(1, n):
            t = float(i) / float(n)
            p = a + t * ab
            cc = int(np.round((p[0] - min_xy[0]) / res))
            rr = int(np.round((p[1] - min_xy[1]) / res))
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if (dr * dr + dc * dc) > (r_cells * r_cells):
                        continue
                    r2 = rr + dr
                    c2 = cc + dc
                    if r2 < 0 or r2 >= h or c2 < 0 or c2 >= w:
                        continue
                    if occ[r2, c2]:
                        return True
        return False

    a = np.asarray(start_xy, dtype=np.float32)
    b = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return False

    for i in range(obs.shape[0]):
        c = obs[i, :2]
        r_eff = float(obs[i, 2] + robot_radius + margin)
        dist, t = point_segment_distance_and_t(a, b, c)
        if 0.0 < t < 1.0 and dist < r_eff:
            return True
    return False


def line_of_sight_blocked_confidence(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.12,
    occ_grid: Optional[np.ndarray] = None,
    occ_observed: Optional[np.ndarray] = None,
    occ_min_xy: Optional[np.ndarray] = None,
    occ_resolution: Optional[float] = None,
    blocked_conf_threshold: float = 0.65,
) -> Tuple[bool, float, int, int]:
    """
    Returns:
      blocked, blocked_confidence, observed_cells, sampled_cells
    """
    if occ_grid is not None and occ_min_xy is not None and occ_resolution is not None:
        occ = np.asarray(occ_grid, dtype=bool)
        if occ_observed is None:
            observed = np.ones_like(occ, dtype=bool)
        else:
            observed = np.asarray(occ_observed, dtype=bool)
            if observed.shape != occ.shape:
                observed = np.ones_like(occ, dtype=bool)
        min_xy = np.asarray(occ_min_xy, dtype=np.float32).reshape(2)
        res = float(max(1e-6, occ_resolution))
        h, w = occ.shape
        a = np.asarray(start_xy, dtype=np.float32)
        b = np.asarray(goal_xy, dtype=np.float32)
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-6:
            return False, 0.0, 0, 0
        step = max(0.5 * res, 0.02)
        n = max(2, int(np.ceil(L / step)))
        r_eff = max(0.0, float(robot_radius) + float(margin))
        r_cells = int(np.ceil(r_eff / res))

        sampled: set[tuple[int, int]] = set()
        observed_cells = 0
        occ_cells = 0
        for i in range(1, n):
            t = float(i) / float(n)
            p = a + t * ab
            cc = int(np.round((p[0] - min_xy[0]) / res))
            rr = int(np.round((p[1] - min_xy[1]) / res))
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if (dr * dr + dc * dc) > (r_cells * r_cells):
                        continue
                    r2 = rr + dr
                    c2 = cc + dc
                    if r2 < 0 or r2 >= h or c2 < 0 or c2 >= w:
                        continue
                    sampled.add((r2, c2))

        for rr, cc in sampled:
            if observed[rr, cc]:
                observed_cells += 1
                if occ[rr, cc]:
                    occ_cells += 1
        blocked_conf = float(occ_cells / max(1, observed_cells))
        blocked = bool(blocked_conf >= float(blocked_conf_threshold))
        return blocked, blocked_conf, int(observed_cells), int(len(sampled))

    blocked_geom = line_of_sight_blocked(
        start_xy=start_xy,
        goal_xy=goal_xy,
        obstacles_xyr=obstacles_xyr,
        robot_radius=robot_radius,
        margin=margin,
        occ_grid=None,
        occ_min_xy=None,
        occ_resolution=None,
    )
    return bool(blocked_geom), (1.0 if blocked_geom else 0.0), 1, 1


def _clip_target_to_bounds(
    target_xy: np.ndarray,
    bounds_x_range: Tuple[float, float],
    bounds_y_range: Tuple[float, float],
    margin: float,
) -> Tuple[np.ndarray, bool]:
    x_min = float(min(bounds_x_range[0], bounds_x_range[1]) + margin)
    x_max = float(max(bounds_x_range[0], bounds_x_range[1]) - margin)
    y_min = float(min(bounds_y_range[0], bounds_y_range[1]) + margin)
    y_max = float(max(bounds_y_range[0], bounds_y_range[1]) - margin)
    if x_min > x_max:
        cx = 0.5 * (x_min + x_max)
        x_min = cx
        x_max = cx
    if y_min > y_max:
        cy = 0.5 * (y_min + y_max)
        y_min = cy
        y_max = cy
    t = np.asarray(target_xy, dtype=np.float32).copy()
    before = t.copy()
    t[0] = float(np.clip(t[0], x_min, x_max))
    t[1] = float(np.clip(t[1], y_min, y_max))
    projected = bool(np.linalg.norm(t - before) > 1e-6)
    return t, projected


def project_target_with_invariants(
    candidate_xy: np.ndarray,
    prev_target_xy: Optional[np.ndarray],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    bounds_x_range: Tuple[float, float],
    bounds_y_range: Tuple[float, float],
    bound_margin: float = 0.10,
    corridor_max_dev: float = 0.50,
    jump_max: float = 0.45,
    path_xy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Enforce target-chain invariants by projection:
      1) bounds
      2) corridor
      3) jump continuity
    """
    info: Dict[str, float] = {
        "bound_projected": 0.0,
        "corridor_projected": 0.0,
        "jump_projected": 0.0,
        "corridor_dev_before": 0.0,
        "jump_before": 0.0,
    }
    t = np.asarray(candidate_xy, dtype=np.float32).reshape(2).copy()
    t, p_bound = _clip_target_to_bounds(
        target_xy=t,
        bounds_x_range=bounds_x_range,
        bounds_y_range=bounds_y_range,
        margin=float(max(0.0, bound_margin)),
    )
    info["bound_projected"] = 1.0 if p_bound else 0.0

    corr_lim = float(max(0.0, corridor_max_dev))
    p_corr = False
    if corr_lim > 1e-6:
        pth = None if path_xy is None else np.asarray(path_xy, dtype=np.float32)
        if pth is not None and pth.ndim == 2 and pth.shape[0] >= 2:
            d = np.linalg.norm(pth - t[None, :], axis=1)
            idx = int(np.argmin(d))
            info["corridor_dev_before"] = float(d[idx])
            if float(d[idx]) > corr_lim:
                t = pth[idx].copy()
                p_corr = True
        else:
            start = np.asarray(start_xy, dtype=np.float32).reshape(2)
            goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
            _, lat = line_signed_lateral(start_xy=start, goal_xy=goal, point_xy=t)
            info["corridor_dev_before"] = float(abs(lat))
            if abs(float(lat)) > corr_lim:
                dvec = goal - start
                L = float(np.linalg.norm(dvec))
                if L > 1e-9:
                    dn = dvec / L
                    perp = np.array([-dn[1], dn[0]], dtype=np.float32)
                    rel = t - start
                    along = float(np.clip(np.dot(rel, dn), 0.0, L))
                    lat_clip = float(np.clip(np.dot(rel, perp), -corr_lim, corr_lim))
                    t = start + along * dn + lat_clip * perp
                    p_corr = True
    info["corridor_projected"] = 1.0 if p_corr else 0.0

    p_jump = False
    if prev_target_xy is not None and jump_max > 1e-6:
        prev = np.asarray(prev_target_xy, dtype=np.float32).reshape(2)
        dvec = t - prev
        dnorm = float(np.linalg.norm(dvec))
        info["jump_before"] = dnorm
        if dnorm > float(jump_max):
            t = prev + (float(jump_max) / max(dnorm, 1e-9)) * dvec
            p_jump = True
    info["jump_projected"] = 1.0 if p_jump else 0.0

    t, p_bound2 = _clip_target_to_bounds(
        target_xy=t,
        bounds_x_range=bounds_x_range,
        bounds_y_range=bounds_y_range,
        margin=float(max(0.0, bound_margin)),
    )
    if p_bound2:
        info["bound_projected"] = 1.0

    return t.astype(np.float32), info


def compute_boundary_recover_target(
    base_xy: np.ndarray,
    goal_xy: np.ndarray,
    bounds_x_range: Tuple[float, float],
    bounds_y_range: Tuple[float, float],
    bound_margin: float = 0.10,
    path_xy: Optional[np.ndarray] = None,
    lookahead_m: float = 0.60,
) -> np.ndarray:
    base = np.asarray(base_xy, dtype=np.float32).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    base_in, _ = _clip_target_to_bounds(
        target_xy=base,
        bounds_x_range=bounds_x_range,
        bounds_y_range=bounds_y_range,
        margin=float(max(0.0, bound_margin)),
    )
    pth = None if path_xy is None else np.asarray(path_xy, dtype=np.float32)
    if pth is not None and pth.ndim == 2 and pth.shape[0] >= 2:
        target, _ = select_path_lookahead_target(
            path_xy=pth,
            current_xy=base_in,
            lookahead_m=float(max(0.1, lookahead_m)),
            min_index=0,
        )
    else:
        d = goal - base_in
        L = float(np.linalg.norm(d))
        if L > 1e-6:
            target = base_in + (float(max(0.1, lookahead_m)) / L) * d
        else:
            x_min = float(min(bounds_x_range[0], bounds_x_range[1]))
            x_max = float(max(bounds_x_range[0], bounds_x_range[1]))
            y_min = float(min(bounds_y_range[0], bounds_y_range[1]))
            y_max = float(max(bounds_y_range[0], bounds_y_range[1]))
            target = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
    target, _ = _clip_target_to_bounds(
        target_xy=target,
        bounds_x_range=bounds_x_range,
        bounds_y_range=bounds_y_range,
        margin=float(max(0.0, bound_margin)),
    )
    return target.astype(np.float32)


def _passes_distribution_constraints(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float,
    min_line_blockers: int,
    blocker_t_range: Tuple[float, float],
    blocker_extra_margin: float,
    require_mixed_sides: bool,
) -> bool:
    if min_line_blockers <= 0 and not require_mixed_sides:
        return True

    a = np.asarray(start_xy, dtype=np.float32)
    b = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return False

    t_min = float(min(blocker_t_range[0], blocker_t_range[1]))
    t_max = float(max(blocker_t_range[0], blocker_t_range[1]))
    blocker_count = 0
    has_left = False
    has_right = False
    extra = float(max(0.0, blocker_extra_margin))
    for i in range(obs.shape[0]):
        c = obs[i, :2]
        r_eff = float(obs[i, 2] + robot_radius + margin + extra)
        dist, t = point_segment_distance_and_t(a, b, c)
        t_line, lat = line_signed_lateral(a, b, c)
        if t_min <= t <= t_max and dist < r_eff:
            blocker_count += 1
        if t_min <= t_line <= t_max:
            if lat > 0.0:
                has_left = True
            elif lat < 0.0:
                has_right = True

    if blocker_count < int(min_line_blockers):
        return False
    if require_mixed_sides and not (has_left and has_right):
        return False
    return True


def _bfs_path_exists(occ: np.ndarray, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> bool:
    if occ[start_rc] or occ[goal_rc]:
        return False
    if start_rc == goal_rc:
        return True
    h, w = occ.shape
    q = deque([start_rc])
    visited = np.zeros_like(occ, dtype=np.uint8)
    visited[start_rc] = 1
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while q:
        r, c = q.popleft()
        for dr, dc in nbrs:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if visited[rr, cc] or occ[rr, cc]:
                continue
            if dr != 0 and dc != 0:
                if occ[r, cc] or occ[rr, c]:
                    continue
            if (rr, cc) == goal_rc:
                return True
            visited[rr, cc] = 1
            q.append((rr, cc))
    return False


def _global_path_feasible(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    inflation_margin: float,
    grid_resolution: float,
    grid_padding: float,
    max_grid_cells: int = 240_000,
) -> bool:
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return True

    min_xy = np.minimum(start, goal)
    max_xy = np.maximum(start, goal)
    if obs.shape[0] > 0:
        min_xy = np.minimum(min_xy, np.min(obs[:, :2] - obs[:, 2:3], axis=0))
        max_xy = np.maximum(max_xy, np.max(obs[:, :2] + obs[:, 2:3], axis=0))
    min_xy = min_xy - float(grid_padding)
    max_xy = max_xy + float(grid_padding)

    res = max(1e-3, float(grid_resolution))
    size = np.maximum(max_xy - min_xy, res)
    nx = int(np.ceil(size[0] / res)) + 1
    ny = int(np.ceil(size[1] / res)) + 1
    if nx * ny > int(max_grid_cells):
        return False

    xs = min_xy[0] + np.arange(nx, dtype=np.float32) * res
    ys = min_xy[1] + np.arange(ny, dtype=np.float32) * res
    xx, yy = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), dtype=bool)

    inflate = float(robot_radius + inflation_margin)
    for i in range(obs.shape[0]):
        cx, cy, r = float(obs[i, 0]), float(obs[i, 1]), float(obs[i, 2])
        rr = r + inflate
        occ |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= (rr * rr)

    def to_rc(pt: np.ndarray) -> Tuple[int, int]:
        c = int(np.clip(np.round((pt[0] - min_xy[0]) / res), 0, nx - 1))
        r = int(np.clip(np.round((pt[1] - min_xy[1]) / res), 0, ny - 1))
        return r, c

    start_rc = to_rc(start)
    goal_rc = to_rc(goal)
    return _bfs_path_exists(occ=occ, start_rc=start_rc, goal_rc=goal_rc)


def _build_occ_grid(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    inflation_margin: float,
    grid_resolution: float,
    grid_padding: float,
    max_grid_cells: int = 240_000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, int, int]:
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return (
            np.zeros((2, 2), dtype=bool),
            np.minimum(start, goal) - float(grid_padding),
            max(1e-3, float(grid_resolution)),
            2,
            2,
        )

    min_xy = np.minimum(start, goal)
    max_xy = np.maximum(start, goal)
    min_xy = np.minimum(min_xy, np.min(obs[:, :2] - obs[:, 2:3], axis=0))
    max_xy = np.maximum(max_xy, np.max(obs[:, :2] + obs[:, 2:3], axis=0))
    min_xy = min_xy - float(grid_padding)
    max_xy = max_xy + float(grid_padding)

    res = max(1e-3, float(grid_resolution))
    size = np.maximum(max_xy - min_xy, res)
    nx = int(np.ceil(size[0] / res)) + 1
    ny = int(np.ceil(size[1] / res)) + 1
    if nx * ny > int(max_grid_cells):
        return None, None, res, 0, 0

    xs = min_xy[0] + np.arange(nx, dtype=np.float32) * res
    ys = min_xy[1] + np.arange(ny, dtype=np.float32) * res
    xx, yy = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), dtype=bool)

    inflate = float(robot_radius + inflation_margin)
    for i in range(obs.shape[0]):
        cx, cy, r = float(obs[i, 0]), float(obs[i, 1]), float(obs[i, 2])
        rr = r + inflate
        occ |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= (rr * rr)

    return occ, min_xy.astype(np.float32), res, nx, ny


def _grid_to_rc(pt: np.ndarray, min_xy: np.ndarray, res: float, nx: int, ny: int) -> Tuple[int, int]:
    c = int(np.clip(np.round((pt[0] - min_xy[0]) / res), 0, nx - 1))
    r = int(np.clip(np.round((pt[1] - min_xy[1]) / res), 0, ny - 1))
    return r, c


def _grid_to_xy(r: int, c: int, min_xy: np.ndarray, res: float) -> np.ndarray:
    x = float(min_xy[0] + c * res)
    y = float(min_xy[1] + r * res)
    return np.array([x, y], dtype=np.float32)


def _inflate_occ_grid(occ: np.ndarray, inflate_cells: int) -> np.ndarray:
    """
    Binary disk dilation on occupancy grid.
    Used for occ-grid planning branch so robot footprint traversability is respected.
    """
    occ_b = np.asarray(occ, dtype=bool)
    r = int(max(0, inflate_cells))
    if r <= 0 or (not np.any(occ_b)):
        return occ_b.copy()

    h, w = occ_b.shape
    out = np.zeros_like(occ_b, dtype=bool)
    rr2 = r * r

    for dr in range(-r, r + 1):
        dc_lim = int(np.floor(np.sqrt(max(0, rr2 - dr * dr))))
        r_src0 = max(0, -dr)
        r_src1 = min(h, h - dr)
        r_dst0 = max(0, dr)
        r_dst1 = min(h, h + dr)
        if r_src0 >= r_src1:
            continue
        for dc in range(-dc_lim, dc_lim + 1):
            c_src0 = max(0, -dc)
            c_src1 = min(w, w - dc)
            c_dst0 = max(0, dc)
            c_dst1 = min(w, w + dc)
            if c_src0 >= c_src1:
                continue
            out[r_dst0:r_dst1, c_dst0:c_dst1] |= occ_b[r_src0:r_src1, c_src0:c_src1]
    return out


def _bfs_path_cells(
    occ: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    if occ[start_rc] or occ[goal_rc]:
        return None
    if start_rc == goal_rc:
        return [start_rc]

    h, w = occ.shape
    q = deque([start_rc])
    visited = np.zeros_like(occ, dtype=np.uint8)
    visited[start_rc] = 1
    parent_r = -np.ones((h, w), dtype=np.int32)
    parent_c = -np.ones((h, w), dtype=np.int32)
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    found = False
    while q:
        r, c = q.popleft()
        for dr, dc in nbrs:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if visited[rr, cc] or occ[rr, cc]:
                continue
            # For diagonal expansion, forbid corner-cutting through obstacle corners.
            if dr != 0 and dc != 0:
                if occ[r, cc] or occ[rr, c]:
                    continue
            visited[rr, cc] = 1
            parent_r[rr, cc] = r
            parent_c[rr, cc] = c
            if (rr, cc) == goal_rc:
                found = True
                q.clear()
                break
            q.append((rr, cc))
        if found:
            break

    if not found:
        return None

    path: List[Tuple[int, int]] = []
    cur = goal_rc
    while True:
        path.append(cur)
        if cur == start_rc:
            break
        pr = int(parent_r[cur])
        pc = int(parent_c[cur])
        if pr < 0 or pc < 0:
            return None
        cur = (pr, pc)
    path.reverse()
    return path


def plan_global_path_xy(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    inflation_margin: float = 0.10,
    grid_resolution: float = 0.10,
    grid_padding: float = 0.60,
    max_grid_cells: int = 240_000,
    occ_grid: Optional[np.ndarray] = None,
    occ_min_xy: Optional[np.ndarray] = None,
    occ_resolution: Optional[float] = None,
) -> Optional[np.ndarray]:
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)

    if occ_grid is not None and occ_min_xy is not None and occ_resolution is not None:
        occ_raw = np.asarray(occ_grid, dtype=bool)
        min_xy = np.asarray(occ_min_xy, dtype=np.float32).reshape(2)
        res = float(max(1e-6, occ_resolution))
        inflate_cells = int(
            np.ceil(
                max(0.0, float(robot_radius) + float(inflation_margin))
                / max(res, 1e-6)
            )
        )
        occ = _inflate_occ_grid(occ=occ_raw, inflate_cells=inflate_cells)
        ny, nx = occ.shape
        start_rc = _grid_to_rc(start, min_xy=min_xy, res=res, nx=nx, ny=ny)
        goal_rc = _grid_to_rc(goal, min_xy=min_xy, res=res, nx=nx, ny=ny)
        if occ[start_rc] or occ[goal_rc]:
            carve_r = int(np.ceil(max(1.2 * float(robot_radius), float(res)) / float(res)))
            h, w = occ.shape
            for rr0, cc0 in (start_rc, goal_rc):
                for dr in range(-carve_r, carve_r + 1):
                    for dc in range(-carve_r, carve_r + 1):
                        rr = rr0 + dr
                        cc = cc0 + dc
                        if rr < 0 or rr >= h or cc < 0 or cc >= w:
                            continue
                        if (dr * dr + dc * dc) <= (carve_r * carve_r):
                            occ[rr, cc] = False
        rc_path = _bfs_path_cells(occ=occ, start_rc=start_rc, goal_rc=goal_rc)
        if rc_path is None or len(rc_path) == 0:
            return None
        xy_path = np.stack([_grid_to_xy(r=r, c=c, min_xy=min_xy, res=res) for r, c in rc_path], axis=0).astype(np.float32)
        xy_path[0] = start
        xy_path[-1] = goal
        return xy_path

    if obs.size == 0:
        return np.stack([start, goal], axis=0).astype(np.float32)

    occ, min_xy, res, nx, ny = _build_occ_grid(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=obs,
        robot_radius=robot_radius,
        inflation_margin=inflation_margin,
        grid_resolution=grid_resolution,
        grid_padding=grid_padding,
        max_grid_cells=max_grid_cells,
    )
    if occ is None or min_xy is None:
        return None

    start_rc = _grid_to_rc(start, min_xy=min_xy, res=res, nx=nx, ny=ny)
    goal_rc = _grid_to_rc(goal, min_xy=min_xy, res=res, nx=nx, ny=ny)
    # Sensor-only obstacles can occasionally mark the robot footprint cell as occupied.
    # Carve small free disks around start/goal to keep BFS numerically stable.
    if occ[start_rc] or occ[goal_rc]:
        carve_r = int(np.ceil(max(1.2 * float(robot_radius), float(res)) / float(res)))
        h, w = occ.shape
        for rr0, cc0 in (start_rc, goal_rc):
            for dr in range(-carve_r, carve_r + 1):
                for dc in range(-carve_r, carve_r + 1):
                    rr = rr0 + dr
                    cc = cc0 + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    if (dr * dr + dc * dc) <= (carve_r * carve_r):
                        occ[rr, cc] = False
    rc_path = _bfs_path_cells(occ=occ, start_rc=start_rc, goal_rc=goal_rc)
    if rc_path is None or len(rc_path) == 0:
        return None

    xy_path = np.stack([_grid_to_xy(r=r, c=c, min_xy=min_xy, res=res) for r, c in rc_path], axis=0).astype(np.float32)
    xy_path[0] = start
    xy_path[-1] = goal
    return xy_path


def select_path_lookahead_target(
    path_xy: np.ndarray,
    current_xy: np.ndarray,
    lookahead_m: float = 0.9,
    min_index: int = 0,
) -> Tuple[np.ndarray, int]:
    path = np.asarray(path_xy, dtype=np.float32)
    cur = np.asarray(current_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] == 0:
        return cur.astype(np.float32), 0
    if path.shape[0] == 1:
        return path[0].astype(np.float32), 0

    idx0 = int(np.clip(min_index, 0, path.shape[0] - 1))
    dists = np.linalg.norm(path[idx0:] - cur[None, :], axis=1)
    idx = int(np.argmin(dists)) + idx0
    if idx >= path.shape[0] - 1:
        return path[-1].astype(np.float32), path.shape[0] - 1

    remain = max(0.0, float(lookahead_m))
    p = path[idx].copy()
    p_idx = idx
    for j in range(idx, path.shape[0] - 1):
        a = path[j]
        b = path[j + 1]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        if remain <= seg_len:
            t = remain / seg_len
            p = a + t * seg
            return p.astype(np.float32), int(j)
        remain -= seg_len
        p = b
        p_idx = j + 1
    return p.astype(np.float32), int(p_idx)


def build_reference_traj_from_path(
    path_xy: np.ndarray,
    start_index: int,
    horizon: int,
    step_m: float,
) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] == 0 or horizon <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if path.shape[0] == 1:
        return np.repeat(path[0:1], repeats=horizon, axis=0)

    idx0 = int(np.clip(start_index, 0, path.shape[0] - 1))
    path_sub = path[idx0:]
    if path_sub.shape[0] == 1:
        return np.repeat(path_sub[0:1], repeats=horizon, axis=0)

    seg = path_sub[1:] - path_sub[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])
    total_len = float(cum[-1])

    ds = max(1e-3, float(step_m))
    s_query = np.arange(1, horizon + 1, dtype=np.float32) * ds
    s_query = np.clip(s_query, 0.0, total_len)

    ref = np.zeros((horizon, 2), dtype=np.float32)
    j = 0
    for i, s in enumerate(s_query):
        while j < cum.shape[0] - 2 and cum[j + 1] < s:
            j += 1
        s0 = float(cum[j])
        s1 = float(cum[j + 1]) if (j + 1) < cum.shape[0] else s0 + 1e-6
        p0 = path_sub[j]
        p1 = path_sub[min(j + 1, path_sub.shape[0] - 1)]
        if s1 - s0 < 1e-6:
            ref[i] = p1
        else:
            t = (float(s) - s0) / (s1 - s0)
            ref[i] = p0 + t * (p1 - p0)
    return ref


def compute_path_remaining(
    path_xy: np.ndarray,
    current_xy: np.ndarray,
    min_index: int = 0,
) -> Tuple[float, int]:
    path = np.asarray(path_xy, dtype=np.float32)
    cur = np.asarray(current_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] == 0:
        return float("inf"), 0
    if path.shape[0] == 1:
        return float(np.linalg.norm(cur - path[0])), 0

    idx0 = int(np.clip(min_index, 0, path.shape[0] - 1))
    dists = np.linalg.norm(path[idx0:] - cur[None, :], axis=1)
    idx = int(np.argmin(dists)) + idx0
    if idx >= path.shape[0] - 1:
        return float(np.linalg.norm(cur - path[-1])), path.shape[0] - 1

    seg = path[idx + 1 :] - path[idx:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    remain = float(np.linalg.norm(cur - path[idx])) + float(np.sum(seg_len))
    return remain, idx


def sample_obstacles_with_constraints(
    rng: np.random.Generator,
    base_obstacles_xyr: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obs_x_range: Tuple[float, float],
    obs_y_range: Tuple[float, float],
    min_obs_obs_dist: float,
    min_start_obs_dist: float,
    min_goal_obs_dist: float,
    attempts: int = 300,
    require_global_path: bool = False,
    robot_radius: float = 0.28,
    path_inflation_margin: float = 0.10,
    path_grid_resolution: float = 0.10,
    path_grid_padding: float = 0.60,
    path_max_grid_cells: int = 240_000,
    min_line_blockers: int = 0,
    blocker_t_range: Tuple[float, float] = (0.2, 0.9),
    blocker_extra_margin: float = 0.0,
    require_mixed_sides: bool = False,
) -> Tuple[bool, np.ndarray]:
    obs = np.asarray(base_obstacles_xyr, dtype=np.float32).copy()
    n = obs.shape[0]
    for _ in range(max(1, int(attempts))):
        xy = np.stack(
            [
                rng.uniform(obs_x_range[0], obs_x_range[1], size=(n,)),
                rng.uniform(obs_y_range[0], obs_y_range[1], size=(n,)),
            ],
            axis=1,
        ).astype(np.float32)
        ok = True
        for i in range(n):
            ri = float(obs[i, 2])
            if np.linalg.norm(xy[i] - start_xy) < (min_start_obs_dist + ri):
                ok = False
                break
            if np.linalg.norm(xy[i] - goal_xy) < (min_goal_obs_dist + ri):
                ok = False
                break
        if not ok:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                min_d = min_obs_obs_dist + float(obs[i, 2]) + float(obs[j, 2])
                if np.linalg.norm(xy[i] - xy[j]) < min_d:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        obs[:, :2] = xy
        if not _passes_distribution_constraints(
            start_xy=start_xy,
            goal_xy=goal_xy,
            obstacles_xyr=obs,
            robot_radius=robot_radius,
            margin=path_inflation_margin,
            min_line_blockers=min_line_blockers,
            blocker_t_range=blocker_t_range,
            blocker_extra_margin=blocker_extra_margin,
            require_mixed_sides=require_mixed_sides,
        ):
            continue
        if require_global_path:
            if not _global_path_feasible(
                start_xy=start_xy,
                goal_xy=goal_xy,
                obstacles_xyr=obs,
                robot_radius=robot_radius,
                inflation_margin=path_inflation_margin,
                grid_resolution=path_grid_resolution,
                grid_padding=path_grid_padding,
                max_grid_cells=path_max_grid_cells,
            ):
                continue
        return True, obs
    return False, np.asarray(base_obstacles_xyr, dtype=np.float32).copy()


def sample_obstacles_adaptive(
    rng: np.random.Generator,
    base_obstacles_xyr: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obs_x_range: Tuple[float, float],
    obs_y_range: Tuple[float, float],
    min_obs_obs_dist: float,
    min_start_obs_dist: float,
    min_goal_obs_dist: float,
    attempts: int = 300,
    require_global_path: bool = True,
    robot_radius: float = 0.28,
    path_inflation_margin: float = 0.10,
    path_grid_resolution: float = 0.10,
    path_grid_padding: float = 0.60,
    path_max_grid_cells: int = 240_000,
    expand_schedule: Sequence[float] = (0.0, 0.35, 0.70),
    attempt_factors: Sequence[float] = (1.0, 1.7, 2.5),
    relax_schedule: Sequence[float] = (1.0, 1.0, 0.95),
    min_line_blockers: int = 0,
    blocker_t_range: Tuple[float, float] = (0.2, 0.9),
    blocker_extra_margin: float = 0.0,
    require_mixed_sides: bool = False,
) -> Tuple[bool, np.ndarray, int]:
    base_obs = np.asarray(base_obstacles_xyr, dtype=np.float32).copy()
    nstg = min(len(expand_schedule), len(attempt_factors), len(relax_schedule))
    if nstg <= 0:
        ok, obs = sample_obstacles_with_constraints(
            rng=rng,
            base_obstacles_xyr=base_obs,
            start_xy=start_xy,
            goal_xy=goal_xy,
            obs_x_range=obs_x_range,
            obs_y_range=obs_y_range,
            min_obs_obs_dist=min_obs_obs_dist,
            min_start_obs_dist=min_start_obs_dist,
            min_goal_obs_dist=min_goal_obs_dist,
            attempts=attempts,
            require_global_path=require_global_path,
            robot_radius=robot_radius,
            path_inflation_margin=path_inflation_margin,
            path_grid_resolution=path_grid_resolution,
            path_grid_padding=path_grid_padding,
            path_max_grid_cells=path_max_grid_cells,
            min_line_blockers=min_line_blockers,
            blocker_t_range=blocker_t_range,
            blocker_extra_margin=blocker_extra_margin,
            require_mixed_sides=require_mixed_sides,
        )
        return ok, obs, 0

    for si in range(nstg):
        ex = float(max(0.0, expand_schedule[si]))
        af = float(max(0.2, attempt_factors[si]))
        rf = float(max(0.8, min(1.0, relax_schedule[si])))
        xr = (float(obs_x_range[0] - ex), float(obs_x_range[1] + ex))
        yr = (float(obs_y_range[0] - ex), float(obs_y_range[1] + ex))
        ok, obs = sample_obstacles_with_constraints(
            rng=rng,
            base_obstacles_xyr=base_obs,
            start_xy=start_xy,
            goal_xy=goal_xy,
            obs_x_range=xr,
            obs_y_range=yr,
            min_obs_obs_dist=float(min_obs_obs_dist) * rf,
            min_start_obs_dist=float(min_start_obs_dist) * rf,
            min_goal_obs_dist=float(min_goal_obs_dist) * rf,
            attempts=max(1, int(attempts * af)),
            require_global_path=require_global_path,
            robot_radius=robot_radius,
            path_inflation_margin=path_inflation_margin,
            path_grid_resolution=path_grid_resolution,
            path_grid_padding=path_grid_padding,
            path_max_grid_cells=path_max_grid_cells,
            min_line_blockers=min_line_blockers,
            blocker_t_range=blocker_t_range,
            blocker_extra_margin=blocker_extra_margin,
            require_mixed_sides=require_mixed_sides,
        )
        if ok:
            return True, obs, si
    return False, base_obs, nstg - 1


def compute_auto_waypoint(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.12,
    lateral_extra: float = 0.28,
) -> Optional[np.ndarray]:
    a = np.asarray(start_xy, dtype=np.float32)
    b = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    dvec = b - a
    L = float(np.linalg.norm(dvec))
    if L < 1e-6:
        return None

    d = dvec / L
    p = np.array([-d[1], d[0]], dtype=np.float32)

    blockers = []
    for i in range(obs.shape[0]):
        c = obs[i, :2]
        r_eff = float(obs[i, 2] + robot_radius + margin)
        dist, t = point_segment_distance_and_t(a, b, c)
        if 0.0 < t < 1.0 and dist < r_eff:
            blockers.append((t, i, r_eff))
    if not blockers:
        return None

    blockers.sort(key=lambda x: x[0])
    _, bi, r_eff = blockers[0]
    c = obs[bi, :2]
    offset = r_eff + max(0.0, float(lateral_extra))
    along_shift = -0.10 * L

    cand = [
        c + p * offset + d * along_shift,
        c - p * offset + d * along_shift,
    ]
    best = None
    best_score = 1e18
    for w in cand:
        path_len = float(np.linalg.norm(a - w) + np.linalg.norm(w - b))
        clr_pen = 0.0
        for j in range(obs.shape[0]):
            rr = float(obs[j, 2] + robot_radius + margin)
            dj = float(np.linalg.norm(w - obs[j, :2]))
            if dj < rr:
                clr_pen += (rr - dj) ** 2
        score = path_len + 30.0 * clr_pen
        if score < best_score:
            best_score = score
            best = w
    if best is None:
        return None
    return np.asarray(best, dtype=np.float32)


def compute_auto_waypoint_with_side(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.12,
    lateral_extra: float = 0.28,
    preferred_side: int = 0,
    waypoint_max_lateral: Optional[float] = None,
    waypoint_min_forward: float = 0.0,
    waypoint_max_forward: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Returns (waypoint_xy, side), where side in {-1, 0, +1}.
    +1 means using +perp side, -1 means using -perp side.
    """
    a = np.asarray(start_xy, dtype=np.float32)
    b = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    dvec = b - a
    L = float(np.linalg.norm(dvec))
    if L < 1e-6:
        return None, 0

    d = dvec / L
    p = np.array([-d[1], d[0]], dtype=np.float32)

    blockers = []
    for i in range(obs.shape[0]):
        c = obs[i, :2]
        r_eff = float(obs[i, 2] + robot_radius + margin)
        dist, t = point_segment_distance_and_t(a, b, c)
        if 0.0 < t < 1.0 and dist < r_eff:
            blockers.append((t, i, r_eff))
    if not blockers:
        return None, 0

    blockers.sort(key=lambda x: x[0])
    _, bi, r_eff = blockers[0]
    c = obs[bi, :2]
    offset = r_eff + max(0.0, float(lateral_extra))
    along_back = -0.08 * L
    along_fwd = 0.04 * L

    preferred = 0
    if preferred_side > 0:
        preferred = 1
    elif preferred_side < 0:
        preferred = -1

    side_order = [1, -1] if preferred == 0 else [preferred, -preferred]
    best = None
    best_side = 0
    best_score = 1e18
    for side in side_order:
        for fwd in (along_back, along_fwd):
            lat_scale = 1.0 if fwd == along_back else 1.20
            w = c + float(side) * p * (offset * lat_scale) + d * fwd
            t_line, lat_line = line_signed_lateral(start_xy=a, goal_xy=b, point_xy=w)
            fwd_m = float(t_line) * L
            if waypoint_max_lateral is not None and abs(float(lat_line)) > float(max(0.0, waypoint_max_lateral)):
                continue
            if fwd_m < float(max(0.0, waypoint_min_forward)):
                continue
            if waypoint_max_forward is not None and fwd_m > float(max(0.0, waypoint_max_forward)):
                continue
            path_len = float(np.linalg.norm(a - w) + np.linalg.norm(w - b))
            clr_pen = 0.0
            for j in range(obs.shape[0]):
                rr = float(obs[j, 2] + robot_radius + margin)
                dj = float(np.linalg.norm(w - obs[j, :2]))
                if dj < rr:
                    clr_pen += (rr - dj) ** 2

            blocked_a_w = line_of_sight_blocked(
                start_xy=a,
                goal_xy=w,
                obstacles_xyr=obs,
                robot_radius=robot_radius,
                margin=margin,
            )
            blocked_w_b = line_of_sight_blocked(
                start_xy=w,
                goal_xy=b,
                obstacles_xyr=obs,
                robot_radius=robot_radius,
                margin=margin,
            )
            los_pen = (2.0 if blocked_a_w else 0.0) + (4.0 if blocked_w_b else 0.0)

            # Strongly prefer requested side, unless it is clearly invalid.
            pref_pen = 0.0 if (preferred == 0 or side == preferred) else 0.25
            score = path_len + 30.0 * clr_pen + 8.0 * los_pen + pref_pen
            if score < best_score:
                best_score = score
                best = w
                best_side = int(side)
    if best is None:
        return None, 0
    return np.asarray(best, dtype=np.float32), best_side
