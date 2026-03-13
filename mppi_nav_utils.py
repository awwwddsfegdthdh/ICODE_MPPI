from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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
) -> bool:
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
) -> Optional[np.ndarray]:
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)

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
