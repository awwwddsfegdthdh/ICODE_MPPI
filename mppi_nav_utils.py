from __future__ import annotations

import heapq
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
    geom_conf = 0.0
    obs_geom = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs_geom.ndim == 2 and obs_geom.shape[0] > 0:
        a = np.asarray(start_xy, dtype=np.float32)
        b = np.asarray(goal_xy, dtype=np.float32)
        best_clear = float("inf")
        for i in range(obs_geom.shape[0]):
            c = obs_geom[i, :2]
            r_eff = float(obs_geom[i, 2] + float(robot_radius) + float(margin))
            dist, t = point_segment_distance_and_t(a, b, c)
            if 0.0 < t < 1.0:
                clear = float(dist - r_eff)
                if clear < best_clear:
                    best_clear = clear
        if np.isfinite(best_clear):
            if best_clear <= 0.0:
                geom_conf = 1.0
            else:
                geom_conf = float(np.exp(-best_clear / 0.05))

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
        occ_conf = float(occ_cells / max(1, observed_cells))
        blocked_conf = float(max(occ_conf, geom_conf))
        blocked = bool(blocked_conf >= float(blocked_conf_threshold))
        return blocked, blocked_conf, int(observed_cells), int(len(sampled))

    blocked_geom = bool(geom_conf >= float(blocked_conf_threshold))
    return blocked_geom, float(geom_conf), 1, 1


def blocked_confidence_range_semantics(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    heading_xy: Optional[np.ndarray],
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.12,
    occ_grid: Optional[np.ndarray] = None,
    occ_observed: Optional[np.ndarray] = None,
    occ_min_xy: Optional[np.ndarray] = None,
    occ_resolution: Optional[float] = None,
    corridor_len: float = 1.2,
    corridor_half_width: float = 0.45,
    front_fov_deg: float = 80.0,
    front_range: float = 0.9,
    blocked_conf_threshold: float = 0.65,
) -> Tuple[bool, float, Dict[str, float]]:
    """
    Range semantics for blocked confidence:
      - goal corridor occupancy/risk
      - forward sector occupancy/risk
    """
    base = np.asarray(start_xy, dtype=np.float32).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    eps = 1e-6
    gvec = goal - base
    gnorm = float(np.linalg.norm(gvec))
    if gnorm > eps:
        gdir = gvec / gnorm
    else:
        gdir = np.array([1.0, 0.0], dtype=np.float32)
    gperp = np.array([-gdir[1], gdir[0]], dtype=np.float32)

    if heading_xy is not None:
        hvec = np.asarray(heading_xy, dtype=np.float32).reshape(2) - base
        hnorm = float(np.linalg.norm(hvec))
        if hnorm > eps:
            hdir = hvec / hnorm
        else:
            hdir = gdir.copy()
    else:
        hdir = gdir.copy()

    corridor_len_eff = float(max(0.2, corridor_len))
    corridor_half_width_eff = float(max(0.08, corridor_half_width))
    front_half_fov = float(np.deg2rad(max(10.0, min(170.0, front_fov_deg)) * 0.5))
    front_range_eff = float(max(0.2, front_range))

    corridor_geom_conf = 0.0
    sector_geom_conf = 0.0
    corridor_min_clear = float("inf")
    sector_min_clear = float("inf")

    if obs.ndim == 2 and obs.shape[0] > 0:
        for i in range(obs.shape[0]):
            c = obs[i, :2]
            r_eff = float(obs[i, 2] + float(robot_radius) + float(margin))
            rel = c - base
            along = float(np.dot(rel, gdir))
            lat = float(abs(np.dot(rel, gperp)))
            if 0.0 < along < corridor_len_eff:
                clear_lat = float(lat - (corridor_half_width_eff + r_eff))
                if clear_lat < corridor_min_clear:
                    corridor_min_clear = clear_lat

            dist = float(np.linalg.norm(rel))
            if dist <= (front_range_eff + r_eff):
                if dist > eps:
                    cang = float(np.clip(np.dot(rel / max(dist, eps), hdir), -1.0, 1.0))
                    ang = float(np.arccos(cang))
                else:
                    ang = 0.0
                if ang <= front_half_fov:
                    clear_rad = float(dist - r_eff)
                    if clear_rad < sector_min_clear:
                        sector_min_clear = clear_rad

        if np.isfinite(corridor_min_clear):
            if corridor_min_clear <= 0.0:
                corridor_geom_conf = 1.0
            else:
                corridor_geom_conf = float(np.exp(-corridor_min_clear / 0.06))
        if np.isfinite(sector_min_clear):
            if sector_min_clear <= 0.0:
                sector_geom_conf = 1.0
            else:
                sector_geom_conf = float(np.exp(-sector_min_clear / 0.08))

    occ_corr_conf = 0.0
    occ_sector_conf = 0.0
    occ_corr_obs = 0
    occ_corr_cnt = 0
    occ_sector_obs = 0
    occ_sector_cnt = 0
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

        def _sample_occ(point_xy: np.ndarray) -> Tuple[bool, bool]:
            cc = int(np.round((float(point_xy[0]) - float(min_xy[0])) / res))
            rr = int(np.round((float(point_xy[1]) - float(min_xy[1])) / res))
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                return False, False
            return bool(observed[rr, cc]), bool(occ[rr, cc])

        s_along = float(max(res, 0.08))
        s_lat = float(max(res, 0.08))
        n_along = max(2, int(np.ceil(corridor_len_eff / s_along)))
        n_lat = max(3, int(np.ceil((2.0 * corridor_half_width_eff) / s_lat)))
        for ia in range(n_along + 1):
            along = float(ia) / float(max(1, n_along)) * corridor_len_eff
            for il in range(n_lat + 1):
                lat = (-corridor_half_width_eff) + float(il) / float(max(1, n_lat)) * (2.0 * corridor_half_width_eff)
                p = base + along * gdir + lat * gperp
                seen, occ_hit = _sample_occ(p)
                if seen:
                    occ_corr_cnt += 1
                    if occ_hit:
                        occ_corr_obs += 1

        s_r = float(max(res, 0.08))
        n_r = max(2, int(np.ceil(front_range_eff / s_r)))
        n_ang = max(7, int(np.ceil((2.0 * front_half_fov) / np.deg2rad(6.0))))
        hperp = np.array([-hdir[1], hdir[0]], dtype=np.float32)
        for ir in range(1, n_r + 1):
            rr = float(ir) / float(max(1, n_r)) * front_range_eff
            for ia in range(n_ang + 1):
                a = (-front_half_fov) + float(ia) / float(max(1, n_ang)) * (2.0 * front_half_fov)
                dir_a = float(np.cos(a)) * hdir + float(np.sin(a)) * hperp
                p = base + rr * dir_a
                seen, occ_hit = _sample_occ(p)
                if seen:
                    occ_sector_cnt += 1
                    if occ_hit:
                        occ_sector_obs += 1

        occ_corr_conf = float(occ_corr_obs / max(1, occ_corr_cnt))
        occ_sector_conf = float(occ_sector_obs / max(1, occ_sector_cnt))

    blocked_conf = float(max(corridor_geom_conf, sector_geom_conf, occ_corr_conf, occ_sector_conf))
    blocked = bool(blocked_conf >= float(blocked_conf_threshold))
    info = {
        "corridor_geom_conf": float(corridor_geom_conf),
        "sector_geom_conf": float(sector_geom_conf),
        "occ_corridor_conf": float(occ_corr_conf),
        "occ_sector_conf": float(occ_sector_conf),
        "corridor_min_clear": float(corridor_min_clear if np.isfinite(corridor_min_clear) else 1e6),
        "sector_min_clear": float(sector_min_clear if np.isfinite(sector_min_clear) else 1e6),
        "occ_corridor_observed": float(occ_corr_cnt),
        "occ_sector_observed": float(occ_sector_cnt),
    }
    return blocked, blocked_conf, info


def _point_min_clearance(
    point_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float,
) -> float:
    p = np.asarray(point_xy, dtype=np.float32).reshape(2)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return float("inf")
    r_eff = obs[:, 2] + float(robot_radius) + float(margin)
    d = np.linalg.norm(obs[:, :2] - p[None, :], axis=1) - r_eff
    return float(np.min(d)) if d.size > 0 else float("inf")


def _segment_min_clearance_sampled(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float,
    sample_step_m: float = 0.06,
) -> float:
    a = np.asarray(start_xy, dtype=np.float32).reshape(2)
    b = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return float("inf")
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return _point_min_clearance(
            point_xy=a,
            obstacles_xyr=obs,
            robot_radius=robot_radius,
            margin=margin,
        )
    step = float(max(0.02, sample_step_m))
    n = max(3, int(np.ceil(L / step)) + 1)
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    pts = a[None, :] + t[:, None] * ab[None, :]
    r_eff = obs[:, 2] + float(robot_radius) + float(margin)
    d = np.linalg.norm(pts[:, None, :] - obs[None, :, :2], axis=2) - r_eff[None, :]
    return float(np.min(d)) if d.size > 0 else float("inf")


def waypoint_is_feasible(
    base_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.12,
    min_seg_clearance: float = 0.01,
    min_point_clearance: float = 0.0,
    goal_min_dist: float = 0.20,
) -> bool:
    base = np.asarray(base_xy, dtype=np.float32).reshape(2)
    wp = np.asarray(waypoint_xy, dtype=np.float32).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    if (not np.all(np.isfinite(base))) or (not np.all(np.isfinite(wp))) or (not np.all(np.isfinite(goal))):
        return False
    if float(np.linalg.norm(wp - base)) < 0.05:
        return False
    if float(np.linalg.norm(wp - goal)) < float(max(0.0, goal_min_dist)):
        return False
    blocked_bw = line_of_sight_blocked(
        start_xy=base,
        goal_xy=wp,
        obstacles_xyr=obstacles_xyr,
        robot_radius=robot_radius,
        margin=margin,
    )
    if blocked_bw:
        return False
    seg_clear = _segment_min_clearance_sampled(
        start_xy=base,
        goal_xy=wp,
        obstacles_xyr=obstacles_xyr,
        robot_radius=robot_radius,
        margin=margin,
    )
    if seg_clear < float(min_seg_clearance):
        return False
    pt_clear = _point_min_clearance(
        point_xy=wp,
        obstacles_xyr=obstacles_xyr,
        robot_radius=robot_radius,
        margin=margin,
    )
    if pt_clear < float(min_point_clearance):
        return False
    return True


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


def project_target_to_passable_point(
    candidate_xy: np.ndarray,
    base_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.06,
    min_seg_clearance: float = 0.06,
    path_xy: Optional[np.ndarray] = None,
    min_progress: float = 0.16,
    backtrack_points: int = 14,
    ray_samples: int = 8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Guard active target passability from current base pose.

    If base->candidate is not passable (LOS blocked or sampled clearance too low),
    project target to a passable point by:
      1) backtracking on reference path (preferred),
      2) fallback to ray shrink from base to candidate.
    """
    cand = np.asarray(candidate_xy, dtype=np.float32).reshape(2)
    base = np.asarray(base_xy, dtype=np.float32).reshape(2)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    info: Dict[str, float] = {
        "passability_projected": 0.0,
        "passability_clear_before": 0.0,
        "passability_clear_after": 0.0,
    }

    margin_eff = float(max(0.0, margin))
    clear_req = float(max(0.0, min_seg_clearance))
    min_prog = float(max(0.0, min_progress))

    def seg_blocked_and_clear(pt: np.ndarray) -> Tuple[bool, float]:
        blocked = line_of_sight_blocked(
            start_xy=base,
            goal_xy=pt,
            obstacles_xyr=obs,
            robot_radius=robot_radius,
            margin=margin_eff,
        )
        clear = _segment_min_clearance_sampled(
            start_xy=base,
            goal_xy=pt,
            obstacles_xyr=obs,
            robot_radius=robot_radius,
            margin=margin_eff,
        )
        return bool(blocked), float(clear)

    blocked_before, clear_before = seg_blocked_and_clear(cand)
    info["passability_clear_before"] = float(clear_before)
    info["passability_clear_after"] = float(clear_before)

    if (not blocked_before) and clear_before >= clear_req:
        return cand.astype(np.float32), info

    best: Optional[np.ndarray] = None
    best_clear = float(clear_before)

    pth = None if path_xy is None else np.asarray(path_xy, dtype=np.float32)
    if pth is not None and pth.ndim == 2 and pth.shape[0] >= 2:
        d_base = np.linalg.norm(pth - base[None, :], axis=1)
        d_cand = np.linalg.norm(pth - cand[None, :], axis=1)
        idx_base = int(np.argmin(d_base))
        idx_cand = int(np.argmin(d_cand))
        hi = max(idx_base, idx_cand)
        lo = min(idx_base, idx_cand)
        if backtrack_points > 0:
            lo = max(lo, hi - int(max(1, backtrack_points)))
        for idx in range(int(hi), int(lo) - 1, -1):
            pt = np.asarray(pth[idx], dtype=np.float32).reshape(2)
            if float(np.linalg.norm(pt - base)) < min_prog:
                continue
            blocked, clear = seg_blocked_and_clear(pt)
            if (not blocked) and clear >= clear_req:
                best = pt
                best_clear = float(clear)
                break

    if best is None:
        vec = cand - base
        dist = float(np.linalg.norm(vec))
        if dist > 1e-6:
            alpha_min = float(np.clip(min_prog / max(dist, 1e-9), 0.0, 1.0))
            ns = int(max(2, ray_samples))
            for a in np.linspace(1.0, alpha_min, num=ns, dtype=np.float32):
                pt = base + float(a) * vec
                if float(np.linalg.norm(pt - base)) < min_prog:
                    continue
                blocked, clear = seg_blocked_and_clear(pt)
                if (not blocked) and clear >= clear_req:
                    best = np.asarray(pt, dtype=np.float32)
                    best_clear = float(clear)
                    break

    if best is not None and float(np.linalg.norm(best - cand)) > 1e-6:
        info["passability_projected"] = 1.0
        info["passability_clear_after"] = float(best_clear)
        return np.asarray(best, dtype=np.float32), info

    return cand.astype(np.float32), info


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


def _clearance_map_from_occ(occ: np.ndarray, res: float) -> np.ndarray:
    """
    Approximate Euclidean clearance map (meters) from binary occupancy.
    Occupied cells have clearance 0.
    """
    occ_b = np.asarray(occ, dtype=bool)
    if occ_b.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    h, w = occ_b.shape
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    inf = np.float32(1.0e6)
    d = np.full((h, w), inf, dtype=np.float32)
    d[occ_b] = 0.0
    rt2 = np.float32(np.sqrt(2.0))

    # Forward pass
    for r in range(h):
        for c in range(w):
            if d[r, c] <= 0.0:
                continue
            best = float(d[r, c])
            if r > 0:
                best = min(best, float(d[r - 1, c]) + 1.0)
                if c > 0:
                    best = min(best, float(d[r - 1, c - 1]) + float(rt2))
                if c + 1 < w:
                    best = min(best, float(d[r - 1, c + 1]) + float(rt2))
            if c > 0:
                best = min(best, float(d[r, c - 1]) + 1.0)
            d[r, c] = np.float32(best)

    # Backward pass
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if d[r, c] <= 0.0:
                continue
            best = float(d[r, c])
            if r + 1 < h:
                best = min(best, float(d[r + 1, c]) + 1.0)
                if c > 0:
                    best = min(best, float(d[r + 1, c - 1]) + float(rt2))
                if c + 1 < w:
                    best = min(best, float(d[r + 1, c + 1]) + float(rt2))
            if c + 1 < w:
                best = min(best, float(d[r, c + 1]) + 1.0)
            d[r, c] = np.float32(best)

    return (d * float(max(1e-6, res))).astype(np.float32)


def _bfs_path_cells(
    occ: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    clearance_map: Optional[np.ndarray] = None,
    min_clearance: float = 0.0,
) -> Optional[List[Tuple[int, int]]]:
    clear = None if clearance_map is None else np.asarray(clearance_map, dtype=np.float32)
    clear_floor = float(max(0.0, min_clearance))

    def _cell_ok(rr: int, cc: int) -> bool:
        if occ[rr, cc]:
            return False
        if clear is not None and clear_floor > 0.0:
            if float(clear[rr, cc]) < clear_floor:
                return False
        return True

    if (not _cell_ok(start_rc[0], start_rc[1])) or (not _cell_ok(goal_rc[0], goal_rc[1])):
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
            if visited[rr, cc] or (not _cell_ok(rr, cc)):
                continue
            # For diagonal expansion, forbid corner-cutting through obstacle corners.
            if dr != 0 and dc != 0:
                if (not _cell_ok(r, cc)) or (not _cell_ok(rr, c)):
                    continue
            if clear is not None and clear_floor > 0.0:
                edge_clear = float(min(float(clear[r, c]), float(clear[rr, cc])))
                if dr != 0 and dc != 0:
                    edge_clear = min(edge_clear, float(clear[r, cc]), float(clear[rr, c]))
                if edge_clear < clear_floor:
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


def _astar_path_cells(
    occ: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    clearance_map: Optional[np.ndarray] = None,
    min_clearance: float = 0.0,
    clearance_soft: float = 0.0,
    clearance_cost_weight: float = 0.0,
) -> Optional[List[Tuple[int, int]]]:
    clear = None if clearance_map is None else np.asarray(clearance_map, dtype=np.float32)
    clear_floor = float(max(0.0, min_clearance))
    clear_soft_eff = float(max(clear_floor + 1e-3, clearance_soft))
    clear_cost_w = float(max(0.0, clearance_cost_weight))

    def _cell_ok(rr: int, cc: int) -> bool:
        if occ[rr, cc]:
            return False
        if clear is not None and clear_floor > 0.0:
            if float(clear[rr, cc]) < clear_floor:
                return False
        return True

    if (not _cell_ok(start_rc[0], start_rc[1])) or (not _cell_ok(goal_rc[0], goal_rc[1])):
        return None
    if start_rc == goal_rc:
        return [start_rc]

    h, w = occ.shape
    g = np.full((h, w), np.inf, dtype=np.float32)
    closed = np.zeros((h, w), dtype=np.uint8)
    parent_r = -np.ones((h, w), dtype=np.int32)
    parent_c = -np.ones((h, w), dtype=np.int32)
    nbrs = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (-1, -1, np.sqrt(2.0)),
    )

    sr, sc = int(start_rc[0]), int(start_rc[1])
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    g[sr, sc] = 0.0

    def _heur(rr: int, cc: int) -> float:
        return float(np.hypot(float(rr - gr), float(cc - gc)))

    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (_heur(sr, sc), 0.0, sr, sc))

    found = False
    while open_heap:
        f_cur, g_cur, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        if g_cur > float(g[r, c]) + 1e-9:
            continue
        closed[r, c] = 1
        if (r, c) == (gr, gc):
            found = True
            break

        for dr, dc, move_cost in nbrs:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if closed[rr, cc] or (not _cell_ok(rr, cc)):
                continue
            # For diagonal expansion, forbid corner-cutting through obstacle corners.
            if dr != 0 and dc != 0:
                if (not _cell_ok(r, cc)) or (not _cell_ok(rr, c)):
                    continue
            edge_clear = float("inf")
            if clear is not None:
                edge_clear = float(min(float(clear[r, c]), float(clear[rr, cc])))
                if dr != 0 and dc != 0:
                    edge_clear = min(edge_clear, float(clear[r, cc]), float(clear[rr, c]))
                if clear_floor > 0.0 and edge_clear < clear_floor:
                    continue
            clear_pen = 0.0
            if clear is not None and clear_cost_w > 0.0:
                short = float(np.clip((clear_soft_eff - edge_clear) / max(clear_soft_eff - clear_floor, 1e-6), 0.0, 1.0))
                clear_pen = clear_cost_w * short * short
            cand = float(g_cur + move_cost + clear_pen)
            if cand + 1e-9 < float(g[rr, cc]):
                g[rr, cc] = cand
                parent_r[rr, cc] = r
                parent_c[rr, cc] = c
                heapq.heappush(open_heap, (cand + _heur(rr, cc), cand, rr, cc))

    if not found:
        return None

    path: List[Tuple[int, int]] = []
    cur = (gr, gc)
    while True:
        path.append(cur)
        if cur == (sr, sc):
            break
        pr = int(parent_r[cur])
        pc = int(parent_c[cur])
        if pr < 0 or pc < 0:
            return None
        cur = (pr, pc)
    path.reverse()
    return path


def _hybrid_astar_path_xy(
    occ: np.ndarray,
    min_xy: np.ndarray,
    res: float,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    start_yaw: Optional[float] = None,
    n_theta: int = 48,
    step_cells: float = 2.5,
    turn_penalty: float = 0.20,
    max_expansions: int = 20_000,
) -> Optional[np.ndarray]:
    """
    Lightweight heading-aware planner on 2D occupancy grid.
    It enforces forward motion primitives and heading continuity so the
    generated path is more driveable than pure 2D A*/BFS centerline paths.
    """
    occ_b = np.asarray(occ, dtype=bool)
    if occ_b.ndim != 2 or occ_b.shape[0] <= 1 or occ_b.shape[1] <= 1:
        return None
    h, w = occ_b.shape
    mn = np.asarray(min_xy, dtype=np.float32).reshape(2)
    rr = float(max(1e-6, res))
    start = np.asarray(start_xy, dtype=np.float32).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        return None

    def _to_rc(pt_xy: np.ndarray) -> Tuple[int, int]:
        c = int(np.clip(np.round((pt_xy[0] - mn[0]) / rr), 0, w - 1))
        r = int(np.clip(np.round((pt_xy[1] - mn[1]) / rr), 0, h - 1))
        return r, c

    def _to_xy(r0: int, c0: int) -> np.ndarray:
        return np.array([mn[0] + c0 * rr, mn[1] + r0 * rr], dtype=np.float32)

    def _wrap_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    theta_bins = int(max(12, n_theta))
    dtheta = float(2.0 * np.pi / theta_bins)

    def _to_tidx(yaw: float) -> int:
        a = float((yaw + np.pi) % (2.0 * np.pi))
        tid = int(np.floor(a / dtheta)) % theta_bins
        return tid

    def _tidx_to_yaw(tid: int) -> float:
        return float(-np.pi + (float(tid) + 0.5) * dtheta)

    start_rc = _to_rc(start)
    goal_rc = _to_rc(goal)
    if occ_b[start_rc] or occ_b[goal_rc]:
        return None

    if start_yaw is None or (not np.isfinite(float(start_yaw))):
        dsg = goal - start
        if float(np.linalg.norm(dsg)) > 1e-6:
            yaw0 = float(np.arctan2(dsg[1], dsg[0]))
        else:
            yaw0 = 0.0
    else:
        yaw0 = float(start_yaw)
    start_tid = _to_tidx(yaw0)

    # Primitive length and heading increments.
    step_len = float(max(1.2 * rr, float(step_cells) * rr))
    turn_step = max(1, int(round(theta_bins / 36)))  # ~10deg for 72 bins
    motion_prims = (-turn_step, 0, turn_step)

    def _cell_free(r0: int, c0: int) -> bool:
        return (0 <= r0 < h) and (0 <= c0 < w) and (not occ_b[r0, c0])

    def _segment_clear(p0: np.ndarray, p1: np.ndarray) -> bool:
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            r0, c0 = _to_rc(p0)
            return _cell_free(r0, c0)
        n = max(2, int(np.ceil(L / max(0.5 * rr, 1e-3))))
        for i in range(n + 1):
            t = float(i) / float(n)
            p = p0 + t * d
            r0, c0 = _to_rc(p)
            if not _cell_free(r0, c0):
                return False
        return True

    start_key = (int(start_rc[0]), int(start_rc[1]), int(start_tid))
    goal_xy_np = goal.astype(np.float32)

    g_cost: Dict[Tuple[int, int, int], float] = {start_key: 0.0}
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    parent_xy: Dict[Tuple[int, int, int], np.ndarray] = {start_key: start.astype(np.float32)}
    open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []

    def _heur(p_xy: np.ndarray) -> float:
        return float(np.linalg.norm(p_xy - goal_xy_np))

    heapq.heappush(open_heap, (_heur(start), 0.0, start_key))
    visited = set()
    best_goal_key: Optional[Tuple[int, int, int]] = None
    best_goal_dist = float("inf")
    expansions = 0
    goal_gate = float(max(1.5 * rr, 0.12))

    while open_heap and expansions < int(max_expansions):
        f_cur, g_cur, key = heapq.heappop(open_heap)
        if key in visited:
            continue
        visited.add(key)
        expansions += 1

        kr, kc, kt = key
        p0 = parent_xy.get(key, _to_xy(kr, kc))
        yaw = _tidx_to_yaw(kt)
        d_goal = float(np.linalg.norm(p0 - goal_xy_np))
        if d_goal < best_goal_dist:
            best_goal_dist = d_goal
            best_goal_key = key
        if d_goal <= goal_gate:
            best_goal_key = key
            break

        for dtid in motion_prims:
            nt = int((kt + dtid) % theta_bins)
            yaw_n = _tidx_to_yaw(nt)
            p1 = p0 + step_len * np.array([np.cos(yaw_n), np.sin(yaw_n)], dtype=np.float32)
            nr, nc = _to_rc(p1)
            nkey = (int(nr), int(nc), int(nt))
            if not _cell_free(nr, nc):
                continue
            if not _segment_clear(p0, p1):
                continue
            move_cost = float(step_len + float(turn_penalty) * abs(float(dtid)) / max(1.0, float(turn_step)))
            cand_g = float(g_cur + move_cost)
            old_g = g_cost.get(nkey, float("inf"))
            if cand_g + 1e-9 < old_g:
                g_cost[nkey] = cand_g
                parent[nkey] = key
                parent_xy[nkey] = p1.astype(np.float32)
                h_n = _heur(p1)
                heapq.heappush(open_heap, (cand_g + h_n, cand_g, nkey))

    if best_goal_key is None:
        return None

    # Reconstruct continuous path from parent_xy.
    chain: List[np.ndarray] = []
    cur = best_goal_key
    while True:
        p = parent_xy.get(cur)
        if p is None:
            p = _to_xy(int(cur[0]), int(cur[1]))
        chain.append(np.asarray(p, dtype=np.float32))
        if cur == start_key:
            break
        if cur not in parent:
            return None
        cur = parent[cur]
    chain.reverse()
    if len(chain) <= 0:
        return None
    path = np.stack(chain, axis=0).astype(np.float32)
    path[0] = start.astype(np.float32)
    path[-1] = goal.astype(np.float32)
    return path


def path_min_clearance_sampled(
    path_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.0,
    sample_step_m: float = 0.05,
) -> float:
    pth = np.asarray(path_xy, dtype=np.float32)
    if pth.ndim != 2 or pth.shape[0] <= 0:
        return float("-inf")
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    if obs.size == 0:
        return float("inf")
    min_clear = float("inf")
    for i in range(max(0, int(pth.shape[0] - 1))):
        c = _segment_min_clearance_sampled(
            start_xy=pth[i],
            goal_xy=pth[i + 1],
            obstacles_xyr=obs,
            robot_radius=robot_radius,
            margin=margin,
            sample_step_m=sample_step_m,
        )
        if c < min_clear:
            min_clear = float(c)
    return float(min_clear)


def path_is_passable_sampled(
    path_xy: np.ndarray,
    obstacles_xyr: np.ndarray,
    robot_radius: float,
    margin: float = 0.0,
    min_seg_clearance: float = 0.01,
    sample_step_m: float = 0.05,
) -> bool:
    min_clear = path_min_clearance_sampled(
        path_xy=path_xy,
        obstacles_xyr=obstacles_xyr,
        robot_radius=robot_radius,
        margin=margin,
        sample_step_m=sample_step_m,
    )
    return bool(np.isfinite(min_clear) and (float(min_clear) >= float(min_seg_clearance)))


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
    planner: str = "astar",
    start_yaw: Optional[float] = None,
    passability_check: bool = True,
    passability_margin: float = 0.0,
    passability_min_clearance: float = 0.01,
    hybrid_n_theta: int = 48,
    hybrid_step_cells: float = 2.5,
    hybrid_turn_penalty: float = 0.20,
    hybrid_max_expansions: int = 20_000,
    debug_info: Optional[Dict[str, object]] = None,
) -> Optional[np.ndarray]:
    if debug_info is not None:
        debug_info.clear()
        debug_info["fail_cause"] = "none"

    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    obs = np.asarray(obstacles_xyr, dtype=np.float32)
    # Unified traversability semantics:
    # - planner occupancy inflation and post-passability check share the same
    #   margin floor to avoid "search passable but post-check reject" divergence.
    clearance_margin = float(max(0.0, float(inflation_margin), float(passability_margin)))
    clearance_floor = float(max(0.0, float(passability_min_clearance)))

    if occ_grid is not None and occ_min_xy is not None and occ_resolution is not None:
        occ_raw = np.asarray(occ_grid, dtype=bool)
        min_xy = np.asarray(occ_min_xy, dtype=np.float32).reshape(2)
        res = float(max(1e-6, occ_resolution))
        inflate_cells = int(
            np.ceil(
                max(0.0, float(robot_radius) + float(clearance_margin))
                / max(res, 1e-6)
            )
        )
        occ = _inflate_occ_grid(occ=occ_raw, inflate_cells=inflate_cells)
        clear_map = _clearance_map_from_occ(occ=occ, res=res)
        clear_soft = float(clearance_floor + max(2.0 * res, 0.08))
        clear_weight = 0.35
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
        planner_mode = str(planner).strip().lower()
        if planner_mode == "bfs":
            rc_path = _bfs_path_cells(
                occ=occ,
                start_rc=start_rc,
                goal_rc=goal_rc,
                clearance_map=clear_map,
                min_clearance=clearance_floor,
            )
            xy_path = None
        elif planner_mode in ("hybrid_astar", "hybrid", "se2"):
            xy_path = _hybrid_astar_path_xy(
                occ=occ,
                min_xy=min_xy,
                res=res,
                start_xy=start,
                goal_xy=goal,
                start_yaw=start_yaw,
                n_theta=int(hybrid_n_theta),
                step_cells=float(hybrid_step_cells),
                turn_penalty=float(hybrid_turn_penalty),
                max_expansions=int(hybrid_max_expansions),
            )
            # Robust fallback: keep global-guide alive when heading-aware search
            # cannot produce a path in tight maps.
            if xy_path is None:
                rc_path = _astar_path_cells(
                    occ=occ,
                    start_rc=start_rc,
                    goal_rc=goal_rc,
                    clearance_map=clear_map,
                    min_clearance=clearance_floor,
                    clearance_soft=clear_soft,
                    clearance_cost_weight=clear_weight,
                )
                if rc_path is None:
                    rc_path = _bfs_path_cells(
                        occ=occ,
                        start_rc=start_rc,
                        goal_rc=goal_rc,
                        clearance_map=clear_map,
                        min_clearance=clearance_floor,
                    )
                if rc_path is not None and len(rc_path) > 0:
                    xy_path = np.stack(
                        [_grid_to_xy(r=r, c=c, min_xy=min_xy, res=res) for r, c in rc_path],
                        axis=0,
                    ).astype(np.float32)
            rc_path = None
        else:
            rc_path = _astar_path_cells(
                occ=occ,
                start_rc=start_rc,
                goal_rc=goal_rc,
                clearance_map=clear_map,
                min_clearance=clearance_floor,
                clearance_soft=clear_soft,
                clearance_cost_weight=clear_weight,
            )
            if rc_path is None:
                rc_path = _bfs_path_cells(
                    occ=occ,
                    start_rc=start_rc,
                    goal_rc=goal_rc,
                    clearance_map=clear_map,
                    min_clearance=clearance_floor,
                )
            xy_path = None
        if xy_path is None:
            if rc_path is None or len(rc_path) == 0:
                if debug_info is not None:
                    debug_info["fail_cause"] = "search_fail"
                return None
            xy_path = np.stack([_grid_to_xy(r=r, c=c, min_xy=min_xy, res=res) for r, c in rc_path], axis=0).astype(np.float32)
        xy_path[0] = start
        xy_path[-1] = goal
        if passability_check and obstacles_xyr is not None and np.asarray(obstacles_xyr).size > 0:
            if not path_is_passable_sampled(
                path_xy=xy_path,
                obstacles_xyr=obstacles_xyr,
                robot_radius=float(robot_radius),
                margin=float(clearance_margin),
                min_seg_clearance=float(clearance_floor),
            ):
                if debug_info is not None:
                    debug_info["fail_cause"] = "passability_reject"
                return None
        if debug_info is not None:
            debug_info["fail_cause"] = "none"
        return xy_path

    if obs.size == 0:
        if debug_info is not None:
            debug_info["fail_cause"] = "none"
        return np.stack([start, goal], axis=0).astype(np.float32)

    occ, min_xy, res, nx, ny = _build_occ_grid(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=obs,
        robot_radius=robot_radius,
        inflation_margin=clearance_margin,
        grid_resolution=grid_resolution,
        grid_padding=grid_padding,
        max_grid_cells=max_grid_cells,
    )
    if occ is None or min_xy is None:
        if debug_info is not None:
            debug_info["fail_cause"] = "grid_build_fail"
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
    clear_map = _clearance_map_from_occ(occ=occ, res=res)
    clear_soft = float(clearance_floor + max(2.0 * res, 0.08))
    clear_weight = 0.35
    planner_mode = str(planner).strip().lower()
    if planner_mode == "bfs":
        rc_path = _bfs_path_cells(
            occ=occ,
            start_rc=start_rc,
            goal_rc=goal_rc,
            clearance_map=clear_map,
            min_clearance=clearance_floor,
        )
        xy_path = None
    elif planner_mode in ("hybrid_astar", "hybrid", "se2"):
        xy_path = _hybrid_astar_path_xy(
            occ=occ,
            min_xy=min_xy,
            res=res,
            start_xy=start,
            goal_xy=goal,
            start_yaw=start_yaw,
            n_theta=int(hybrid_n_theta),
            step_cells=float(hybrid_step_cells),
            turn_penalty=float(hybrid_turn_penalty),
            max_expansions=int(hybrid_max_expansions),
        )
        if xy_path is None:
            rc_path = _astar_path_cells(
                occ=occ,
                start_rc=start_rc,
                goal_rc=goal_rc,
                clearance_map=clear_map,
                min_clearance=clearance_floor,
                clearance_soft=clear_soft,
                clearance_cost_weight=clear_weight,
            )
            if rc_path is None:
                rc_path = _bfs_path_cells(
                    occ=occ,
                    start_rc=start_rc,
                    goal_rc=goal_rc,
                    clearance_map=clear_map,
                    min_clearance=clearance_floor,
                )
            if rc_path is not None and len(rc_path) > 0:
                xy_path = np.stack(
                    [_grid_to_xy(r=r, c=c, min_xy=min_xy, res=res) for r, c in rc_path],
                    axis=0,
                ).astype(np.float32)
        rc_path = None
    else:
        rc_path = _astar_path_cells(
            occ=occ,
            start_rc=start_rc,
            goal_rc=goal_rc,
            clearance_map=clear_map,
            min_clearance=clearance_floor,
            clearance_soft=clear_soft,
            clearance_cost_weight=clear_weight,
        )
        if rc_path is None:
            rc_path = _bfs_path_cells(
                occ=occ,
                start_rc=start_rc,
                goal_rc=goal_rc,
                clearance_map=clear_map,
                min_clearance=clearance_floor,
            )
        xy_path = None
    if xy_path is None:
        if rc_path is None or len(rc_path) == 0:
            if debug_info is not None:
                debug_info["fail_cause"] = "search_fail"
            return None

        xy_path = np.stack([_grid_to_xy(r=r, c=c, min_xy=min_xy, res=res) for r, c in rc_path], axis=0).astype(np.float32)
    xy_path[0] = start
    xy_path[-1] = goal
    if passability_check and obs.size > 0:
        if not path_is_passable_sampled(
            path_xy=xy_path,
            obstacles_xyr=obs,
            robot_radius=float(robot_radius),
            margin=float(clearance_margin),
            min_seg_clearance=float(clearance_floor),
        ):
            if debug_info is not None:
                debug_info["fail_cause"] = "passability_reject"
            return None
    if debug_info is not None:
        debug_info["fail_cause"] = "none"
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
    waypoint_goal_min_dist: float = 0.20,
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

    preferred = 0
    if preferred_side > 0:
        preferred = 1
    elif preferred_side < 0:
        preferred = -1

    side_order = [1, -1] if preferred == 0 else [preferred, -preferred]
    blocker_count = min(3, len(blockers))
    blocker_ids = [int(blockers[k][1]) for k in range(blocker_count)]
    blocker_refs = [float(blockers[k][2]) for k in range(blocker_count)]

    forward_offsets = np.asarray([-0.14, -0.08, -0.02, 0.05, 0.12], dtype=np.float32) * float(L)
    lateral_scales = (1.00, 1.25, 1.55, 1.85)
    clearance_req = float(max(0.02, 0.10 * float(robot_radius)))

    candidates: List[Tuple[float, int, np.ndarray, float, bool, bool, float, float, float]] = []
    for bi, r_eff in zip(blocker_ids, blocker_refs):
        c = obs[int(bi), :2]
        offset_base = float(r_eff + max(0.0, float(lateral_extra)))
        for side in side_order:
            for fwd in forward_offsets:
                for lat_scale in lateral_scales:
                    w = c + float(side) * p * (offset_base * float(lat_scale)) + d * float(fwd)
                    t_line, lat_line = line_signed_lateral(start_xy=a, goal_xy=b, point_xy=w)
                    fwd_m = float(t_line) * L
                    if waypoint_max_lateral is not None and abs(float(lat_line)) > float(max(0.0, waypoint_max_lateral)):
                        continue
                    if fwd_m < float(max(0.0, waypoint_min_forward)):
                        continue
                    if waypoint_max_forward is not None and fwd_m > float(max(0.0, waypoint_max_forward)):
                        continue
                    if float(np.linalg.norm(w - b)) < float(max(0.0, waypoint_goal_min_dist)):
                        continue

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
                    seg1_clear = _segment_min_clearance_sampled(
                        start_xy=a,
                        goal_xy=w,
                        obstacles_xyr=obs,
                        robot_radius=robot_radius,
                        margin=margin,
                    )
                    seg2_clear = _segment_min_clearance_sampled(
                        start_xy=w,
                        goal_xy=b,
                        obstacles_xyr=obs,
                        robot_radius=robot_radius,
                        margin=margin,
                    )
                    point_clear = _point_min_clearance(
                        point_xy=w,
                        obstacles_xyr=obs,
                        robot_radius=robot_radius,
                        margin=margin,
                    )
                    min_seg_clear = float(min(seg1_clear, seg2_clear))
                    path_len = float(np.linalg.norm(a - w) + np.linalg.norm(w - b))

                    clear_short = float(max(0.0, clearance_req - min_seg_clear))
                    point_short = float(max(0.0, 0.8 * clearance_req - point_clear))
                    los_pen = (3.0 if blocked_a_w else 0.0) + (5.0 if blocked_w_b else 0.0)
                    pref_pen = 0.0 if (preferred == 0 or side == preferred) else 0.30
                    fwd_pen = float(max(0.0, 0.18 - fwd_m))
                    score = (
                        path_len
                        + 55.0 * (clear_short**2)
                        + 35.0 * (point_short**2)
                        + 7.5 * los_pen
                        + 0.45 * fwd_pen
                        + pref_pen
                        - 0.30 * float(np.clip(min_seg_clear, 0.0, 0.30))
                    )
                    candidates.append(
                        (
                            score,
                            int(side),
                            np.asarray(w, dtype=np.float32),
                            float(min_seg_clear),
                            bool(blocked_a_w),
                            bool(blocked_w_b),
                            float(path_len),
                            float(point_clear),
                            float(fwd_m),
                        )
                    )

    if len(candidates) <= 0:
        return None, 0

    # Passability-first gating:
    # 1) both segments LOS-clear + enough sampled clearance
    # 2) both segments LOS-clear + weak clearance floor
    # 3) fallback to all candidates when scene is very tight
    tier1 = [
        c for c in candidates
        if (not c[4]) and (not c[5]) and (c[3] >= clearance_req)
    ]
    tier2 = [
        c for c in candidates
        if (not c[4]) and (not c[5]) and (c[3] >= -0.005)
    ]
    pool = tier1 if len(tier1) > 0 else (tier2 if len(tier2) > 0 else candidates)
    pool.sort(key=lambda x: x[0])
    best = pool[0]
    return np.asarray(best[2], dtype=np.float32), int(best[1])
