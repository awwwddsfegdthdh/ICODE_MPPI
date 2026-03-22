import numpy as np

from mppi_nav_utils import line_of_sight_blocked


def test_los_consistency_map_vs_geom():
    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([2.0, 0.0], dtype=np.float32)
    obstacles = np.array([[1.0, 0.0, 0.20]], dtype=np.float32)

    blocked_geom = line_of_sight_blocked(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=obstacles,
        robot_radius=0.2,
        margin=0.0,
    )

    res = 0.05
    x_min, y_min = -1.0, -1.0
    nx, ny = 80, 80
    occ = np.zeros((ny, nx), dtype=bool)
    cx = int(round((1.0 - x_min) / res))
    cy = int(round((0.0 - y_min) / res))
    rr = int(round((0.20 + 0.2) / res))
    for r in range(ny):
        for c in range(nx):
            if (r - cy) ** 2 + (c - cx) ** 2 <= rr * rr:
                occ[r, c] = True

    blocked_map = line_of_sight_blocked(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=np.zeros((0, 3), dtype=np.float32),
        robot_radius=0.2,
        margin=0.0,
        occ_grid=occ,
        occ_min_xy=np.array([x_min, y_min], dtype=np.float32),
        occ_resolution=res,
    )

    assert blocked_geom is True
    assert blocked_map is True
