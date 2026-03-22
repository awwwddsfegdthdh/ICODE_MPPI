import numpy as np

from mppi_nav_utils import plan_global_path_xy


def test_bfs_path_on_occ_grid():
    res = 0.05
    x_min, y_min = -1.0, -1.0
    nx, ny = 100, 80
    occ = np.zeros((ny, nx), dtype=bool)

    wall_x = int(round((0.8 - x_min) / res))
    for r in range(10, ny - 10):
        if 36 <= r <= 44:
            continue
        occ[r, wall_x] = True

    path = plan_global_path_xy(
        start_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal_xy=np.array([1.8, 0.0], dtype=np.float32),
        obstacles_xyr=np.zeros((0, 3), dtype=np.float32),
        robot_radius=0.18,
        occ_grid=occ,
        occ_min_xy=np.array([x_min, y_min], dtype=np.float32),
        occ_resolution=res,
    )
    assert path is not None
    assert path.shape[0] >= 2


def test_bfs_occ_grid_respects_robot_radius_clearance():
    res = 0.05
    x_min, y_min = -1.0, -1.0
    nx, ny = 120, 90
    occ = np.zeros((ny, nx), dtype=bool)

    wall_x = int(round((0.8 - x_min) / res))
    for r in range(0, ny):
        # Keep a narrow doorway (~0.25m).
        if 42 <= r <= 46:
            continue
        occ[r, wall_x] = True

    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([1.8, 0.0], dtype=np.float32)
    min_xy = np.array([x_min, y_min], dtype=np.float32)

    path_small = plan_global_path_xy(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=np.zeros((0, 3), dtype=np.float32),
        robot_radius=0.05,
        inflation_margin=0.0,
        occ_grid=occ,
        occ_min_xy=min_xy,
        occ_resolution=res,
    )
    path_big = plan_global_path_xy(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=np.zeros((0, 3), dtype=np.float32),
        robot_radius=0.20,
        inflation_margin=0.0,
        occ_grid=occ,
        occ_min_xy=min_xy,
        occ_resolution=res,
    )
    assert path_small is not None
    assert path_big is None


def test_bfs_no_corner_cut_through_diagonal_gap():
    # Obstacles adjacent to start in +x and +y directions.
    # Diagonal shortcut should be disallowed; BFS must detour.
    res = 0.1
    x_min, y_min = -0.2, -0.2
    nx, ny = 8, 8
    occ = np.zeros((ny, nx), dtype=bool)

    # world (0.1, 0.0) -> (r=2,c=3), world (0.0,0.1) -> (r=3,c=2)
    occ[2, 3] = True
    occ[3, 2] = True

    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([0.3, 0.3], dtype=np.float32)
    path = plan_global_path_xy(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=np.zeros((0, 3), dtype=np.float32),
        robot_radius=0.0,
        inflation_margin=0.0,
        occ_grid=occ,
        occ_min_xy=np.array([x_min, y_min], dtype=np.float32),
        occ_resolution=res,
    )
    assert path is not None
    # Direct diagonal (start -> mid -> goal) would be 3 points.
    assert path.shape[0] > 3
