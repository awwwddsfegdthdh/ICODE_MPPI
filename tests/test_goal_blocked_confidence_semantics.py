import numpy as np

from mppi_nav_utils import line_of_sight_blocked_confidence


def test_unknown_cells_do_not_force_goal_blocked() -> None:
    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([2.0, 0.0], dtype=np.float32)
    occ = np.ones((80, 80), dtype=bool)
    observed = np.zeros_like(occ, dtype=bool)

    blocked, conf, observed_cells, sampled_cells = line_of_sight_blocked_confidence(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=np.zeros((0, 3), dtype=np.float32),
        robot_radius=0.2,
        margin=0.0,
        occ_grid=occ,
        occ_observed=observed,
        occ_min_xy=np.array([-1.0, -1.0], dtype=np.float32),
        occ_resolution=0.05,
        blocked_conf_threshold=0.65,
    )

    assert blocked is False
    assert conf == 0.0
    assert observed_cells == 0
    assert sampled_cells > 0
