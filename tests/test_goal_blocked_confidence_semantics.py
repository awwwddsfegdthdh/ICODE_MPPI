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


def test_geometry_evidence_can_block_when_map_conf_is_low() -> None:
    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([2.0, 0.0], dtype=np.float32)
    occ = np.zeros((80, 80), dtype=bool)
    observed = np.ones_like(occ, dtype=bool)
    # A narrow obstacle that intersects the start-goal segment.
    obstacles = np.array([[1.0, 0.02, 0.22]], dtype=np.float32)

    blocked, conf, observed_cells, sampled_cells = line_of_sight_blocked_confidence(
        start_xy=start,
        goal_xy=goal,
        obstacles_xyr=obstacles,
        robot_radius=0.2,
        margin=0.0,
        occ_grid=occ,
        occ_observed=observed,
        occ_min_xy=np.array([-1.0, -1.0], dtype=np.float32),
        occ_resolution=0.05,
        blocked_conf_threshold=0.65,
    )

    assert blocked is True
    assert conf >= 0.65
    assert observed_cells > 0
    assert sampled_cells > 0
