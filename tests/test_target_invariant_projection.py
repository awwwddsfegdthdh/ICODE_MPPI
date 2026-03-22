import numpy as np

from mppi_nav_utils import line_signed_lateral, project_target_with_invariants


def test_target_projection_enforces_bounds_corridor_and_jump() -> None:
    target, info = project_target_with_invariants(
        candidate_xy=np.array([5.0, 2.0], dtype=np.float32),
        prev_target_xy=np.array([0.2, 0.0], dtype=np.float32),
        start_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal_xy=np.array([3.0, 0.0], dtype=np.float32),
        bounds_x_range=(-0.8, 3.2),
        bounds_y_range=(-2.0, 2.0),
        bound_margin=0.10,
        corridor_max_dev=0.50,
        jump_max=0.45,
        path_xy=None,
    )

    assert -0.7 <= float(target[0]) <= 3.1
    assert -1.9 <= float(target[1]) <= 1.9
    _, lat = line_signed_lateral(
        start_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal_xy=np.array([3.0, 0.0], dtype=np.float32),
        point_xy=target,
    )
    assert abs(float(lat)) <= 0.5001
    assert float(np.linalg.norm(target - np.array([0.2, 0.0], dtype=np.float32))) <= 0.4501
    assert float(info["bound_projected"]) > 0.5
    assert float(info["corridor_projected"]) > 0.5
    assert float(info["jump_projected"]) > 0.5
