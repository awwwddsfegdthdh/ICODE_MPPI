import numpy as np
import torch

from mppi import DiffDriveKinematicModel, MPPIController


def _controller() -> MPPIController:
    model = DiffDriveKinematicModel()
    return MPPIController(
        dynamics_model=model,
        num_samples=16,
        horizon=3,
        device="cpu",
        cost_task_goal=0.0,
        cost_task_final=0.0,
        cost_task_progress=0.0,
        cost_task_heading=0.0,
        cost_task_reverse=0.0,
        cost_task_path_track=0.0,
        cost_task_path_terminal=0.0,
        cost_task_path_progress=10.0,
        cost_task_lateral=0.0,
        cost_task_goal_visibility=0.0,
        cost_safe_collision=0.0,
        cost_safe_near=0.0,
        cost_safe_bounds=0.0,
        cost_safe_bounds_terminal=0.0,
        cost_ctrl_effort=0.0,
        cost_ctrl_smooth=0.0,
        cost_ctrl_spin=0.0,
        cost_ctrl_wheel_diff=0.0,
        cost_terminal_progress_deficit=0.0,
        cost_terminal_near_goal_stall=0.0,
        cost_terminal_stop=0.0,
        cost_terminal_overshoot=0.0,
    )


def test_legacy_redundant_terms_removed_from_controller() -> None:
    ctrl = _controller()
    assert not hasattr(ctrl, "w_away_goal")
    assert not hasattr(ctrl, "w_goal_motion_away")
    assert not hasattr(ctrl, "w_reverse_away")
    assert not hasattr(ctrl, "w_backward_step")
    assert not hasattr(ctrl, "w_path_backtrack")


def test_signed_path_progress_without_backtrack_duplicate_penalty() -> None:
    ctrl = _controller()

    # rollout A: forward then backward (sum path motion = 0)
    # rollout B: no motion (sum path motion = 0)
    states = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    actions = torch.zeros((2, 3, 2), dtype=torch.float32)
    target_xy = torch.tensor([2.0, 0.0], dtype=torch.float32)
    obstacles = torch.zeros((0, 3), dtype=torch.float32)
    init_dist = torch.tensor([2.0, 2.0], dtype=torch.float32)
    init_pos_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    reference_traj = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    cost, terms = ctrl.compute_cost(
        states=states,
        actions=actions,
        target_xy=target_xy,
        obstacles=obstacles,
        init_dist=init_dist,
        init_pos_xy=init_pos_xy,
        reference_traj=reference_traj,
        return_terms=True,
    )
    assert set(terms.keys()) == {"task", "safety", "control", "terminal", "total"}
    assert np.isclose(float(cost[0].item()), float(cost[1].item()), atol=1e-6)
