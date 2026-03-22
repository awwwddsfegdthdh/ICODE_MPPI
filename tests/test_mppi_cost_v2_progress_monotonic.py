import torch

from mppi import DiffDriveKinematicModel, MPPIController


def test_progress_better_rollout_has_lower_cost() -> None:
    ctrl = MPPIController(
        dynamics_model=DiffDriveKinematicModel(),
        num_samples=16,
        horizon=3,
        device="cpu",
        cost_task_goal=0.0,
        cost_task_final=40.0,
        cost_task_progress=80.0,
        cost_task_heading=0.0,
        cost_task_reverse=0.0,
        cost_task_path_track=0.0,
        cost_task_path_terminal=0.0,
        cost_task_path_progress=0.0,
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

    # rollout 0 has better progress and closer terminal distance than rollout 1
    states = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    actions = torch.zeros((2, 3, 2), dtype=torch.float32)
    target_xy = torch.tensor([2.0, 0.0], dtype=torch.float32)
    obstacles = torch.zeros((0, 3), dtype=torch.float32)
    init_dist = torch.tensor([2.0, 2.0], dtype=torch.float32)
    init_pos_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)

    cost = ctrl.compute_cost(
        states=states,
        actions=actions,
        target_xy=target_xy,
        obstacles=obstacles,
        init_dist=init_dist,
        init_pos_xy=init_pos_xy,
    )
    assert float(cost[0].item()) < float(cost[1].item())
