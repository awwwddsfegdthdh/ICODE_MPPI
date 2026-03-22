from run_icode_mppi_e1_test import build_argparser, run_episode


def main() -> None:
    args = build_argparser().parse_args()
    args.planner_model = "kinematic"
    args.kinematic_lock = True
    args.sensor_use_depth = False
    args.max_steps = 20
    args.num_samples = 256
    args.horizon = 20
    metrics = run_episode(args)
    if "supervisor_summary" not in metrics:
        raise RuntimeError("stage3 supervisor summary missing in metrics")
    if "action_post_delta_ratio" not in metrics:
        raise RuntimeError("stage4 action consistency metric missing in metrics")
    if "cost_group_mean" not in metrics:
        raise RuntimeError("stage4 cost-group summary missing in metrics")
    print(metrics)


if __name__ == "__main__":
    main()
