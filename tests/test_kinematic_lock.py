from run_icode_mppi_e1_test import build_argparser


def test_defaults_are_kinematic_locked_and_depth_off():
    args = build_argparser().parse_args([])
    assert args.planner_model == "kinematic"
    assert args.kinematic_lock is True
    assert args.sensor_use_depth is False
