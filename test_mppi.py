from run_icode_mppi_e1_test import build_argparser, run_episode


def main() -> None:
    args = build_argparser().parse_args()
    args.max_steps = 20
    args.num_samples = 256
    args.horizon = 20
    metrics = run_episode(args)
    print(metrics)


if __name__ == "__main__":
    main()
