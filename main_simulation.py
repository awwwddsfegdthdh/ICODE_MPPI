import json

from run_icode_mppi_e1_test import build_argparser, run_episode


def main() -> None:
    args = build_argparser().parse_args()
    metrics = run_episode(args)
    print("Saved outputs to:", args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
