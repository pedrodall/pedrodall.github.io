import argparse
from pathlib import Path

from spiral_gflownet_core import SWEEP_CONFIGS, train_autoregressive_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one autoregressive spiral GFlowNet run.")
    parser.add_argument("--config-name", required=True, choices=sorted(SWEEP_CONFIGS.keys()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--snapshot-every", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SWEEP_CONFIGS[args.config_name]
    summary = train_autoregressive_run(
        config=config,
        run_dir=Path(args.output_dir),
        num_steps_override=args.num_steps,
        snapshot_every_override=args.snapshot_every,
        validation_samples_override=args.validation_samples,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )
    print(f"completed run: {summary['config_name']}")
    print(f"summary: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
