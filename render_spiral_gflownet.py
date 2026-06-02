import argparse
from pathlib import Path

from spiral_gflownet_core import render_autoregressive_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render transparent density GIFs from saved spiral GFlowNet snapshots.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--snapshot-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = render_autoregressive_run(
        run_dir=Path(args.run_dir),
        transparent=args.transparent,
        snapshot_stride=args.snapshot_stride,
    )
    print(f"aggregate_gif: {artifacts['aggregate_gif']}")
    print(f"selected_gif: {artifacts['selected_gif']}")


if __name__ == "__main__":
    main()
