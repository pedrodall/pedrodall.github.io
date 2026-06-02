import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from spiral_gflownet_core import (
    DEFAULT_OUTPUT_ROOT,
    SWEEP_CONFIG_NAMES,
    best_summary,
    build_leaderboard_key,
    load_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autoregressive spiral GFlowNet sweep and render the winner.")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--snapshot-every", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--transparent", action="store_true")
    return parser.parse_args()


def train_command(config_name: str, run_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "train_spiral_gflownet.py",
        "--config-name",
        config_name,
        "--output-dir",
        str(run_dir),
        "--torch-num-threads",
        "2",
        "--torch-num-interop-threads",
        "1",
    ]
    if args.num_steps is not None:
        command.extend(["--num-steps", str(args.num_steps)])
    if args.snapshot_every is not None:
        command.extend(["--snapshot-every", str(args.snapshot_every)])
    if args.validation_samples is not None:
        command.extend(["--validation-samples", str(args.validation_samples)])
    return command


def render_command(run_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "render_spiral_gflownet.py",
        "--run-dir",
        str(run_dir),
    ]
    if args.transparent:
        command.append("--transparent")
    return command


def run_subprocess(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"

    run_dirs = {
        config_name: output_root / f"run_{config_name}"
        for config_name in SWEEP_CONFIG_NAMES
    }

    max_workers = min(args.max_workers, len(SWEEP_CONFIG_NAMES))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for config_name in SWEEP_CONFIG_NAMES:
            futures.append(
                executor.submit(
                    run_subprocess,
                    train_command(config_name, run_dirs[config_name], args),
                    env,
                )
            )
        for future in futures:
            future.result()

    summaries = [load_json(run_dirs[name] / "summary.json") for name in SWEEP_CONFIG_NAMES]
    summaries.sort(key=build_leaderboard_key)
    leaderboard = {"runs": summaries}
    write_json(output_root / "leaderboard.json", leaderboard)

    winner = best_summary(summaries)
    winner_run_dir = Path(winner["run_dir"])
    (output_root / "best_run.txt").write_text(str(winner_run_dir) + "\n", encoding="utf-8")

    run_subprocess(render_command(winner_run_dir, args), env)

    print(f"winner: {winner['config_name']}")
    print(f"leaderboard: {output_root / 'leaderboard.json'}")
    print(f"best_run: {winner_run_dir}")


if __name__ == "__main__":
    main()
