"""
Shared utilities for the spiral conditional GFlowNet experiment.

This module contains:
- problem geometry and context generation
- autoregressive and iterative policy definitions
- training and evaluation helpers
- snapshot and artifact I/O
- rendering helpers for transparent density GIFs
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

plt.switch_backend("Agg")


C = 6 * np.pi
a = 0.2
b = 0.08
sigma = 0.12

DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_STEPS = 1000
DEFAULT_SNAPSHOT_EVERY = 5
DEFAULT_SEED = 0
DEFAULT_VALIDATION_SAMPLES = 256
DEFAULT_GRID_SIZE = 200
DEFAULT_N_TRAIN_CONTEXTS = 12
DEFAULT_N_UNIFORM_CONTEXTS = 80
DEFAULT_SELECTED_TRAIN_CONTEXT_COUNT = 4
DEFAULT_FPS = 8

T = 6
reward_eps = 1e-12
polar_eps = 1e-6
log_std_min = -3.5
log_std_max = 0.0
plot_margin = 4 * sigma
xmin = -(a + b * C + plot_margin)
xmax = a + b * C + plot_margin
ymin = xmin
ymax = xmax
iterative_step_scale = (xmax - xmin) / (2 * T)

DEFAULT_OUTPUT_ROOT = Path("outputs/spiral_gflownet_sweep")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden_dim: int
    context_K: int
    policy_lr: float
    logz_lr: float
    batch_size: int = DEFAULT_BATCH_SIZE
    num_steps: int = DEFAULT_NUM_STEPS
    snapshot_every: int = DEFAULT_SNAPSHOT_EVERY
    seed: int = DEFAULT_SEED
    validation_samples: int = DEFAULT_VALIDATION_SAMPLES
    n_train_contexts: int = DEFAULT_N_TRAIN_CONTEXTS
    n_uniform_contexts: int = DEFAULT_N_UNIFORM_CONTEXTS
    selected_train_context_count: int = DEFAULT_SELECTED_TRAIN_CONTEXT_COUNT
    grid_size: int = DEFAULT_GRID_SIZE
    fps: int = DEFAULT_FPS


SWEEP_CONFIGS: dict[str, ExperimentConfig] = {
    "ar_small_stable": ExperimentConfig(
        name="ar_small_stable",
        hidden_dim=64,
        context_K=4,
        policy_lr=1e-3,
        logz_lr=1e-2,
    ),
    "ar_base_stable": ExperimentConfig(
        name="ar_base_stable",
        hidden_dim=128,
        context_K=4,
        policy_lr=1e-3,
        logz_lr=1e-2,
    ),
    "ar_base_faster": ExperimentConfig(
        name="ar_base_faster",
        hidden_dim=128,
        context_K=4,
        policy_lr=3e-3,
        logz_lr=3e-2,
    ),
    "ar_fourier_richer": ExperimentConfig(
        name="ar_fourier_richer",
        hidden_dim=128,
        context_K=6,
        policy_lr=1e-3,
        logz_lr=1e-2,
    ),
    "ar_large_conservative": ExperimentConfig(
        name="ar_large_conservative",
        hidden_dim=256,
        context_K=6,
        policy_lr=3e-4,
        logz_lr=3e-3,
    ),
}

SWEEP_CONFIG_NAMES = list(SWEEP_CONFIGS.keys())


def set_seed(seed_value: int) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def configure_worker_threads(num_threads: int = 2, interop_threads: int = 1) -> None:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(interop_threads)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def spiral_center(contexts: torch.Tensor) -> torch.Tensor:
    radius = a + b * contexts
    x = radius * torch.cos(contexts)
    y = radius * torch.sin(contexts)
    return torch.stack([x, y], dim=-1)


def principal_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def context_polar_center(contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    radius = a + b * contexts
    log_r = torch.log(radius + polar_eps)
    theta = principal_angle(contexts)
    return log_r, theta


def reward_fn(x: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
    centers = spiral_center(contexts)
    sq_dist = ((x - centers) ** 2).sum(dim=-1)
    return torch.exp(-sq_dist / (2 * sigma**2))


def true_density_fn(x: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
    normalizer = 2.0 * math.pi * sigma**2
    return reward_fn(x, contexts) / normalizer


def make_context_features(contexts: torch.Tensor, context_K: int) -> torch.Tensor:
    scaled = contexts / C
    features = [scaled.unsqueeze(-1)]
    for k in range(context_K):
        angle = (2**k) * 2.0 * math.pi * scaled
        features.append(torch.sin(angle).unsqueeze(-1))
        features.append(torch.cos(angle).unsqueeze(-1))
    return torch.cat(features, dim=-1)


def make_angle_features(theta: torch.Tensor, context_K: int) -> torch.Tensor:
    scaled = theta / math.pi
    features = [scaled.unsqueeze(-1)]
    for k in range(context_K):
        angle = (2**k) * theta
        features.append(torch.sin(angle).unsqueeze(-1))
        features.append(torch.cos(angle).unsqueeze(-1))
    return torch.cat(features, dim=-1)


def raw_to_theta(raw_theta: torch.Tensor) -> torch.Tensor:
    return math.pi * torch.tanh(raw_theta)


def theta_to_raw(theta: torch.Tensor) -> torch.Tensor:
    scaled = torch.clamp(theta / math.pi, -1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))


def polar_to_xy(log_r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r = torch.exp(log_r)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def xy_to_polar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = x[:, 0]
    x2 = x[:, 1]
    r = torch.sqrt(x1**2 + x2**2 + polar_eps)
    log_r = torch.log(r)
    theta = torch.atan2(x2, x1)
    return log_r, theta


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def clamp_log_std(log_std: torch.Tensor) -> torch.Tensor:
    return torch.clamp(log_std, min=log_std_min, max=log_std_max)


class AutoregressivePolicy(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.log_prob_scale = 1.0
        context_dim = 1 + 2 * config.context_K
        angle_dim = 1 + 2 * config.context_K
        self.theta_net = build_mlp(context_dim, 2, config.hidden_dim)
        self.log_r_net = build_mlp(context_dim + angle_dim, 2, config.hidden_dim)
        self.logz_net = build_mlp(context_dim, 1, config.hidden_dim)

    def context_features(self, contexts: torch.Tensor) -> torch.Tensor:
        return make_context_features(contexts, self.config.context_K)

    def sample_and_log_prob(self, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_features = self.context_features(contexts)
        base_log_r, base_theta = context_polar_center(contexts)
        base_theta_raw = theta_to_raw(base_theta)

        theta_stats = self.theta_net(context_features)
        theta_raw_mean = base_theta_raw + 0.5 * torch.tanh(theta_stats[:, 0])
        theta_raw_log_std = clamp_log_std(theta_stats[:, 1])
        theta_raw_dist = Normal(theta_raw_mean, theta_raw_log_std.exp())
        theta_raw = theta_raw_dist.rsample()
        theta = raw_to_theta(theta_raw)
        theta_log_abs_det = math.log(math.pi) + torch.log1p(-torch.tanh(theta_raw) ** 2 + polar_eps)
        log_prob_theta = theta_raw_dist.log_prob(theta_raw) - theta_log_abs_det

        angle_features = make_angle_features(theta, self.config.context_K)
        log_r_input = torch.cat([context_features, angle_features], dim=-1)
        log_r_stats = self.log_r_net(log_r_input)
        log_r_mean = base_log_r + 0.5 * torch.tanh(log_r_stats[:, 0])
        log_r_log_std = clamp_log_std(log_r_stats[:, 1])
        log_r_dist = Normal(log_r_mean, log_r_log_std.exp())
        log_r = log_r_dist.rsample()
        log_prob_log_r = log_r_dist.log_prob(log_r)

        samples = polar_to_xy(log_r, theta)
        log_prob = log_prob_theta + log_prob_log_r - 2.0 * log_r
        logz = self.logz_net(context_features).squeeze(-1)
        return samples, log_prob, logz

    def log_prob_exact(self, x: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        context_features = self.context_features(contexts)
        base_log_r, base_theta = context_polar_center(contexts)
        base_theta_raw = theta_to_raw(base_theta)
        log_r, theta = xy_to_polar(x)

        theta_stats = self.theta_net(context_features)
        theta_raw_mean = base_theta_raw + 0.5 * torch.tanh(theta_stats[:, 0])
        theta_raw_log_std = clamp_log_std(theta_stats[:, 1])
        theta_raw_dist = Normal(theta_raw_mean, theta_raw_log_std.exp())
        theta_raw = theta_to_raw(theta)
        theta_log_abs_det = math.log(math.pi) + torch.log1p(-torch.tanh(theta_raw) ** 2 + polar_eps)
        log_prob_theta = theta_raw_dist.log_prob(theta_raw) - theta_log_abs_det

        angle_features = make_angle_features(theta, self.config.context_K)
        log_r_input = torch.cat([context_features, angle_features], dim=-1)
        log_r_stats = self.log_r_net(log_r_input)
        log_r_mean = base_log_r + 0.5 * torch.tanh(log_r_stats[:, 0])
        log_r_log_std = clamp_log_std(log_r_stats[:, 1])
        log_r_dist = Normal(log_r_mean, log_r_log_std.exp())
        log_prob_log_r = log_r_dist.log_prob(log_r)
        return log_prob_theta + log_prob_log_r - 2.0 * log_r

    def logz(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.logz_net(self.context_features(contexts)).squeeze(-1)


class IterativePolicy(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.log_prob_scale = 1.0
        context_dim = 1 + 2 * config.context_K
        self.forward_net = build_mlp(context_dim + 2 + 1, 4, config.hidden_dim)
        self.backward_net = build_mlp(context_dim + 2 + 1, 4, config.hidden_dim)
        self.logz_net = build_mlp(context_dim, 1, config.hidden_dim)

    def transition_dist(self, contexts: torch.Tensor, x: torch.Tensor, step: int) -> Normal:
        batch = contexts.shape[0]
        context_features = make_context_features(contexts, self.config.context_K)
        step_feature = torch.full((batch, 1), fill_value=step / max(T - 1, 1), device=contexts.device)
        net_input = torch.cat([context_features, x, step_feature], dim=-1)
        step_stats = self.forward_net(net_input)
        delta_mean = iterative_step_scale * torch.tanh(step_stats[:, :2])
        delta_log_std = clamp_log_std(step_stats[:, 2:] - 1.5)
        next_mean = x + delta_mean
        return Normal(next_mean, delta_log_std.exp())

    def backward_dist(self, contexts: torch.Tensor, x_t: torch.Tensor, step: int) -> Normal:
        if step <= 1:
            raise ValueError("backward_dist is only defined for step >= 2")

        batch = contexts.shape[0]
        context_features = make_context_features(contexts, self.config.context_K)
        step_feature = torch.full((batch, 1), fill_value=(step - 1) / max(T - 1, 1), device=contexts.device)
        net_input = torch.cat([context_features, x_t, step_feature], dim=-1)
        step_stats = self.backward_net(net_input)
        delta_mean = iterative_step_scale * torch.tanh(step_stats[:, :2])
        delta_log_std = clamp_log_std(step_stats[:, 2:] - 1.5)
        prev_mean = x_t + delta_mean
        return Normal(prev_mean, delta_log_std.exp())


def build_arc_length_lookup(num_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, C, num_points, dtype=np.float64)
    radius = a + b * theta
    speed = np.sqrt(radius**2 + b**2)
    delta_theta = np.diff(theta)
    delta_arc = 0.5 * (speed[1:] + speed[:-1]) * delta_theta
    cumulative_arc = np.concatenate([[0.0], np.cumsum(delta_arc)])
    cumulative_arc /= cumulative_arc[-1]
    return theta, cumulative_arc


def arc_fraction_to_theta(fractions: np.ndarray, theta_lookup: np.ndarray, arc_lookup: np.ndarray) -> np.ndarray:
    return np.interp(fractions, arc_lookup, theta_lookup).astype(np.float32)


def make_context_splits(config: ExperimentConfig, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_lookup, arc_lookup = build_arc_length_lookup()
    train_fracs = np.linspace(0.0, 1.0, config.n_train_contexts)
    test_fracs = 0.5 * (train_fracs[:-1] + train_fracs[1:])
    uniform_fracs = np.linspace(0.0, 1.0, config.n_uniform_contexts)

    train_np = arc_fraction_to_theta(train_fracs, theta_lookup, arc_lookup)
    test_np = arc_fraction_to_theta(test_fracs, theta_lookup, arc_lookup)
    uniform_np = arc_fraction_to_theta(uniform_fracs, theta_lookup, arc_lookup)

    train_contexts = torch.tensor(train_np, dtype=torch.float32, device=device)
    test_contexts = torch.tensor(test_np, dtype=torch.float32, device=device)
    uniform_contexts = torch.tensor(uniform_np, dtype=torch.float32, device=device)
    return train_contexts, test_contexts, uniform_contexts


def select_train_contexts_for_panels(train_contexts: torch.Tensor, count: int) -> torch.Tensor:
    indices = np.linspace(0, train_contexts.shape[0] - 1, count, dtype=int)
    unique_indices: list[int] = []
    for idx in indices.tolist():
        if idx not in unique_indices:
            unique_indices.append(idx)
    return train_contexts[unique_indices]


def build_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return xs, ys, grid_points


def ensure_grid_tensor(grid_points: np.ndarray | torch.Tensor, device: str) -> torch.Tensor:
    if isinstance(grid_points, torch.Tensor):
        return grid_points.to(device=device, dtype=torch.float32)
    return torch.tensor(grid_points, dtype=torch.float32, device=device)


def induced_density_grid_autoregressive(
    model: AutoregressivePolicy,
    grid_points: np.ndarray | torch.Tensor,
    context_value: float,
    device: str,
    grid_size: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    grid_tensor = ensure_grid_tensor(grid_points, device)
    densities = []
    context_array = torch.full((chunk_size,), float(context_value), device=device)

    with torch.no_grad():
        for start in range(0, grid_tensor.shape[0], chunk_size):
            end = min(start + chunk_size, grid_tensor.shape[0])
            chunk = grid_tensor[start:end]
            contexts = context_array[: end - start]
            log_prob = model.log_prob_exact(chunk, contexts)
            densities.append(torch.exp(log_prob).detach().cpu())

    flat_density = torch.cat(densities, dim=0).numpy()
    return flat_density.reshape(grid_size, grid_size)


def aggregated_density_grid_autoregressive(
    model: AutoregressivePolicy,
    grid_points: np.ndarray | torch.Tensor,
    context_values: np.ndarray,
    device: str,
    grid_size: int,
) -> np.ndarray:
    aggregate = np.zeros((grid_size, grid_size), dtype=np.float32)
    for context_value in context_values:
        aggregate += induced_density_grid_autoregressive(model, grid_points, float(context_value), device, grid_size)
    return aggregate


def true_density_grid(
    grid_points: np.ndarray | torch.Tensor,
    context_value: float,
    device: str,
    grid_size: int,
) -> np.ndarray:
    context_array = np.full((len(grid_points),), context_value, dtype=np.float32)
    x_tensor = ensure_grid_tensor(grid_points, device)
    c_tensor = torch.tensor(context_array, dtype=torch.float32, device=device)
    density = true_density_fn(x_tensor, c_tensor)
    return density.detach().cpu().numpy().reshape(grid_size, grid_size)


def aggregated_true_density_grid(
    grid_points: np.ndarray | torch.Tensor,
    context_values: np.ndarray,
    device: str,
    grid_size: int,
) -> np.ndarray:
    aggregate = np.zeros((grid_size, grid_size), dtype=np.float32)
    for context_value in context_values:
        aggregate += true_density_grid(grid_points, float(context_value), device, grid_size)
    return aggregate


def spiral_xy(context_value: float) -> tuple[float, float]:
    radius = a + b * context_value
    return radius * math.cos(context_value), radius * math.sin(context_value)


def build_spiral_path(num_points: int = 400) -> np.ndarray:
    contexts = np.linspace(0.0, C, num_points)
    coords = [spiral_xy(value) for value in contexts]
    return np.array(coords, dtype=np.float32)


def build_optimizer(model: AutoregressivePolicy, config: ExperimentConfig) -> torch.optim.Optimizer:
    logz_params = list(model.logz_net.parameters())
    logz_param_ids = {id(param) for param in logz_params}
    policy_params = [param for param in model.parameters() if id(param) not in logz_param_ids]
    return torch.optim.Adam(
        [
            {"params": policy_params, "lr": config.policy_lr},
            {"params": logz_params, "lr": config.logz_lr},
        ]
    )


def tb_loss(
    model: AutoregressivePolicy,
    contexts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples, log_prob, logz = model.sample_and_log_prob(contexts)
    rewards = reward_fn(samples, contexts).clamp_min(reward_eps)
    residual = logz + model.log_prob_scale * log_prob - rewards.log()
    return (residual**2).mean(), samples, rewards, logz


def evaluate_autoregressive(
    model: AutoregressivePolicy,
    validation_contexts: torch.Tensor,
    validation_samples: int,
) -> dict[str, float]:
    model.eval()
    repeated_contexts = validation_contexts.repeat_interleave(validation_samples)
    with torch.no_grad():
        samples, log_prob, logz = model.sample_and_log_prob(repeated_contexts)
        rewards = reward_fn(samples, repeated_contexts).clamp_min(reward_eps)
        induced_rewards = torch.exp(logz + model.log_prob_scale * log_prob)
        residual = logz + model.log_prob_scale * log_prob - rewards.log()
        score = torch.abs(rewards - induced_rewards).mean()

    return {
        "validation_mean_abs_induced_reward_gap": float(score.item()),
        "validation_tb_loss": float((residual**2).mean().item()),
        "validation_mean_reward": float(rewards.mean().item()),
        "validation_mean_induced_reward": float(induced_rewards.mean().item()),
    }


def snapshot_path(run_dir: Path, step: int) -> Path:
    return run_dir / "snapshots" / f"step_{step:06d}.pt"


def save_snapshot(
    run_dir: Path,
    step: int,
    model: AutoregressivePolicy,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    metrics: dict[str, float],
) -> None:
    payload = {
        "step": step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": to_dict(config),
        "metrics": metrics,
    }
    path = snapshot_path(run_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def list_snapshot_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "snapshots").glob("step_*.pt"))


def load_snapshot(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def train_autoregressive_run(
    config: ExperimentConfig,
    run_dir: Path,
    num_steps_override: int | None = None,
    snapshot_every_override: int | None = None,
    validation_samples_override: int | None = None,
    torch_num_threads: int = 2,
    torch_num_interop_threads: int = 1,
) -> dict[str, Any]:
    device = default_device()
    configure_worker_threads(torch_num_threads, torch_num_interop_threads)
    set_seed(config.seed)

    num_steps = num_steps_override or config.num_steps
    snapshot_every = snapshot_every_override or config.snapshot_every
    validation_samples = validation_samples_override or config.validation_samples

    train_contexts, validation_contexts, uniform_contexts = make_context_splits(config, device)
    selected_train_contexts = select_train_contexts_for_panels(train_contexts, config.selected_train_context_count)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "config.json",
        {
            **to_dict(config),
            "num_steps": num_steps,
            "snapshot_every": snapshot_every,
            "validation_samples": validation_samples,
            "device": device,
        },
    )

    model = AutoregressivePolicy(config).to(device)
    optimizer = build_optimizer(model, config)
    metrics_path = run_dir / "metrics.jsonl"

    def emit_metrics(step: int, train_loss: float, train_reward: float) -> dict[str, float]:
        validation = evaluate_autoregressive(model, validation_contexts, validation_samples)
        metrics = {
            "step": float(step),
            "train_tb_loss": float(train_loss),
            "train_mean_reward": float(train_reward),
            **validation,
        }
        append_jsonl(metrics_path, metrics)
        save_snapshot(run_dir, step, model, optimizer, config, metrics)
        return metrics

    initial_metrics = emit_metrics(step=0, train_loss=float("nan"), train_reward=float("nan"))
    latest_metrics = initial_metrics

    model.train()
    for step in range(1, num_steps + 1):
        batch_indices = torch.randint(0, train_contexts.shape[0], (config.batch_size,), device=device)
        batch_contexts = train_contexts[batch_indices]
        loss, samples, rewards, _ = tb_loss(model, batch_contexts)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 1000 == 0:
            print(
                f"[{config.name}] step={step:05d} "
                f"loss={loss.item():.4f} reward={rewards.mean().item():.4f}"
            )

        if step % snapshot_every == 0:
            latest_metrics = emit_metrics(step, float(loss.item()), float(rewards.mean().item()))

    summary = {
        "config_name": config.name,
        "run_dir": str(run_dir),
        "num_steps": num_steps,
        "snapshot_every": snapshot_every,
        "snapshot_count": len(list_snapshot_paths(run_dir)),
        "selected_train_contexts": selected_train_contexts.detach().cpu().tolist(),
        "uniform_contexts": uniform_contexts.detach().cpu().tolist(),
        **latest_metrics,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def load_config_for_run(run_dir: Path) -> ExperimentConfig:
    payload = load_json(run_dir / "config.json")
    config_name = payload["name"]
    base = SWEEP_CONFIGS.get(config_name)
    if base is None:
        raise ValueError(f"unknown config in run directory: {config_name}")
    return ExperimentConfig(
        name=base.name,
        hidden_dim=payload["hidden_dim"],
        context_K=payload["context_K"],
        policy_lr=payload["policy_lr"],
        logz_lr=payload["logz_lr"],
        batch_size=payload["batch_size"],
        num_steps=payload["num_steps"],
        snapshot_every=payload["snapshot_every"],
        seed=payload["seed"],
        validation_samples=payload["validation_samples"],
        n_train_contexts=payload["n_train_contexts"],
        n_uniform_contexts=payload["n_uniform_contexts"],
        selected_train_context_count=payload["selected_train_context_count"],
        grid_size=payload["grid_size"],
        fps=payload["fps"],
    )


def build_leaderboard_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["validation_mean_abs_induced_reward_gap"]),
        float(summary["validation_tb_loss"]),
        -float(summary["validation_mean_reward"]),
    )


def best_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return min(summaries, key=build_leaderboard_key)


def transparent_axes(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))


def figure_rgba(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape(height, width, 4)


def save_gif_from_png_frames(frame_dir: Path, gif_path: Path, fps: int) -> None:
    frames = [imageio.imread(path) for path in sorted(frame_dir.glob("frame_*.png"))]
    imageio.mimsave(gif_path, frames, fps=fps)


def render_autoregressive_run(
    run_dir: Path,
    transparent: bool = False,
    snapshot_stride: int = 1,
) -> dict[str, str]:
    config = load_config_for_run(run_dir)
    device = default_device()
    model = AutoregressivePolicy(config).to(device)
    _, _, grid_points = build_grid(config.grid_size)
    grid_tensor = ensure_grid_tensor(grid_points, device)
    spiral_path = build_spiral_path()

    train_contexts, _, uniform_contexts = make_context_splits(config, device)
    selected_train_contexts = select_train_contexts_for_panels(train_contexts, config.selected_train_context_count)
    uniform_contexts_np = uniform_contexts.detach().cpu().numpy()
    selected_train_contexts_np = selected_train_contexts.detach().cpu().numpy()

    snapshot_files = list_snapshot_paths(run_dir)[::snapshot_stride]
    aggregate_dir = run_dir / "frames_aggregate"
    selected_dir = run_dir / "frames_selected"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, snapshot_file in enumerate(snapshot_files):
        snapshot = load_snapshot(snapshot_file)
        model.load_state_dict(snapshot["model_state_dict"])
        model.eval()
        step = int(snapshot["step"])

        aggregate_grid = aggregated_density_grid_autoregressive(
            model,
            grid_tensor,
            uniform_contexts_np,
            device,
            config.grid_size,
        )
        vmax = max(np.quantile(aggregate_grid, 0.995), 1e-8)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        if transparent:
            transparent_axes(fig, ax)
        ax.imshow(
            aggregate_grid,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.plot(spiral_path[:, 0], spiral_path[:, 1], color="white", lw=1.0, alpha=0.7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Aggregated model density across arc-length contexts\nstep = {step}")
        fig.tight_layout()
        aggregate_path = aggregate_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(aggregate_path, dpi=160, transparent=transparent)
        plt.close(fig)

        panel_grids = [
            induced_density_grid_autoregressive(model, grid_tensor, float(context_value), device, config.grid_size)
            for context_value in selected_train_contexts_np
        ]
        vmax = max(max(np.quantile(grid, 0.995), 1e-8) for grid in panel_grids)
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        axes = axes.flatten()
        for ax, context_value, grid in zip(axes, selected_train_contexts_np, panel_grids):
            if transparent:
                transparent_axes(fig, ax)
            center_x, center_y = spiral_xy(float(context_value))
            ax.imshow(
                grid,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            ax.scatter([center_x], [center_y], s=60, color="cyan", edgecolor="white", linewidth=0.8)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"c = {float(context_value):.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.suptitle(f"Model density on selected training contexts\nstep = {step}", y=0.98)
        fig.tight_layout()
        selected_path = selected_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(selected_path, dpi=160, transparent=transparent)
        plt.close(fig)

    aggregate_gif = run_dir / "aggregate_density_training.gif"
    selected_gif = run_dir / "selected_train_contexts_density_training.gif"
    save_gif_from_png_frames(aggregate_dir, aggregate_gif, config.fps)
    save_gif_from_png_frames(selected_dir, selected_gif, config.fps)

    return {
        "aggregate_gif": str(aggregate_gif),
        "selected_gif": str(selected_gif),
        "aggregate_frames_dir": str(aggregate_dir),
        "selected_frames_dir": str(selected_dir),
    }
