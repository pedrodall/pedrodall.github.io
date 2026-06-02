"""
Conditional continuous GFlowNet toy experiment.

We define a family of unnormalized rewards R_c(x) over R^2.
For each context c, R_c is a Gaussian-shaped reward centered at a point
on an Archimedean spiral. The goal is to train a conditional policy that
amortizes sampling across contexts: after seeing only a small discrete set
of training contexts, the policy should sample near the correct spiral point
for unseen intermediate contexts.

This script is intentionally simple. It uses a lightweight TB-style objective.
For the iterative policy, we learn a backward policy for all reverse steps
except the final one, which is constrained to return to the initial state
with probability 1. The result should be treated as a first visualization
experiment rather than a fully specified continuous GFlowNet implementation.
"""

import os
import math
import tempfile

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

n_train_contexts = 12
n_uniform_contexts = 80

batch_size = 512
num_steps = 1000
policy_lr = 1e-2
logz_lr = 1e-1

hidden_dim = 128
context_K = 4

T = 6

grid_size = 200
n_samples_gif = 500

fps = 8
seed = 0
snapshot_every = 5
gif_frame_stride = 5
iterative_gif_frame_stride = 25
selected_train_context_count = 4

output_dir = "outputs/spiral_gflownet"
device = "cuda" if torch.cuda.is_available() else "cpu"


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
iterative_render_grid_size = 100
iterative_mc_chunk_size = 2048


def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def spiral_center(contexts):
    radius = a + b * contexts
    x = radius * torch.cos(contexts)
    y = radius * torch.sin(contexts)
    return torch.stack([x, y], dim=-1)


def principal_angle(theta):
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def context_polar_center(contexts):
    radius = a + b * contexts
    log_r = torch.log(radius + polar_eps)
    theta = principal_angle(contexts)
    return log_r, theta


def reward_fn(x, contexts):
    centers = spiral_center(contexts)
    sq_dist = ((x - centers) ** 2).sum(dim=-1)
    return torch.exp(-sq_dist / (2 * sigma**2))


def true_density_fn(x, contexts):
    normalizer = 2.0 * math.pi * sigma**2
    return reward_fn(x, contexts) / normalizer


def make_context_features(contexts):
    scaled = contexts / C
    features = [scaled.unsqueeze(-1)]
    for k in range(context_K):
        angle = (2**k) * 2.0 * math.pi * scaled
        features.append(torch.sin(angle).unsqueeze(-1))
        features.append(torch.cos(angle).unsqueeze(-1))
    return torch.cat(features, dim=-1)


def make_angle_features(theta):
    scaled = theta / math.pi
    features = [scaled.unsqueeze(-1)]
    for k in range(context_K):
        angle = (2**k) * theta
        features.append(torch.sin(angle).unsqueeze(-1))
        features.append(torch.cos(angle).unsqueeze(-1))
    return torch.cat(features, dim=-1)


def raw_to_theta(raw_theta):
    return math.pi * torch.tanh(raw_theta)


def theta_to_raw(theta):
    scaled = torch.clamp(theta / math.pi, -1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))


def polar_to_xy(log_r, theta):
    r = torch.exp(log_r)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def xy_to_polar(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    r = torch.sqrt(x1**2 + x2**2 + polar_eps)
    log_r = torch.log(r)
    theta = torch.atan2(x2, x1)
    return log_r, theta


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def clamp_log_std(log_std):
    return torch.clamp(log_std, min=log_std_min, max=log_std_max)


class AutoregressivePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_prob_scale = 1.0
        context_dim = 1 + 2 * context_K
        angle_dim = 1 + 2 * context_K
        self.theta_net = build_mlp(context_dim, 2)
        self.log_r_net = build_mlp(context_dim + angle_dim, 2)
        self.logz_net = build_mlp(context_dim, 1)

    def sample_and_log_prob(self, contexts):
        context_features = make_context_features(contexts)
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

        angle_features = make_angle_features(theta)
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

    def log_prob_exact(self, x, contexts):
        context_features = make_context_features(contexts)
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

        angle_features = make_angle_features(theta)
        log_r_input = torch.cat([context_features, angle_features], dim=-1)
        log_r_stats = self.log_r_net(log_r_input)
        log_r_mean = base_log_r + 0.5 * torch.tanh(log_r_stats[:, 0])
        log_r_log_std = clamp_log_std(log_r_stats[:, 1])
        log_r_dist = Normal(log_r_mean, log_r_log_std.exp())
        log_prob_log_r = log_r_dist.log_prob(log_r)

        return log_prob_theta + log_prob_log_r - 2.0 * log_r

    def logz(self, contexts):
        context_features = make_context_features(contexts)
        return self.logz_net(context_features).squeeze(-1)


class IterativePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_prob_scale = 1.0
        context_dim = 1 + 2 * context_K
        self.forward_net = build_mlp(context_dim + 2 + 1, 4)
        self.backward_net = build_mlp(context_dim + 2 + 1, 4)
        self.logz_net = build_mlp(context_dim, 1)

    def transition_dist(self, contexts, x, step):
        batch = contexts.shape[0]
        context_features = make_context_features(contexts)
        step_feature = torch.full(
            (batch, 1),
            fill_value=step / max(T - 1, 1),
            device=contexts.device,
        )
        net_input = torch.cat([context_features, x, step_feature], dim=-1)
        step_stats = self.forward_net(net_input)
        delta_mean = iterative_step_scale * torch.tanh(step_stats[:, :2])
        delta_log_std = clamp_log_std(step_stats[:, 2:] - 1.5)
        next_mean = x + delta_mean
        dist = Normal(next_mean, delta_log_std.exp())
        return dist

    def backward_dist(self, contexts, x_t, step):
        # The final reverse step x_1 -> x_0 is deterministic by construction,
        # so we only learn p_B(x_{t-1} | x_t, c) for t >= 2.
        if step <= 1:
            raise ValueError("backward_dist is only defined for step >= 2")

        batch = contexts.shape[0]
        context_features = make_context_features(contexts)
        step_feature = torch.full(
            (batch, 1),
            fill_value=(step - 1) / max(T - 1, 1),
            device=contexts.device,
        )
        net_input = torch.cat([context_features, x_t, step_feature], dim=-1)
        step_stats = self.backward_net(net_input)
        delta_mean = iterative_step_scale * torch.tanh(step_stats[:, :2])
        delta_log_std = clamp_log_std(step_stats[:, 2:] - 1.5)
        prev_mean = x_t + delta_mean
        dist = Normal(prev_mean, delta_log_std.exp())
        return dist

    def sample_and_log_prob(self, contexts):
        batch = contexts.shape[0]
        states = [torch.zeros(batch, 2, device=contexts.device)]
        log_pf_terms = []

        for step in range(T):
            step_dist = self.transition_dist(contexts, states[-1], step)
            x_next = step_dist.rsample()
            log_pf_terms.append(step_dist.log_prob(x_next).sum(dim=-1))
            states.append(x_next)

        log_pb_terms = []
        for step in range(T, 1, -1):
            step_dist = self.backward_dist(contexts, states[step], step)
            log_pb_terms.append(step_dist.log_prob(states[step - 1]).sum(dim=-1))

        log_pf = torch.stack(log_pf_terms, dim=0).sum(dim=0)
        if log_pb_terms:
            log_pb = torch.stack(log_pb_terms, dim=0).sum(dim=0)
        else:
            log_pb = torch.zeros_like(log_pf)

        log_prob = log_pf - log_pb
        logz = self.logz_net(make_context_features(contexts)).squeeze(-1)
        return states[-1], log_prob, logz


def make_context_splits():
    theta_lookup, arc_lookup = build_arc_length_lookup()
    train_fracs = np.linspace(0.0, 1.0, n_train_contexts)
    test_fracs = 0.5 * (train_fracs[:-1] + train_fracs[1:])
    uniform_fracs = np.linspace(0.0, 1.0, n_uniform_contexts)

    train_np = arc_fraction_to_theta(train_fracs, theta_lookup, arc_lookup)
    test_np = arc_fraction_to_theta(test_fracs, theta_lookup, arc_lookup)
    uniform_np = arc_fraction_to_theta(uniform_fracs, theta_lookup, arc_lookup)

    train_contexts = torch.tensor(train_np, dtype=torch.float32, device=device)
    test_contexts = torch.tensor(test_np, dtype=torch.float32, device=device)
    uniform_contexts = torch.tensor(uniform_np, dtype=torch.float32, device=device)
    return train_contexts, test_contexts, uniform_contexts


def tb_loss(model, contexts):
    samples, log_prob, logz = model.sample_and_log_prob(contexts)
    rewards = reward_fn(samples, contexts).clamp_min(reward_eps)
    residual = logz + model.log_prob_scale * log_prob - rewards.log()
    return (residual**2).mean(), samples, rewards


def train_model(model, train_contexts, label, capture_snapshots=False):
    logz_params = list(model.logz_net.parameters())
    logz_param_ids = {id(param) for param in logz_params}
    policy_params = [
        param for param in model.parameters() if id(param) not in logz_param_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": policy_params, "lr": policy_lr},
            {"params": logz_params, "lr": logz_lr},
        ]
    )
    model.train()
    snapshots = []

    if capture_snapshots:
        snapshots.append({"step": 0, "state_dict": cpu_state_dict(model)})

    for step in range(1, num_steps + 1):
        batch_indices = torch.randint(0, train_contexts.shape[0], (batch_size,), device=device)
        batch_contexts = train_contexts[batch_indices]

        loss, samples, rewards = tb_loss(model, batch_contexts)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0 or step == 1:
            mean_radius = samples.norm(dim=-1).mean().item()
            mean_reward = rewards.mean().item()
            print(
                f"[{label}] step={step:05d} "
                f"loss={loss.item():.4f} "
                f"reward={mean_reward:.4f} "
                f"radius={mean_radius:.4f}"
            )

        if capture_snapshots and step % snapshot_every == 0:
            snapshots.append({"step": step, "state_dict": cpu_state_dict(model)})

    return snapshots


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_cpu_state_dict(model, state_dict):
    model.load_state_dict(state_dict)


def select_train_contexts_for_panels(train_contexts):
    indices = np.linspace(
        0,
        train_contexts.shape[0] - 1,
        selected_train_context_count,
        dtype=int,
    )
    unique_indices = []
    for idx in indices.tolist():
        if idx not in unique_indices:
            unique_indices.append(idx)
    return train_contexts[unique_indices]


def induced_reward_grid_autoregressive(model, grid_points, context_value, chunk_size=8192):
    densities = []
    context_array = torch.full((chunk_size,), float(context_value), device=device)

    with torch.no_grad():
        for start in range(0, grid_points.shape[0], chunk_size):
            end = min(start + chunk_size, grid_points.shape[0])
            chunk = grid_points[start:end]
            contexts = context_array[: end - start]
            log_prob = model.log_prob_exact(chunk, contexts)
            density = torch.exp(log_prob)
            densities.append(density.detach().cpu())

    flat_density = torch.cat(densities, dim=0).numpy()
    return flat_density.reshape(grid_size, grid_size)


def aggregated_induced_reward_grid(model, grid_points, context_values):
    aggregate = np.zeros((grid_size, grid_size), dtype=np.float32)
    for context_value in context_values:
        aggregate += induced_reward_grid_autoregressive(model, grid_points, float(context_value))
    return aggregate


def estimate_iterative_density_grid_mc(model, grid_points, context_value, chunk_size=iterative_mc_chunk_size):
    grid_tensor = ensure_grid_tensor(grid_points)
    densities = []

    with torch.no_grad():
        for start in range(0, grid_tensor.shape[0], chunk_size):
            end = min(start + chunk_size, grid_tensor.shape[0])
            x_terminal = grid_tensor[start:end]
            batch = x_terminal.shape[0]
            contexts = torch.full((batch,), float(context_value), device=device)

            states = [None] * (T + 1)
            states[T] = x_terminal
            log_pb = torch.zeros(batch, device=device)

            for step in range(T, 1, -1):
                backward_dist = model.backward_dist(contexts, states[step], step)
                states[step - 1] = backward_dist.rsample()
                log_pb += backward_dist.log_prob(states[step - 1]).sum(dim=-1)

            states[0] = torch.zeros(batch, 2, device=device)

            log_pf = torch.zeros(batch, device=device)
            for step in range(T):
                forward_dist = model.transition_dist(contexts, states[step], step)
                log_pf += forward_dist.log_prob(states[step + 1]).sum(dim=-1)

            density = torch.exp(log_pf - log_pb)
            densities.append(density.detach().cpu())

    side = int(round(math.sqrt(grid_tensor.shape[0])))
    flat_density = torch.cat(densities, dim=0).numpy()
    return flat_density.reshape(side, side)


def aggregated_iterative_density_grid(model, grid_points, context_values):
    side = int(round(math.sqrt(grid_points.shape[0])))
    aggregate = np.zeros((side, side), dtype=np.float32)
    for context_value in context_values:
        aggregate += estimate_iterative_density_grid_mc(model, grid_points, float(context_value))
    return aggregate


def ensure_grid_tensor(grid_points):
    if isinstance(grid_points, torch.Tensor):
        return grid_points.to(device=device, dtype=torch.float32)
    return torch.tensor(grid_points, dtype=torch.float32, device=device)


def true_reward_grid(grid_points, context_value):
    context_array = np.full((grid_points.shape[0],), context_value, dtype=np.float32)
    x_tensor = ensure_grid_tensor(grid_points)
    c_tensor = torch.tensor(context_array, dtype=torch.float32, device=device)
    density = true_density_fn(x_tensor, c_tensor)
    return density.detach().cpu().numpy().reshape(grid_size, grid_size)


def aggregated_true_reward_grid(grid_points, context_values):
    aggregate = np.zeros((grid_size, grid_size), dtype=np.float32)
    for context_value in context_values:
        aggregate += true_reward_grid(grid_points, float(context_value))
    return aggregate


def render_autoregressive_aggregate_training_gif(
    model,
    snapshots,
    context_values,
    path,
    title_prefix,
    grid_points,
    spiral_path,
):
    snapshots = snapshots[::gif_frame_stride]
    aggregate_frames = []
    for snapshot in snapshots:
        load_cpu_state_dict(model, snapshot["state_dict"])
        model.eval()
        aggregate_frames.append(
            {
                "step": snapshot["step"],
                "grid": aggregated_induced_reward_grid(model, grid_points, context_values),
            }
        )

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, frame_data in enumerate(aggregate_frames):
            vmax = max(np.quantile(frame_data["grid"], 0.995), 1e-8)
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.imshow(
                frame_data["grid"],
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
            ax.set_title(f"{title_prefix}\nstep = {frame_data['step']}")
            fig.tight_layout()

            frame = figure_to_array(fig)
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(frame_path, frame)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    save_gif(frames, path)


def render_autoregressive_selected_contexts_training_gif(
    model,
    snapshots,
    selected_contexts,
    path,
    title_prefix,
    grid_points,
):
    snapshots = snapshots[::gif_frame_stride]
    selected_contexts_np = selected_contexts.detach().cpu().numpy()
    panel_frames = []
    for snapshot in snapshots:
        load_cpu_state_dict(model, snapshot["state_dict"])
        model.eval()
        grids = []
        for context_value in selected_contexts_np:
            grids.append(
                {
                    "context": float(context_value),
                    "grid": induced_reward_grid_autoregressive(model, grid_points, float(context_value)),
                }
            )
        panel_frames.append({"step": snapshot["step"], "grids": grids})

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, frame_data in enumerate(panel_frames):
            vmax = max(
                np.quantile(panel["grid"], 0.995)
                for panel in frame_data["grids"]
            )
            vmax = max(vmax, 1e-8)
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            axes = axes.flatten()

            for ax, panel_data in zip(axes, frame_data["grids"]):
                context_value = panel_data["context"]
                center_x, center_y = spiral_xy(context_value)
                ax.imshow(
                    panel_data["grid"],
                    extent=[xmin, xmax, ymin, ymax],
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax,
                )
                ax.scatter([center_x], [center_y], s=60, color="cyan", edgecolor="white", linewidth=0.8)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_title(f"c = {context_value:.2f}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            fig.suptitle(f"{title_prefix}\nstep = {frame_data['step']}", y=0.98)
            fig.tight_layout()

            frame = figure_to_array(fig)
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(frame_path, frame)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    save_gif(frames, path)


def render_iterative_aggregate_training_gif(
    model,
    snapshots,
    context_values,
    path,
    title_prefix,
    grid_points,
    spiral_path,
):
    snapshots = snapshots[::iterative_gif_frame_stride]
    side = int(round(math.sqrt(grid_points.shape[0])))
    xs = np.linspace(xmin, xmax, side)
    ys = np.linspace(ymin, ymax, side)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    aggregate_frames = []
    for snapshot in snapshots:
        load_cpu_state_dict(model, snapshot["state_dict"])
        model.eval()
        aggregate_frames.append(
            {
                "step": snapshot["step"],
                "grid": aggregated_iterative_density_grid(model, grid_points, context_values),
            }
        )

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, frame_data in enumerate(aggregate_frames):
            vmax = max(np.quantile(frame_data["grid"], 0.995), 1e-8)
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.imshow(
                frame_data["grid"],
                extent=extent,
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
            ax.set_title(f"{title_prefix}\nstep = {frame_data['step']}")
            fig.tight_layout()

            frame = figure_to_array(fig)
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(frame_path, frame)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    save_gif(frames, path)


def render_iterative_selected_contexts_training_gif(
    model,
    snapshots,
    selected_contexts,
    path,
    title_prefix,
    grid_points,
):
    snapshots = snapshots[::iterative_gif_frame_stride]
    selected_contexts_np = selected_contexts.detach().cpu().numpy()
    side = int(round(math.sqrt(grid_points.shape[0])))
    xs = np.linspace(xmin, xmax, side)
    ys = np.linspace(ymin, ymax, side)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    panel_frames = []
    for snapshot in snapshots:
        load_cpu_state_dict(model, snapshot["state_dict"])
        model.eval()
        grids = []
        for context_value in selected_contexts_np:
            grids.append(
                {
                    "context": float(context_value),
                    "grid": estimate_iterative_density_grid_mc(model, grid_points, float(context_value)),
                }
            )
        panel_frames.append({"step": snapshot["step"], "grids": grids})

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, frame_data in enumerate(panel_frames):
            vmax = max(np.quantile(panel["grid"], 0.995) for panel in frame_data["grids"])
            vmax = max(vmax, 1e-8)
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            axes = axes.flatten()

            for ax, panel_data in zip(axes, frame_data["grids"]):
                context_value = panel_data["context"]
                center_x, center_y = spiral_xy(context_value)
                ax.imshow(
                    panel_data["grid"],
                    extent=extent,
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax,
                )
                ax.scatter([center_x], [center_y], s=60, color="cyan", edgecolor="white", linewidth=0.8)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_title(f"c = {context_value:.2f}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            fig.suptitle(f"{title_prefix}\nstep = {frame_data['step']}", y=0.98)
            fig.tight_layout()

            frame = figure_to_array(fig)
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(frame_path, frame)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    save_gif(frames, path)


def save_final_reference_pngs(model, grid_points, uniform_contexts, selected_contexts):
    uniform_contexts_np = uniform_contexts.detach().cpu().numpy()
    selected_contexts_np = selected_contexts.detach().cpu().numpy()

    aggregate_true = aggregated_true_reward_grid(grid_points, uniform_contexts_np)
    aggregate_model = aggregated_induced_reward_grid(model, grid_points, uniform_contexts_np)
    vmax = max(aggregate_true.max(), aggregate_model.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, grid, title in [
        (axes[0], aggregate_true, "True aggregated density"),
        (axes[1], aggregate_model, "Model aggregated density"),
    ]:
        ax.imshow(
            grid,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_final_aggregate_reference.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()
    for ax, context_value in zip(axes, selected_contexts_np):
        grid = induced_reward_grid_autoregressive(model, grid_points, float(context_value))
        center_x, center_y = spiral_xy(float(context_value))
        ax.imshow(
            grid,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="magma",
        )
        ax.scatter([center_x], [center_y], s=60, color="cyan", edgecolor="white", linewidth=0.8)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Final model density, c = {float(context_value):.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "11_final_selected_contexts_reference.png"), dpi=160)
    plt.close(fig)


def sample_policy(model, contexts, num_samples):
    model.eval()
    all_samples = []
    with torch.no_grad():
        for context in contexts:
            repeated_context = context.repeat(num_samples)
            samples, _, _ = model.sample_and_log_prob(repeated_context)
            all_samples.append(samples.detach().cpu().numpy())
    return all_samples


def build_grid():
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return xs, ys, grid_x, grid_y, grid_points


def build_arc_length_lookup(num_points=20000):
    theta = np.linspace(0.0, C, num_points, dtype=np.float64)
    radius = a + b * theta
    speed = np.sqrt(radius**2 + b**2)
    delta_theta = np.diff(theta)
    delta_arc = 0.5 * (speed[1:] + speed[:-1]) * delta_theta
    cumulative_arc = np.concatenate([[0.0], np.cumsum(delta_arc)])
    cumulative_arc /= cumulative_arc[-1]
    return theta, cumulative_arc


def arc_fraction_to_theta(fractions, theta_lookup, arc_lookup):
    return np.interp(fractions, arc_lookup, theta_lookup).astype(np.float32)


def reward_on_grid(grid_points, context_value):
    context_array = np.full((grid_points.shape[0],), context_value, dtype=np.float32)
    x_tensor = ensure_grid_tensor(grid_points)
    c_tensor = torch.tensor(context_array, dtype=torch.float32, device=device)
    rewards = reward_fn(x_tensor, c_tensor)
    return rewards.detach().cpu().numpy().reshape(grid_size, grid_size)


def save_gif(frames, path):
    imageio.mimsave(path, frames, fps=fps)
    print(f"saved {path}")


def figure_to_array(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape(height, width, 4)[..., :3]


def render_reward_gif(contexts, path, title_prefix, grid_points, spiral_path):
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, context_value in enumerate(contexts):
            reward_grid = reward_on_grid(grid_points, float(context_value))
            center_x, center_y = spiral_xy(float(context_value))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(
                reward_grid,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            ax.plot(spiral_path[:, 0], spiral_path[:, 1], color="white", lw=1.0, alpha=0.6)
            ax.scatter([center_x], [center_y], s=80, color="red", edgecolor="white", linewidth=0.8)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{title_prefix}\ncontext = {context_value:.2f}")
            fig.tight_layout()

            frame = figure_to_array(fig)
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(frame_path, frame)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    save_gif(frames, path)


def render_sample_gif(contexts, samples_per_context, path, title_prefix, grid_points, spiral_path):
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, (context_value, samples) in enumerate(zip(contexts, samples_per_context)):
            reward_grid = reward_on_grid(grid_points, float(context_value))
            center_x, center_y = spiral_xy(float(context_value))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(
                reward_grid,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                alpha=0.65,
            )
            ax.plot(spiral_path[:, 0], spiral_path[:, 1], color="white", lw=1.0, alpha=0.6)
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                s=10,
                color="deepskyblue",
                alpha=0.35,
                edgecolors="none",
            )
            ax.scatter([center_x], [center_y], s=80, color="red", edgecolor="white", linewidth=0.8)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{title_prefix}\ncontext = {context_value:.2f}")
            fig.tight_layout()

            frame = figure_to_array(fig)
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(frame_path, frame)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    save_gif(frames, path)


def spiral_xy(context_value):
    radius = a + b * context_value
    return radius * math.cos(context_value), radius * math.sin(context_value)


def build_spiral_path(num_points=400):
    contexts = np.linspace(0.0, C, num_points)
    coords = [spiral_xy(value) for value in contexts]
    return np.array(coords, dtype=np.float32)


def main():
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"device: {device}")
    print(f"output_dir: {output_dir}")

    train_contexts, test_contexts, uniform_contexts = make_context_splits()
    print(f"train contexts: {train_contexts.shape[0]}")
    print(f"test contexts: {test_contexts.shape[0]}")
    print(f"uniform contexts: {uniform_contexts.shape[0]}")

    autoregressive_policy = AutoregressivePolicy().to(device)
    snapshots = train_model(
        autoregressive_policy,
        train_contexts,
        "autoregressive",
        capture_snapshots=True,
    )
    iterative_policy = IterativePolicy().to(device)
    iterative_snapshots = train_model(
        iterative_policy,
        train_contexts,
        "iterative",
        capture_snapshots=True,
    )

    grid_data = build_grid()
    grid_points = grid_data[-1]
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    iterative_xs = np.linspace(xmin, xmax, iterative_render_grid_size)
    iterative_ys = np.linspace(ymin, ymax, iterative_render_grid_size)
    iterative_grid_x, iterative_grid_y = np.meshgrid(iterative_xs, iterative_ys)
    iterative_grid_points = np.stack(
        [iterative_grid_x.ravel(), iterative_grid_y.ravel()],
        axis=-1,
    )
    iterative_grid_points_tensor = torch.tensor(
        iterative_grid_points,
        dtype=torch.float32,
        device=device,
    )
    spiral_path = build_spiral_path()
    selected_train_contexts = select_train_contexts_for_panels(train_contexts)

    render_autoregressive_aggregate_training_gif(
        autoregressive_policy,
        snapshots,
        uniform_contexts.detach().cpu().numpy(),
        os.path.join(output_dir, "12_autoregressive_aggregate_density_training.gif"),
        "Aggregated model density across arc-length contexts",
        grid_points_tensor,
        spiral_path,
    )
    render_autoregressive_selected_contexts_training_gif(
        autoregressive_policy,
        snapshots,
        selected_train_contexts,
        os.path.join(output_dir, "13_autoregressive_selected_train_contexts_density_training.gif"),
        "Model density on selected training contexts",
        grid_points_tensor,
    )
    render_iterative_aggregate_training_gif(
        iterative_policy,
        iterative_snapshots,
        uniform_contexts.detach().cpu().numpy(),
        os.path.join(output_dir, "14_iterative_aggregate_density_training.gif"),
        "Iterative model density across arc-length contexts",
        iterative_grid_points_tensor,
        spiral_path,
    )
    render_iterative_selected_contexts_training_gif(
        iterative_policy,
        iterative_snapshots,
        selected_train_contexts,
        os.path.join(output_dir, "15_iterative_selected_train_contexts_density_training.gif"),
        "Iterative model density on selected training contexts",
        iterative_grid_points_tensor,
    )
    save_final_reference_pngs(
        autoregressive_policy,
        grid_points_tensor,
        uniform_contexts,
        selected_train_contexts,
    )

    print("done")


if __name__ == "__main__":
    main()
