"""Langevin simulation for the ToggleBasic dynamics.

The default physical and numerical parameters are read from
``results/ToggleBasic/config.yaml`` and ``results/ToggleBasic/problem.yaml``.
The SDE is integrated with Euler--Maruyama:

    dX = f(X) dt + sqrt(2 D) dW.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from scipy.optimize import root

from dynamics import ToggleBasic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "ToggleBasic"


def reflect_into_bounds(x, lower, upper):
    """Reflect every coordinate into [lower, upper], including large overshoots."""
    span = upper - lower
    offset = torch.remainder(x - lower, 2.0 * span)
    return lower + torch.where(offset <= span, offset, 2.0 * span - offset)


def simulate(force, particles, steps, burn_in, sample_stride, dt, diffusion, lower, upper, device, seed):
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = lower + (upper - lower) * torch.rand((particles, 2), device=device, generator=generator)
    noise_scale = math.sqrt(2.0 * diffusion * dt)
    samples = []

    for step in range(steps):
        x = x + dt * force.force(x) + noise_scale * torch.randn(
            x.shape, device=device, dtype=x.dtype, generator=generator
        )
        x = reflect_into_bounds(x, lower, upper)

        if step >= burn_in and (step - burn_in) % sample_stride == 0:
            samples.append(x.cpu().numpy().copy())

        if (step + 1) % max(steps // 10, 1) == 0:
            print(f"Simulation progress: {step + 1}/{steps}")

    if not samples:
        raise ValueError("No samples were collected; burn_in must be smaller than steps.")
    return np.concatenate(samples, axis=0)


def save_figures(samples, force, lower, upper, diffusion, bins, output_dir):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bounds = ((float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1])))
    histogram, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=bounds, density=True)
    density = histogram.T
    potential = -diffusion * np.log(np.maximum(density, 1e-12))
    potential -= potential.min()
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.pcolormesh(x_edges, y_edges, density, shading="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="stationary density")
    ax.set(xlabel="LacI", ylabel="TetR", title="ToggleBasic Langevin stationary density")
    fig.savefig(output_dir / "stationary_density.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    contours = ax.contourf(grid_x, grid_y, potential, levels=40, cmap="magma_r")
    fig.colorbar(contours, ax=ax, label="U = -D log p (shifted)")
    ax.set(xlabel="LacI", ylabel="TetR", title="ToggleBasic effective landscape")
    fig.savefig(output_dir / "effective_landscape.png", dpi=220)
    plt.close(fig)

    return density, potential, x_edges, y_edges


def stable_fixed_points(force, search_max=200.0):
    """Find deterministic stable fixed points for the red reference markers."""
    candidates = []
    for lac_i in np.linspace(0.0, search_max, 9):
        for tet_r in np.linspace(0.0, search_max, 9):
            solution = root(lambda y: force.force(y).detach().cpu().numpy(), (lac_i, tet_r))
            point = solution.x
            if not (solution.success and np.all(point >= 0.0) and np.all(point <= 1000.0)):
                continue
            if np.linalg.norm(solution.fun) > 1e-5 or any(np.linalg.norm(point - item) < 1e-3 for item in candidates):
                continue
            candidates.append(point)
    stable = []
    for point in candidates:
        state = torch.tensor(point, dtype=torch.float32, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(force.force, state).detach().cpu().numpy()
        if np.all(np.linalg.eigvals(jacobian).real < 0.0):
            stable.append(point)
    return np.asarray(stable)


def terminal_step_comparison(force, particles, target_steps, dt, diffusion, init_min, init_max, device, seed):
    """Simulate one shared ensemble and retain terminal snapshots."""
    target_steps = sorted(set(target_steps))
    if not target_steps or target_steps[0] <= 0:
        raise ValueError("--terminal-steps must contain positive integers.")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = init_min + (init_max - init_min) * torch.rand((particles, 2), device=device, generator=generator)
    lower = torch.full((2,), init_min, dtype=torch.float32, device=device)
    upper = torch.full((2,), init_max, dtype=torch.float32, device=device)
    noise_scale = math.sqrt(2.0 * diffusion * dt)
    snapshots = {}
    for step in range(1, target_steps[-1] + 1):
        x = x + dt * force.force(x) + noise_scale * torch.randn(x.shape, device=device, generator=generator)
        x = reflect_into_bounds(x, lower, upper)
        if step in target_steps:
            snapshots[step] = x.cpu().numpy().copy()
    return snapshots


def save_terminal_comparison(snapshots, attractors, growth_rate, output_path, axis_min, axis_max):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    steps = list(snapshots)
    fig, axes = plt.subplots(1, len(steps), figsize=(6 * len(steps), 5), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, step in zip(axes, steps):
        values = snapshots[step]
        assigned = np.argmin(np.linalg.norm(values[:, None, :] - attractors[None, :, :], axis=2), axis=1)
        fractions = [(assigned == index).mean() for index in range(len(attractors))]
        ax.scatter(values[:, 0], values[:, 1], s=2, alpha=0.30, color="royalblue", linewidths=0)
        ax.scatter(attractors[:, 0], attractors[:, 1], marker="x", s=130, linewidths=2.5, color="#d62728")
        ax.set_title(f"gr={growth_rate}, sim={step}\nfrac={fractions[0]:.3f}/{fractions[1]:.3f}")
        ax.set(xlim=(axis_min, axis_max), ylim=(axis_min, axis_max), xlabel="x1 / LacI")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("x2 / TetR")
    fig.suptitle(f"ToggleBasic SDE terminal samples (N={len(next(iter(snapshots.values())))})")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Langevin simulation for ToggleBasic.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--diffusion", type=float, default=None)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--terminal-comparison", action="store_true", help="Plot terminal particle snapshots at multiple simulation steps.")
    parser.add_argument("--terminal-steps", default="500,5000,50000", help="Comma-separated terminal steps used with --terminal-comparison.")
    parser.add_argument("--init-min", type=float, default=0.0)
    parser.add_argument("--init-max", type=float, default=200.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = args.config_dir.resolve()
    with (config_dir / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with (config_dir / "problem.yaml").open(encoding="utf-8") as file:
        problem = yaml.safe_load(file)

    growth_rate = float(config["force_params"]["ToggleBasic"]["growth_rate"])
    particles = args.particles or int(config["batch_size"])
    steps = args.steps or int(config["sim_steps"])
    burn_in = args.burn_in if args.burn_in is not None else steps // 2
    dt = args.dt or float(config["sim_dt"])
    diffusion = args.diffusion or float(problem["noise_strength"])
    if not (0 <= burn_in < steps):
        raise ValueError("--burn-in must satisfy 0 <= burn-in < steps.")
    if args.sample_stride <= 0 or dt <= 0 or diffusion <= 0:
        raise ValueError("sample-stride, dt, and diffusion must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    lower = torch.tensor([problem["x_min"], problem["y_min"]], dtype=torch.float32, device=device)
    upper = torch.tensor([problem["x_max"], problem["y_max"]], dtype=torch.float32, device=device)
    force = ToggleBasic(growth_rate=growth_rate)
    output_dir = config_dir / f"langevin_growth_rate_{growth_rate:g}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ToggleBasic Langevin simulation: growth_rate={growth_rate}, D={diffusion}, dt={dt}")
    if args.terminal_comparison:
        target_steps = [int(value) for value in args.terminal_steps.split(",")]
        terminal_particles = args.particles or 5000
        if not args.init_min < args.init_max:
            raise ValueError("--init-min must be smaller than --init-max.")
        snapshots = terminal_step_comparison(
            force, terminal_particles, target_steps, dt, diffusion,
            args.init_min, args.init_max, device, args.seed,
        )
        attractors = stable_fixed_points(force, search_max=args.init_max)
        if len(attractors) != 2:
            raise RuntimeError(f"Expected two stable ToggleBasic attractors, found {len(attractors)}.")
        image_path = output_dir / "terminal_samples_by_sim_steps.png"
        save_terminal_comparison(snapshots, attractors, growth_rate, image_path, args.init_min, args.init_max)
        np.savez_compressed(
            output_dir / "terminal_samples_by_sim_steps.npz",
            attractors=attractors,
            **{f"step_{step}": values for step, values in snapshots.items()},
        )
        print(f"Saved terminal-sample comparison: {image_path}")
        return
    samples = simulate(force, particles, steps, burn_in, args.sample_stride, dt, diffusion, lower, upper, device, args.seed)
    density, potential, x_edges, y_edges = save_figures(samples, force, lower.cpu(), upper.cpu(), diffusion, args.bins, output_dir)
    np.savez_compressed(
        output_dir / "langevin_samples_and_landscape.npz",
        samples=samples,
        density=density,
        potential=potential,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    metadata = {
        "growth_rate": growth_rate, "diffusion": diffusion, "dt": dt,
        "particles": particles, "steps": steps, "burn_in": burn_in,
        "sample_stride": args.sample_stride, "samples_collected": int(len(samples)),
        "seed": args.seed, "bounds": {"LacI": [float(lower[0]), float(upper[0])], "TetR": [float(lower[1]), float(upper[1])]},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Collected {len(samples)} samples; mean={samples.mean(axis=0)}, std={samples.std(axis=0)}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
