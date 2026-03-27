import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm, qr, solve_continuous_lyapunov
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, Union


@dataclass
class DDGAResult:
    limit_cycle: np.ndarray
    covariance: np.ndarray
    pre_solution: np.ndarray
    period: float
    period_detection: str
    pre_solution_mode: str
    pre_solution_requested_mode: str
    pre_solution_support_fraction: float
    covariance_mode: str = "paper_local_adaptive"
    covariance_stability_floor: float = 0.0
    covariance_shift_mean: float = 0.0
    covariance_shift_max: float = 0.0
    covariance_shift_fraction: float = 0.0
    phase_kernel_mode: str = "proxy_speed"
    phase_empirical_use_drift: bool = False
    phase_empirical_valid_fraction: float = 0.0
    phase_empirical_projection_distance_mean: float = 0.0
    phase_empirical_projection_distance_q90: float = 0.0
    phase_empirical_mean_increment_mean: float = 1.0
    phase_empirical_std_mean: float = 0.0
    phase_empirical_blend_mean: float = 0.0


def ensure_positive_definite(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetrize and add the smallest necessary jitter for positive definiteness."""
    matrix = np.real(matrix)
    matrix = 0.5 * (matrix + matrix.T)

    if not np.all(np.isfinite(matrix)):
        return eps * np.eye(matrix.shape[0], dtype=float)

    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        pass

    eigvals = np.linalg.eigvalsh(matrix)
    min_eig = np.min(np.real(eigvals))
    if min_eig <= eps:
        matrix = matrix + (abs(min_eig) + eps) * np.eye(matrix.shape[0], dtype=float)

    for scale in (0.0, eps, 1e-7, 1e-6, 1e-5, 1e-4):
        trial = matrix + scale * np.eye(matrix.shape[0], dtype=float)
        try:
            np.linalg.cholesky(trial)
            return trial
        except np.linalg.LinAlgError:
            continue

    return matrix + 1e-3 * np.eye(matrix.shape[0], dtype=float)


def safe_inverse_and_det(matrix: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Robust inverse/determinant pair for nearly singular covariance matrices."""
    matrix = ensure_positive_definite(matrix, eps=eps)

    for scale in (0.0, eps, 1e-8, 1e-6, 1e-4):
        trial = matrix + scale * np.eye(matrix.shape[0], dtype=float)
        try:
            inv_cov = np.linalg.inv(trial)
            det_cov = np.linalg.det(trial)
            if np.isfinite(det_cov) and det_cov > eps:
                return inv_cov, float(det_cov)
        except np.linalg.LinAlgError:
            continue

    regularized = matrix + 1e-3 * np.eye(matrix.shape[0], dtype=float)
    return np.linalg.pinv(regularized), max(float(np.linalg.det(regularized)), eps)


def _normalize_pre_solution(pre_solution: np.ndarray) -> np.ndarray:
    pre_solution = np.real(pre_solution)
    pre_solution = np.maximum(pre_solution, 0.0)
    total = pre_solution.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("DDGA pre-solution became non-finite or degenerate.")
    return pre_solution / total


def _drop_duplicate_endpoint(limit_cycle: np.ndarray) -> np.ndarray:
    """Avoid counting the same phase twice when the solver returns a closed endpoint."""
    if limit_cycle.shape[0] <= 2:
        return limit_cycle

    step_norms = np.linalg.norm(np.diff(limit_cycle, axis=0), axis=1)
    positive_steps = step_norms[step_norms > 0]
    reference_step = float(np.median(positive_steps)) if positive_steps.size else 0.0
    duplicate_tol = max(1e-10, 0.5 * reference_step)

    if np.linalg.norm(limit_cycle[0] - limit_cycle[-1]) <= duplicate_tol:
        return limit_cycle[:-1]
    return limit_cycle


def _regularize_pre_solution(pre_solution: np.ndarray, floor_ratio: float) -> np.ndarray:
    """Blend with a uniform density to avoid numerically collapsed cycle weights."""
    if floor_ratio is None or floor_ratio <= 0:
        return pre_solution
    floor_ratio = float(np.clip(floor_ratio, 0.0, 1.0))
    uniform = np.full_like(pre_solution, 1.0 / len(pre_solution), dtype=float)
    return (1.0 - floor_ratio) * pre_solution + floor_ratio * uniform


def _pre_solution_support_fraction(pre_solution: np.ndarray, threshold_ratio: float = 0.25) -> float:
    baseline = threshold_ratio * float(np.mean(pre_solution))
    return float(np.mean(pre_solution >= baseline))


def _smooth_periodic(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Apply periodic Gaussian smoothing along the limit-cycle index."""
    if sigma is None or sigma <= 0:
        return signal

    signal = np.asarray(signal, dtype=float)
    n = signal.shape[0]
    if n < 3:
        return signal

    radius = int(max(1, np.ceil(3 * sigma)))
    radius = min(radius, (n - 1) // 2)
    if radius <= 0:
        return signal

    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)

    padded = np.concatenate([signal[-radius:], signal, signal[:radius]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _smooth_periodic_matrix_stack(stack: np.ndarray, sigma: float) -> np.ndarray:
    """Apply periodic smoothing to each matrix entry along the cycle index."""
    if sigma is None or sigma <= 0:
        return np.asarray(stack, dtype=float)

    stack = np.asarray(stack, dtype=float)
    smoothed = np.empty_like(stack)
    for idx in np.ndindex(stack.shape[1:]):
        slc = (slice(None),) + idx
        smoothed[slc] = _smooth_periodic(stack[slc], sigma)
    return smoothed


def _compute_legacy_pre_solution(gs: np.ndarray, dt: float, noise_strength: float) -> np.ndarray:
    int_gs2 = np.cumsum(gs**2 * dt)
    int_exp = np.exp(-int_gs2 / noise_strength)
    int_whole = np.cumsum(gs * int_exp / noise_strength * dt)

    c0 = (1.0 - int_exp[-1]) / max(int_whole[-1], 1e-16)
    pre_solution = (1.0 / np.maximum(int_exp, 1e-300)) * (1.0 - c0 * int_whole)
    return _normalize_pre_solution(pre_solution)


def _chang_cooper_delta(w: np.ndarray) -> np.ndarray:
    """Stable Chang-Cooper weighting for 1D advection-diffusion fluxes."""
    delta = np.empty_like(w, dtype=float)
    small = np.abs(w) < 1e-6
    delta[small] = 0.5 - w[small] / 12.0

    large = ~small
    exp_w = np.exp(np.clip(w[large], -50.0, 50.0))
    denom = np.maximum(exp_w - 1.0, 1e-16)
    delta[large] = 1.0 / np.maximum(w[large], 1e-16) - 1.0 / denom
    return np.clip(delta, 0.0, 1.0)


def _compute_flux_fpe_pre_solution(gs: np.ndarray, dt: float, noise_strength: float) -> np.ndarray:
    """
    Solve the steady 1D periodic advection-diffusion equation along the cycle.

    The drift is the tangential speed g and the effective tangential diffusion follows the
    current DDGA ansatz D_tan = D * g^2, but the stationary density is obtained numerically
    to avoid the exponential stiffness of the legacy closed form.
    """
    n = len(gs)
    if n == 1:
        return np.array([1.0], dtype=float)

    g_half = 0.5 * (gs + np.roll(gs, -1))
    diffusion_half = noise_strength * np.maximum(g_half**2, 1e-16)
    w = g_half * dt / diffusion_half
    delta = _chang_cooper_delta(w)

    forward_coeff = g_half * delta + diffusion_half / dt
    backward_coeff = g_half * (1.0 - delta) - diffusion_half / dt

    matrix = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    for i in range(n - 1):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n

        matrix[i, im1] += -forward_coeff[im1]
        matrix[i, i] += forward_coeff[i] - forward_coeff[im1]
        matrix[i, i] += -backward_coeff[im1]
        matrix[i, ip1] += backward_coeff[i]

    matrix[-1, :] = 1.0
    rhs[-1] = 1.0

    pre_solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return _normalize_pre_solution(pre_solution)


def _default_phase_kernel_diagnostics(mode: str = "proxy_speed") -> dict[str, float | bool | str]:
    return {
        "phase_kernel_mode": mode,
        "phase_empirical_use_drift": False,
        "phase_empirical_valid_fraction": 0.0,
        "phase_empirical_projection_distance_mean": 0.0,
        "phase_empirical_projection_distance_q90": 0.0,
        "phase_empirical_mean_increment_mean": 1.0,
        "phase_empirical_std_mean": 0.0,
        "phase_empirical_blend_mean": 0.0,
    }


def _proxy_phase_increment_variance(
    gs: np.ndarray,
    dt: float,
    noise_strength: float,
    speed_floor_ratio: float = 0.05,
) -> np.ndarray:
    median_speed = float(np.median(gs))
    speed_floor = max(1e-8, speed_floor_ratio * max(median_speed, 1e-8))
    gs_safe = np.maximum(gs, speed_floor)
    return 2.0 * noise_strength / np.maximum(gs_safe**2 * max(dt, 1e-16), 1e-16)


def _wrapped_phase_offset(target: np.ndarray, source: np.ndarray, n: int) -> np.ndarray:
    return ((target - source + n // 2) % n) - n // 2


def _estimate_empirical_phase_kernel(
    limit_cycle: np.ndarray,
    force_limit_cycle: np.ndarray,
    gs: np.ndarray,
    dt: float,
    noise_strength: float,
    samples_per_bin: int,
    count_prior: float,
    smooth_sigma: float,
    projection_distance_factor: float,
    use_drift: bool,
    seed: int,
    speed_floor_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool | str]]:
    n = len(limit_cycle)
    diagnostics = _default_phase_kernel_diagnostics(
        "empirical_drift_diffusion" if use_drift else "empirical_diffusion"
    )
    diagnostics["phase_empirical_use_drift"] = bool(use_drift)

    proxy_mean = np.ones(n, dtype=float)
    proxy_var = _proxy_phase_increment_variance(
        gs,
        dt,
        noise_strength,
        speed_floor_ratio=speed_floor_ratio,
    )

    if n < 2 or samples_per_bin <= 0:
        diagnostics["phase_empirical_blend_mean"] = 0.0
        diagnostics["phase_empirical_std_mean"] = float(np.mean(np.sqrt(proxy_var)))
        return proxy_mean, proxy_var, diagnostics

    rng = np.random.default_rng(int(seed))
    noise_scale = np.sqrt(max(2.0 * noise_strength * dt, 0.0))
    samples_per_bin = int(max(samples_per_bin, 1))

    cycle_segments = np.diff(np.vstack([limit_cycle, limit_cycle[0]]), axis=0)
    cycle_step_scale = float(max(np.median(np.linalg.norm(cycle_segments, axis=1)), 1e-8))
    noise_step_scale = float(np.sqrt(max(2.0 * noise_strength * dt * limit_cycle.shape[1], 1e-16)))
    distance_cap = float(
        max(cycle_step_scale, noise_step_scale) * max(projection_distance_factor, 0.0)
    )

    x0 = np.repeat(limit_cycle, samples_per_bin, axis=0)
    drift = np.repeat(force_limit_cycle, samples_per_bin, axis=0)
    x1 = x0 + drift * dt + noise_scale * rng.normal(size=x0.shape)

    tree = cKDTree(limit_cycle)
    proj_dist, proj_idx = tree.query(x1, k=1)
    proj_dist = proj_dist.reshape(n, samples_per_bin)
    proj_idx = proj_idx.reshape(n, samples_per_bin)

    valid_mask = np.isfinite(proj_dist)
    if distance_cap > 0:
        valid_mask &= proj_dist <= distance_cap

    source_idx = np.broadcast_to(np.arange(n, dtype=int)[:, None], proj_idx.shape)
    phase_increments = _wrapped_phase_offset(proj_idx, source_idx, n).astype(float)

    empirical_mean = proxy_mean.copy()
    empirical_var = proxy_var.copy()
    valid_counts = np.sum(valid_mask, axis=1).astype(float)

    for i in range(n):
        if valid_counts[i] <= 0:
            continue
        increments_i = phase_increments[i, valid_mask[i]]
        empirical_mean[i] = float(np.mean(increments_i))
        empirical_var[i] = float(np.var(increments_i)) if increments_i.size > 1 else proxy_var[i]

    if smooth_sigma and smooth_sigma > 0:
        empirical_mean = _smooth_periodic(empirical_mean, smooth_sigma)
        empirical_var = _smooth_periodic(empirical_var, smooth_sigma)

    empirical_var = np.maximum(empirical_var, 1e-8)
    count_prior = max(float(count_prior), 0.0)
    blend = valid_counts / np.maximum(valid_counts + count_prior, 1e-8)

    if use_drift:
        mean_increment = (1.0 - blend) * proxy_mean + blend * empirical_mean
    else:
        mean_increment = proxy_mean
    var_increment = (1.0 - blend) * proxy_var + blend * empirical_var
    var_increment = np.maximum(var_increment, 1e-8)

    valid_projection_distances = proj_dist[valid_mask]
    diagnostics["phase_empirical_valid_fraction"] = float(np.mean(valid_mask))
    diagnostics["phase_empirical_projection_distance_mean"] = (
        float(np.mean(valid_projection_distances)) if valid_projection_distances.size else 0.0
    )
    diagnostics["phase_empirical_projection_distance_q90"] = (
        float(np.quantile(valid_projection_distances, 0.9)) if valid_projection_distances.size else 0.0
    )
    diagnostics["phase_empirical_mean_increment_mean"] = float(np.mean(empirical_mean))
    diagnostics["phase_empirical_std_mean"] = float(np.mean(np.sqrt(var_increment)))
    diagnostics["phase_empirical_blend_mean"] = float(np.mean(blend))

    return mean_increment, var_increment, diagnostics


def _build_phase_transition_kernel(
    gs: np.ndarray,
    dt: float,
    noise_strength: float,
    speed_floor_ratio: float = 0.05,
    wrap_std_cutoff_ratio: float = 0.2,
    max_bandwidth: int = 128,
    phase_drift: np.ndarray | None = None,
    phase_increment_variance: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Approximate the phase dynamics on the periodic orbit by a wrapped Gaussian Markov chain.

    On the time parametrization x = x*(t), a tangential perturbation dx maps to a phase perturbation
    dtheta ≈ dx_tan / ||f||, so the phase diffusion scale is D / ||f||^2.
    """
    n = len(gs)
    if n < 2:
        return [(np.array([0], dtype=int), np.array([1.0], dtype=float))]

    if phase_drift is None:
        phase_drift = np.ones(n, dtype=float)
    else:
        phase_drift = np.asarray(phase_drift, dtype=float)

    if phase_increment_variance is None:
        phase_increment_variance = _proxy_phase_increment_variance(
            gs,
            dt,
            noise_strength,
            speed_floor_ratio=speed_floor_ratio,
        )
    else:
        phase_increment_variance = np.asarray(phase_increment_variance, dtype=float)

    transitions: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        mean_increment = float(phase_drift[i])
        std_index = float(np.sqrt(max(phase_increment_variance[i], 1e-16)))
        if std_index >= wrap_std_cutoff_ratio * n:
            targets = np.arange(n, dtype=int)
            probs = np.full(n, 1.0 / n, dtype=float)
            transitions.append((targets, probs))
            continue

        if std_index <= 1e-6:
            targets = np.array([(i + int(np.round(mean_increment))) % n], dtype=int)
            probs = np.array([1.0], dtype=float)
            transitions.append((targets, probs))
            continue

        radius = int(min(max_bandwidth, max(2, np.ceil(6.0 * std_index))))
        center = int(np.round(mean_increment))
        offsets = np.arange(center - radius, center + radius + 1, dtype=int)
        targets = (i + offsets) % n
        probs = np.exp(-0.5 * ((offsets - mean_increment) / std_index) ** 2)
        probs = probs / np.sum(probs)
        transitions.append((targets.astype(int), probs.astype(float)))

    return transitions


def _compute_phase_sde_pre_solution(
    gs: np.ndarray,
    dt: float,
    noise_strength: float,
    phase_drift: np.ndarray | None = None,
    phase_increment_variance: np.ndarray | None = None,
    diagnostics: dict[str, float | bool | str] | None = None,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float | bool | str]]]:
    transitions = _build_phase_transition_kernel(
        gs,
        dt,
        noise_strength,
        phase_drift=phase_drift,
        phase_increment_variance=phase_increment_variance,
    )
    n = len(gs)
    weights = np.full(n, 1.0 / n, dtype=float)

    for _ in range(4000):
        next_weights = np.zeros(n, dtype=float)
        for i, (targets, probs) in enumerate(transitions):
            next_weights[targets] += weights[i] * probs
        next_weights = _normalize_pre_solution(next_weights)
        if np.linalg.norm(next_weights - weights, ord=1) < 1e-12:
            weights = next_weights
            break
        weights = next_weights

    weights = _normalize_pre_solution(weights)
    if not return_diagnostics:
        return weights
    return weights, (diagnostics or _default_phase_kernel_diagnostics())


def _regularize_normal_jacobian(
    jac_normal: np.ndarray,
    stability_floor: float,
) -> tuple[np.ndarray, float]:
    """
    The DDGA paper assumes local equilibrium on the normal plane, which requires the normal dynamics
    to be locally stable. Near a bifurcation this assumption breaks pointwise, so we enforce a
    cycle-informed negative relaxation floor on the symmetric part before solving the Lyapunov equation.
    """
    if jac_normal.shape[0] == 0:
        return jac_normal, 0.0

    stability_floor = max(float(stability_floor), 1e-8)
    sym_part = 0.5 * (jac_normal + jac_normal.T)
    max_real_sym_eig = float(np.max(np.linalg.eigvalsh(sym_part)))
    shift = max(0.0, max_real_sym_eig + stability_floor)
    if shift == 0.0:
        return jac_normal, 0.0
    stabilized = jac_normal - shift * np.eye(jac_normal.shape[0], dtype=float)
    return stabilized, shift


def _estimate_normal_stability_floor(
    jac_normal_stack: np.ndarray,
    min_stability: float = 1e-2,
    stable_quantile: float = 0.25,
) -> float:
    """
    Estimate a transverse relaxation floor from the stable segments of the cycle.

    This approximates the finite-time Floquet contraction scale that the DDGA local-equilibrium
    assumption needs, while avoiding the near-zero margins that produce numerically gigantic
    covariances on slow-fast segments.
    """
    min_stability = max(float(min_stability), 1e-8)
    stable_quantile = float(np.clip(stable_quantile, 0.0, 1.0))

    if jac_normal_stack.shape[1] == 0:
        return min_stability

    max_sym_eigs = np.array(
        [
            float(np.max(np.linalg.eigvalsh(0.5 * (jac_normal + jac_normal.T))))
            for jac_normal in jac_normal_stack
        ],
        dtype=float,
    )
    stable_magnitudes = -max_sym_eigs[max_sym_eigs < 0.0]
    if stable_magnitudes.size == 0:
        return min_stability
    return max(min_stability, float(np.quantile(stable_magnitudes, stable_quantile)))


def _solve_local_normal_covariance(
    jac_normal: np.ndarray,
    noise_strength: float,
    stability_floor: float,
) -> tuple[np.ndarray, float]:
    """
    Paper-aligned local normal-plane covariance:
    solve A_n S + S A_n^T + 2 D I = 0 after enforcing the local stability assumption numerically.
    """
    normal_dim = jac_normal.shape[0]
    if normal_dim == 0:
        return np.zeros((0, 0), dtype=float), 0.0

    jac_stable, shift = _regularize_normal_jacobian(jac_normal, stability_floor=stability_floor)
    sigma_normal = -solve_continuous_lyapunov(
        jac_stable,
        2.0 * noise_strength * np.eye(normal_dim, dtype=float),
    )
    sigma_normal = np.real(sigma_normal)
    sigma_normal = 0.5 * (sigma_normal + sigma_normal.T)
    sigma_normal = ensure_positive_definite(sigma_normal)
    return sigma_normal, shift


def _lift_covariance_from_normal_plane(
    normal_basis: np.ndarray,
    sigma_normal: np.ndarray,
    tangent_unit: np.ndarray,
    noise_strength: float,
    tangent_scale: float,
) -> np.ndarray:
    """
    Paper-aligned uplift from the normal plane to the full space.

    The paper uses the unit tangential direction only to make the full covariance positive definite,
    after the tangential information has already been handled by the pre-solution. In the discrete
    mixture implementation, we keep only a small tangential regularizer to avoid double-counting
    phase spread that is already encoded by the cycle weights.
    """
    sigma_full = normal_basis @ sigma_normal @ normal_basis.T
    sigma_full = sigma_full + float(max(tangent_scale, 0.0)) * noise_strength * np.outer(tangent_unit, tangent_unit)
    return ensure_positive_definite(sigma_full)


def _build_limit_cycle_frames(
    force_limit_cycle: np.ndarray,
    jacobian_limit_cycle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a continuous tangent/normal frame and the normal-plane Jacobians along the cycle."""
    len_limit_cycle, dim = force_limit_cycle.shape
    tangent_vectors = np.zeros((len_limit_cycle, dim), dtype=float)
    normal_bases = np.zeros((len_limit_cycle, dim, max(dim - 1, 0)), dtype=float)
    jac_normal_stack = np.zeros((len_limit_cycle, max(dim - 1, 0), max(dim - 1, 0)), dtype=float)
    q_last_step = np.zeros((dim, dim), dtype=float)

    for i in range(len_limit_cycle):
        tangent_norm = np.linalg.norm(force_limit_cycle[i])
        if tangent_norm < 1e-12:
            raise RuntimeError("Tangential velocity is too small on the limit cycle.")

        tangent_vector = force_limit_cycle[i] / tangent_norm
        if dim == 2:
            normal_vec = np.array([-tangent_vector[1], tangent_vector[0]], dtype=float)
            q = np.column_stack((tangent_vector, normal_vec))
        else:
            q = np.column_stack((tangent_vector, np.eye(dim, dtype=float)[:, 1:dim]))

        q_this_step, _ = qr(q)
        if i > 0 and dim > 1:
            direction = np.sign(np.diag(q_this_step[:, 1:].T @ q_last_step[:, 1:]))
            direction[direction == 0.0] = 1.0
            q_this_step[:, 1:] *= direction

        q_last_step = q_this_step.copy()
        tangent_vectors[i] = tangent_vector
        if dim > 1:
            normal_bases[i] = q_this_step[:, 1:]
            jac_normal_stack[i] = q_this_step[:, 1:].T @ jacobian_limit_cycle[i] @ q_this_step[:, 1:]

    return tangent_vectors, normal_bases, jac_normal_stack


def _compute_covariance_stack(
    force_limit_cycle: np.ndarray,
    jacobian_limit_cycle: np.ndarray,
    noise_strength: float,
    mode: str,
    min_stability: float,
    stability_quantile: float,
    jacobian_smooth_sigma: float,
    tangent_scale: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute DDGA covariance matrices along the limit cycle.

    `paper_local_adaptive` follows the DDGA paper's normal-plane Lyapunov solve, with an adaptive
    stability floor estimated from the stable parts of the cycle. `legacy` preserves the old local
    solve for direct comparison.
    """
    len_limit_cycle, dim = force_limit_cycle.shape
    sigma_all = np.zeros((len_limit_cycle, dim, dim), dtype=float)
    tangent_vectors, normal_bases, jac_normal_stack = _build_limit_cycle_frames(
        force_limit_cycle,
        jacobian_limit_cycle,
    )

    if jacobian_smooth_sigma and jacobian_smooth_sigma > 0 and dim > 1:
        jac_normal_stack = _smooth_periodic_matrix_stack(jac_normal_stack, jacobian_smooth_sigma)

    if mode == "legacy":
        covariance_shifts = np.zeros(len_limit_cycle, dtype=float)
        for i in range(len_limit_cycle):
            sigma_normal = -solve_continuous_lyapunov(
                jac_normal_stack[i],
                2.0 * noise_strength * np.eye(max(dim - 1, 0), dtype=float),
            )
            sigma_normal = np.real(sigma_normal)
            sigma_normal = 0.5 * (sigma_normal + sigma_normal.T)
            sigma_all[i] = ensure_positive_definite(
                normal_bases[i] @ sigma_normal @ normal_bases[i].T
                + noise_strength * np.outer(force_limit_cycle[i], force_limit_cycle[i])
            )
        return sigma_all, 0.0, covariance_shifts

    if mode != "paper_local_adaptive":
        raise ValueError(f"Unsupported DDGA covariance_mode={mode}")

    stability_floor = _estimate_normal_stability_floor(
        jac_normal_stack,
        min_stability=min_stability,
        stable_quantile=stability_quantile,
    )
    covariance_shifts = np.zeros(len_limit_cycle, dtype=float)
    for i in range(len_limit_cycle):
        sigma_normal, covariance_shifts[i] = _solve_local_normal_covariance(
            jac_normal_stack[i],
            noise_strength,
            stability_floor=stability_floor,
        )
        sigma_all[i] = _lift_covariance_from_normal_plane(
            normal_bases[i],
            sigma_normal,
            tangent_vectors[i],
            noise_strength,
            tangent_scale=tangent_scale,
        )

    return sigma_all, stability_floor, covariance_shifts


def _compute_pre_solution(
    gs: np.ndarray,
    dt: float,
    noise_strength: float,
    mode: str,
    limit_cycle: np.ndarray | None = None,
    force_limit_cycle: np.ndarray | None = None,
    phase_empirical_samples_per_bin: int = 32,
    phase_empirical_count_prior: float = 16.0,
    phase_empirical_smooth_sigma: float = 2.0,
    phase_empirical_projection_distance_factor: float = 3.0,
    phase_empirical_use_drift: bool = False,
    phase_empirical_seed: int = 1234,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float | bool | str]]]:
    if mode == "legacy":
        pre_solution = _compute_legacy_pre_solution(gs, dt, noise_strength)
        if return_diagnostics:
            return pre_solution, _default_phase_kernel_diagnostics("legacy")
        return pre_solution
    if mode == "uniform":
        pre_solution = np.full(len(gs), 1.0 / len(gs), dtype=float)
        if return_diagnostics:
            return pre_solution, _default_phase_kernel_diagnostics("uniform")
        return pre_solution
    if mode == "flux_fpe":
        pre_solution = _compute_flux_fpe_pre_solution(gs, dt, noise_strength)
        if return_diagnostics:
            return pre_solution, _default_phase_kernel_diagnostics("flux_fpe")
        return pre_solution
    if mode == "phase_sde":
        return _compute_phase_sde_pre_solution(
            gs,
            dt,
            noise_strength,
            diagnostics=_default_phase_kernel_diagnostics("proxy_speed"),
            return_diagnostics=return_diagnostics,
        )
    if mode == "phase_sde_empirical":
        if limit_cycle is None or force_limit_cycle is None:
            raise ValueError("DDGA empirical phase mode requires limit_cycle and force_limit_cycle.")
        mean_increment, var_increment, diagnostics = _estimate_empirical_phase_kernel(
            limit_cycle,
            force_limit_cycle,
            gs,
            dt,
            noise_strength,
            samples_per_bin=phase_empirical_samples_per_bin,
            count_prior=phase_empirical_count_prior,
            smooth_sigma=phase_empirical_smooth_sigma,
            projection_distance_factor=phase_empirical_projection_distance_factor,
            use_drift=phase_empirical_use_drift,
            seed=phase_empirical_seed,
        )
        return _compute_phase_sde_pre_solution(
            gs,
            dt,
            noise_strength,
            phase_drift=mean_increment,
            phase_increment_variance=var_increment,
            diagnostics=diagnostics,
            return_diagnostics=return_diagnostics,
        )
    raise ValueError(f"Unsupported DDGA pre_solution_mode={mode}")


def _pre_solution_fallback_order(mode: str) -> list[str]:
    order = [mode]
    for candidate in ("phase_sde", "uniform"):
        if candidate not in order:
            order.append(candidate)
    return order


def _estimate_period_from_endpoint(
    path: np.ndarray,
    force_origin: np.ndarray,
    dt: float,
    start_fraction: float = 0.3,
) -> Union[float, None]:
    centered_path = path - path[-1]
    distance_path = np.linalg.norm(centered_path, axis=1)
    threshold_force = np.max(np.linalg.norm(force_origin, axis=1))

    start_idx = int(start_fraction * len(path))
    selected_distance = distance_path[start_idx:]
    near_points = np.where(selected_distance < 3 * threshold_force * dt)[0] + start_idx

    period_time = []
    for i in range(1, len(near_points)):
        if near_points[i] - near_points[i - 1] != 1:
            period_time.append(near_points[i - 1])

    if len(period_time) < 2:
        return None

    period = np.mean(np.diff(period_time)) * dt
    if not np.isfinite(period) or period <= dt:
        return None
    return float(period)


def _estimate_period_from_peaks(
    path: np.ndarray,
    time: np.ndarray,
    start_fraction: float = 0.8,
    coordinate: Union[int, None] = None,
) -> Tuple[float, np.ndarray]:
    start_idx = int(start_fraction * len(path))
    if start_idx >= len(path) - 2:
        raise RuntimeError("Peak-based DDGA period detection has too little tail data.")

    if coordinate is None:
        coordinate = int(np.argmax(np.std(path[start_idx:], axis=0)))

    signal = path[:, coordinate]
    signal_tail = signal[start_idx:]

    peak_indices = []
    for i in range(1, len(signal_tail) - 1):
        if signal_tail[i - 1] < signal_tail[i] and signal_tail[i] >= signal_tail[i + 1]:
            peak_indices.append(start_idx + i)

    peak_indices = np.array(peak_indices, dtype=int)
    if len(peak_indices) < 2:
        raise RuntimeError(
            "Failed to detect the period from trajectory peaks. Try increasing steps or adjusting the initial condition."
        )

    peak_times = time[peak_indices]
    period = np.mean(np.diff(peak_times))
    if not np.isfinite(period) or period <= 0:
        raise RuntimeError("Peak-based DDGA period detection produced an invalid period.")

    return float(period), peak_indices

def gaussian_land_dim2(V, Sigma, cycle, phi, range_1, range_2, num):
    """
    Generate 2D Gaussian landscape from DDGA results.
    
    Args:
        V: n*2 projection matrix
        Sigma: n*n*(time steps) covariance matrix set
        cycle: (time steps)*n limit cycle time series
        phi: (time steps)*1 pre-solution
        range_1/range_2: 1*2 range matrix as [min(range_dimension), max(range_dimension)]
        num: Resolution of the landscape
        
    Returns:
        P_DDGA, mesh_1, mesh_2: landscape data and mesh grids
    """
    # Generate mesh grid
    x = np.linspace(range_1[0], range_1[1], num)
    y = np.linspace(range_2[0], range_2[1], num)
    mesh_1, mesh_2 = np.meshgrid(x, y, indexing='xy')

    # Calculate projection parameters
    mu_proj = np.zeros((len(phi), 2))
    sigma0_proj = np.zeros((2, 2, len(phi)))
    for i in range(len(phi)):
        # Fix: ensure correct dimension of cycle[i, :]
        cycle_vec = cycle[i, :].reshape(-1, 1) if cycle[i, :].ndim == 1 else cycle[i, :]
        mu_proj[i, :] = (V.T @ cycle_vec).flatten()
        sigma0_proj[:, :, i] = V.T @ Sigma[:, :, i] @ V

    # Initialize probability matrix
    P_DDGA = np.zeros(mesh_1.shape)

    # Gaussian kernel calculation
    for k in range(len(phi)):
        inv_cov, det_sig = safe_inverse_and_det(sigma0_proj[:, :, k])

        # Calculate constants
        Cons1 = 1 / np.sqrt((2*np.pi)**2 * det_sig)
        Cons2 = np.exp(-0.5)

        # Calculate gaussian function
        dx = mesh_1 - mu_proj[k, 0]
        dy = mesh_2 - mu_proj[k, 1]
        quadratic_form = inv_cov[0,0]*dx**2 + inv_cov[1,1]*dy**2 + 2*inv_cov[0,1]*dx*dy
        
        # Calculate Z
        Z = Cons1 * (Cons2 ** quadratic_form)
        Z = np.real(Z)
        Z[Z < 0] = 0

        # Accumulate probability
        P_DDGA += Z * phi[k]

    return P_DDGA, mesh_1, mesh_2

def analyze_limit_cycle(
    system_function: Callable[[torch.Tensor], torch.Tensor], 
    dim: int, 
    x_min: float, 
    x_max: float, 
    noise_strength: float, 
    dt: float = 0.01, 
    steps: int = 20000,
    plot_verification: bool = False,
    plot_landscape: bool = False,
    init_state: Union[np.ndarray, list, tuple, None] = None,
    period_detection: str = "auto",
    peak_coordinate: Union[int, None] = None,
    period_start_fraction: float = 0.8,
    pre_solution_mode: str = "phase_sde",
    pre_solution_require_full_cycle: bool = True,
    pre_solution_min_support_fraction: float = 0.85,
    pre_solution_smooth_sigma: float = 0.0,
    pre_solution_floor_ratio: float = 0.0,
    phase_empirical_samples_per_bin: int = 32,
    phase_empirical_count_prior: float = 16.0,
    phase_empirical_smooth_sigma: float = 2.0,
    phase_empirical_projection_distance_factor: float = 3.0,
    phase_empirical_use_drift: bool = False,
    phase_empirical_seed: int = 1234,
    covariance_mode: str = "paper_local_adaptive",
    covariance_min_stability: float = 1e-2,
    covariance_stability_quantile: float = 0.5,
    covariance_jacobian_smooth_sigma: float = 0.0,
    covariance_tangent_scale: float = 0.1,
    return_details: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], DDGAResult]:
    """
    Analyze limit cycle of a dynamical system with noise.
    
    Args:
        system_function: Function defining the system dynamics
        dim: Dimension of the system
        x_min: Minimum value for initial condition sampling
        x_max: Maximum value for initial condition sampling
        noise_strength: Strength of noise in the system
        dt: Time step for integration
        steps: Number of integration steps
        plot_verification: Whether to plot verification graph
        plot_landscape: Whether to plot landscape
        
    Returns:
        Tuple of limit cycle trajectory and covariance matrices
    """
    
    def ode_fun(t: float, y: np.ndarray) -> np.ndarray:
        """Convert system function to numpy format for scipy integration."""
        with torch.no_grad():
            y_torch = torch.tensor(y, dtype=torch.float64)
            dy = system_function(y_torch).cpu().numpy()
        return dy

    # Simulate long trajectory
    time = np.arange(0, steps*dt, dt)
    if init_state is not None and len(init_state) == dim:
        x0 = np.asarray(init_state, dtype=float)
    else:
        x0 = np.random.uniform(x_min, x_max, size=dim)

    sol = solve_ivp(
        ode_fun,
        [time[0], time[-1]],
        x0,
        t_eval=time,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    x = sol.y.T
    path = x

    # Calculate drift force
    force_origin = np.zeros((steps, dim))
    for i in range(steps):
        force_origin[i] = ode_fun(0, path[i])

    period = None
    cycle_start = path[-1]
    detection_used = "unknown"

    if period_detection not in {"auto", "endpoint", "peak"}:
        raise ValueError(f"Unsupported DDGA period_detection={period_detection}")

    if period_detection in {"auto", "endpoint"}:
        period = _estimate_period_from_endpoint(path, force_origin, dt)
        if period is not None:
            detection_used = "endpoint"

    if period is None and period_detection in {"auto", "peak"}:
        period, peak_indices = _estimate_period_from_peaks(
            path,
            time,
            start_fraction=period_start_fraction,
            coordinate=peak_coordinate,
        )
        cycle_start = path[peak_indices[-1]]
        detection_used = f"peak:{peak_coordinate if peak_coordinate is not None else 'auto'}"

    if period is None:
        raise RuntimeError("DDGA failed to detect a valid limit-cycle period.")

    # Generate limit cycle
    t_limit_cycle = np.arange(0, period + dt, dt)
    
    sol = solve_ivp(
        ode_fun,
        [t_limit_cycle[0], t_limit_cycle[-1]],
        cycle_start,
        t_eval=t_limit_cycle,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    sol_limit_cycle = sol.y.T
    limit_cycle = _drop_duplicate_endpoint(sol_limit_cycle)
    len_limit_cycle = len(limit_cycle)

    # Force and Jacobian along limit cycle
    force_limit_cycle = np.zeros((len_limit_cycle, dim))
    jacobian_limit_cycle = np.zeros((len_limit_cycle, dim, dim))
    for i in range(len_limit_cycle):
        force_limit_cycle[i] = ode_fun(0, limit_cycle[i])
        jacobian_limit_cycle[i] = torch.autograd.functional.jacobian(
            system_function, 
            torch.tensor(limit_cycle[i], dtype=torch.float64)
        ).cpu().numpy()

    # Pre-solution calculation
    gs = np.linalg.norm(force_limit_cycle, axis=1)
    requested_pre_solution_mode = pre_solution_mode
    support_fraction = 0.0
    selected_pre_solution_mode = requested_pre_solution_mode
    phase_kernel_diagnostics = _default_phase_kernel_diagnostics("proxy_speed")
    candidate_modes = _pre_solution_fallback_order(requested_pre_solution_mode)

    for candidate_mode in candidate_modes:
        candidate_pre_solution, candidate_phase_kernel_diagnostics = _compute_pre_solution(
            gs,
            dt,
            noise_strength,
            candidate_mode,
            limit_cycle=limit_cycle,
            force_limit_cycle=force_limit_cycle,
            phase_empirical_samples_per_bin=phase_empirical_samples_per_bin,
            phase_empirical_count_prior=phase_empirical_count_prior,
            phase_empirical_smooth_sigma=phase_empirical_smooth_sigma,
            phase_empirical_projection_distance_factor=phase_empirical_projection_distance_factor,
            phase_empirical_use_drift=phase_empirical_use_drift,
            phase_empirical_seed=phase_empirical_seed,
            return_diagnostics=True,
        )
        candidate_support_fraction = _pre_solution_support_fraction(candidate_pre_solution)
        selected_pre_solution_mode = candidate_mode
        pre_solution = candidate_pre_solution
        support_fraction = candidate_support_fraction
        phase_kernel_diagnostics = candidate_phase_kernel_diagnostics
        if (not pre_solution_require_full_cycle) or (candidate_support_fraction >= pre_solution_min_support_fraction):
            break

    if pre_solution_smooth_sigma and pre_solution_smooth_sigma > 0:
        pre_solution = _smooth_periodic(pre_solution, pre_solution_smooth_sigma)
        pre_solution = _normalize_pre_solution(pre_solution)
    if pre_solution_floor_ratio and pre_solution_floor_ratio > 0:
        pre_solution = _regularize_pre_solution(pre_solution, pre_solution_floor_ratio)
        pre_solution = _normalize_pre_solution(pre_solution)
    support_fraction = _pre_solution_support_fraction(pre_solution)

    # Covariance calculation
    sigma_all, covariance_stability_floor, covariance_shifts = _compute_covariance_stack(
        force_limit_cycle,
        jacobian_limit_cycle,
        noise_strength,
        mode=covariance_mode,
        min_stability=covariance_min_stability,
        stability_quantile=covariance_stability_quantile,
        jacobian_smooth_sigma=covariance_jacobian_smooth_sigma,
        tangent_scale=covariance_tangent_scale,
    )
    covariance_shift_mean = float(np.mean(covariance_shifts)) if covariance_shifts.size else 0.0
    covariance_shift_max = float(np.max(covariance_shifts)) if covariance_shifts.size else 0.0
    covariance_shift_fraction = float(np.mean(covariance_shifts > 0.0)) if covariance_shifts.size else 0.0

    # Plot pre-solution only when requested
    if plot_verification:
        fig, ax = plt.subplots(figsize=(8, 6))
        time_axis = np.arange(pre_solution.shape[0], dtype=float) * dt
        if time_axis.size > 1 and period > 0:
            time_axis *= period / max(time_axis[-1], dt)
        line = ax.plot(time_axis, pre_solution, linewidth=2.5, color=[160/255, 201/255, 235/255])
        
        ax.set_xlim([time_axis[0], time_axis[-1] if time_axis.size > 1 else max(period, dt)])
        ax.set_xlabel('Time', fontname='Arial', fontsize=24)
        ax.set_ylabel('Pre-Solution', fontname='Arial', fontsize=24)
        
        ax.legend(['Pre-Solution'], loc='upper right')
        ax.grid(True)
        
        # Set font properties
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontname('Arial')
            item.set_fontsize(20)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # Set background color
        ax.set_facecolor([1, 1, 1])
        fig.set_facecolor([1, 1, 1])
        
        plt.show()

    # Plot landscape only when requested
    if plot_landscape:
        # Add missing parameter definitions
        range_1 = [-1.5, 1.5]
        range_2 = [-1.5, 1.5]
        num = 300
        
        V = np.eye(2)
        # Fix: ensure correct Sigma dimension
        P_DDGA, mesh_1, mesh_2 = gaussian_land_dim2(V, sigma_all.transpose(1,2,0), 
                                                  limit_cycle, pre_solution, 
                                                  range_1, range_2, num)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(mesh_1, mesh_2, P_DDGA, cmap='viridis', antialiased=True)
        ax.set_xlim(range_1)
        ax.set_ylim(range_2)
        ax.view_init(elev=59, azim=-29)
        
        # Set font properties
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.line.set_linewidth(1.5)
        ax.yaxis.line.set_linewidth(1.5)
        ax.zaxis.line.set_linewidth(1.5)
        
        # Set background color
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        plt.show()

    if return_details:
        return DDGAResult(
            limit_cycle=limit_cycle,
            covariance=sigma_all,
            pre_solution=pre_solution,
            period=float(period),
            period_detection=detection_used,
            pre_solution_mode=selected_pre_solution_mode,
            pre_solution_requested_mode=requested_pre_solution_mode,
            pre_solution_support_fraction=support_fraction,
            covariance_mode=covariance_mode,
            covariance_stability_floor=float(covariance_stability_floor),
            covariance_shift_mean=covariance_shift_mean,
            covariance_shift_max=covariance_shift_max,
            covariance_shift_fraction=covariance_shift_fraction,
            phase_kernel_mode=str(phase_kernel_diagnostics.get("phase_kernel_mode", "proxy_speed")),
            phase_empirical_use_drift=bool(phase_kernel_diagnostics.get("phase_empirical_use_drift", False)),
            phase_empirical_valid_fraction=float(phase_kernel_diagnostics.get("phase_empirical_valid_fraction", 0.0)),
            phase_empirical_projection_distance_mean=float(
                phase_kernel_diagnostics.get("phase_empirical_projection_distance_mean", 0.0)
            ),
            phase_empirical_projection_distance_q90=float(
                phase_kernel_diagnostics.get("phase_empirical_projection_distance_q90", 0.0)
            ),
            phase_empirical_mean_increment_mean=float(
                phase_kernel_diagnostics.get("phase_empirical_mean_increment_mean", 1.0)
            ),
            phase_empirical_std_mean=float(phase_kernel_diagnostics.get("phase_empirical_std_mean", 0.0)),
            phase_empirical_blend_mean=float(phase_kernel_diagnostics.get("phase_empirical_blend_mean", 0.0)),
        )

    return limit_cycle, sigma_all
if __name__ == "__main__":
    # Define dynamical system function (PyTorch version)
    def drift_f(y):
        lambda_ = 2.0
        m1, m2 = -1.5, 1.5
        x, y = y[0], y[1]
        
        dx1 = (lambda_*x - y + lambda_*m1*x**3 + 
            (m2 - m1 + m1*m2)*x**2*y + 
            lambda_*m1*m2*x*y**2 + m2*y**3)
        
        dx2 = (x + lambda_*y - x**3 + lambda_*m1*x**2*y + 
            (m1*m2 - m1 - 1)*x*y**2 + 
            lambda_*m1*m2*y**3)
        
        return torch.stack([dx1, dx2])

    # Call analysis function
    LC, Sigma_all = analyze_limit_cycle(
        system_function=drift_f, dim=2, x_min=-1.5, x_max=1.5, noise_strength=0.1, plot_verification=True, plot_landscape=True
    )
