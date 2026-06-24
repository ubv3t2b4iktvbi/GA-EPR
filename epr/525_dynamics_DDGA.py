from pathlib import Path

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import qr, solve_continuous_lyapunov


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "results" / "predatorprey" / "config.yaml"
DEFAULT_CONFIG = {
    "dynamics": {
        "D": 1.0,
        "K": 6.67826211,
        "a": 2.539936705,
        "h": 2.030471009,
        "d": 0.5,
        "c": 1.0,
        "growth_rate": 1.0,
        "initial_offset": 0.85,
        "dt": 0.1,
        "steps": 20000,
    },
    "real_data": {
        "excel_path": "C2 - 3.xlsx",
        "excel_paths": [],
        "excel_glob": None,
        "real_n_scale": 1.0,
        "target_lower": [0.0, 0.0],
        "target_upper": [8.0, 4.0],
        "outlier_iqr_factor": 0.6,
    },
    "covariance": {
        "eigenvalue_floor": 1e-1,
        "correction_tolerance": 1e-12,
        "low_landscape": {
            "floor_factor": 15.0,
            "n_min": 2.7,
            "n_max": 4.1,
            "p_min": 0.6,
            "p_max": 0.95,
        },
        "lower_left": {
            "floor_factor": 0.1,
            "n_min": 1.2,
            "n_max": 2.2,
            "p_min": 0.0,
            "p_max": 0.9,
        },
    },
    "landscape": {
        "range_1": [0.0, 5.0],
        "range_2": [0.0, 3.0],
        "grid_num": 300,
        "display_real_point_offset": [-0.1, 0.05],
        "real_point_style": {
            "color": "white",
            "edgecolors": "black",
            "linewidths": 0.55,
            "size": 22,
            "alpha": 0.85,
        },
    },
    "energy_test": {
        "ref_sample_size": 200,
        "bootstraps": 200,
        "alpha": 0.05,
        "bad_quantile": 0.1,
        "bad_delta": 0.1,
        "sample_bounds": {
            "x_min": 0.0,
            "x_max": 5.0,
            "y_min": 0.0,
            "y_max": 3.0,
        },
    },
}


def merge_config(defaults, overrides):
    merged = defaults.copy()
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path=CONFIG_PATH):
    if not config_path.exists():
        print(f"Config file not found, using defaults: {config_path}")
        return DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return merge_config(DEFAULT_CONFIG, loaded)


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_real_data_paths(real_data_config):
    paths = []
    excel_paths = real_data_config.get("excel_paths") or []
    excel_glob = real_data_config.get("excel_glob")
    excel_path = real_data_config.get("excel_path")

    if excel_paths:
        paths.extend(resolve_project_path(path_value) for path_value in excel_paths)
    elif excel_glob:
        pattern = str(resolve_project_path(excel_glob))
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    elif excel_path:
        paths.append(resolve_project_path(excel_path))

    resolved_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.suffix.lower() != ".xlsx":
            continue
        if resolved not in seen:
            resolved_paths.append(resolved)
            seen.add(resolved)

    if not resolved_paths:
        raise ValueError(
            "No real-data .xlsx files were configured. Set real_data.excel_path, "
            "real_data.excel_paths, or real_data.excel_glob."
        )
    missing_paths = [path for path in resolved_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Configured real-data files do not exist:\n{missing_text}")
    return resolved_paths


def load_real_points(excel_path):
    data_frame = pd.read_excel(excel_path)
    selected = data_frame[["prey", "predator"]].apply(pd.to_numeric, errors="coerce").dropna()
    selected = selected[(selected["prey"] > 0.0) & (selected["predator"] > 0.0)]
    excel_rows = selected.index.to_numpy(dtype=int) + 2
    points = selected.to_numpy(dtype=float)
    if len(points) == 0:
        raise ValueError(f"No positive prey/predator rows were found in {excel_path}.")
    return points, excel_rows


def load_real_points_many(excel_paths):
    points_list = []
    excel_rows_list = []
    source_list = []
    for excel_path in excel_paths:
        points, excel_rows = load_real_points(excel_path)
        points_list.append(points)
        excel_rows_list.append(excel_rows)
        source_list.extend([excel_path.name] * len(points))
    return (
        np.vstack(points_list),
        np.concatenate(excel_rows_list),
        np.asarray(source_list, dtype=object),
    )


def normalize_points(points, target_lower, target_upper):
    data_lower = np.min(points, axis=0)
    data_upper = np.max(points, axis=0)
    data_span = data_upper - data_lower
    if np.any(data_span <= 0.0):
        raise ValueError("Both dimensions must vary before normalization.")
    target_lower = np.asarray(target_lower, dtype=float)
    target_upper = np.asarray(target_upper, dtype=float)
    normalized = target_lower + (
        (points - data_lower) / data_span
    ) * (target_upper - target_lower)
    return normalized, data_lower, data_upper


def filter_outliers_iqr(points, factor):
    points = np.asarray(points, dtype=float)
    q1 = np.percentile(points, 25.0, axis=0)
    q3 = np.percentile(points, 75.0, axis=0)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    flat_dimensions = iqr <= 1e-12
    lower[flat_dimensions] = -np.inf
    upper[flat_dimensions] = np.inf

    inlier_mask = np.all((points >= lower) & (points <= upper), axis=1)
    filtered_points = points[inlier_mask]
    if len(filtered_points) == 0:
        raise ValueError("The outlier filter removed every real data point.")
    return filtered_points, inlier_mask, lower, upper


def point_to_closed_curve_distances(points, curve_points):
    closed_curve = np.vstack([curve_points, curve_points[0]])
    segment_starts = closed_curve[:-1]
    segment_vectors = closed_curve[1:] - closed_curve[:-1]
    segment_norm_sq = np.sum(segment_vectors * segment_vectors, axis=1)
    segment_norm_sq = np.maximum(segment_norm_sq, 1e-12)

    relative_vectors = points[:, None, :] - segment_starts[None, :, :]
    projection_ratio = np.sum(
        relative_vectors * segment_vectors[None, :, :],
        axis=2,
    ) / segment_norm_sq[None, :]
    projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
    projected_points = (
        segment_starts[None, :, :]
        + projection_ratio[:, :, None] * segment_vectors[None, :, :]
    )
    distances = np.linalg.norm(points[:, None, :] - projected_points, axis=2)
    return np.min(distances, axis=1)


def energy_distance(x, y):
    """
    Energy distance statistic between two 2D samples.
    Larger values indicate stronger distributional mismatch.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = x.shape[0], y.shape[0]
    d_xy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2).sum()
    d_xx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2).sum()
    d_yy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2).sum()
    return (2.0 / (n * m)) * d_xy - (1.0 / (n * n)) * d_xx - (1.0 / (m * m)) * d_yy


def normalize_for_energy_test(points, ref_points):
    points = np.asarray(points, dtype=float)
    ref_points = np.asarray(ref_points, dtype=float)
    lower = np.min(ref_points, axis=0)
    upper = np.max(ref_points, axis=0)
    span = upper - lower
    span[span <= 1e-12] = 1.0
    return (points - lower) / span


def sample_ddga_mixture(means, covariances, weights, sample_size, rng, bounds=None, max_attempts=200):
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    if bounds is None:
        component_ids = rng.choice(len(weights), size=sample_size, p=weights)
        samples = np.empty((sample_size, means.shape[1]), dtype=float)
        for component_id in np.unique(component_ids):
            mask = component_ids == component_id
            samples[mask] = rng.multivariate_normal(
                means[component_id],
                covariances[component_id],
                size=int(np.sum(mask)),
            )
        return samples

    bounds = np.asarray(bounds, dtype=float)
    samples = []
    remaining = sample_size
    attempts = 0
    while remaining > 0 and attempts < max_attempts:
        attempts += 1
        batch = sample_ddga_mixture(
            means,
            covariances,
            weights,
            max(remaining * 2, 50),
            rng,
            bounds=None,
        )
        in_bounds = np.all((batch >= bounds[:, 0]) & (batch <= bounds[:, 1]), axis=1)
        if np.any(in_bounds):
            accepted = batch[in_bounds][:remaining]
            samples.append(accepted)
            remaining -= len(accepted)

    if remaining > 0:
        fallback = sample_ddga_mixture(
            means,
            covariances,
            weights,
            remaining,
            rng,
            bounds=None,
        )
        fallback = np.clip(fallback, bounds[:, 0], bounds[:, 1])
        samples.append(fallback)

    return np.vstack(samples)


def ddga_energy_distance_test(
    real_points,
    means,
    covariances,
    weights,
    ref_sample_size,
    num_bootstrap,
    bounds=None,
    normalize=True,
    rng_seed=12345,
):
    """
    Right-tailed Monte Carlo test for H0: real_points are compatible with
    the DDGA Gaussian mixture distribution.
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(real_points, dtype=float)
    y_ref = sample_ddga_mixture(means, covariances, weights, ref_sample_size, rng, bounds=bounds)
    if normalize:
        ref_points = x
        x_test = normalize_for_energy_test(x, ref_points)
        y_ref_test = normalize_for_energy_test(y_ref, ref_points)
    else:
        x_test = x
        y_ref_test = y_ref

    observed_stat = energy_distance(x_test, y_ref_test)
    null_stats = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        xb = sample_ddga_mixture(means, covariances, weights, len(x), rng, bounds=bounds)
        yb = sample_ddga_mixture(means, covariances, weights, ref_sample_size, rng, bounds=bounds)
        if normalize:
            xb = normalize_for_energy_test(xb, ref_points)
            yb = normalize_for_energy_test(yb, ref_points)
        null_stats[i] = energy_distance(xb, yb)

    p_value = (np.sum(null_stats >= observed_stat) + 1.0) / (num_bootstrap + 1.0)
    return observed_stat, p_value, null_stats


def ddga_energy_distance_similarity_test(
    real_points,
    means,
    covariances,
    weights,
    ref_sample_size,
    num_bootstrap,
    baseline_samples=200,
    alpha=0.05,
    gamma=0.1,
    delta_c=0.1,
    bounds=None,
    normalize=True,
    rng_seed=12345,
):
    """
    Similarity test:
      H0: EnergyDistance(real_points, DDGA) >= A + delta  (not close enough)
      H1: EnergyDistance(real_points, DDGA) <  A + delta  (close enough)

    Rejecting H0 supports that real_points are close enough to the DDGA mixture
    under the chosen similarity threshold.
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(real_points, dtype=float)
    ref_points = x
    if normalize:
        x_test = normalize_for_energy_test(x, ref_points)
    else:
        x_test = x

    y_ref = sample_ddga_mixture(means, covariances, weights, ref_sample_size, rng, bounds=bounds)
    y_ref_test = normalize_for_energy_test(y_ref, ref_points) if normalize else y_ref
    observed_stat = energy_distance(x_test, y_ref_test)

    baseline_stats = np.empty(baseline_samples, dtype=float)
    for i in range(baseline_samples):
        x0 = sample_ddga_mixture(means, covariances, weights, len(x), rng, bounds=bounds)
        y0 = sample_ddga_mixture(means, covariances, weights, ref_sample_size, rng, bounds=bounds)
        if normalize:
            x0 = normalize_for_energy_test(x0, ref_points)
            y0 = normalize_for_energy_test(y0, ref_points)
        baseline_stats[i] = energy_distance(x0, y0)

    baseline_quantile = float(np.quantile(baseline_stats, 1.0 - gamma))
    delta = float(delta_c * baseline_quantile)
    similarity_threshold = baseline_quantile + delta

    boot_stats = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        yb = sample_ddga_mixture(means, covariances, weights, ref_sample_size, rng, bounds=bounds)
        yb_test = normalize_for_energy_test(yb, ref_points) if normalize else yb
        boot_stats[i] = energy_distance(x_test, yb_test)

    upper_ci = float(np.quantile(boot_stats, 1.0 - alpha))
    reject_h0 = upper_ci < similarity_threshold
    return observed_stat, upper_ci, baseline_quantile, delta, similarity_threshold, reject_h0


def gaussian_mixture_logpdf(points, means, covariances, weights):
    points = np.asarray(points, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    component_log_terms = []
    for mean, covariance, weight in zip(means, covariances, weights):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("DDGA covariance must be positive definite for log-probability testing.")
        diff = points - mean
        solved = np.linalg.solve(covariance, diff.T).T
        maha = np.sum(diff * solved, axis=1)
        log_norm = -0.5 * (points.shape[1] * np.log(2.0 * np.pi) + logdet)
        component_log_terms.append(np.log(weight) + log_norm - 0.5 * maha)

    stacked = np.vstack(component_log_terms).T
    max_log = np.max(stacked, axis=1, keepdims=True)
    return (max_log[:, 0] + np.log(np.sum(np.exp(stacked - max_log), axis=1)))


def ddga_logprob_bad_rate_test(
    real_points,
    means,
    covariances,
    weights,
    ref_sample_size=500,
    bad_quantile=0.1,
    extra_delta=0.1,
    bounds=None,
    rng_seed=54321,
):
    rng = np.random.default_rng(rng_seed)
    reference = sample_ddga_mixture(means, covariances, weights, ref_sample_size, rng, bounds=bounds)
    reference_logp = gaussian_mixture_logpdf(reference, means, covariances, weights)
    real_logp = gaussian_mixture_logpdf(real_points, means, covariances, weights)
    cutoff = float(np.quantile(reference_logp, bad_quantile))
    bad_rate = float(np.mean(real_logp < cutoff))
    threshold = float(bad_quantile + extra_delta)
    compatible = bad_rate <= threshold
    return bad_rate, cutoff, bad_quantile, threshold, compatible


def print_ddga_energy_distance_test(real_points, means, covariances, weights, config, bounds=None):
    ref_sample_size = int(config["ref_sample_size"])
    bootstraps = int(config["bootstraps"])
    alpha = float(config.get("alpha", 0.05))
    test_mode = str(config.get("test_mode", "difference"))
    bad_quantile = float(config["bad_quantile"])
    bad_delta = float(config["bad_delta"])
    if test_mode == "similarity":
        stat, upper_ci, baseline_q, delta, threshold, reject_h0 = ddga_energy_distance_similarity_test(
            real_points,
            means,
            covariances,
            weights,
            ref_sample_size=ref_sample_size,
            num_bootstrap=bootstraps,
            baseline_samples=int(config.get("similarity_baseline_samples", 200)),
            alpha=alpha,
            gamma=float(config.get("similarity_gamma", 0.1)),
            delta_c=float(config.get("similarity_delta_c", 0.1)),
            bounds=bounds,
        )
        verdict = "REJECT H0 (close enough)" if reject_h0 else "FAIL TO REJECT H0 (not close enough)"
        print(
            f"[EnergyDistance][DDGA][similarity] T_obs={stat:.4f}, "
            f"U_1-a={upper_ci:.4f}, A={baseline_q:.4f}, delta={delta:.4f}, "
            f"threshold={threshold:.4f}, alpha={alpha:.3f} => {verdict} "
            "(H0: real points are not close enough to DDGA mixture)"
        )
    else:
        stat, p_value, _ = ddga_energy_distance_test(
            real_points,
            means,
            covariances,
            weights,
            ref_sample_size=ref_sample_size,
            num_bootstrap=bootstraps,
            bounds=bounds,
        )
        reject_h0 = p_value < alpha
        verdict = "REJECT H0" if reject_h0 else "FAIL TO REJECT H0"
        print(
            f"[EnergyDistance][DDGA] T_ed={stat:.4f}, right-tailed p={p_value:.4f} "
            f"alpha={alpha:.3f} => {verdict} "
            "(H0: normalized real points ~ DDGA Gaussian mixture)"
        )

    bad_rate, cutoff, bad_quantile, threshold, compatible = ddga_logprob_bad_rate_test(
        real_points,
        means,
        covariances,
        weights,
        ref_sample_size=ref_sample_size,
        bad_quantile=bad_quantile,
        extra_delta=bad_delta,
        bounds=bounds,
    )
    verdict = "COMPATIBLE" if compatible else "TOO MANY BAD POINTS"
    print(
        f"[LogProbBadRate][DDGA] R={bad_rate:.4f}, cq={cutoff:.4f}, "
        f"r0={bad_quantile:.4f}, delta={bad_delta:.4f}, "
        f"threshold={threshold:.4f} => {verdict}"
    )


def gaussian_land_dim2(V, Sigma, cycle, phi, range_1, range_2, num):
    """
    V: n*2 projection matrix
    Sigma: n*n*(time steps) covariance matrix set
    cycle: (time steps)*n limit cycle time series
    phi: (time steps,) pre-solution
    range_1/range_2: [min, max]
    num: grid resolution
    """

    def gauss(A, x, y):
        return A[0, 0] * x**2 + A[1, 1] * y**2 + 2 * A[0, 1] * x * y

    sigma0_proj = np.zeros((2, 2, len(phi)), dtype=float)
    mu_proj = np.zeros((len(phi), 2), dtype=float)

    for i in range(len(phi)):
        mu_proj[i, :] = (V.T @ cycle[i, :].reshape(-1, 1)).ravel()
        sigma0_proj[:, :, i] = V.T @ Sigma[:, :, i] @ V

    mesh_1, mesh_2 = np.meshgrid(
        np.linspace(range_1[0], range_1[1], num),
        np.linspace(range_2[0], range_2[1], num),
    )

    P_DDGA = np.zeros((num, num), dtype=float)

    for k in range(len(phi)):
        sig = sigma0_proj[:, :, k]
        inv_cov = np.linalg.inv(sig)

        cons1 = 1.0 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(sig))
        cons2 = np.exp(-0.5)

        Z = cons1 * (cons2 ** gauss(
            inv_cov,
            mesh_1 - mu_proj[k, 0],
            mesh_2 - mu_proj[k, 1],
        ))
        P_DDGA = P_DDGA + Z * phi[k]

    return P_DDGA, mesh_1, mesh_2


def coexistence_point(r_n, K, a, h, d, c):
    N_star = d * h / (c - d)
    P_star = (r_n / a) * (h + N_star) * (1.0 - N_star / K)
    return N_star, P_star


def drift_f(_, x, r_n, K, a, h, d, c):
    N, P = x
    response = N / (h + N)
    return np.array([
        r_n * N * (1.0 - N / K)
        - a * response * P,
        (-d + c * response) * P,
    ], dtype=float)


def jacobian_f(x, r_n, K, a, h, d, c):
    N, P = x
    response = N / (h + N)
    response_prime = h / (h + N) ** 2
    return np.array([
        [
            r_n * (1.0 - 2.0 * N / K)
            - a * P * response_prime,
            -a * response,
        ],
        [
            c * P * response_prime,
            -d + c * response,
        ],
    ], dtype=float)


def main():
    # ============================================================
    # Parameter Setting
    # ============================================================
    config = load_config()
    dynamics_config = config["dynamics"]
    real_data_config = config["real_data"]
    covariance_config = config["covariance"]
    landscape_config = config["landscape"]
    energy_test_config = config["energy_test"]

    dim = 2
    D = float(dynamics_config["D"])
    d0 = D
    covariance_eigenvalue_floor = float(covariance_config["eigenvalue_floor"])
    covariance_correction_tolerance = float(covariance_config["correction_tolerance"])

    growth_rate = float(dynamics_config["growth_rate"])
    r_n = float(growth_rate)
    K = float(dynamics_config["K"])
    a = float(dynamics_config["a"])
    h = float(dynamics_config["h"])
    d = float(dynamics_config["d"])
    c = float(dynamics_config["c"])
    initial_offset = float(dynamics_config["initial_offset"])

    dt = float(dynamics_config["dt"])
    steps = int(dynamics_config["steps"])
    time = dt * np.arange(1, steps + 1)

    # ============================================================
    # Find the limit cycle
    # ============================================================
    N_star, P_star = coexistence_point(r_n, K, a, h, d, c)
    x_init = np.array([N_star + initial_offset, P_star], dtype=float)

    sol = solve_ivp(
        fun=lambda t, x: drift_f(t, x, r_n, K, a, h, d, c),
        t_span=(time[0], time[-1]),
        y0=x_init,
        t_eval=time,
        method="RK45",
    )
    path = sol.y.T

    Force_origin = np.zeros((steps, dim), dtype=float)
    for i in range(steps):
        Force_origin[i, :] = drift_f(0.0, path[i, :], r_n, K, a, h, d, c)

    cen_path = path - path[-1, :]
    dis_path = np.linalg.norm(cen_path, axis=1)

    thres_force = np.max(np.linalg.norm(Force_origin, axis=1))

    start_idx = int(0.3 * steps)
    near_points = np.where(dis_path[start_idx:] < 3 * thres_force * dt)[0]
    period_time = np.zeros(max(len(near_points) - 1, 0), dtype=int)

    for i in range(len(near_points) - 1):
        if near_points[i + 1] - near_points[i] != 1:
            period_time[i] = near_points[i]

    period_time = period_time[period_time != 0]
    if len(period_time) < 2:
        raise RuntimeError("Failed to detect the period. Try increasing steps or changing parameters.")

    Period = np.mean(np.diff(period_time)) * dt
    print(f"Estimated period = {Period:.6f}")

    t_cycle = np.arange(0.0, Period + dt, dt)
    sol_cycle = solve_ivp(
        fun=lambda t, x: drift_f(t, x, r_n, K, a, h, d, c),
        t_span=(t_cycle[0], t_cycle[-1]),
        y0=path[-1, :],
        t_eval=t_cycle,
        method="RK45",
    )

    Limit_cycle = sol_cycle.y.T
    len_LC = len(Limit_cycle)

    # Min-max normalize real points first, then filter outliers in normalized coordinates.
    excel_paths = resolve_real_data_paths(real_data_config)
    loaded_real_points, loaded_excel_rows, loaded_sources = load_real_points_many(excel_paths)
    normalized_real_points, real_min, real_max = normalize_points(
        loaded_real_points,
        target_lower=real_data_config["target_lower"],
        target_upper=real_data_config["target_upper"],
    )
    real_n_scale = float(real_data_config["real_n_scale"])
    outlier_iqr_factor = float(real_data_config["outlier_iqr_factor"])
    normalized_real_points[:, 0] *= real_n_scale
    normalized_real_points, inlier_mask, outlier_lower, outlier_upper = filter_outliers_iqr(
        normalized_real_points,
        factor=outlier_iqr_factor,
    )
    real_points = loaded_real_points[inlier_mask]
    removed_indices = np.flatnonzero(~inlier_mask)
    removed_excel_rows = loaded_excel_rows[removed_indices]
    removed_points = loaded_real_points[removed_indices]
    removed_sources = loaded_sources[removed_indices]
    real_point_distances = point_to_closed_curve_distances(
        normalized_real_points,
        Limit_cycle,
    )
    distance_sum = float(np.sum(real_point_distances))
    print(
        f"Loaded {len(loaded_real_points)} real points from "
        f"{len(excel_paths)} file(s): {', '.join(path.name for path in excel_paths)}"
    )
    print(
        "Outlier filter after normalization: "
        f"method=iqr, factor={outlier_iqr_factor:.10g}, "
        f"kept={len(real_points)}, removed={len(removed_indices)}"
    )
    print(
        "IQR inlier bounds after normalization (N, P): "
        f"lower=({outlier_lower[0]:.10g}, {outlier_lower[1]:.10g}), "
        f"upper=({outlier_upper[0]:.10g}, {outlier_upper[1]:.10g})"
    )
    if len(removed_indices) > 0:
        print("Removed outlier points:")
        for index, source, excel_row, point in zip(
            removed_indices,
            removed_sources,
            removed_excel_rows,
            removed_points,
        ):
            print(
                f"  data_index={int(index)}, file={source}, excel_row={int(excel_row)}, "
                f"prey={point[0]:.10g}, predator={point[1]:.10g}"
            )
    else:
        print("Removed outlier points: none")
    print(
        "Real-data minimum (prey, predator) = "
        f"({real_min[0]:.10g}, {real_min[1]:.10g})"
    )
    print(
        "Real-data maximum (prey, predator) = "
        f"({real_max[0]:.10g}, {real_max[1]:.10g})"
    )
    print(f"Normalized real-point range = {np.min(normalized_real_points, axis=0)} to {np.max(normalized_real_points, axis=0)}")
    print(f"Applied real-data N scale factor = {real_n_scale:.10g}")
    print(f"Sum of normalized Euclidean distances = {distance_sum:.10g}")

    Force_LC = np.zeros((len_LC, dim), dtype=float)
    Jacobian_LC = np.zeros((len_LC, dim, dim), dtype=float)
    for i in range(len_LC):
        Force_LC[i, :] = drift_f(0.0, Limit_cycle[i, :], r_n, K, a, h, d, c)
        Jacobian_LC[i, :, :] = jacobian_f(Limit_cycle[i, :], r_n, K, a, h, d, c)

    # ============================================================
    # Pre-solution
    # ============================================================
    gs = np.linalg.norm(Force_LC, axis=1)
    int_gs2 = np.cumsum(gs * gs / len_LC * (len_LC * dt))
    int_exp = np.exp(-int_gs2 / D)
    int_whole = np.cumsum(gs * int_exp / D / len_LC * (len_LC * dt))

    C0 = (1 - int_exp[-1]) / int_whole[-1]
    pre_solution = (1.0 / int_exp) * (1 - C0 * int_whole)
    pre_solution = pre_solution / np.sum(pre_solution)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(
        t_cycle,
        pre_solution,
        linewidth=2.5,
        color=(160 / 255, 201 / 255, 235 / 255),
        label="Pre-Solution",
    )
    ax1.set_xlim([0, Period])
    ax1.set_facecolor((1, 1, 1))
    ax1.tick_params(labelsize=20)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    ax1.set_xlabel("t", fontname="Arial", fontsize=24)
    ax1.set_ylabel("Pre-solution", fontname="Arial", fontsize=24)
    lgd = ax1.legend(
        prop={"family": "Arial", "size": 18},
        loc="upper right",
        frameon=True,
    )
    lgd.get_frame().set_facecolor((240 / 255, 240 / 255, 240 / 255))
    fig1.patch.set_facecolor((1, 1, 1))
    ax1.grid(True)

    # ============================================================
    # Covariance
    # ============================================================
    Sigma_all = np.zeros((len_LC, dim, dim), dtype=float)
    Q_last_step = np.zeros((dim, dim), dtype=float)
    d0_values = np.full(len_LC, d0, dtype=float)
    d1_corrections = np.zeros(len_LC, dtype=float)
    min_eigenvalues_after_rank1 = np.zeros(len_LC, dtype=float)
    covariance_floor_by_point = np.full(len_LC, covariance_eigenvalue_floor, dtype=float)
    low_landscape_config = covariance_config["low_landscape"]
    low_landscape_floor_factor = float(low_landscape_config["floor_factor"])
    low_landscape_region_n_min = float(low_landscape_config["n_min"])
    low_landscape_region_n_max = float(low_landscape_config["n_max"])
    low_landscape_region_p_min = float(low_landscape_config["p_min"])
    low_landscape_region_p_max = float(low_landscape_config["p_max"])
    lower_right_floor_mask = (
        (Limit_cycle[:, 0] >= low_landscape_region_n_min)
        & (Limit_cycle[:, 0] <= low_landscape_region_n_max)
        & (Limit_cycle[:, 1] >= low_landscape_region_p_min)
        & (Limit_cycle[:, 1] <= low_landscape_region_p_max)
    )
    covariance_floor_by_point[lower_right_floor_mask] = (
        covariance_eigenvalue_floor * low_landscape_floor_factor
    )
    lower_left_config = covariance_config["lower_left"]
    lower_left_floor_factor = float(lower_left_config["floor_factor"])
    lower_left_region_n_min = float(lower_left_config["n_min"])
    lower_left_region_n_max = float(lower_left_config["n_max"])
    lower_left_region_p_min = float(lower_left_config["p_min"])
    lower_left_region_p_max = float(lower_left_config["p_max"])
    lower_left_floor_mask = (
        (Limit_cycle[:, 0] <= lower_left_region_n_max)
        & (Limit_cycle[:, 0] >= lower_left_region_n_min)
        & (Limit_cycle[:, 1] >= lower_left_region_p_min)
        & (Limit_cycle[:, 1] <= lower_left_region_p_max)
    )
    covariance_floor_by_point[lower_left_floor_mask] = (
        covariance_eigenvalue_floor * lower_left_floor_factor
    )
    regularized_covariance_count = 0
    max_d1 = 0.0

    for i in range(len_LC):
        tan_vec = Force_LC[i, :].reshape(-1, 1) / np.linalg.norm(Force_LC[i, :], 2)
        Q = np.hstack([tan_vec, np.vstack([np.zeros((1, dim - 1)), np.eye(dim - 1)])])
        Q_this_step, _ = qr(Q, mode="economic")

        if i > 1:
            direction = np.sign(np.sum(Q_this_step[:, 1:] * Q_this_step[:, 1:], axis=0))
            direction[direction == 0] = 1
            Q_this_step[:, 1:] = Q_this_step[:, 1:] * direction

        Q_last_step = Q_this_step.copy()

        Jac_normal = Q_this_step[:, 1:].T @ Jacobian_LC[i, :, :] @ Q_this_step[:, 1:]

        # MATLAB lyap(A, Q): A X + X A^T + Q = 0.
        # scipy solve_continuous_lyapunov(A, Q) solves A X + X A^T = Q.
        Sigma_normal = solve_continuous_lyapunov(
            Jac_normal,
            -2 * D * np.eye(dim - 1),
        )

        # Complete covariance before pointwise regularization:
        # Sigma = Q Sigma_normal Q^T + d0 v1 v1^T + d1 I.
        # The rank-1 term supplies tangent variance at every point. The d1 I
        # term is added below only for points that are still not invertible.
        Sigma_normal_projected = (
            Q_this_step[:, 1:] @ Sigma_normal @ Q_this_step[:, 1:].T
        )
        Sigma_rank1_correction = d0_values[i] * (tan_vec @ tan_vec.T)
        Sigma = Sigma_normal_projected + Sigma_rank1_correction
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Fall back to the second correction only if rank-1 correction is
        # insufficient. The added d1 is the minimum needed for inversion.
        min_eigenvalue = np.min(np.linalg.eigvalsh(Sigma))
        min_eigenvalues_after_rank1[i] = min_eigenvalue
        d1 = max(0.0, covariance_floor_by_point[i] - min_eigenvalue)
        if d1 > covariance_correction_tolerance:
            Sigma_identity_correction = d1 * np.eye(dim)
            Sigma = Sigma + Sigma_identity_correction
            d1_corrections[i] = d1
            regularized_covariance_count += 1
            max_d1 = max(max_d1, d1)

        Sigma_all[i, :, :] = Sigma

    print(
        "Covariance regularization: "
        f"{regularized_covariance_count}/{len_LC} matrices required d1 * I; "
        f"max d1 = {max_d1:.6e}"
    )
    # print(
    #     "Low-landscape covariance floor factor: "
    #     f"{low_landscape_floor_factor:.3g} for {np.count_nonzero(lower_right_floor_mask)} "
    #     f"points where N >= {low_landscape_region_n_min:.3g} and "
    #     f"{low_landscape_region_p_min:.3g} <= P <= {low_landscape_region_p_max:.3g}"
    # )
    # print(
    #     "Lower-left covariance floor factor: "
    #     f"{lower_left_floor_factor:.3g} for {np.count_nonzero(lower_left_floor_mask)} "
    #     f"points where N <= {lower_left_region_n_max:.3g} and "
    #     f"P <= {lower_left_region_p_max:.3g}"
    # )
    corrected_idx = np.where(d1_corrections > covariance_correction_tolerance)[0]
    print(f"Corrected covariance points: {len(corrected_idx)}")
    for idx in corrected_idx:
        print(
            f"  idx={idx:4d}, "
            f"N={Limit_cycle[idx, 0]:.8f}, "
            f"P={Limit_cycle[idx, 1]:.8f}, "
            f"floor={covariance_floor_by_point[idx]:.6e}, "
            f"min_eig_before_d1={min_eigenvalues_after_rank1[idx]:.6e}, "
            f"d1={d1_corrections[idx]:.6e}"
        )

    # ============================================================
    # Landscape from DDGA
    # ============================================================
    range_1 = [float(value) for value in landscape_config["range_1"]]
    range_2 = [float(value) for value in landscape_config["range_2"]]
    grid_num = int(landscape_config["grid_num"])
    P_DDGA, mesh_1, mesh_2 = gaussian_land_dim2(
        np.eye(2),
        np.transpose(Sigma_all, (1, 2, 0)),
        Limit_cycle,
        pre_solution,
        range_1,
        range_2,
        grid_num,
    )
    P_DDGA_safe = np.maximum(P_DDGA, 1e-12)
    U_DDGA = -D * np.log(P_DDGA_safe)
    U_DDGA = U_DDGA - np.nanmin(U_DDGA)
    sample_bounds_config = energy_test_config["sample_bounds"]
    ddga_sample_bounds = np.array([
        [float(sample_bounds_config["x_min"]), float(sample_bounds_config["x_max"])],
        [float(sample_bounds_config["y_min"]), float(sample_bounds_config["y_max"])],
    ], dtype=float)
    print_ddga_energy_distance_test(
        normalized_real_points,
        Limit_cycle,
        Sigma_all,
        pre_solution,
        config=energy_test_config,
        bounds=ddga_sample_bounds,
    )
    probability_interpolator = RegularGridInterpolator(
        (mesh_2[:, 0], mesh_1[0, :]),
        P_DDGA,
        bounds_error=False,
        fill_value=np.nan,
    )
    energy_interpolator = RegularGridInterpolator(
        (mesh_2[:, 0], mesh_1[0, :]),
        U_DDGA,
        bounds_error=False,
        fill_value=np.nan,
    )
    display_real_points = normalized_real_points.copy()
    display_real_point_offset = np.asarray(
        landscape_config["display_real_point_offset"],
        dtype=float,
    )
    display_real_points[:, 0] += display_real_point_offset[0]
    display_real_points[:, 1] += display_real_point_offset[1]
    interpolation_points = np.column_stack([
        display_real_points[:, 1],
        display_real_points[:, 0],
    ])
    real_probability = probability_interpolator(interpolation_points)
    real_energy = energy_interpolator(interpolation_points)
    probability_mask = np.isfinite(real_probability)
    energy_mask = np.isfinite(real_energy)
    probability_z_offset = 0.04 * float(np.ptp(P_DDGA))
    real_point_style = landscape_config.get("real_point_style", {})
    real_point_color = real_point_style.get("color", "white")
    real_point_edgecolors = real_point_style.get("edgecolors", "black")
    real_point_linewidths = float(real_point_style.get("linewidths", 0.55))
    real_point_size = float(real_point_style.get("size", 22))
    real_point_alpha = float(real_point_style.get("alpha", 0.85))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(
        mesh_1,
        mesh_2,
        P_DDGA,
        cmap="turbo",
        linewidth=0,
        edgecolor="none",
        antialiased=True,
    )
    ax2.scatter(
        display_real_points[probability_mask, 0],
        display_real_points[probability_mask, 1],
        real_probability[probability_mask] + probability_z_offset,
        color=real_point_color,
        edgecolors=real_point_edgecolors,
        linewidths=real_point_linewidths,
        s=real_point_size,
        alpha=real_point_alpha,
        depthshade=False,
        label="Normalized real points",
    )
    ax2.set_xlim(range_1)
    ax2.set_ylim(range_2)
    ax2.view_init(elev=39, azim=-33)
    ax2.tick_params(labelsize=20)
    ax2.set_box_aspect((1, 1, 0.8))

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    surf3 = ax3.pcolormesh(
        mesh_1,
        mesh_2,
        U_DDGA,
        cmap="turbo",
        shading="auto",
    )
    ax3.contour(
        mesh_1,
        mesh_2,
        U_DDGA,
        40,
        cmap="turbo",
        linewidths=0.45,
        alpha=0.75,
    )
    ax3.plot(
        Limit_cycle[:, 0],
        Limit_cycle[:, 1],
        color="black",
        linewidth=1.8,
        label="Limit cycle",
    )
    ax3.scatter(
        [N_star],
        [P_star],
        color="black",
        marker="x",
        s=60,
        label="Coexistence point",
    )
    correction_scatter = None
    if len(corrected_idx) > 0:
        correction_scatter = ax3.scatter(
            Limit_cycle[corrected_idx, 0],
            Limit_cycle[corrected_idx, 1],
            c=d1_corrections[corrected_idx],
            cmap="Reds",
            marker="o",
            s=35,
            edgecolors="red",
            linewidths=0.35,
            label="Covariance corrected points",
        )
    ax3.scatter(
        display_real_points[:, 0],
        display_real_points[:, 1],
        color=real_point_color,
        edgecolors=real_point_edgecolors,
        linewidths=real_point_linewidths,
        s=real_point_size,
        alpha=real_point_alpha,
        label="Normalized real points",
    )
    ax3.set_xlim(range_1)
    ax3.set_ylim(range_2)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("N", fontname="Arial", fontsize=24)
    ax3.set_ylabel("P", fontname="Arial", fontsize=24)
    ax3.set_title("DDGA energy landscape", fontname="Arial", fontsize=20)
    ax3.tick_params(labelsize=20)
    ax3.legend(
        prop={"family": "Arial", "size": 8},
        loc="upper right",
        markerscale=0.6,
        handlelength=1.1,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    cbar3 = fig3.colorbar(surf3, ax=ax3, pad=0.02)
    cbar3.set_label("U = -D log P_DDGA", fontname="Arial", fontsize=18)
    if correction_scatter is not None:
        cbar3_corr = fig3.colorbar(correction_scatter, ax=ax3, pad=0.12)
        cbar3_corr.set_label("d1 correction", fontname="Arial", fontsize=18)
    fig3.tight_layout()

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection="3d")
    ax4.plot_surface(
        mesh_1,
        mesh_2,
        U_DDGA,
        cmap="turbo",
        linewidth=0,
        edgecolor="none",
        antialiased=True,
    )
    ax4.scatter(
        display_real_points[energy_mask, 0],
        display_real_points[energy_mask, 1],
        real_energy[energy_mask] + 0.2,
        color=real_point_color,
        edgecolors=real_point_edgecolors,
        linewidths=real_point_linewidths,
        s=real_point_size,
        alpha=real_point_alpha,
        depthshade=False,
        label="Normalized real points",
    )
    ax4.set_xlim(range_1)
    ax4.set_ylim(range_2)
    ax4.set_xlabel("N", fontname="Arial", fontsize=18)
    ax4.set_ylabel("P", fontname="Arial", fontsize=18)
    ax4.set_zlabel("U = -D log P_DDGA", fontname="Arial", fontsize=18)
    ax4.set_title("DDGA energy landscape", fontname="Arial", fontsize=20)
    ax4.view_init(elev=39, azim=-33)
    ax4.tick_params(labelsize=16)
    ax4.set_box_aspect((1, 1, 0.8))

    output_dir = Path(__file__).resolve().parents[1] / "results" / "predatorprey"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = excel_paths[0].stem if len(excel_paths) == 1 else f"combined_{len(excel_paths)}files"
    distance_text = f"{distance_sum:.8f}"
    figure_paths = {
        "figure1": output_dir / f"{input_name}_{distance_text}_figure1_pre-solution.png",
        "figure2": output_dir / f"{input_name}_{distance_text}_figure2_3D-probability-density.png",
        "figure3": output_dir / f"{input_name}_{distance_text}_figure3_2D-landscape.png",
        "figure4": output_dir / f"{input_name}_{distance_text}_figure4_3D-landscape.png",
    }
    fig1.savefig(figure_paths["figure1"], dpi=300, bbox_inches="tight")
    fig2.savefig(figure_paths["figure2"], dpi=300, bbox_inches="tight")
    fig3.savefig(figure_paths["figure3"], dpi=300, bbox_inches="tight")
    fig4.savefig(figure_paths["figure4"], dpi=300, bbox_inches="tight")
    for figure_name, figure_path in figure_paths.items():
        print(f"Saved {figure_name}: {figure_path}")

    plt.show()


if __name__ == "__main__":
    main()
