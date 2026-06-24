from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import qr, solve_continuous_lyapunov


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETER_CSV = PROJECT_ROOT / "results" / "predatorprey" / "parameter_candidates.csv"
DEFAULT_BOUNDS_CSV = PROJECT_ROOT / "results" / "predatorprey" / "energy_bounds_grid.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "results" / "predatorprey" / "config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "normalized_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "predatorprey" / "energy_cluster_search"


@dataclass(frozen=True)
class Candidate:
    group: str
    K: float
    a: float
    h: float


def candidate_cache_signature(candidate: Candidate) -> str:
    return f"K={candidate.K:.12g};a={candidate.a:.12g};h={candidate.h:.12g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster predator-prey groups by pairwise parameter-to-data loss, "
            "merge each cluster's data, choose representative parameters by "
            "EnergyDistance test, and iterate reassignment."
        )
    )
    parser.add_argument("--parameter-csv", type=Path, default=DEFAULT_PARAMETER_CSV)
    parser.add_argument(
        "--bounds-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV of bounds to evaluate. If omitted, uses "
            "energy_test.bounds_source from config.yaml. Use bounds_source=real_data "
            "to derive one bounds row from the current source/cluster data."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 4])
    parser.add_argument(
        "--test-mode",
        choices=["similarity", "difference", "bad_rate"],
        default=None,
        help=(
            "Decision rule for this run. If omitted, uses energy_test.test_mode from "
            "config.yaml. 'similarity' uses an EnergyDistance upper bound; "
            "'difference' requires p-value >= alpha; 'bad_rate' requires the "
            "low-log-probability real-point rate to be at most q + bad_delta."
        ),
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--bad-quantile",
        type=float,
        default=None,
        help="Override energy_test.bad_quantile for --test-mode bad_rate.",
    )
    parser.add_argument(
        "--bad-delta",
        type=float,
        default=None,
        help="Override energy_test.bad_delta for --test-mode bad_rate.",
    )
    parser.add_argument("--ref-sample-size", type=int, default=None)
    parser.add_argument("--bootstraps", type=int, default=None)
    parser.add_argument("--similarity-baseline-samples", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--pairwise-cache",
        type=Path,
        default=None,
        help="CSV cache for pairwise loss rows. Defaults to output-dir/pairwise_loss.csv.",
    )
    parser.add_argument(
        "--merged-cache",
        type=Path,
        default=None,
        help="CSV cache for merged-cluster candidate rows. Defaults to output-dir/merged_cluster_cache.csv.",
    )
    parser.add_argument(
        "--no-reuse-pairwise-loss",
        action="store_true",
        help="Ignore any existing pairwise loss cache and recompute all loss[i,j] entries.",
    )
    parser.add_argument(
        "--no-reuse-merged-cluster",
        action="store_true",
        help="Ignore any existing merged-cluster cache and recompute merged candidate tests.",
    )
    parser.add_argument(
        "--pairwise-loss-mode",
        choices=["energy", "full"],
        default="energy",
        help=(
            "How to compute loss[i,j] for k-medoids initialization. "
            "'energy' uses observed T_ed only; 'full' uses the configured hypothesis test."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum cluster representative/reassignment iterations after k-medoids initialization.",
    )
    parser.add_argument(
        "--max-initializations",
        type=int,
        default=25,
        help="Maximum k-medoids initial groupings to try for each k before declaring no feasible solution.",
    )
    parser.add_argument(
        "--max-bounds",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of bounds rows to evaluate.",
    )
    return parser.parse_args()


def load_ddga_module():
    path = PROJECT_ROOT / "epr" / "525_dynamics_DDGA.py"
    spec = importlib.util.spec_from_file_location("ddga_525", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_to_normalized_path(group: str, data_dir: Path) -> Path:
    file_stem = group.replace("-", " - ")
    return data_dir / f"{file_stem}_normalized.xlsx"


def load_points_for_groups(groups: list[str], data_dir: Path) -> np.ndarray:
    frames = []
    missing = []
    for group in groups:
        path = group_to_normalized_path(group, data_dir)
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_excel(path)
        required = {"prey", "predator"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain prey and predator columns.")
        points = frame[["prey", "predator"]].apply(pd.to_numeric, errors="coerce").dropna()
        points = points[(points["prey"] > 0.0) & (points["predator"] > 0.0)]
        if len(points) > 0:
            frames.append(points)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing normalized data files:\n{missing_text}")
    if not frames:
        raise ValueError(f"No valid points found for groups: {groups}")
    return pd.concat(frames, ignore_index=True).to_numpy(dtype=float)


def load_candidates(path: Path) -> list[Candidate]:
    table = pd.read_csv(path)
    required = {"group", "K", "a", "h"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return [
        Candidate(str(row.group), float(row.K), float(row.a), float(row.h))
        for row in table.itertuples(index=False)
    ]


def load_bounds(path: Path, max_bounds: int | None) -> list[np.ndarray]:
    table = pd.read_csv(path)
    required = {"x_min", "x_max", "y_min", "y_max"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if max_bounds is not None:
        table = table.head(max_bounds)
    bounds = []
    for row in table.itertuples(index=False):
        bounds.append(
            np.array(
                [
                    [float(row.x_min), float(row.x_max)],
                    [float(row.y_min), float(row.y_max)],
                ],
                dtype=float,
            )
        )
    return bounds


def bounds_payload(bounds_list: list[np.ndarray]) -> list:
    return [
        [
            [float(bounds[0, 0]), float(bounds[0, 1])],
            [float(bounds[1, 0]), float(bounds[1, 1])],
        ]
        for bounds in bounds_list
    ]


def bounds_list_signature(bounds_list: list[np.ndarray]) -> str:
    payload = json.dumps(bounds_payload(bounds_list), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def filter_outliers_iqr(points: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def real_data_evaluation_data(points: np.ndarray, config: dict) -> tuple[np.ndarray, list[np.ndarray], dict]:
    factor = float(config.get("real_data", {}).get("outlier_iqr_factor", 0.6))
    filtered_points, inlier_mask, _, _ = filter_outliers_iqr(points, factor)
    lower = np.min(filtered_points, axis=0)
    upper = np.max(filtered_points, axis=0)

    flat_dimensions = (upper - lower) <= 1e-12
    if np.any(flat_dimensions):
        pad = np.maximum(np.abs(lower[flat_dimensions]) * 1.0e-6, 1.0e-6)
        lower[flat_dimensions] -= pad
        upper[flat_dimensions] += pad

    bounds = np.array(
        [
            [float(lower[0]), float(upper[0])],
            [float(lower[1]), float(upper[1])],
        ],
        dtype=float,
    )
    metadata = {
        "bounds_source": "real_data",
        "test_points_source": "iqr_inliers",
        "bounds_iqr_factor": factor,
        "bounds_inlier_count": int(len(filtered_points)),
        "bounds_outlier_count": int(len(points) - np.sum(inlier_mask)),
        "bounds_total_count": int(len(points)),
        "raw_num_points": int(len(points)),
        "test_num_points": int(len(filtered_points)),
    }
    return filtered_points, [bounds], metadata


def load_config_sample_bounds(config: dict) -> list[np.ndarray] | None:
    sample_bounds = config.get("energy_test", {}).get("sample_bounds")
    if not sample_bounds:
        return None

    required = {"x_min", "x_max", "y_min", "y_max"}
    missing = required.difference(sample_bounds)
    if missing:
        raise ValueError(
            f"config energy_test.sample_bounds is missing keys: {', '.join(sorted(missing))}"
        )
    return [
        np.array(
            [
                [float(sample_bounds["x_min"]), float(sample_bounds["x_max"])],
                [float(sample_bounds["y_min"]), float(sample_bounds["y_max"])],
            ],
            dtype=float,
        )
    ]


def normalize_bounds_source(value: str | None) -> str:
    if value is None:
        return "real_data"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "data": "real_data",
        "cluster_data": "real_data",
        "merged_data": "real_data",
        "sample_bounds": "config",
        "config_sample_bounds": "config",
        "bounds_csv": "csv",
    }
    return aliases.get(normalized, normalized)


def resolve_bounds_policy(args: argparse.Namespace, config: dict) -> tuple[list[np.ndarray] | None, str]:
    if args.bounds_csv is not None:
        bounds_list = load_bounds(args.bounds_csv, args.max_bounds)
        print(f"using {len(bounds_list)} bounds from {args.bounds_csv}", flush=True)
        return bounds_list, "csv"

    bounds_source = normalize_bounds_source(config.get("energy_test", {}).get("bounds_source"))
    if bounds_source == "real_data":
        print(
            "using per-source/per-cluster bounds from IQR-filtered real data",
            flush=True,
        )
        return None, bounds_source

    if bounds_source == "config":
        bounds_list = load_config_sample_bounds(config)
        if bounds_list is None:
            raise ValueError("energy_test.bounds_source=config requires energy_test.sample_bounds.")
        print(
            "using 1 bounds from config energy_test.sample_bounds: "
            f"({bounds_list[0][0, 0]},{bounds_list[0][0, 1]},"
            f"{bounds_list[0][1, 0]},{bounds_list[0][1, 1]})",
            flush=True,
        )
        return bounds_list, bounds_source

    if bounds_source == "csv":
        bounds_list = load_bounds(DEFAULT_BOUNDS_CSV, args.max_bounds)
        print(f"using {len(bounds_list)} bounds from {DEFAULT_BOUNDS_CSV}", flush=True)
        return bounds_list, bounds_source

    raise ValueError(
        "energy_test.bounds_source must be one of real_data, config, or csv; "
        f"got {bounds_source!r}."
    )


def evaluation_data_and_bounds(
    real_points: np.ndarray,
    static_bounds_list: list[np.ndarray] | None,
    bounds_source: str,
    config: dict,
) -> tuple[np.ndarray, list[np.ndarray], str, dict]:
    if bounds_source == "real_data":
        test_points, bounds_list, metadata = real_data_evaluation_data(real_points, config)
    elif static_bounds_list is not None:
        test_points = real_points
        bounds_list = static_bounds_list
        metadata = {
            "bounds_source": bounds_source,
            "test_points_source": "all_points",
            "bounds_iqr_factor": np.nan,
            "bounds_inlier_count": np.nan,
            "bounds_outlier_count": np.nan,
            "bounds_total_count": len(real_points),
            "raw_num_points": int(len(real_points)),
            "test_num_points": int(len(real_points)),
        }
    else:
        raise ValueError(f"No static bounds were provided for bounds_source={bounds_source!r}.")
    return test_points, bounds_list, bounds_list_signature(bounds_list), metadata


def build_ddga_mixture(candidate: Candidate, config: dict, ddga):
    dynamics = config["dynamics"]
    covariance = config["covariance"]
    dim = 2
    D = float(dynamics["D"])
    d0 = D
    r_n = float(dynamics["growth_rate"])
    d = float(dynamics["d"])
    c = float(dynamics["c"])
    initial_offset = float(dynamics["initial_offset"])
    dt = float(dynamics["dt"])
    steps = int(dynamics["steps"])
    time = dt * np.arange(1, steps + 1)

    N_star, P_star = ddga.coexistence_point(r_n, candidate.K, candidate.a, candidate.h, d, c)
    x_init = np.array([N_star + initial_offset, P_star], dtype=float)
    sol = solve_ivp(
        fun=lambda t, x: ddga.drift_f(t, x, r_n, candidate.K, candidate.a, candidate.h, d, c),
        t_span=(time[0], time[-1]),
        y0=x_init,
        t_eval=time,
        method="RK45",
    )
    path = sol.y.T
    force_origin = np.array(
        [ddga.drift_f(0.0, state, r_n, candidate.K, candidate.a, candidate.h, d, c) for state in path]
    )
    cen_path = path - path[-1, :]
    dis_path = np.linalg.norm(cen_path, axis=1)
    thres_force = np.max(np.linalg.norm(force_origin, axis=1))
    start_idx = int(0.3 * steps)
    near_points = np.where(dis_path[start_idx:] < 3 * thres_force * dt)[0]
    period_time = np.zeros(max(len(near_points) - 1, 0), dtype=int)
    for i in range(len(near_points) - 1):
        if near_points[i + 1] - near_points[i] != 1:
            period_time[i] = near_points[i]
    period_time = period_time[period_time != 0]
    if len(period_time) < 2:
        raise RuntimeError(f"Failed to detect period for {candidate.group}.")

    period = np.mean(np.diff(period_time)) * dt
    t_cycle = np.arange(0.0, period + dt, dt)
    sol_cycle = solve_ivp(
        fun=lambda t, x: ddga.drift_f(t, x, r_n, candidate.K, candidate.a, candidate.h, d, c),
        t_span=(t_cycle[0], t_cycle[-1]),
        y0=path[-1, :],
        t_eval=t_cycle,
        method="RK45",
    )
    limit_cycle = sol_cycle.y.T
    len_lc = len(limit_cycle)
    force_lc = np.array(
        [ddga.drift_f(0.0, state, r_n, candidate.K, candidate.a, candidate.h, d, c) for state in limit_cycle]
    )
    jacobian_lc = np.array(
        [ddga.jacobian_f(state, r_n, candidate.K, candidate.a, candidate.h, d, c) for state in limit_cycle]
    )

    gs = np.linalg.norm(force_lc, axis=1)
    int_gs2 = np.cumsum(gs * gs / len_lc * (len_lc * dt))
    int_exp = np.exp(-int_gs2 / D)
    int_whole = np.cumsum(gs * int_exp / D / len_lc * (len_lc * dt))
    c0 = (1 - int_exp[-1]) / int_whole[-1]
    pre_solution = (1.0 / int_exp) * (1 - c0 * int_whole)
    pre_solution = pre_solution / np.sum(pre_solution)

    sigma_all = np.zeros((len_lc, dim, dim), dtype=float)
    covariance_floor = np.full(len_lc, float(covariance["eigenvalue_floor"]), dtype=float)
    low = covariance["low_landscape"]
    low_mask = (
        (limit_cycle[:, 0] >= float(low["n_min"]))
        & (limit_cycle[:, 0] <= float(low["n_max"]))
        & (limit_cycle[:, 1] >= float(low["p_min"]))
        & (limit_cycle[:, 1] <= float(low["p_max"]))
    )
    covariance_floor[low_mask] = float(covariance["eigenvalue_floor"]) * float(low["floor_factor"])
    lower_left = covariance["lower_left"]
    lower_left_mask = (
        (limit_cycle[:, 0] >= float(lower_left["n_min"]))
        & (limit_cycle[:, 0] <= float(lower_left["n_max"]))
        & (limit_cycle[:, 1] >= float(lower_left["p_min"]))
        & (limit_cycle[:, 1] <= float(lower_left["p_max"]))
    )
    covariance_floor[lower_left_mask] = (
        float(covariance["eigenvalue_floor"]) * float(lower_left["floor_factor"])
    )

    for i in range(len_lc):
        tan_vec = force_lc[i, :].reshape(-1, 1) / np.linalg.norm(force_lc[i, :], 2)
        q = np.hstack([tan_vec, np.vstack([np.zeros((1, dim - 1)), np.eye(dim - 1)])])
        q_this_step, _ = qr(q, mode="economic")
        jac_normal = q_this_step[:, 1:].T @ jacobian_lc[i, :, :] @ q_this_step[:, 1:]
        sigma_normal = solve_continuous_lyapunov(jac_normal, -2 * D * np.eye(dim - 1))
        sigma = q_this_step[:, 1:] @ sigma_normal @ q_this_step[:, 1:].T
        sigma = sigma + d0 * (tan_vec @ tan_vec.T)
        sigma = 0.5 * (sigma + sigma.T)
        min_eigenvalue = np.min(np.linalg.eigvalsh(sigma))
        d1 = max(0.0, covariance_floor[i] - min_eigenvalue)
        if d1 > float(covariance["correction_tolerance"]):
            sigma = sigma + d1 * np.eye(dim)
        sigma_all[i, :, :] = sigma

    return limit_cycle, sigma_all, pre_solution


def get_candidate_mixture(candidate: Candidate, config: dict, ddga, mixture_cache: dict[str, tuple]):
    if candidate.group not in mixture_cache:
        mixture_cache[candidate.group] = build_ddga_mixture(candidate, config, ddga)
    return mixture_cache[candidate.group]


def evaluate_candidate(
    candidate: Candidate,
    real_points: np.ndarray,
    bounds_list: list[np.ndarray],
    config: dict,
    ddga,
    mixture_cache: dict[str, tuple],
    progress_label: str | None = None,
):
    mixture_start = perf_counter()
    if progress_label is not None and candidate.group not in mixture_cache:
        print(f"{progress_label} building DDGA mixture", flush=True)
    means, covariances, weights = get_candidate_mixture(candidate, config, ddga, mixture_cache)
    if progress_label is not None:
        if candidate.group in mixture_cache:
            print(
                f"{progress_label} DDGA mixture ready "
                f"elapsed={perf_counter() - mixture_start:.1f}s",
                flush=True,
            )
    energy_config = config["energy_test"]
    ref_sample_size = int(energy_config["ref_sample_size"])
    bootstraps = int(energy_config["bootstraps"])
    alpha = float(energy_config.get("alpha", 0.05))
    test_mode = str(energy_config.get("test_mode", "difference"))
    rows = []
    for bounds_index, bounds in enumerate(bounds_list, start=1):
        bounds_start = perf_counter()
        if progress_label is not None:
            print(
                f"{progress_label} bounds={bounds_index}/{len(bounds_list)} "
                f"({bounds[0, 0]},{bounds[0, 1]},{bounds[1, 0]},{bounds[1, 1]})",
                flush=True,
            )
        row = {
            "candidate_group": candidate.group,
            "K": candidate.K,
            "a": candidate.a,
            "h": candidate.h,
            "x_min": bounds[0, 0],
            "x_max": bounds[0, 1],
            "y_min": bounds[1, 0],
            "y_max": bounds[1, 1],
            "test_mode": test_mode,
        }
        if test_mode == "similarity":
            stat, upper_ci, baseline_q, delta, threshold, reject_h0 = (
                ddga.ddga_energy_distance_similarity_test(
                    real_points,
                    means,
                    covariances,
                    weights,
                    ref_sample_size=ref_sample_size,
                    num_bootstrap=bootstraps,
                    baseline_samples=int(energy_config.get("similarity_baseline_samples", 200)),
                    alpha=alpha,
                    gamma=float(energy_config.get("similarity_gamma", 0.1)),
                    delta_c=float(energy_config.get("similarity_delta_c", 0.1)),
                    bounds=bounds,
                    rng_seed=int(energy_config.get("rng_seed", 12345)),
                )
            )
            row.update(
                {
                    "T_ed": stat,
                    "p_value": np.nan,
                    "upper_ci": upper_ci,
                    "baseline_A": baseline_q,
                    "similarity_delta": delta,
                    "similarity_threshold": threshold,
                    "similarity_margin": threshold - upper_ci,
                    "bad_rate": np.nan,
                    "bad_rate_cutoff": np.nan,
                    "bad_rate_quantile": np.nan,
                    "bad_rate_threshold": np.nan,
                    "bad_rate_margin": np.nan,
                    "reject_h0": reject_h0,
                    "passed": reject_h0,
                }
            )
        elif test_mode == "bad_rate":
            bad_rate, cutoff, bad_quantile, threshold, compatible = (
                ddga.ddga_logprob_bad_rate_test(
                    real_points,
                    means,
                    covariances,
                    weights,
                    ref_sample_size=ref_sample_size,
                    bad_quantile=float(energy_config.get("bad_quantile", 0.1)),
                    extra_delta=float(energy_config.get("bad_delta", 0.1)),
                    bounds=bounds,
                    rng_seed=int(energy_config.get("rng_seed", 12345)),
                )
            )
            row.update(
                {
                    "T_ed": np.nan,
                    "p_value": np.nan,
                    "upper_ci": np.nan,
                    "baseline_A": np.nan,
                    "similarity_delta": np.nan,
                    "similarity_threshold": np.nan,
                    "similarity_margin": np.nan,
                    "bad_rate": bad_rate,
                    "bad_rate_cutoff": cutoff,
                    "bad_rate_quantile": bad_quantile,
                    "bad_rate_threshold": threshold,
                    "bad_rate_margin": threshold - bad_rate,
                    "reject_h0": not compatible,
                    "passed": compatible,
                }
            )
        else:
            stat, p_value, _ = ddga.ddga_energy_distance_test(
                real_points,
                means,
                covariances,
                weights,
                ref_sample_size=ref_sample_size,
                num_bootstrap=bootstraps,
                bounds=bounds,
                rng_seed=int(energy_config.get("rng_seed", 12345)),
            )
            row.update(
                {
                    "T_ed": stat,
                    "p_value": p_value,
                    "upper_ci": np.nan,
                    "baseline_A": np.nan,
                    "similarity_delta": np.nan,
                    "similarity_threshold": np.nan,
                    "similarity_margin": np.nan,
                    "bad_rate": np.nan,
                    "bad_rate_cutoff": np.nan,
                    "bad_rate_quantile": np.nan,
                    "bad_rate_threshold": np.nan,
                    "bad_rate_margin": np.nan,
                    "reject_h0": p_value < alpha,
                    "passed": p_value >= alpha,
                }
            )
        rows.append(row)
        if progress_label is not None:
            print(
                f"{progress_label} bounds={bounds_index}/{len(bounds_list)} "
                f"done elapsed={perf_counter() - bounds_start:.1f}s",
                flush=True,
            )
    return rows


def make_error_row(candidate: Candidate, test_mode: str, error: Exception) -> dict:
    return {
        "candidate_group": candidate.group,
        "K": candidate.K,
        "a": candidate.a,
        "h": candidate.h,
        "x_min": np.nan,
        "x_max": np.nan,
        "y_min": np.nan,
        "y_max": np.nan,
        "test_mode": test_mode,
        "T_ed": np.inf,
        "p_value": np.nan,
        "upper_ci": np.nan,
        "baseline_A": np.nan,
        "similarity_delta": np.nan,
        "similarity_threshold": np.nan,
        "similarity_margin": -np.inf,
        "bad_rate": np.nan,
        "bad_rate_cutoff": np.nan,
        "bad_rate_quantile": np.nan,
        "bad_rate_threshold": np.nan,
        "bad_rate_margin": -np.inf,
        "reject_h0": False,
        "passed": False,
        "error": str(error),
    }


def finite_float(value, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def truthy_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def selection_key(row: dict, test_mode: str) -> tuple:
    if test_mode == "similarity":
        return (
            not truthy_bool(row.get("passed", False)),
            -finite_float(row.get("similarity_margin"), -np.inf),
            finite_float(row.get("T_ed"), np.inf),
        )
    if test_mode == "bad_rate":
        return (
            not truthy_bool(row.get("passed", False)),
            -finite_float(row.get("bad_rate_margin"), -np.inf),
            finite_float(row.get("bad_rate"), np.inf),
        )
    return (
        -finite_float(row.get("p_value"), -np.inf),
        finite_float(row.get("T_ed"), np.inf),
    )


def select_best_row(rows: list[dict], test_mode: str) -> dict:
    if not rows:
        raise ValueError("Cannot select from an empty row list.")
    return min(rows, key=lambda row: selection_key(row, test_mode))


def row_loss(row: dict, test_mode: str) -> float:
    if "error" in row:
        return np.inf
    if test_mode == "similarity":
        margin = finite_float(row.get("similarity_margin"), -np.inf)
        t_ed = finite_float(row.get("T_ed"), np.inf)
        if not np.isfinite(margin) or not np.isfinite(t_ed):
            return np.inf
        passed_penalty = 0.0 if truthy_bool(row.get("passed", False)) else 1_000_000.0
        return passed_penalty - margin + 1.0e-6 * t_ed

    if test_mode == "bad_rate":
        margin = finite_float(row.get("bad_rate_margin"), -np.inf)
        bad_rate = finite_float(row.get("bad_rate"), np.inf)
        if not np.isfinite(margin) or not np.isfinite(bad_rate):
            return np.inf
        passed_penalty = 0.0 if truthy_bool(row.get("passed", False)) else 1_000_000.0
        return passed_penalty - margin + 1.0e-6 * bad_rate

    p_value = finite_float(row.get("p_value"), -np.inf)
    t_ed = finite_float(row.get("T_ed"), np.inf)
    if not np.isfinite(p_value) or not np.isfinite(t_ed):
        return np.inf
    return -p_value + 1.0e-6 * t_ed


def evaluate_candidate_safely(
    candidate: Candidate,
    real_points: np.ndarray,
    bounds_list: list[np.ndarray],
    config: dict,
    ddga,
    mixture_cache: dict[str, tuple],
    progress_label: str | None = None,
) -> list[dict]:
    test_mode = str(config["energy_test"].get("test_mode", "difference"))
    try:
        return evaluate_candidate(
            candidate,
            real_points,
            bounds_list,
            config,
            ddga,
            mixture_cache,
            progress_label,
        )
    except Exception as exc:
        return [make_error_row(candidate, test_mode, exc)]


def evaluate_candidate_energy_loss(
    candidate: Candidate,
    real_points: np.ndarray,
    bounds_list: list[np.ndarray],
    config: dict,
    ddga,
    mixture_cache: dict[str, tuple],
) -> list[dict]:
    means, covariances, weights = get_candidate_mixture(candidate, config, ddga, mixture_cache)
    energy_config = config["energy_test"]
    rng = np.random.default_rng(int(energy_config.get("rng_seed", 12345)))
    ref_sample_size = int(energy_config["ref_sample_size"])
    x = np.asarray(real_points, dtype=float)
    x_test = ddga.normalize_for_energy_test(x, x)
    rows = []
    for bounds in bounds_list:
        y_ref = ddga.sample_ddga_mixture(
            means,
            covariances,
            weights,
            ref_sample_size,
            rng,
            bounds=bounds,
        )
        y_ref_test = ddga.normalize_for_energy_test(y_ref, x)
        stat = ddga.energy_distance(x_test, y_ref_test)
        rows.append(
            {
                "candidate_group": candidate.group,
                "K": candidate.K,
                "a": candidate.a,
                "h": candidate.h,
                "x_min": bounds[0, 0],
                "x_max": bounds[0, 1],
                "y_min": bounds[1, 0],
                "y_max": bounds[1, 1],
                "test_mode": "pairwise_energy",
                "T_ed": stat,
                "p_value": np.nan,
                "upper_ci": np.nan,
                "baseline_A": np.nan,
                "similarity_delta": np.nan,
                "similarity_threshold": np.nan,
                "similarity_margin": np.nan,
                "bad_rate": np.nan,
                "bad_rate_cutoff": np.nan,
                "bad_rate_quantile": np.nan,
                "bad_rate_threshold": np.nan,
                "bad_rate_margin": np.nan,
                "reject_h0": np.nan,
                "passed": np.nan,
            }
        )
    return rows


def select_pairwise_loss_row(rows: list[dict], test_mode: str, pairwise_loss_mode: str) -> dict:
    if pairwise_loss_mode == "energy":
        return min(rows, key=lambda row: finite_float(row.get("T_ed"), np.inf))
    return select_best_row(rows, test_mode)


def pairwise_row_loss(row: dict, test_mode: str, pairwise_loss_mode: str) -> float:
    if pairwise_loss_mode == "energy":
        return finite_float(row.get("T_ed"), np.inf)
    return row_loss(row, test_mode)


def load_pairwise_cache(
    path: Path,
    candidates: list[Candidate],
    pairwise_loss_mode: str,
    evaluation_signature: str,
) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}

    table = pd.read_csv(path)
    required = {"source_group", "candidate_group", "selection_loss", "bounds_signature"}
    missing = required.difference(table.columns)
    if missing:
        print(f"ignoring pairwise cache {path}: missing columns {sorted(missing)}", flush=True)
        return {}
    if "pairwise_loss_mode" in table.columns:
        table = table[table["pairwise_loss_mode"].astype(str) == pairwise_loss_mode]
    elif pairwise_loss_mode != "full":
        print(f"ignoring pairwise cache {path}: cache has no pairwise_loss_mode column", flush=True)
        return {}
    if "pairwise_evaluation_signature" in table.columns:
        table = table[table["pairwise_evaluation_signature"].astype(str) == evaluation_signature]
    else:
        print(
            f"ignoring pairwise cache {path}: cache has no pairwise_evaluation_signature column",
            flush=True,
        )
        return {}

    valid_groups = {candidate.group for candidate in candidates}
    cache = {}
    for row in table.to_dict(orient="records"):
        source_group = str(row.get("source_group"))
        candidate_group = str(row.get("candidate_group"))
        if source_group not in valid_groups or candidate_group not in valid_groups:
            continue
        loss = finite_float(row.get("selection_loss"), np.inf)
        if not np.isfinite(loss):
            continue
        cache[(source_group, candidate_group, str(row.get("bounds_signature")))] = row
    print(f"loaded {len(cache)} cached pairwise loss rows from {path}", flush=True)
    return cache


def write_pairwise_cache(path: Path, rows: list[dict]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_csv(path, index=False)
    except OSError as exc:
        print(f"warning: could not write pairwise cache {path}: {exc}", flush=True)


def merged_evaluation_signature(
    config: dict,
    bounds_source: str,
    static_bounds_list: list[np.ndarray] | None,
) -> str:
    energy_config = config["energy_test"]
    real_data_config = config.get("real_data", {})
    relevant_config = {
        "test_mode": str(energy_config.get("test_mode", "difference")),
        "ref_sample_size": int(energy_config["ref_sample_size"]),
        "bootstraps": int(energy_config["bootstraps"]),
        "alpha": float(energy_config.get("alpha", 0.05)),
        "similarity_baseline_samples": int(energy_config.get("similarity_baseline_samples", 200)),
        "similarity_gamma": float(energy_config.get("similarity_gamma", 0.1)),
        "similarity_delta_c": float(energy_config.get("similarity_delta_c", 0.1)),
        "bad_quantile": float(energy_config.get("bad_quantile", 0.1)),
        "bad_delta": float(energy_config.get("bad_delta", 0.1)),
        "rng_seed": int(energy_config.get("rng_seed", 12345)),
        "bounds_source": bounds_source,
        "static_bounds": bounds_payload(static_bounds_list) if static_bounds_list is not None else None,
        "outlier_iqr_factor": float(real_data_config.get("outlier_iqr_factor", 0.6)),
        "real_data_test_points": "iqr_inliers" if bounds_source == "real_data" else "all_points",
    }
    payload = json.dumps(relevant_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_merged_cache(
    path: Path,
    evaluation_signature: str,
) -> dict[tuple[str, str, str, str, str], list[dict]]:
    if not path.exists():
        return {}

    table = pd.read_csv(path)
    required = {
        "cluster_signature",
        "candidate_group",
        "candidate_signature",
        "evaluation_signature",
        "bounds_signature",
    }
    missing = required.difference(table.columns)
    if missing:
        print(f"ignoring merged cache {path}: missing columns {sorted(missing)}", flush=True)
        return {}

    table = table[table["evaluation_signature"].astype(str) == evaluation_signature]
    cache = {}
    for key_values, group in table.groupby(
        [
            "cluster_signature",
            "candidate_group",
            "candidate_signature",
            "evaluation_signature",
            "bounds_signature",
        ],
        sort=False,
    ):
        cache[tuple(str(value) for value in key_values)] = group.to_dict(orient="records")
    print(f"loaded {len(cache)} cached merged-cluster candidate rows from {path}", flush=True)
    return cache


def write_merged_cache(path: Path, rows_by_key: dict[tuple[str, str, str, str, str], list[dict]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for cached_rows in rows_by_key.values() for row in cached_rows]
    try:
        pd.DataFrame(rows).to_csv(path, index=False)
    except OSError as exc:
        print(f"warning: could not write merged cache {path}: {exc}", flush=True)


def build_pairwise_loss_matrix(
    candidates: list[Candidate],
    group_points: list[np.ndarray],
    static_bounds_list: list[np.ndarray] | None,
    bounds_source: str,
    config: dict,
    ddga,
    mixture_cache: dict[str, tuple],
    pairwise_cache_path: Path | None,
    reuse_pairwise_cache: bool,
    pairwise_loss_mode: str,
    pairwise_cache_signature: str,
) -> tuple[np.ndarray, list[dict]]:
    test_mode = str(config["energy_test"].get("test_mode", "difference"))
    n = len(candidates)
    total_jobs = n * n
    completed_jobs = 0
    start_time = perf_counter()
    cached_rows = {}
    if reuse_pairwise_cache and pairwise_cache_path is not None:
        cached_rows = load_pairwise_cache(
            pairwise_cache_path,
            candidates,
            pairwise_loss_mode,
            pairwise_cache_signature,
        )
    pairwise_rows_by_key = dict(cached_rows)
    loss_matrix = np.full((n, n), np.inf, dtype=float)
    pairwise_rows = []
    for group_index, real_points in enumerate(group_points):
        group = candidates[group_index].group
        source_test_points, source_bounds_list, source_bounds_signature, source_bounds_metadata = (
            evaluation_data_and_bounds(
                real_points,
                static_bounds_list,
                bounds_source,
                config,
            )
        )
        source_bounds = source_bounds_list[0]
        print(
            f"pairwise loss source={group} ({group_index + 1}/{n}) "
            f"test_points={len(source_test_points)}/{len(real_points)} "
            f"bounds=({source_bounds[0, 0]},{source_bounds[0, 1]},"
            f"{source_bounds[1, 0]},{source_bounds[1, 1]}) "
            f"source={source_bounds_metadata['bounds_source']} "
            f"inliers={source_bounds_metadata['bounds_inlier_count']}/"
            f"{source_bounds_metadata['bounds_total_count']}",
            flush=True,
        )
        for candidate_index, candidate in enumerate(candidates):
            completed_jobs += 1
            elapsed = perf_counter() - start_time
            print(
                f"  pairwise {completed_jobs}/{total_jobs}: "
                f"source={group} candidate={candidate.group} "
                f"({candidate_index + 1}/{n}) elapsed={elapsed:.1f}s",
                flush=True,
            )
            cached = cached_rows.get((group, candidate.group, source_bounds_signature))
            if cached is not None:
                best = cached.copy()
                loss = finite_float(best.get("selection_loss"), np.inf)
                best["cache_status"] = "reused"
                print(f"    reused cached loss={loss:.6f}", flush=True)
            else:
                if pairwise_loss_mode == "energy":
                    rows = evaluate_candidate_energy_loss(
                        candidate,
                        source_test_points,
                        source_bounds_list,
                        config,
                        ddga,
                        mixture_cache,
                    )
                else:
                    rows = evaluate_candidate_safely(
                        candidate,
                        source_test_points,
                        source_bounds_list,
                        config,
                        ddga,
                        mixture_cache,
                    )
                best = select_pairwise_loss_row(rows, test_mode, pairwise_loss_mode).copy()
                loss = pairwise_row_loss(best, test_mode, pairwise_loss_mode)
                best["cache_status"] = "computed"
            loss_matrix[group_index, candidate_index] = loss
            best.update(
                {
                    "source_group": group,
                    "source_group_index": group_index,
                    "candidate_index": candidate_index,
                    "selection_loss": loss,
                    "pairwise_loss_mode": pairwise_loss_mode,
                    "pairwise_evaluation_signature": pairwise_cache_signature,
                    "bounds_signature": source_bounds_signature,
                    **source_bounds_metadata,
                    "stage": "pairwise_loss",
                }
            )
            pairwise_rows.append(best)
            pairwise_rows_by_key[(group, candidate.group, source_bounds_signature)] = best.copy()

        if pairwise_cache_path is not None:
            write_pairwise_cache(pairwise_cache_path, list(pairwise_rows_by_key.values()))

    bad_rows = np.where(~np.any(np.isfinite(loss_matrix), axis=1))[0]
    if len(bad_rows) > 0:
        groups = ", ".join(candidates[i].group for i in bad_rows)
        raise RuntimeError(f"No finite pairwise loss was available for groups: {groups}")
    return loss_matrix, pairwise_rows


def assign_groups_to_representatives(loss_matrix: np.ndarray, representative_indices: list[int]) -> np.ndarray:
    representatives = np.asarray(representative_indices, dtype=int)
    subloss = loss_matrix[:, representatives]
    labels = np.argmin(subloss, axis=1)
    for cluster_id, representative_index in enumerate(representatives):
        labels[representative_index] = cluster_id
    return labels.astype(int)


def assignment_objective(loss_matrix: np.ndarray, labels: np.ndarray, representative_indices: list[int]) -> float:
    representatives = np.asarray(representative_indices, dtype=int)
    selected = representatives[labels]
    losses = loss_matrix[np.arange(len(labels)), selected]
    return float(np.sum(losses))


def choose_initial_medoids(loss_matrix: np.ndarray, k: int) -> tuple[list[int], np.ndarray, float]:
    n = loss_matrix.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"k must be between 1 and {n}; got {k}")

    best_medoids = None
    best_labels = None
    best_objective = np.inf
    for medoids in combinations(range(n), k):
        medoids = list(medoids)
        labels = assign_groups_to_representatives(loss_matrix, medoids)
        objective = assignment_objective(loss_matrix, labels, medoids)
        if objective < best_objective:
            best_medoids = medoids
            best_labels = labels
            best_objective = objective

    if best_medoids is None or best_labels is None or not np.isfinite(best_objective):
        raise RuntimeError(f"Could not find finite k-medoids initialization for k={k}.")
    return best_medoids, best_labels, best_objective


def initial_medoid_solutions(
    loss_matrix: np.ndarray,
    k: int,
    max_solutions: int,
) -> list[tuple[list[int], np.ndarray, float, int]]:
    n = loss_matrix.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"k must be between 1 and {n}; got {k}")

    solutions = []
    for medoids in combinations(range(n), k):
        medoids = list(medoids)
        labels = assign_groups_to_representatives(loss_matrix, medoids)
        objective = assignment_objective(loss_matrix, labels, medoids)
        if np.isfinite(objective):
            solutions.append((objective, medoids, labels))

    if not solutions:
        raise RuntimeError(f"Could not find finite k-medoids initialization for k={k}.")

    solutions.sort(key=lambda item: item[0])
    if max_solutions > 0:
        solutions = solutions[:max_solutions]
    return [
        (medoids, labels, float(objective), rank)
        for rank, (objective, medoids, labels) in enumerate(solutions, start=1)
    ]


def merged_points(member_indices: list[int], group_points: list[np.ndarray]) -> np.ndarray:
    return np.vstack([group_points[index] for index in member_indices])


def evaluate_clusters(
    labels: np.ndarray,
    k: int,
    iteration: int,
    candidates: list[Candidate],
    group_points: list[np.ndarray],
    static_bounds_list: list[np.ndarray] | None,
    bounds_source: str,
    config: dict,
    ddga,
    mixture_cache: dict[str, tuple],
    merged_cache_path: Path | None,
    merged_cache_rows_by_key: dict[tuple[str, str, str, str, str], list[dict]],
    merged_cache_signature: str,
) -> tuple[list[int], list[dict], list[dict], list[dict]]:
    test_mode = str(config["energy_test"].get("test_mode", "difference"))
    representative_indices = []
    summary_rows = []
    detail_rows = []
    failure_rows = []
    group_to_index = {candidate.group: index for index, candidate in enumerate(candidates)}

    for cluster_id in range(k):
        member_indices = np.where(labels == cluster_id)[0].tolist()
        if not member_indices:
            raise RuntimeError(f"k={k} iteration={iteration} cluster={cluster_id} is empty.")

        cluster_candidates = [candidates[index] for index in member_indices]
        cluster_groups = [candidate.group for candidate in cluster_candidates]
        cluster_signature = ";".join(cluster_groups)
        real_points = merged_points(member_indices, group_points)
        cluster_test_points, cluster_bounds_list, cluster_bounds_signature, cluster_bounds_metadata = (
            evaluation_data_and_bounds(
                real_points,
                static_bounds_list,
                bounds_source,
                config,
            )
        )
        cluster_bounds = cluster_bounds_list[0]
        cluster_rows = []
        print(
            f"k={k} iter={iteration} cluster={cluster_id}: "
            f"{len(cluster_groups)} groups, "
            f"test_points={len(cluster_test_points)}/{len(real_points)} "
            f"bounds=({cluster_bounds[0, 0]},{cluster_bounds[0, 1]},"
            f"{cluster_bounds[1, 0]},{cluster_bounds[1, 1]}) "
            f"source={cluster_bounds_metadata['bounds_source']} "
            f"inliers={cluster_bounds_metadata['bounds_inlier_count']}/"
            f"{cluster_bounds_metadata['bounds_total_count']}",
            flush=True,
        )

        for local_candidate_index, candidate_index in enumerate(member_indices, start=1):
            candidate = candidates[candidate_index]
            candidate_signature = candidate_cache_signature(candidate)
            cache_key = (
                cluster_signature,
                candidate.group,
                candidate_signature,
                merged_cache_signature,
                cluster_bounds_signature,
            )
            print(
                f"  merged cluster candidate={candidate.group} "
                f"({local_candidate_index}/{len(member_indices)})",
                flush=True,
            )
            cached_rows = merged_cache_rows_by_key.get(cache_key)
            if cached_rows is not None:
                rows = [row.copy() for row in cached_rows]
                print(
                    f"    reused cached merged rows={len(rows)} "
                    f"signature={merged_cache_signature}",
                    flush=True,
                )
                merged_cache_status = "reused"
            else:
                rows = evaluate_candidate_safely(
                    candidate,
                    cluster_test_points,
                    cluster_bounds_list,
                    config,
                    ddga,
                    mixture_cache,
                    progress_label=f"    candidate={candidate.group}",
                )
                cache_rows = []
                for row in rows:
                    cached_row = row.copy()
                    cached_row.update(
                        {
                            "cluster_signature": cluster_signature,
                            "candidate_signature": candidate_signature,
                            "evaluation_signature": merged_cache_signature,
                            "bounds_signature": cluster_bounds_signature,
                            **cluster_bounds_metadata,
                            "num_groups": len(cluster_groups),
                            "num_points": len(cluster_test_points),
                        }
                    )
                    cache_rows.append(cached_row)
                merged_cache_rows_by_key[cache_key] = cache_rows
                if merged_cache_path is not None:
                    write_merged_cache(merged_cache_path, merged_cache_rows_by_key)
                merged_cache_status = "computed"
            for row in rows:
                row.update(
                    {
                        "k": k,
                        "iteration": iteration,
                        "cluster_id": cluster_id,
                        "cluster_groups": ";".join(cluster_groups),
                        "num_groups": len(cluster_groups),
                        "num_points": len(cluster_test_points),
                        "candidate_index": candidate_index,
                        "cluster_signature": cluster_signature,
                        "candidate_signature": candidate_signature,
                        "evaluation_signature": merged_cache_signature,
                        "bounds_signature": cluster_bounds_signature,
                        **cluster_bounds_metadata,
                        "merged_cache_status": merged_cache_status,
                        "stage": "merged_cluster",
                    }
                )
            cluster_rows.extend(rows)
            detail_rows.extend(rows)

        passed_rows = [row for row in cluster_rows if truthy_bool(row.get("passed", False))]
        if not passed_rows:
            best_failed = select_best_row(cluster_rows, test_mode).copy()
            best_failed.update(
                {
                    "valid_representative": False,
                    "failure_reason": f"no_candidate_passed_{test_mode}_test",
                    "best_failed_candidate": best_failed["candidate_group"],
                    "best_failed_margin": best_failed.get("similarity_margin", np.nan),
                    "best_failed_T_ed": best_failed.get("T_ed", np.nan),
                    "best_failed_bad_rate": best_failed.get("bad_rate", np.nan),
                    "best_failed_bad_rate_threshold": best_failed.get("bad_rate_threshold", np.nan),
                }
            )
            failure_rows.append(best_failed)
            if test_mode == "similarity":
                print(
                    f"  no valid representative: best_failed={best_failed['candidate_group']} "
                    f"margin={best_failed.get('similarity_margin', np.nan):.4f} "
                    f"T_ed={best_failed.get('T_ed', np.nan):.4f} "
                    f"passed={best_failed.get('passed')}",
                    flush=True,
                )
            elif test_mode == "bad_rate":
                print(
                    f"  no valid representative: best_failed={best_failed['candidate_group']} "
                    f"R={best_failed.get('bad_rate', np.nan):.4f} "
                    f"threshold={best_failed.get('bad_rate_threshold', np.nan):.4f} "
                    f"passed={best_failed.get('passed')}",
                    flush=True,
                )
            else:
                print(
                    f"  no valid representative: best_failed={best_failed['candidate_group']} "
                    f"p={best_failed.get('p_value', np.nan):.4f} "
                    f"T_ed={best_failed.get('T_ed', np.nan):.4f} "
                    f"passed={best_failed.get('passed')}",
                    flush=True,
                )
            continue

        best = select_best_row(passed_rows, test_mode).copy()
        representative_index = group_to_index[str(best["candidate_group"])]
        representative_indices.append(representative_index)
        best.update(
            {
                "valid_representative": True,
                "representative_index": representative_index,
                "representative_group": best["candidate_group"],
                "selection_loss": row_loss(best, test_mode),
            }
        )
        summary_rows.append(best)

        if test_mode == "similarity":
            print(
                f"  representative={best['candidate_group']} "
                f"margin={best['similarity_margin']:.4f} "
                f"T_ed={best['T_ed']:.4f} passed={best['passed']} "
                f"bounds=({best['x_min']},{best['x_max']},{best['y_min']},{best['y_max']})"
            )
        elif test_mode == "bad_rate":
            print(
                f"  representative={best['candidate_group']} "
                f"R={best['bad_rate']:.4f} threshold={best['bad_rate_threshold']:.4f} "
                f"margin={best['bad_rate_margin']:.4f} passed={best['passed']} "
                f"bounds=({best['x_min']},{best['x_max']},{best['y_min']},{best['y_max']})"
            )
        else:
            print(
                f"  representative={best['candidate_group']} p={best['p_value']:.4f} "
                f"T_ed={best['T_ed']:.4f} passed={best['passed']} "
                f"bounds=({best['x_min']},{best['x_max']},{best['y_min']},{best['y_max']})"
            )

    return representative_indices, summary_rows, detail_rows, failure_rows


def build_assignment_rows(
    k: int,
    labels: np.ndarray,
    representative_indices: list[int],
    candidates: list[Candidate],
    loss_matrix: np.ndarray,
) -> list[dict]:
    rows = []
    for group_index, label in enumerate(labels):
        representative_index = representative_indices[int(label)]
        candidate = candidates[group_index]
        representative = candidates[representative_index]
        rows.append(
            {
                "k": k,
                "group": candidate.group,
                "group_index": group_index,
                "cluster_id": int(label),
                "representative_group": representative.group,
                "representative_index": representative_index,
                "loss_to_representative": loss_matrix[group_index, representative_index],
                "K": candidate.K,
                "a": candidate.a,
                "h": candidate.h,
                "representative_K": representative.K,
                "representative_a": representative.a,
                "representative_h": representative.h,
            }
        )
    return rows


def safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_cluster_assignment_rows(
    k: int,
    labels: np.ndarray,
    cluster_rows: list[dict],
    candidates: list[Candidate],
    loss_matrix: np.ndarray,
) -> list[dict]:
    rows = []
    cluster_by_id = {}
    for row in cluster_rows:
        cluster_id = safe_int(row.get("cluster_id"))
        if cluster_id is not None:
            cluster_by_id[cluster_id] = row

    for group_index, label in enumerate(labels):
        cluster_id = int(label)
        cluster_row = cluster_by_id.get(cluster_id, {})
        representative_index = safe_int(cluster_row.get("representative_index"))
        candidate = candidates[group_index]
        representative = candidates[representative_index] if representative_index is not None else None
        rows.append(
            {
                "k": k,
                "group": candidate.group,
                "group_index": group_index,
                "cluster_id": cluster_id,
                "cluster_status": cluster_row.get("cluster_status", ""),
                "cluster_groups": cluster_row.get("cluster_groups", ""),
                "representative_group": representative.group if representative is not None else "",
                "representative_index": representative_index if representative_index is not None else np.nan,
                "best_failed_candidate": cluster_row.get("best_failed_candidate", ""),
                "failure_reason": cluster_row.get("failure_reason", ""),
                "loss_to_representative": (
                    loss_matrix[group_index, representative_index]
                    if representative_index is not None
                    else np.nan
                ),
                "initial_rank": cluster_row.get("initial_rank", np.nan),
                "initial_representatives": cluster_row.get("initial_representatives", ""),
                "initial_objective": cluster_row.get("initial_objective", np.nan),
                "attempt_status": cluster_row.get("attempt_status", ""),
                "solution_status": cluster_row.get("solution_status", ""),
                "iteration": cluster_row.get("iteration", np.nan),
                "K": candidate.K,
                "a": candidate.a,
                "h": candidate.h,
                "representative_K": representative.K if representative is not None else np.nan,
                "representative_a": representative.a if representative is not None else np.nan,
                "representative_h": representative.h if representative is not None else np.nan,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ddga = load_ddga_module()
    config = ddga.load_config(args.config)
    if args.test_mode is not None:
        config.setdefault("energy_test", {})["test_mode"] = args.test_mode
    if args.alpha is not None:
        config["energy_test"]["alpha"] = args.alpha
    if args.bad_quantile is not None:
        config["energy_test"]["bad_quantile"] = args.bad_quantile
    if args.bad_delta is not None:
        config["energy_test"]["bad_delta"] = args.bad_delta
    if args.ref_sample_size is not None:
        config["energy_test"]["ref_sample_size"] = args.ref_sample_size
    if args.bootstraps is not None:
        config["energy_test"]["bootstraps"] = args.bootstraps
    if args.similarity_baseline_samples is not None:
        config["energy_test"]["similarity_baseline_samples"] = args.similarity_baseline_samples
    config["energy_test"]["rng_seed"] = args.random_seed
    test_mode = str(config["energy_test"].get("test_mode", "difference"))
    message = (
        f"energy test mode={test_mode} "
        f"alpha={config['energy_test'].get('alpha', 0.05)}"
    )
    if test_mode == "bad_rate":
        message += (
            f" bad_quantile={config['energy_test'].get('bad_quantile', 0.1)}"
            f" bad_delta={config['energy_test'].get('bad_delta', 0.1)}"
        )
    print(message, flush=True)

    candidates = load_candidates(args.parameter_csv)
    static_bounds_list, bounds_source = resolve_bounds_policy(args, config)
    evaluation_signature = merged_evaluation_signature(config, bounds_source, static_bounds_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    detail_path = args.output_dir / "details.csv"
    pairwise_path = args.pairwise_cache or args.output_dir / "pairwise_loss.csv"
    merged_cache_path = args.merged_cache or args.output_dir / "merged_cluster_cache.csv"
    assignments_path = args.output_dir / "assignments.csv"
    cluster_summary_path = args.output_dir / "cluster_summary.csv"
    cluster_assignments_path = args.output_dir / "cluster_assignments.csv"
    initialization_path = args.output_dir / "initialization.csv"
    failures_path = args.output_dir / "failures.csv"

    group_points = [
        load_points_for_groups([candidate.group], args.data_dir)
        for candidate in candidates
    ]
    mixture_cache: dict[str, tuple] = {}
    loss_matrix, pairwise_rows = build_pairwise_loss_matrix(
        candidates,
        group_points,
        static_bounds_list,
        bounds_source,
        config,
        ddga,
        mixture_cache,
        pairwise_path,
        not args.no_reuse_pairwise_loss,
        args.pairwise_loss_mode,
        evaluation_signature,
    )
    if args.no_reuse_merged_cluster:
        merged_cache_rows_by_key = {}
    else:
        merged_cache_rows_by_key = load_merged_cache(merged_cache_path, evaluation_signature)

    summary_rows = []
    detail_rows = []
    assignment_rows = []
    cluster_summary_rows = []
    cluster_assignment_rows = []
    initialization_rows = []
    failure_rows = []
    for k in args.k_values:
        feasible_found = False
        initial_solutions = initial_medoid_solutions(
            loss_matrix,
            k,
            args.max_initializations,
        )
        for initial_representatives, labels, initial_objective, initial_rank in initial_solutions:
            initial_representative_groups = ";".join(
                candidates[index].group for index in initial_representatives
            )
            print(
                f"k={k} try={initial_rank}/{len(initial_solutions)} initial medoids="
                f"{initial_representative_groups} objective={initial_objective:.6f}",
                flush=True,
            )

            final_representatives = initial_representatives
            final_summary_rows = []
            final_labels = labels
            final_objective = initial_objective
            converged = False
            attempt_failed = False
            failed_cluster_ids = []
            passed_cluster_count = 0
            failed_cluster_count = 0

            for iteration in range(args.max_iterations + 1):
                (
                    representatives,
                    iteration_summary_rows,
                    iteration_detail_rows,
                    iteration_failure_rows,
                ) = evaluate_clusters(
                    labels,
                    k,
                    iteration,
                    candidates,
                    group_points,
                    static_bounds_list,
                    bounds_source,
                    config,
                    ddga,
                    mixture_cache,
                    merged_cache_path,
                    merged_cache_rows_by_key,
                    evaluation_signature,
                )
                passed_cluster_count = len(iteration_summary_rows)
                failed_cluster_count = len(iteration_failure_rows)
                for row in iteration_detail_rows:
                    row.update(
                        {
                            "initial_rank": initial_rank,
                            "initial_representatives": initial_representative_groups,
                            "initial_objective": initial_objective,
                            "feasible_attempt": len(iteration_failure_rows) == 0,
                        }
                    )
                detail_rows.extend(iteration_detail_rows)

                if iteration_failure_rows:
                    attempt_failed = True
                    failed_cluster_ids = [
                        int(row["cluster_id"]) for row in iteration_failure_rows
                    ]
                    failed_cluster_id_text = ";".join(
                        str(value) for value in failed_cluster_ids
                    )
                    partial_representative_groups = ";".join(
                        str(row["representative_group"]) for row in iteration_summary_rows
                    )
                    for row in iteration_failure_rows:
                        row.update(
                            {
                                "initial_rank": initial_rank,
                                "initial_representatives": initial_representative_groups,
                                "initial_objective": initial_objective,
                                "attempt_status": "failed",
                            }
                        )
                    failure_rows.extend(iteration_failure_rows)
                    attempt_cluster_rows = []
                    for row in iteration_summary_rows:
                        cluster_row = row.copy()
                        cluster_row.update(
                            {
                                "initial_rank": initial_rank,
                                "initial_representatives": initial_representative_groups,
                                "initial_objective": initial_objective,
                                "attempt_status": "failed",
                                "solution_status": "partial",
                                "cluster_status": "passed",
                                "final": False,
                                "failed_cluster_ids": failed_cluster_id_text,
                                "passed_cluster_count": passed_cluster_count,
                                "failed_cluster_count": failed_cluster_count,
                                "total_clusters": k,
                                "representatives": partial_representative_groups,
                                "assignment_objective": np.nan,
                                "converged": False,
                            }
                        )
                        attempt_cluster_rows.append(cluster_row)
                    for row in iteration_failure_rows:
                        cluster_row = row.copy()
                        cluster_row.update(
                            {
                                "attempt_status": "failed",
                                "solution_status": "partial",
                                "cluster_status": "failed",
                                "final": False,
                                "representative_group": "",
                                "representative_index": np.nan,
                                "selection_loss": np.nan,
                                "failed_cluster_ids": failed_cluster_id_text,
                                "passed_cluster_count": passed_cluster_count,
                                "failed_cluster_count": failed_cluster_count,
                                "total_clusters": k,
                                "representatives": partial_representative_groups,
                                "assignment_objective": np.nan,
                                "converged": False,
                            }
                        )
                        attempt_cluster_rows.append(cluster_row)
                    cluster_summary_rows.extend(attempt_cluster_rows)
                    cluster_assignment_rows.extend(
                        build_cluster_assignment_rows(
                            k,
                            labels,
                            attempt_cluster_rows,
                            candidates,
                            loss_matrix,
                        )
                    )
                    print(
                        f"k={k} try={initial_rank} failed: clusters without "
                        f"passed representative={failed_cluster_ids}",
                        flush=True,
                    )
                    break

                new_labels = assign_groups_to_representatives(loss_matrix, representatives)
                objective = assignment_objective(loss_matrix, new_labels, representatives)
                converged = np.array_equal(new_labels, labels)
                representative_groups = ";".join(
                    candidates[index].group for index in representatives
                )
                for row in iteration_summary_rows:
                    row.update(
                        {
                            "initial_rank": initial_rank,
                            "initial_representatives": initial_representative_groups,
                            "initial_objective": initial_objective,
                            "assignment_objective": objective,
                            "converged": converged,
                            "representatives": representative_groups,
                        }
                    )
                for row in iteration_detail_rows:
                    row.update(
                        {
                            "assignment_objective": objective,
                            "converged": converged,
                            "representatives": representative_groups,
                        }
                    )

                final_representatives = representatives
                final_summary_rows = iteration_summary_rows
                final_labels = labels
                final_objective = objective

                print(
                    f"k={k} try={initial_rank} iter={iteration} "
                    f"reassignment objective={objective:.6f} converged={converged}",
                    flush=True,
                )
                if converged or iteration == args.max_iterations:
                    break
                labels = new_labels

            initialization_rows.append(
                {
                    "k": k,
                    "initial_rank": initial_rank,
                    "initial_representatives": initial_representative_groups,
                    "initial_representative_indices": ";".join(
                        str(index) for index in initial_representatives
                    ),
                    "initial_objective": initial_objective,
                    "attempt_status": "failed" if attempt_failed else "feasible",
                    "passed_cluster_count": passed_cluster_count,
                    "failed_cluster_count": failed_cluster_count,
                    "failed_cluster_ids": ";".join(str(value) for value in failed_cluster_ids),
                    "final_representatives": (
                        ";".join(candidates[index].group for index in final_representatives)
                        if not attempt_failed
                        else ""
                    ),
                    "final_objective": final_objective if not attempt_failed else np.nan,
                    "converged": converged if not attempt_failed else False,
                }
            )

            if attempt_failed:
                continue

            feasible_found = True
            final_cluster_rows = []
            for row in final_summary_rows:
                row["final"] = True
                row["solution_status"] = "feasible"
                row["attempt_status"] = "feasible"
                row["cluster_status"] = "passed"
                row["failed_cluster_ids"] = ""
                row["passed_cluster_count"] = k
                row["failed_cluster_count"] = 0
                row["total_clusters"] = k
                summary_rows.append(row.copy())
                final_cluster_rows.append(row.copy())
            cluster_summary_rows.extend(final_cluster_rows)
            cluster_assignment_rows.extend(
                build_cluster_assignment_rows(
                    k,
                    final_labels,
                    final_cluster_rows,
                    candidates,
                    loss_matrix,
                )
            )
            assignment_rows.extend(
                build_assignment_rows(k, final_labels, final_representatives, candidates, loss_matrix)
            )
            print(
                f"k={k} feasible solution found with initial try={initial_rank}",
                flush=True,
            )
            break

        if not feasible_found:
            failure_rows.append(
                {
                    "k": k,
                    "cluster_id": np.nan,
                    "failure_reason": "no_feasible_initialization_found",
                    "attempted_initializations": len(initial_solutions),
                    "max_initializations": args.max_initializations,
                }
            )
            print(
                f"k={k} no feasible solution found after "
                f"{len(initial_solutions)} initializations",
                flush=True,
            )

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    write_pairwise_cache(pairwise_path, pairwise_rows)
    pd.DataFrame(assignment_rows).to_csv(assignments_path, index=False)
    pd.DataFrame(cluster_summary_rows).to_csv(cluster_summary_path, index=False)
    pd.DataFrame(cluster_assignment_rows).to_csv(cluster_assignments_path, index=False)
    pd.DataFrame(initialization_rows).to_csv(initialization_path, index=False)
    pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {pairwise_path}")
    print(f"wrote {assignments_path}")
    print(f"wrote {cluster_summary_path}")
    print(f"wrote {cluster_assignments_path}")
    print(f"wrote {initialization_path}")
    print(f"wrote {failures_path}")


if __name__ == "__main__":
    main()
