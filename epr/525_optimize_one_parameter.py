import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks


PARAMETER_NAMES = ("r_n", "K", "a", "h", "d", "c")
DEFAULT_EXCEL_PATH = Path(__file__).resolve().parents[1] / "C2 - 3.xlsx"
DEFAULT_REAL_N_SCALE = 1.0
DEFAULT_PARAMETERS = {
    "r_n": 1.0,
    "K": 3.5,
    "a": 1,
    "h": 1,
    "d": 0.5,
    "c": 1.0,
}
DEFAULT_BOUNDS = {
    "r_n": (0.05, 5.0),
    "K": (1.05, 12.0),
    "a": (0.05, 10.0),
    "h": (0.05, 10.0),
    "d": (0.02, 3.0),
    "c": (0.05, 6.0),
}


def load_real_points(excel_path):
    data_frame = pd.read_excel(excel_path)
    selected = data_frame[["prey", "predator"]].apply(pd.to_numeric, errors="coerce").dropna()
    selected = selected[(selected["prey"] > 0.0) & (selected["predator"] > 0.0)]
    excel_rows = selected.index.to_numpy(dtype=int) + 2
    points = selected.to_numpy(dtype=float)
    if len(points) == 0:
        raise ValueError(f"No positive prey/predator rows were found in {excel_path}.")
    return points, excel_rows


def normalize_points(points, target_lower=(0.0, 0.0), target_upper=(4.0, 3.0)):
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


def drift_f(_, x, r_n, K, a, h, d, c):
    N, P = x
    response = N / (h + N)
    return np.array([
        r_n * N * (1.0 - N / K) - a * response * P,
        (-d + c * response) * P,
    ], dtype=float)


def coexistence_point(params):
    r_n, K, a, h, d, c = params
    if min(params) <= 0.0 or c <= d:
        raise ValueError("Positive coexistence requires positive parameters and c > d.")
    N_star = d * h / (c - d)
    P_star = (r_n / a) * (h + N_star) * (1.0 - N_star / K)
    if not (0.0 < N_star < K and P_star > 0.0):
        raise ValueError("The coexistence point is outside the positive interior.")
    return np.array([N_star, P_star], dtype=float)


def hopf_threshold_k(params):
    _, _, _, h, d, c = params
    if c <= d:
        raise ValueError("Need c > d.")
    N_star = d * h / (c - d)
    return h + 2.0 * N_star


def resample_closed_curve(curve, num_points):
    curve = np.asarray(curve, dtype=float)
    if np.linalg.norm(curve[0] - curve[-1]) > 1e-12:
        curve = np.vstack([curve, curve[0]])
    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    keep = np.concatenate([[True], segment_lengths > 1e-12])
    curve = curve[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    if cumulative[-1] <= 1e-12:
        raise ValueError("The curve has negligible arc length.")
    target = np.linspace(0.0, cumulative[-1], num_points, endpoint=False)
    return np.column_stack([
        np.interp(target, cumulative, curve[:, axis])
        for axis in range(2)
    ])


def simulate_limit_cycle(params, integrate_time, dt, cycle_points, initial_offset):
    params = tuple(float(value) for value in params)
    equilibrium = coexistence_point(params)
    k_hopf = hopf_threshold_k(params)
    if params[1] <= k_hopf:
        raise ValueError("K is not above the Hopf threshold, so no stable limit cycle is expected.")

    y0 = np.array([equilibrium[0] + initial_offset, equilibrium[1]], dtype=float)
    t_eval = np.arange(0.0, integrate_time + 0.5 * dt, dt)
    solution = solve_ivp(
        lambda t, x: drift_f(t, x, *params),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if not solution.success:
        raise ValueError(solution.message)
    trajectory = solution.y.T
    if not np.all(np.isfinite(trajectory)) or np.any(trajectory <= 0.0):
        raise ValueError("The trajectory left the positive finite domain.")
    if np.max(trajectory) > 1e5:
        raise ValueError("The trajectory diverged.")

    tail_start = int(0.5 * len(trajectory))
    prey_tail = trajectory[tail_start:, 0]
    prominence = max(0.02 * np.ptp(prey_tail), 1e-8)
    peaks, _ = find_peaks(prey_tail, prominence=prominence, distance=max(int(2.0 / dt), 1))
    if len(peaks) < 3:
        raise ValueError("No stable periodic orbit was detected.")

    absolute_peaks = tail_start + peaks
    start = absolute_peaks[-2]
    end = absolute_peaks[-1] + 1
    raw_cycle = trajectory[start:end]
    if len(raw_cycle) < 8:
        raise ValueError("Detected cycle contains too few samples.")
    closure_error = np.linalg.norm(raw_cycle[0] - raw_cycle[-1])
    amplitude = np.linalg.norm(np.ptp(raw_cycle, axis=0))
    if amplitude <= 1e-8 or closure_error > 0.12 * amplitude:
        raise ValueError("Detected orbit is not sufficiently closed.")

    period = float(t_eval[end - 1] - t_eval[start])
    return resample_closed_curve(raw_cycle, cycle_points), period


def build_params_from_candidate(candidate, optimized_names, base_params):
    params = dict(base_params)
    for name, value in zip(optimized_names, candidate):
        params[name] = float(value)
    return np.array([params[name] for name in PARAMETER_NAMES], dtype=float), params


class DistanceObjective:
    def __init__(
        self,
        normalized_real_points,
        optimized_names,
        base_params,
        integrate_time,
        dt,
        cycle_points,
        initial_offset,
        invalid_penalty,
        max_cycle_n,
        min_cycle_n,
        max_cycle_p,
    ):
        self.normalized_real_points = normalized_real_points
        self.optimized_names = optimized_names
        self.base_params = base_params
        self.integrate_time = integrate_time
        self.dt = dt
        self.cycle_points = cycle_points
        self.initial_offset = initial_offset
        self.invalid_penalty = invalid_penalty
        self.max_cycle_n = max_cycle_n
        self.min_cycle_n = min_cycle_n
        self.max_cycle_p = max_cycle_p
        self.count = 0
        self.valid = 0

    def __call__(self, candidate):
        self.count += 1
        try:
            params_vector, _ = build_params_from_candidate(
                candidate,
                self.optimized_names,
                self.base_params,
            )
            limit_cycle, _ = simulate_limit_cycle(
                params_vector,
                integrate_time=self.integrate_time,
                dt=self.dt,
                cycle_points=self.cycle_points,
                initial_offset=self.initial_offset,
            )
            min_n = float(np.min(limit_cycle[:, 0]))
            max_n = float(np.max(limit_cycle[:, 0]))
            max_p = float(np.max(limit_cycle[:, 1]))
            if min_n < self.min_cycle_n:
                return self.invalid_penalty * (1.0 + self.min_cycle_n - min_n)
            if max_n > self.max_cycle_n:
                return self.invalid_penalty * (1.0 + max_n - self.max_cycle_n)
            if max_p > self.max_cycle_p:
                return self.invalid_penalty * (1.0 + max_p - self.max_cycle_p)
            distances = point_to_closed_curve_distances(
                self.normalized_real_points,
                limit_cycle,
            )
            loss = float(np.sum(distances))
        except (ValueError, FloatingPointError, OverflowError):
            return self.invalid_penalty
        self.valid += 1
        return loss


def parse_bound(text):
    try:
        left, right = text.split(",", 1)
        lower = float(left)
        upper = float(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Bounds must use the format low,high.") from exc
    if lower >= upper:
        raise argparse.ArgumentTypeError("The lower bound must be smaller than the upper bound.")
    return lower, upper


def parse_named_bound(text):
    try:
        name, bound_text = text.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Named bounds must use the format name:low,high.") from exc
    if name not in PARAMETER_NAMES:
        raise argparse.ArgumentTypeError(
            f"Unknown parameter {name!r}. Choose from: {', '.join(PARAMETER_NAMES)}."
        )
    return name, parse_bound(bound_text)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one or two Holling-II parameters while keeping other parameters fixed; "
            "K is optionally optimized as the exception needed to maintain a limit cycle."
        )
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH)
    parser.add_argument(
        "--parameter",
        choices=PARAMETER_NAMES,
        default=None,
        help="Backward-compatible single primary parameter. Default: a.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=PARAMETER_NAMES,
        default=None,
        help="Optimize one or two primary parameters, e.g. --parameters a h.",
    )
    parser.add_argument(
        "--parameter-bounds",
        type=parse_bound,
        default=None,
        help="Backward-compatible bounds for --parameter or the first --parameters value. Format: low,high",
    )
    parser.add_argument(
        "--bound",
        action="append",
        type=parse_named_bound,
        default=[],
        help="Override one parameter bound. Format: name:low,high. Can be repeated.",
    )
    parser.add_argument("--k-bounds", type=parse_bound, default=None, help="Format: low,high")
    parser.add_argument(
        "--no-optimize-k",
        action="store_true",
        help="Do not vary K unless K is one of the primary optimized parameters.",
    )
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--polish", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU worker processes for candidate evaluations. Use -1 for all cores.",
    )
    parser.add_argument("--integrate-time", type=float, default=700.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--cycle-points", type=int, default=500)
    parser.add_argument(
        "--max-cycle-n",
        type=float,
        default=4,
        help="Reject candidate limit cycles whose maximum N coordinate exceeds this value.",
    )
    parser.add_argument(
        "--min-cycle-n",
        type=float,
        default=0.2,
        help="Reject candidate limit cycles whose minimum N coordinate is below this value.",
    )
    parser.add_argument(
        "--max-cycle-p",
        type=float,
        default=4,
        help="Reject candidate limit cycles whose maximum P coordinate exceeds this value.",
    )
    parser.add_argument("--initial-offset", type=float, default=0.85)
    parser.add_argument("--invalid-penalty", type=float, default=1e6)
    parser.add_argument(
        "--outlier-method",
        choices=("iqr", "none"),
        default="iqr",
        help="Remove real-data outliers before normalization and distance calculation.",
    )
    parser.add_argument(
        "--outlier-iqr-factor",
        type=float,
        default=0.6,
        help="IQR multiplier for --outlier-method iqr.",
    )
    parser.add_argument(
        "--real-n-scale",
        type=float,
        default=DEFAULT_REAL_N_SCALE,
        help="Scale factor applied to normalized real-data N coordinates.",
    )
    parser.add_argument(
        "--min-real-n",
        type=float,
        default=0.2,
        help="Remove processed real points whose normalized N coordinate is below this threshold.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "predatorprey")
    args = parser.parse_args()
    if args.parameter is not None and args.parameters is not None:
        parser.error("Use either --parameter or --parameters, not both.")

    primary_parameters = args.parameters if args.parameters is not None else [args.parameter or "a"]
    if len(primary_parameters) > 2:
        parser.error("--parameters accepts at most two primary parameters.")
    if len(set(primary_parameters)) != len(primary_parameters):
        parser.error("Do not repeat parameter names in --parameters.")
    if args.workers == 0 or args.workers < -1:
        parser.error("--workers must be -1 or a positive integer.")
    if args.outlier_iqr_factor <= 0.0:
        parser.error("--outlier-iqr-factor must be positive.")
    if args.real_n_scale <= 0.0:
        parser.error("--real-n-scale must be positive.")
    if args.min_real_n < 0.0:
        parser.error("--min-real-n must be non-negative.")
    if args.max_cycle_n <= 0.0:
        parser.error("--max-cycle-n must be positive.")
    if args.min_cycle_n < 0.0:
        parser.error("--min-cycle-n must be non-negative.")
    if args.min_cycle_n >= args.max_cycle_n:
        parser.error("--min-cycle-n must be smaller than --max-cycle-n.")
    if args.max_cycle_p <= 0.0:
        parser.error("--max-cycle-p must be positive.")

    args.primary_parameters = primary_parameters
    args.named_bounds = dict(args.bound)
    return args


def main():
    args = parse_args()
    base_params = DEFAULT_PARAMETERS.copy()
    loaded_real_points, loaded_excel_rows = load_real_points(args.excel)
    normalized_real_points, real_min, real_max = normalize_points(
        loaded_real_points,
        target_lower=(0.0, 0.0),
        target_upper=(8.0, 4.0),
    )
    normalized_real_points[:, 0] *= args.real_n_scale
    min_n_mask = normalized_real_points[:, 0] >= args.min_real_n
    if not np.any(min_n_mask):
        raise ValueError("The minimum-N filter removed every real data point.")
    candidate_indices = np.flatnonzero(min_n_mask)
    normalized_real_points = normalized_real_points[min_n_mask]
    if args.outlier_method == "iqr":
        normalized_real_points, iqr_inlier_mask, outlier_lower, outlier_upper = filter_outliers_iqr(
            normalized_real_points,
            factor=args.outlier_iqr_factor,
        )
        kept_indices = candidate_indices[iqr_inlier_mask]
    else:
        kept_indices = candidate_indices
        outlier_lower = np.array([np.nan, np.nan], dtype=float)
        outlier_upper = np.array([np.nan, np.nan], dtype=float)
    inlier_mask = np.zeros(len(loaded_real_points), dtype=bool)
    inlier_mask[kept_indices] = True
    real_points = loaded_real_points[inlier_mask]

    primary_names = list(args.primary_parameters)
    optimized_names = list(primary_names)
    if "K" not in optimized_names and not args.no_optimize_k:
        optimized_names.append("K")

    bounds = []
    for name in optimized_names:
        if name in args.named_bounds:
            bounds.append(args.named_bounds[name])
        elif name == "K" and args.k_bounds is not None:
            bounds.append(args.k_bounds)
        elif name == primary_names[0] and args.parameter_bounds is not None:
            bounds.append(args.parameter_bounds)
        else:
            bounds.append(DEFAULT_BOUNDS[name])

    objective = DistanceObjective(
        normalized_real_points=normalized_real_points,
        optimized_names=optimized_names,
        base_params=base_params,
        integrate_time=args.integrate_time,
        dt=args.dt,
        cycle_points=args.cycle_points,
        initial_offset=args.initial_offset,
        invalid_penalty=args.invalid_penalty,
        max_cycle_n=args.max_cycle_n,
        min_cycle_n=args.min_cycle_n,
        max_cycle_p=args.max_cycle_p,
    )
    updating = "immediate" if args.workers == 1 else "deferred"

    removed_indices = np.flatnonzero(~inlier_mask)
    removed_excel_rows = loaded_excel_rows[removed_indices]
    removed_points = loaded_real_points[removed_indices]
    print(f"Loaded {len(loaded_real_points)} real points from {args.excel}")
    print(
        "Minimum-N filter after normalization: "
        f"min_real_n={args.min_real_n:.10g}, "
        f"removed={np.count_nonzero(~min_n_mask)}"
    )
    print(
        "Outlier filter: "
        f"method={args.outlier_method}, kept={len(real_points)}, "
        f"removed={len(removed_indices)}"
    )
    if args.outlier_method == "iqr":
        print(
            "IQR inlier bounds after normalization (N, P): "
            f"lower=({outlier_lower[0]:.10g}, {outlier_lower[1]:.10g}), "
            f"upper=({outlier_upper[0]:.10g}, {outlier_upper[1]:.10g})"
        )
    if len(removed_indices) > 0:
        print("Removed outlier points:")
        for index, excel_row, point in zip(removed_indices, removed_excel_rows, removed_points):
            print(
                f"  data_index={int(index)}, excel_row={int(excel_row)}, "
                f"prey={point[0]:.10g}, predator={point[1]:.10g}"
            )
    else:
        print("Removed outlier points: none")
    print(f"Real-data minimum used for normalization (prey, predator) = ({real_min[0]:.10g}, {real_min[1]:.10g})")
    print(f"Real-data maximum used for normalization (prey, predator) = ({real_max[0]:.10g}, {real_max[1]:.10g})")
    print(f"Applied real-data N scale factor = {args.real_n_scale:.10g}")
    print(f"Primary optimized parameters: {primary_names}")
    print(f"Optimizing variables: {optimized_names}")
    print(f"Bounds: {dict(zip(optimized_names, bounds))}")
    print(
        "Limit-cycle constraints: "
        f"{args.min_cycle_n:.10g} <= N <= {args.max_cycle_n:.10g}, "
        f"P <= {args.max_cycle_p:.10g}"
    )
    print(f"Workers: {args.workers}, updating: {updating}")

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=args.seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=args.tol,
        polish=args.polish,
        workers=args.workers,
        updating=updating,
        disp=True,
    )

    best_params_vector, best_params = build_params_from_candidate(result.x, optimized_names, base_params)
    best_cycle, best_period = simulate_limit_cycle(
        best_params_vector,
        integrate_time=args.integrate_time,
        dt=args.dt,
        cycle_points=args.cycle_points,
        initial_offset=args.initial_offset,
    )
    best_cycle_min = np.min(best_cycle, axis=0)
    best_cycle_max = np.max(best_cycle, axis=0)
    best_distances = point_to_closed_curve_distances(normalized_real_points, best_cycle)
    best_loss = float(np.sum(best_distances))

    print("\nBest result:")
    for name in PARAMETER_NAMES:
        print(f"  {name} = {best_params[name]:.10g}")
    print(f"  estimated_period = {best_period:.10g}")
    print(
        "  limit_cycle_range = "
        f"N [{best_cycle_min[0]:.10g}, {best_cycle_max[0]:.10g}], "
        f"P [{best_cycle_min[1]:.10g}, {best_cycle_max[1]:.10g}]"
    )
    print(f"  sum_of_normalized_euclidean_distances = {best_loss:.10g}")
    print(f"  optimizer_success = {result.success}")
    evaluations = {
        "count": int(getattr(result, "nfev", objective.count)),
        "valid": objective.valid if args.workers == 1 else None,
    }
    print(f"  evaluations = {evaluations['count']}, valid_cycles = {evaluations['valid']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_tag = "_".join(primary_names)
    output_path = args.output_dir / (
        f"{args.excel.stem}_optimize_{parameter_tag}_"
        f"distance_{best_loss:.8f}.json"
    )
    payload = {
        "excel": str(args.excel),
        "optimized_parameter": primary_names[0] if len(primary_names) == 1 else parameter_tag,
        "primary_optimized_parameters": primary_names,
        "optimized_variables": optimized_names,
        "base_parameters": base_params,
        "best_parameters": {name: float(best_params[name]) for name in PARAMETER_NAMES},
        "sum_of_normalized_euclidean_distances": best_loss,
        "estimated_period": best_period,
        "limit_cycle_constraint": {
            "min_n": float(args.min_cycle_n),
            "max_n": float(args.max_cycle_n),
            "max_p": float(args.max_cycle_p),
        },
        "limit_cycle_range": {
            "min_n": float(best_cycle_min[0]),
            "max_n": float(best_cycle_max[0]),
            "min_p": float(best_cycle_min[1]),
            "max_p": float(best_cycle_max[1]),
        },
        "outlier_filter": {
            "method": args.outlier_method,
            "applied_after": "normalization_real_n_scale_and_min_real_n_filter",
            "iqr_factor": args.outlier_iqr_factor if args.outlier_method == "iqr" else None,
            "loaded_points": int(len(loaded_real_points)),
            "used_points": int(len(real_points)),
            "removed_points": int(len(removed_indices)),
            "removed_indices": removed_indices.tolist(),
            "removed_excel_rows": removed_excel_rows.tolist(),
            "removed_point_details": [
                {
                    "data_index": int(index),
                    "excel_row": int(excel_row),
                    "prey": float(point[0]),
                    "predator": float(point[1]),
                }
                for index, excel_row, point in zip(
                    removed_indices,
                    removed_excel_rows,
                    removed_points,
                )
            ],
            "inlier_lower": outlier_lower.tolist(),
            "inlier_upper": outlier_upper.tolist(),
        },
        "real_data_min": real_min.tolist(),
        "real_data_max": real_max.tolist(),
        "real_data_n_scale": float(args.real_n_scale),
        "minimum_real_n_filter": {
            "threshold": float(args.min_real_n),
            "removed_points": int(np.count_nonzero(~min_n_mask)),
        },
        "bounds": {name: list(bound) for name, bound in zip(optimized_names, bounds)},
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": evaluations,
        "workers": args.workers,
        "updating": updating,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()
