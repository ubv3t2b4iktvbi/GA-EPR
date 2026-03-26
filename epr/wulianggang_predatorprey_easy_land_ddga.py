import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import qr, solve_continuous_lyapunov, cholesky, eigvals


def ensure_positive_definite(mat, eps=1e-8):
    mat = np.real(mat)
    if not np.all(np.isfinite(mat)):
        return eps * np.eye(mat.shape[0])
    mat = 0.5 * (mat + mat.T)

    try:
        cholesky(mat, lower=True)
        return mat
    except np.linalg.LinAlgError:
        pass

    mat = mat + eps * np.eye(mat.shape[0])

    try:
        cholesky(mat, lower=True)
        return mat
    except np.linalg.LinAlgError:
        min_eig = np.min(np.real(eigvals(mat)))
        mat = mat + (abs(min_eig) + eps) * np.eye(mat.shape[0])

    for scale in (1e-7, 1e-6, 1e-5, 1e-4):
        trial = mat + scale * np.eye(mat.shape[0])
        try:
            cholesky(trial, lower=True)
            return trial
        except np.linalg.LinAlgError:
            continue

    return mat + 1e-3 * np.eye(mat.shape[0])


def safe_inverse_and_det(mat, eps=1e-10):
    mat = ensure_positive_definite(mat, eps=eps)

    for scale in (0.0, eps, 1e-8, 1e-6, 1e-4):
        trial = mat + scale * np.eye(mat.shape[0])
        try:
            inv_cov = np.linalg.inv(trial)
            det_cov = np.linalg.det(trial)
            if np.isfinite(det_cov) and det_cov > eps:
                return inv_cov, det_cov
        except np.linalg.LinAlgError:
            continue

    det_cov = max(float(np.linalg.det(mat + 1e-3 * np.eye(mat.shape[0]))), eps)
    inv_cov = np.linalg.pinv(mat + 1e-3 * np.eye(mat.shape[0]))
    return inv_cov, det_cov


def estimate_period_from_peaks(signal, time, start_fraction=0.8):
    # This predator-prey system has slow relaxation oscillations, so
    # peak-to-peak timing is more reliable than endpoint distance.
    start_idx = int(start_fraction * len(signal))
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
    return period, peak_indices


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
        inv_cov, det_cov = safe_inverse_and_det(sigma0_proj[:, :, k])
        cons1 = 1.0 / np.sqrt((2 * np.pi) ** 2 * det_cov)
        cons2 = np.exp(-0.5)

        Z = cons1 * (cons2 ** gauss(
            inv_cov,
            mesh_1 - mu_proj[k, 0],
            mesh_2 - mu_proj[k, 1],
        ))
        Z = np.real(Z)
        Z[Z < 0] = 0
        P_DDGA = P_DDGA + Z * phi[k]

    return P_DDGA, mesh_1, mesh_2


def drift_f(_, x, alpha, beta, kappa):
    x1, x2 = x
    x1 = max(x1, 0.0)
    x2 = max(x2, 0.0)
    denom = x1 + x2

    if denom <= 1e-12:
        interaction = 0.0
        predator_gain = 0.0
    else:
        interaction = x1 * x2 / denom
        predator_gain = kappa * x1 * x2 / denom

    return np.array([
        alpha * x1 * (1 - x1) - interaction,
        -beta * x2 + predator_gain,
    ], dtype=float)


def jacobian_f(x, alpha, beta, kappa):
    x1, x2 = x
    x1 = max(x1, 0.0)
    x2 = max(x2, 0.0)
    denom = x1 + x2

    if denom <= 1e-12:
        return np.array([
            [alpha, 0.0],
            [0.0, -beta],
        ], dtype=float)

    denom2 = denom**2
    return np.array([
        [
            alpha * (1 - 2 * x1) - x2**2 / denom2,
            -x1**2 / denom2,
        ],
        [
            kappa * x2**2 / denom2,
            -beta + kappa * x1**2 / denom2,
        ],
    ], dtype=float)


def main():
    # ============================================================
    # Parameter Setting
    # ============================================================
    dim = 2
    D = 0.1

    alpha = 0.1
    beta = 0.8
    kappa = 0.879

    dt = 0.2
    steps = int(8e4)
    time = dt * np.arange(1, steps + 1)

    # ============================================================
    # Find the limit cycle
    # ============================================================
    x_init = np.array([0.999, 0.0001], dtype=float)

    sol = solve_ivp(
        fun=lambda t, x: drift_f(t, x, alpha, beta, kappa),
        t_span=(time[0], time[-1]),
        y0=x_init,
        t_eval=time,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    path = np.maximum(sol.y.T, 1e-12)

    Period, peak_indices = estimate_period_from_peaks(path[:, 0], time, start_fraction=0.8)
    print(f"Estimated period = {Period:.6f}")

    t_cycle = np.arange(0.0, Period + dt, dt)
    sol_cycle = solve_ivp(
        fun=lambda t, x: drift_f(t, x, alpha, beta, kappa),
        t_span=(t_cycle[0], t_cycle[-1]),
        y0=path[peak_indices[-1], :],
        t_eval=t_cycle,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    Limit_cycle = np.maximum(sol_cycle.y.T, 1e-12)
    len_LC = len(Limit_cycle)

    Force_LC = np.zeros((len_LC, dim), dtype=float)
    Jacobian_LC = np.zeros((len_LC, dim, dim), dtype=float)

    for i in range(len_LC):
        Force_LC[i, :] = drift_f(0.0, Limit_cycle[i, :], alpha, beta, kappa)
        Jacobian_LC[i, :, :] = jacobian_f(Limit_cycle[i, :], alpha, beta, kappa)

    # ============================================================
    # Pre-solution
    # ============================================================
    gs = np.linalg.norm(Force_LC, axis=1)
    int_gs2 = np.cumsum(gs * gs / len_LC * (len_LC * dt))
    int_exp = np.exp(-int_gs2 / D)
    int_whole = np.cumsum(gs * int_exp / D / len_LC * (len_LC * dt))

    C0 = (1 - int_exp[-1]) / int_whole[-1]
    pre_solution = (1.0 / int_exp) * (1 - C0 * int_whole)
    pre_solution = np.real(pre_solution)
    pre_solution = np.maximum(pre_solution, 0)
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

    for i in range(len_LC):
        force_norm = np.linalg.norm(Force_LC[i, :], 2)
        if force_norm < 1e-12:
            raise RuntimeError("Tangential velocity is too small on the limit cycle.")

        tan_vec = Force_LC[i, :].reshape(-1, 1) / force_norm
        Q = np.hstack([tan_vec, np.vstack([np.zeros((1, dim - 1)), np.eye(dim - 1)])])
        Q_this_step, _ = qr(Q, mode="economic")

        if i > 1:
            direction = np.sign(np.sum(Q_this_step[:, 1:] * Q_last_step[:, 1:], axis=0))
            direction[direction == 0] = 1
            Q_this_step[:, 1:] = Q_this_step[:, 1:] * direction

        Q_last_step = Q_this_step.copy()

        Jac_normal = Q_this_step[:, 1:].T @ Jacobian_LC[i, :, :] @ Q_this_step[:, 1:]
        Sigma_normal = solve_continuous_lyapunov(
            Jac_normal,
            -2 * D * np.eye(dim - 1),
        )
        Sigma_normal = np.real(Sigma_normal)
        Sigma_normal = 0.5 * (Sigma_normal + Sigma_normal.T)

        Sigma_i = (
            Q_this_step[:, 1:] @ Sigma_normal @ Q_this_step[:, 1:].T
            + D * np.outer(Force_LC[i, :], Force_LC[i, :])
        )
        Sigma_all[i, :, :] = ensure_positive_definite(Sigma_i)

    # ============================================================
    # Landscape from DDGA
    # ============================================================
    x_padding = max(0.02, 0.1 * (np.max(Limit_cycle[:, 0]) - np.min(Limit_cycle[:, 0])))
    y_padding = max(0.002, 0.1 * (np.max(Limit_cycle[:, 1]) - np.min(Limit_cycle[:, 1])))
    range_1 = [max(0.0, np.min(Limit_cycle[:, 0]) - x_padding), np.max(Limit_cycle[:, 0]) + x_padding]
    range_2 = [max(0.0, np.min(Limit_cycle[:, 1]) - y_padding), np.max(Limit_cycle[:, 1]) + y_padding]

    P_DDGA, mesh_1, mesh_2 = gaussian_land_dim2(
        np.eye(2),
        np.transpose(Sigma_all, (1, 2, 0)),
        Limit_cycle,
        pre_solution,
        range_1,
        range_2,
        250,
    )

    P_DDGA = np.real(P_DDGA)
    P_DDGA[P_DDGA < 0] = 0
    P_display_max = np.quantile(P_DDGA, 0.995)
    P_display = np.minimum(P_DDGA, P_display_max)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(
        mesh_1,
        mesh_2,
        P_display,
        cmap="turbo",
        linewidth=0,
        edgecolor="none",
        antialiased=True,
    )
    ax2.set_xlim(range_1)
    ax2.set_ylim(range_2)
    ax2.view_init(elev=39, azim=-33)
    ax2.tick_params(labelsize=20)
    ax2.set_box_aspect((1, 1, 0.8))
    ax2.set_xlabel("Prey x", fontname="Arial", fontsize=18)
    ax2.set_ylabel("Predator y", fontname="Arial", fontsize=18)
    ax2.set_zlabel("P_DDGA", fontname="Arial", fontsize=18)

    fig3 = plt.figure(figsize=(7.2, 5.8))
    ax3 = plt.axes()
    color_map = "rainbow"
    surf2d = ax3.pcolormesh(
        mesh_1,
        mesh_2,
        P_display,
        cmap=color_map,
        shading="auto",
    )
    ax3.contour(
        mesh_1,
        mesh_2,
        P_display,
        50,
        cmap=color_map,
        linewidths=0.4,
        alpha=0.7,
    )
    ax3.set_aspect("auto")
    cbar = fig3.colorbar(surf2d, ax=ax3, pad=0.02, shrink=0.5)
    cbar.set_label("P_DDGA", fontname="Arial", fontsize=14)
    ax3.set_xlim(range_1)
    ax3.set_ylim(range_2)
    ax3.set_xlabel("Prey x", fontname="Arial", fontsize=16)
    ax3.set_ylabel("Predator y", fontname="Arial", fontsize=16)
    ax3.set_title("DDGA landscape (2D)", fontname="Arial", fontsize=16)
    ax3.grid(False)
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
