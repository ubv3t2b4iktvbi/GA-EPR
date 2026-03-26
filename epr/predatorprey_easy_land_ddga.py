import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import qr, solve_continuous_lyapunov, cholesky, eigvals
from scipy.optimize import brentq


def ensure_positive_definite(mat, eps=1e-8):
    mat = np.real(mat)
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
        return mat + (abs(min_eig) + eps) * np.eye(mat.shape[0])


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
        sig = ensure_positive_definite(sigma0_proj[:, :, k])
        inv_cov = np.linalg.inv(sig)

        cons1 = 1.0 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(sig))
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


def coexistence_equilibrium(r, k, beta, mu, death_rate):
    def predator_nullcline(x):
        return x * np.exp(-beta * x) - death_rate / mu

    x_star = brentq(predator_nullcline, 1e-10, k - 1e-10)
    y_star = r * (1 - x_star / k) * np.exp(beta * x_star)
    return np.array([x_star, y_star], dtype=float)


def find_equilibria(r, k, beta, mu, death_rate):
    equilibria = [
        ("E0", np.array([0.0, 0.0], dtype=float)),
        ("E1", np.array([k, 0.0], dtype=float)),
    ]

    def predator_nullcline(x):
        return x * np.exp(-beta * x) - death_rate / mu

    x_grid = np.linspace(1e-10, k - 1e-10, 2000)
    values = predator_nullcline(x_grid)
    roots = []

    for i in range(len(x_grid) - 1):
        left, right = x_grid[i], x_grid[i + 1]
        f_left, f_right = values[i], values[i + 1]

        if f_left == 0:
            roots.append(left)
            continue

        if f_left * f_right < 0:
            roots.append(brentq(predator_nullcline, left, right))

    unique_roots = []
    for root in roots:
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-6:
            unique_roots.append(root)

    for idx, x_star in enumerate(unique_roots, start=2):
        y_star = r * (1 - x_star / k) * np.exp(beta * x_star)
        if y_star >= 0:
            equilibria.append((f"E{idx}", np.array([x_star, y_star], dtype=float)))

    return equilibria


def drift_f(_, x, r, k, beta, mu, death_rate):
    x1, x2 = x
    exp_term = np.exp(-beta * x1)
    return np.array([
        r * x1 * (1 - x1 / k) - x1 * exp_term * x2,
        x2 * (-death_rate + mu * x1 * exp_term),
    ], dtype=float)


def jacobian_f(x, r, k, beta, mu, death_rate):
    x1, x2 = x
    exp_term = np.exp(-beta * x1)
    return np.array([
        [
            r * (1 - 2 * x1 / k) - x2 * exp_term * (1 - beta * x1),
            -x1 * exp_term,
        ],
        [
            x2 * mu * exp_term * (1 - beta * x1),
            -death_rate + mu * x1 * exp_term,
        ],
    ], dtype=float)


def main():
    # ============================================================
    # Parameter Setting
    # ============================================================
    dim = 2
    D = 0.35

    r = 2
    beta = 0.5
    k = 3.0
    death_rate = 0.4
    mu = 1.2

    dt = 0.01
    steps = int(2e4)
    time = dt * np.arange(1, steps + 1)

    # ============================================================
    # Find the limit cycle
    # ============================================================
    x_init = np.array([1.0, 0.5], dtype=float)

    sol = solve_ivp(
        fun=lambda t, x: drift_f(t, x, r, k, beta, mu, death_rate),
        t_span=(time[0], time[-1]),
        y0=x_init,
        t_eval=time,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    path = sol.y.T

    Force_origin = np.zeros((steps, dim), dtype=float)
    for i in range(steps):
        Force_origin[i, :] = drift_f(0.0, path[i, :], r, k, beta, mu, death_rate)

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
        fun=lambda t, x: drift_f(t, x, r, k, beta, mu, death_rate),
        t_span=(t_cycle[0], t_cycle[-1]),
        y0=path[-1, :],
        t_eval=t_cycle,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    Limit_cycle = sol_cycle.y.T
    len_LC = len(Limit_cycle)
    equilibria = find_equilibria(r, k, beta, mu, death_rate)

    Force_LC = np.zeros((len_LC, dim), dtype=float)
    Jacobian_LC = np.zeros((len_LC, dim, dim), dtype=float)
    for i in range(len_LC):
        Force_LC[i, :] = drift_f(0.0, Limit_cycle[i, :], r, k, beta, mu, death_rate)
        Jacobian_LC[i, :, :] = jacobian_f(Limit_cycle[i, :], r, k, beta, mu, death_rate)

    # ============================================================
    # Phase portrait with vector field and equilibria
    # ============================================================
    equilibrium_points = np.array([point for _, point in equilibria], dtype=float)
    x_max = max(np.max(path[:, 0]), np.max(equilibrium_points[:, 0])) + 0.2
    y_max = max(np.max(path[:, 1]), np.max(equilibrium_points[:, 1])) + 0.2

    x_vals = np.linspace(0.0, x_max, 25)
    y_vals = np.linspace(0.0, y_max, 25)
    mesh_x, mesh_y = np.meshgrid(x_vals, y_vals)
    field_u = r * mesh_x * (1 - mesh_x / k) - mesh_x * np.exp(-beta * mesh_x) * mesh_y
    field_v = mesh_y * (-death_rate + mu * mesh_x * np.exp(-beta * mesh_x))
    field_speed = np.hypot(field_u, field_v)
    field_u_plot = field_u / np.maximum(field_speed, 1e-12)
    field_v_plot = field_v / np.maximum(field_speed, 1e-12)

    fig0 = plt.figure(figsize=(7.2, 5.8))
    ax0 = fig0.add_subplot(111)
    quiver = ax0.quiver(
        mesh_x,
        mesh_y,
        field_u_plot,
        field_v_plot,
        field_speed,
        cmap="viridis",
        angles="xy",
        pivot="mid",
        scale_units="xy",
        scale=18,
        alpha=0.85,
    )
    ax0.plot(path[:, 0], path[:, 1], color=(0.82, 0.82, 0.82), linewidth=1.0, label="Trajectory")
    ax0.plot(Limit_cycle[:, 0], Limit_cycle[:, 1], color="crimson", linewidth=2.0, label="Limit cycle")

    equilibrium_colors = ["black", "dimgray", "goldenrod", "teal", "purple"]
    for idx, (label, point) in enumerate(equilibria):
        color = equilibrium_colors[idx % len(equilibrium_colors)]
        ax0.scatter(point[0], point[1], s=55, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax0.annotate(
            f"{label} = ({point[0]:.2f}, {point[1]:.2f})",
            xy=(point[0], point[1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            color=color,
        )

    cbar = fig0.colorbar(quiver, ax=ax0, pad=0.02)
    cbar.set_label("Vector field speed", fontname="Arial", fontsize=12)
    ax0.set_xlim([0.0, x_max])
    ax0.set_ylim([0.0, y_max])
    ax0.set_xlabel("Prey x", fontname="Arial", fontsize=16)
    ax0.set_ylabel("Predator y", fontname="Arial", fontsize=16)
    ax0.set_title("Vector field, equilibria, and limit cycle", fontname="Arial", fontsize=16)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", frameon=True)
    fig0.tight_layout()

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
    equilibrium = coexistence_equilibrium(r, k, beta, mu, death_rate)
    centered_cycle = Limit_cycle - equilibrium
    x_extent = 1.15 * np.max(np.abs(centered_cycle[:, 0]))
    y_extent = 1.15 * np.max(np.abs(centered_cycle[:, 1]))
    range_1 = [-x_extent, x_extent]
    range_2 = [-y_extent, y_extent]
    P_DDGA, mesh_1, mesh_2 = gaussian_land_dim2(
        np.eye(2),
        np.transpose(Sigma_all, (1, 2, 0)),
        centered_cycle,
        pre_solution,
        range_1,
        range_2,
        300,
    )

    P_DDGA = np.real(P_DDGA)
    P_DDGA[P_DDGA < 0] = 0
    P_plot = np.minimum(P_DDGA, np.quantile(P_DDGA, 0.995))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(
        mesh_1,
        mesh_2,
        P_plot,
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
    ax2.set_zlim(0.0, np.max(P_plot))
    ax2.set_xlabel("Prey x - x*", fontname="Arial", fontsize=18)
    ax2.set_ylabel("Predator y - y*", fontname="Arial", fontsize=18)

    plt.show()


if __name__ == "__main__":
    main()
