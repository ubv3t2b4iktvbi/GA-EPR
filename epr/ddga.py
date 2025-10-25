import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import qr, solve_continuous_lyapunov
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Union

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
        sig = sigma0_proj[:, :, k]
        inv_cov = np.linalg.inv(sig)
        det_sig = np.linalg.det(sig)

        # Calculate constants
        Cons1 = 1 / np.sqrt((2*np.pi)**2 * det_sig)
        Cons2 = np.exp(-0.5)

        # Calculate gaussian function
        dx = mesh_1 - mu_proj[k, 0]
        dy = mesh_2 - mu_proj[k, 1]
        quadratic_form = inv_cov[0,0]*dx**2 + inv_cov[1,1]*dy**2 + 2*inv_cov[0,1]*dx*dy
        
        # Calculate Z
        Z = Cons1 * (Cons2 ** quadratic_form)
        
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
    plot_landscape: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
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
            y_torch = torch.tensor(y, dtype=torch.float32)
            dy = system_function(y_torch).cpu().numpy()
        return dy

    # Simulate long trajectory
    time = np.arange(0, steps*dt, dt)
    x0 = np.random.uniform(x_min, x_max, size=dim)

    sol = solve_ivp(ode_fun, [time[0], time[-1]], x0, t_eval=time)
    x = sol.y.T
    path = x

    # Calculate drift force
    force_origin = np.zeros((steps, dim))
    for i in range(steps):
        force_origin[i] = ode_fun(0, path[i])

    # Distance calculation
    centered_path = path - path[-1]
    distance_path = np.linalg.norm(centered_path, axis=1)

    # Threshold determination
    threshold_force = np.max(np.linalg.norm(force_origin, axis=1))

    # Find near points
    start_idx = int(0.3*steps)
    selected_distance = distance_path[start_idx:]
    near_points = np.where(selected_distance < 3*threshold_force*dt)[0] + start_idx

    # Period calculation
    period_time = []
    for i in range(1, len(near_points)):
        if near_points[i] - near_points[i-1] != 1:
            period_time.append(near_points[i-1])
    period = np.mean(np.diff(period_time)) * dt

    # Generate limit cycle
    t_limit_cycle = np.arange(0, period + dt, dt)
    
    sol = solve_ivp(ode_fun, [t_limit_cycle[0], t_limit_cycle[-1]], path[-1], t_eval=t_limit_cycle)
    sol_limit_cycle = sol.y.T
    limit_cycle = sol_limit_cycle
    len_limit_cycle = len(limit_cycle)

    # Force and Jacobian along limit cycle
    force_limit_cycle = np.zeros((len_limit_cycle, dim))
    jacobian_limit_cycle = np.zeros((len_limit_cycle, dim, dim))
    for i in range(len_limit_cycle):
        force_limit_cycle[i] = ode_fun(0, limit_cycle[i])
        jacobian_limit_cycle[i] = torch.autograd.functional.jacobian(
            system_function, 
            torch.tensor(limit_cycle[i], dtype=torch.float32)
        ).cpu().numpy()

    # Pre-solution calculation
    gs = np.linalg.norm(force_limit_cycle, axis=1)
    int_gs2 = np.cumsum(gs**2 / len_limit_cycle * (len_limit_cycle*dt))
    int_exp = np.exp(-int_gs2 / noise_strength)
    int_whole = np.cumsum(gs * int_exp / noise_strength / len_limit_cycle * (len_limit_cycle*dt))

    c0 = (1 - int_exp[-1]) / int_whole[-1]
    pre_solution = (1 / int_exp) * (1 - c0*int_whole)
    pre_solution /= np.sum(pre_solution)

    # Covariance calculation
    sigma_all = np.zeros((len_limit_cycle, dim, dim))
    q_last_step = np.zeros((dim, dim))

    for i in range(len_limit_cycle):
        tangent_vector = force_limit_cycle[i] / np.linalg.norm(force_limit_cycle[i])
        # Fix: correctly construct matrix to avoid dimension mismatch
        if dim == 2:
            # For 2D case, use perpendicular vector as second basis vector
            normal_vec = np.array([-tangent_vector[1], tangent_vector[0]])
            q = np.column_stack((tangent_vector, normal_vec))
        else:
            # For higher dimensional case, use original method
            q = np.column_stack((tangent_vector, np.eye(dim)[:, 1:dim]))
        q_this_step, _ = qr(q)
        
        if i > 0:
            direction = np.sign(np.diag(q_this_step[:, 1:].T @ q_last_step[:, 1:]))
            q_this_step[:, 1:] *= direction
        
        q_last_step = q_this_step.copy()
        
        jac_normal = q_this_step[:, 1:].T @ jacobian_limit_cycle[i] @ q_this_step[:, 1:]
        sigma_normal = -solve_continuous_lyapunov(
            jac_normal, 
            2*noise_strength*np.eye(dim-1)
        )
        
        # Fix: ensure dimension match
        sigma_all[i] = (
            q_this_step[:, 1:] @ sigma_normal @ q_this_step[:, 1:].T + 
            noise_strength * np.outer(force_limit_cycle[i], force_limit_cycle[i])
        )

    # Plot pre-solution only when requested
    if plot_verification:
        fig, ax = plt.subplots(figsize=(8, 6))
        time_axis = np.arange(0, period + dt, dt)
        line = ax.plot(time_axis, pre_solution, linewidth=2.5, color=[160/255, 201/255, 235/255])
        
        ax.set_xlim([0, period])
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