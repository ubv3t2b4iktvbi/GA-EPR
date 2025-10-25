import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.linalg import kron
from tqdm import tqdm

def analyze_attractors(
    system_function: callable,
    dim: int,
    x_min: float,
    x_max: float,
    noise_strength: float,
    dt: float = 0.01,
    steps: int = 2000,
    rand_num: int = 1000
) -> tuple:
    """
    Explore attractors in a dynamical system and calculate their covariances.
    
    Args:
        system_function: Function defining the dynamical system
        dim: Problem dimension
        x_min: Minimum value for initial condition sampling
        x_max: Maximum value for initial condition sampling
        noise_strength: Strength of noise for covariance calculation
        dt: Time step size
        steps: Number of integration steps
        rand_num: Number of random initial conditions to try
        
    Returns:
        Tuple of attractor array and covariance matrices
        - Attractor array: shape (n_attractors, dim+1) with last column as frequency
        - Covariance matrices: shape (dim, dim, n_attractors)
    """
    
    # Setup time array
    time = np.arange(0, steps + 1) * dt
    
    # Compute simulation ranges with 20% padding
    padding = 0.1 * (x_max - x_min)
    simu_range_1 = x_min - padding
    simu_range_2 = x_max + padding
    
    # Convert system function to numpy compatible
    def ode_fun(t, y):
        try:
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                print(f"Invalid input y at t={t}: {y}")
                return np.zeros_like(y)
            with torch.no_grad():
                y_torch = torch.tensor(y, dtype=torch.float32)
                dy = system_function(y_torch).cpu().numpy()
            if np.any(np.isnan(dy)) or np.any(np.isinf(dy)):
                print(f"Invalid output dydt at t={t}: {dy}")
                print(f"Input y was: {y}")
                return np.zeros_like(y)
            return dy
        except Exception as e:
            print(f"Error in ODE function at t={t}: {e}")
            return np.zeros_like(y)
    
    # Initialize attractor storage
    attractor = np.array([[np.nan] * (dim + 1)])  # Extra column for frequency
    real_num = 0
    
    # Attractor exploration
    for _ in tqdm(range(rand_num), desc="run odes to explore attractors"):
        # Generate random initial conditions
        x0 = np.random.uniform(simu_range_1, simu_range_2, size=dim)

        # Solve ODE
        sol = solve_ivp(ode_fun, [0, steps * dt], x0, method='RK45', t_eval=time)
        x = sol.y.T
        
        # Check if solution converged to an attractor
        end_section = x[int(0.8 * steps):, :]
        mean_end = np.mean(end_section, axis=0)
        drift = np.linalg.norm(x[-1, :] - mean_end)
        
        if drift < 2 * dt:  # Convergence criterion
            real_num += 1
            
            if np.isnan(attractor[0, 0]):  # First attractor
                attractor[0, :dim] = x[-1, :]
                attractor[0, dim] = 1
            else:
                # Find closest existing attractor
                tree = cKDTree(attractor[:, :dim])
                dist, index = tree.query(x[-1, :])
                
                if dist > 10 * dt:  # New attractor found
                    new_point = np.append(x[-1, :], 1)
                    attractor = np.vstack((attractor, new_point))
                else:  # Increment frequency of existing attractor
                    attractor[index, dim] += 1
    attractor[:, dim] /= rand_num
                    
    # attractor[:, dim] = 0
    # real_num = 0 
    
    # for i in range(attractor.shape[0]):
    #     for _ in tqdm(range(rand_num)):
    #         x0 = attractor[i, :dim] + np.random.randn(dim) * 0.1
            
    #         sol = solve_ivp(ode_fun, [0, steps * dt], x0, method='RK45', t_eval=time)
    #         x = sol.y.T

    #         # Check if solution converged to an attractor
    #         end_section = x[int(0.8 * steps):, :]
    #         mean_end = np.mean(end_section, axis=0)
    #         drift = np.linalg.norm(x[-1, :] - mean_end)
            
    #         if drift < 2 * dt:  # Convergence criterion
    #             real_num += 1
    #             # Find closest existing attractor
    #             tree = cKDTree(attractor[:, :dim])
    #             dist, index = tree.query(x[-1, :])
                
    #             if dist > 10 * dt:  # New attractor found
    #                 new_point = np.append(x[-1, :], 1)
    #                 attractor = np.vstack((attractor, new_point))
    #             else:  # Increment frequency of existing attractor
    #                 attractor[index, dim] += 1

    # # Convert frequencies to probabilities
    # if real_num > 0:
    #     attractor[:, dim] /= real_num
    
    # Covariance calculation for each attractor
    attr_num = attractor.shape[0]
    sigma = np.zeros((dim, dim, attr_num))
    eye_dim_np = np.eye(dim)
    
    def get_jacobian(x):
        """Compute Jacobian matrix using torch.autograd.grad"""
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        jacobian = torch.zeros((dim, dim))
        
        for i in range(dim):
            y = system_function(x_torch)[i]
            grad = torch.autograd.grad(y, x_torch, create_graph=True)[0]
            jacobian[i, :] = grad
            
        return jacobian.detach().numpy()
    
    for num in range(attr_num):
        # Get attractor point
        attractor_point = attractor[num, :dim]
        
        # Compute Jacobian at attractor point
        jac_eval = get_jacobian(attractor_point)
        
        # Calculate Kronecker products
        kron1 = kron(jac_eval, eye_dim_np)
        kron2 = kron(eye_dim_np, jac_eval)
        kron_sum = kron1 + kron2
        
        # Create right-hand side vector (-2 * noise_strength * I)
        rhs = -2 * noise_strength * eye_dim_np.reshape(dim**2)
        
        # Solve the linear system
        vec_sigma = np.linalg.solve(kron_sum, rhs)
        
        # Reshape solution into matrix form
        sigma[:, :, num] = vec_sigma.reshape((dim, dim))

    return attractor, sigma

if __name__ == "__main__":
    # Define dynamical system function (PyTorch version)
    class Force:
        def __init__(self):
            self.n = 4
            self.S = 0.5
            self.S_n = self.S ** self.n
            self.k = 1.0
            self.a = 1.0
            
        def activate(self, x):
            return self.a * torch.pow(x, self.n) / (self.S_n + torch.pow(x, self.n))
        
        def restrict(self, x):
            return self.a * self.S_n / (self.S_n + torch.pow(x, self.n))

        def force_2d_multistable(self, x):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                f = -self.k * x + self.activate(x) + torch.flip(self.restrict(x), dims=[1])
                return f.squeeze()
            else:
                f = -self.k * x + self.activate(x) + torch.flip(self.restrict(x), dims=[1])
                return f
            
    force_system = Force()
    attractor, sigma = analyze_attractors(force_system.force_2d_multistable, 2, 0, 3, 0.05)
