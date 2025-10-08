import math
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from normflows.distributions.base import BaseDistribution
from sklearn.decomposition import PCA

from tqdm import tqdm
from wsga import analyze_attractors
from ddga import analyze_limit_cycle
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
class SharedBaseDataset:
    """
    Core dataset with shared resources and caching.
    Implements:
    - Simulation caching
    - Lazy GMM parameter calculation
    - Cross-component resource sharing
    """
    _instance = None  # Singleton instance

    def __new__(cls, args, problem, force):
        """Singleton pattern ensures shared resources across components"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(args, problem, force)
        return cls._instance

    def __init__(self, args, problem, force):
        self.args = args
        self.problem = problem
        self.force = force
        self.device = args.device
        
        # Determine maximum required simulation size
        self.max_required_size = 0

        if args.train_mode in ['hybrid', 'flow_only']:
            flow_sample_size = problem.flow_sample_size
        else:
            flow_sample_size = 0

        if args.pdf_method == "Simulate":
            self.max_required_size = max(problem.dnn_sample_size, flow_sample_size)            
        else:
            self.max_required_size = flow_sample_size
        # Pre-simulated data - will be lazily initialized
        self.simulation_data = None

        # GMM parameters cache
        self._gmm_params = None  
        self._mix = None
        self._q0_dist = None     # Converted flow distribution

        # Optimization: pre-allocate 20% more than typical batch size
        self.base_batch_size = max(1024, int(args.base_batch_size * 1.2))

        # Store projection indices
        self.index_1 = getattr(self.problem, 'index_1', 0)
        self.index_2 = getattr(self.problem, 'index_2', 1)

    def get_simulated_data(self, required_size, noise_strength):
        """
        Get simulated data with pre-simulation optimization:
        - Generate all needed data once on first call
        - Return subsets as needed for efficiency
        """
        # Lazy initialization of the full simulation
        if self.simulation_data is None:
            print(f"Pre-generating {self.max_required_size} simulation samples...")
            self.simulation_data = self._run_simulation(
                self.max_required_size, 
                noise_strength
            )
            self.simulation_data.requires_grad_(True)
            print("Simulation complete and cached.")
        return self.simulation_data[:required_size]

    def _run_simulation(self, batch_size, noise_strength):
        """Core SDE simulation with progress tracking"""
        x = self._uniform_sample((batch_size, self.problem.input_dim))
        z_dist = MultivariateNormal(
            torch.zeros(self.problem.input_dim, device=self.device),
            torch.eye(self.problem.input_dim, device=self.device)
        ).expand((batch_size,))
        padding = 0.1 * (self.problem.x_max - self.problem.x_min)
        self.bound_min = self.problem.x_min - padding
        self.bound_max = self.problem.x_max + padding
        with tqdm(total=self.args.sim_steps, desc=f'Simulating {batch_size} samples') as pbar:
            for step in range(self.args.sim_steps):
                x = self._sde_step(x, z_dist, noise_strength)
                pbar.set_description(f'Simulating {batch_size} samples (Step {step+1}/{self.args.sim_steps})')  
                pbar.update(1)
        return x

    def _sde_step(self, x, z_dist, noise_strength):
        """Single SDE integration step with boundary handling"""
        with torch.no_grad():
            dt = self.args.sim_dt
            z = z_dist.sample()
            x_new = x + dt*self.force(x) + math.sqrt(2*noise_strength*dt)*z
            
            # Reflect boundaries
            x_new = torch.where(x_new < self.bound_min, 
                            2*self.bound_min - x_new, x_new)
            x_new = torch.where(x_new > self.bound_max,
                            2*self.bound_max - x_new, x_new)
        return x_new.detach()

    @property
    def gmm_components(self):
        """Lazy-loaded GMM parameters with validation"""
        if self._gmm_params is None:
            self._gmm_params = self._calculate_gmm()
        return self._gmm_params
    @property
    def mix(self):
        if self._mix is None:
            weights, means, covs = self.gmm_components
            if self.problem.input_dim > 2:
                # self._build_conditional_gmm()
                # self._mix = MixtureSameFamily(Categorical(weights), MultivariateNormal(self.means_obs, self.cov_obs))
                self.max_required_size = self.problem.dnn_sample_size
                self.x_full = self.get_simulated_data(self.problem.dnn_sample_size, 
                                                    self.problem.noise_strength)
                tree = cKDTree(means.detach().cpu().numpy())
                dist, idx = tree.query(self.x_full.detach().cpu().numpy(), k=1)
                # 修改开始：根据每个点离哪个吸引子最近来计算权重
                # 统计每个吸引子附近的点数并归一化
                counts = np.bincount(idx, minlength=weights.shape[0])
                weights = torch.tensor(counts, dtype=torch.float32, device=self.device)
                weights = weights / weights.sum()
                # 修改结束
                self._mix = MixtureSameFamily(Categorical(weights), MultivariateNormal(means, covs))
                pca = PCA()
                Z = pca.fit_transform(means.detach().cpu().numpy())
                self.pca_mean = torch.tensor(pca.mean_).to(self.device)
                self.pca_w = torch.tensor(pca.components_).to(self.device)

                samples = self._mix.sample((10000,))

                # 转换张量为 NumPy 数组
                if hasattr(samples, 'numpy'):
                    samples = samples.cpu().numpy()
                samples = np.dot(samples - pca.mean_, pca.components_.T)
                # # 2D 数据：标准散点图
                plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20)
                plt.title('2D Gaussian Mixture Model Samples')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')  
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
                self.min_project = samples.min()
                self.max_project = samples.max()
            else:
                self._mix = MixtureSameFamily(Categorical(weights), MultivariateNormal(means, covs))
        return self._mix
    @property
    def q0(self):
        if self._q0_dist is None:
            self._q0_dist = convert_nfdist(self.mix)
        return self._q0_dist
    
    def _calculate_gmm(self):
        """Analyze force field to determine GMM parameters"""
        # Implementation of analyze_attractors should return:
        # (weights, means, covariances)
        if self.args.pdf_method == 'DDGA':
            attractors, covariances = analyze_limit_cycle(self.force, 
            self.problem.input_dim, 
            self.problem.x_min, 
            self.problem.x_max,
            self.problem.noise_strength,
            self.args.ddga_delta_t,
            self.args.ddga_num_steps,
            )
        else:
            attractors, covariances = analyze_attractors(self.force, 
            self.problem.input_dim, 
            self.problem.x_min, 
            self.problem.x_max,
            self.problem.noise_strength,
            self.args.wsga_delta_t,
            self.args.wsga_num_steps,
            self.args.rand_num
            )
        # Extract means and weights

        means = torch.tensor(attractors[:, :self.problem.input_dim], dtype=torch.float32, device=self.args.device)  # (n_attractors, dim)
        if self.args.pdf_method == 'DDGA':
            weights = torch.ones_like(means[:, 0], dtype=torch.float32, device=self.args.device) / means.shape[0]  # (n_attractors,)
            covs = torch.tensor(covariances, dtype=torch.float32, device=self.args.device)  # (n_attractors, dim, dim)
        else:
            weights = torch.tensor(attractors[:, -1], dtype=torch.float32, device=self.args.device)  # (n_attractors,)
            covs = torch.tensor(covariances, dtype=torch.float32, device=self.args.device).permute(2, 0, 1)  # (n_attractors, dim, dim)
        
        for cov in covs:
            self._validate_covariance(cov)
            
        return (weights, means, covs)
    
    def _build_conditional_gmm(self):
        """Build conditional GMM parameters using PyTorch operations"""
        weights, means, covs = self.gmm_components
        
        # Indices for observed (index_1, index_2) and unobserved dimensions
        self.obs_idx = [self.index_1, self.index_2]
        all_dims = list(range(self.problem.input_dim))
        self.unobs_idx = [i for i in all_dims if i not in self.obs_idx]
        
        self.weights = weights
        self.means_obs = means[:, self.obs_idx]  # μ_1
        self.means_unobs = means[:, self.unobs_idx]  # μ_2
        
        # Covariance blocks 
        self.cov_obs = covs[:, self.obs_idx][:, :, self.obs_idx]
    
        # Σ_22: covariances between unobserved dimensions
        self.cov_unobs = covs[:, self.unobs_idx][:, :, self.unobs_idx]
        
        # Σ_21: cross-covariances (unobserved x observed)
        self.cov_cross = covs[:, self.unobs_idx][:, :, self.obs_idx]
        
        self._conditional_gmm = True

    # def _sample_conditional(self, x_observed, num_samples):
    #     """ batch sampling from conditional GMM for fixed observation"""
    #     num_components = self.weights.shape[0]
    #     unobs_dim = len(self.unobs_idx)
        
    #     # Precompute conditional distributions for each component
    #     cond_means = []
    #     cond_covs = []
        
    #     for k in range(num_components):
    #         # Conditional mean: μ_2 + Σ_21 Σ_11^(-1) (x_obs - μ_1)
    #         diff = x_observed - self.means_obs[k]
    #         cov_inv = torch.linalg.inv(self.cov_obs[k])
    #         update_term = self.cov_cross[k] @ cov_inv @ diff
    #         cond_mean = self.means_unobs[k] + update_term
    #         cond_means.append(cond_mean)
            
    #         # Conditional covariance: Σ_22 - Σ_21 Σ_11^(-1) Σ_12
    #         cov_cross_T = self.cov_cross[k].T
    #         reduction_term = self.cov_cross[k] @ cov_inv @ cov_cross_T
    #         cond_cov = self.cov_unobs[k] - reduction_term
            
    #         # Regularize covariance matrix
    #         cond_cov += 1e-6 * torch.eye(unobs_dim, device=cond_cov.device)
    #         cond_covs.append(cond_cov)
        
    #     # Sample component counts using multinomial distribution
    #     component_counts = torch.multinomial(self.weights, num_samples, replacement=True)
    #     unique_components, counts = torch.unique(component_counts, return_counts=True)
        
    #     # Prepare output tensor
    #     x_full = torch.zeros(num_samples, self.problem.input_dim, device=self.device)
    #     x_full[:, self.obs_idx] = x_observed
        
    #     # Track current sample position
    #     start_idx = 0
        
    #     # Batch sample for each component that has samples
    #     for comp_idx, count in zip(unique_components, counts):
    #         if count == 0:
    #             continue
                
    #         comp_idx = comp_idx.item()
    #         count = count.item()
            
    #         # Get precomputed parameters for this component
    #         mean = cond_means[comp_idx]
    #         cov = cond_covs[comp_idx]
            
    #         try:
    #             # Try batch sampling with Cholesky decomposition
    #             L = torch.linalg.cholesky(cov)
    #             noise = torch.randn(count, unobs_dim, device=self.device)
    #             samples = mean + noise @ L.T
    #             x_full[start_idx:start_idx+count, self.unobs_idx] = samples
    #         except:
    #             # Fallback to diagonal approximation
    #             diag_std = torch.sqrt(torch.diag(cov)).abs() + 1e-6
    #             noise = torch.randn(count, unobs_dim, device=self.device)
    #             samples = mean + noise * diag_std
    #             x_full[start_idx:start_idx+count, self.unobs_idx] = samples
            
    #         start_idx += count
        
    #     return x_full
    def _sample_conditional(self, x_observed, num_samples):
        """ batch sampling from conditional GMM for fixed observation """
        K = self.weights.shape[0]
        D_unobs = len(self.unobs_idx)
        I = torch.eye(D_unobs, device=self.device)

        # 1) 预计算每个分量的条件参数
        cond_means, cond_covs = [], []
        for k in range(K):
            diff = x_observed - self.means_obs[k]             # (D_obs,)
            Σ11_inv = torch.linalg.inv(self.cov_obs[k])       # (D_obs, D_obs)
            μ2 = self.means_unobs[k]                          # (D_unobs,)
            Σ21 = self.cov_cross[k]                           # (D_unobs, D_obs)
            Σ22 = self.cov_unobs[k]                           # (D_unobs, D_unobs)

            # 条件均值 μ₂ + Σ₂₁ Σ₁₁⁻¹ (x_obs - μ₁)
            m = μ2 + Σ21 @ (Σ11_inv @ diff)

            # 条件协方差 Σ₂₂ - Σ₂₁ Σ₁₁⁻¹ Σ₁₂
            C = Σ22 - Σ21 @ (Σ11_inv @ Σ21.T)
            # 对称化 + 微正则
            C = 0.5 * (C + C.T) + 1e-6 * I

            cond_means.append(m)
            cond_covs.append(C)

        # 2) 为每个样本抽分量
        comp_ids = torch.multinomial(self.weights, num_samples, replacement=True)
        unique_ids, counts = torch.unique(comp_ids, return_counts=True)

        # 3) 逐分量 batch 采样，然后按块拼接
        blocks = []
        for k, cnt in zip(unique_ids.tolist(), counts.tolist()):
            if cnt == 0:
                continue

            m, C = cond_means[k], cond_covs[k]
            try:
                L = torch.linalg.cholesky(C)                # 下三角
                z = torch.randn(cnt, D_unobs, device=self.device)
                samples = m + z @ L.T
            except torch.linalg.LinAlgError:
                # 回退到对角近似
                std = torch.sqrt(torch.diag(C)).clamp_min(1e-6)
                z = torch.randn(cnt, D_unobs, device=self.device)
                samples = m + z * std

            blocks.append(samples)  # shape = (cnt, D_unobs)

        # 4) 把所有 unobs 部分拼起来
        x_full = torch.zeros(num_samples, self.problem.input_dim, device=self.device)
        x_full[:, self.obs_idx] = x_observed

        x_unobs = torch.cat(blocks, dim=0)   # (num_samples, D_unobs)

        x_full[:, self.unobs_idx] = x_unobs

        return x_full


    def _validate_covariance(self, matrix):
        """Ensure covariance matrix is valid"""
        if not torch.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError("Covariance matrix must be symmetric")
        eigvals = torch.linalg.eigvalsh(matrix)
        if (eigvals <= 0).any():
            raise ValueError(f"Non-positive definite matrix. Min eigenvalue: {eigvals.min()}")

    def _uniform_sample(self, size):
        """Generate uniform samples with gradient tracking"""
        return Variable(
            torch.rand(size) * (self.problem.x_max - self.problem.x_min) 
            + self.problem.x_min,
            requires_grad=True
        ).to(self.device)

class convert_nfdist(BaseDistribution):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def log_prob(self, z, context=None):

        return self.dist.log_prob(z)

    def sample(self, num_samples=1, context=None):

        return self.dist.sample((num_samples,))

    def forward(self, num_samples=1, context=None):

        z = self.sample(num_samples)
        log_prob = self.log_prob(z)
        return z, log_prob  
class DNNDataset(Dataset):
    """Dataset for DNN training with dual sampling modes"""
    def __init__(self, base_dataset, sample_size, use_simulation = True):
        self.base = base_dataset
        self.sample_size = sample_size
        self.condition_sample_size = self.base.problem.condition_sample_size
        self.use_simulation = use_simulation
        self.x = None
        self.f = None
        self.fx = None
        self.pdf = None
        self._prepare()
        self.f_proj = None
    def _prepare(self):
        """Lazy initialization based on sampling mode"""
        if self.use_simulation:
            if self.base.problem.input_dim > 2:
                self.x_full = self.base.get_simulated_data(self.sample_size, 
                                                    self.base.problem.noise_strength)
                weights, means, covs = self.base.gmm_components
                pca = PCA()
                Z = pca.fit_transform(means.detach().cpu().numpy())
                self.base.pca_mean = torch.tensor(pca.mean_).to(self.base.device)
                self.base.pca_w = torch.tensor(pca.components_).to(self.base.device)
                self.f_full = self.base.force(self.x_full)
                self.x_full = (self.x_full - self.base.pca_mean) @ self.base.pca_w.T
                self.f_full = self.f_full @ self.base.pca_w.T

                self.x = self.x_full[:, [self.base.index_1, self.base.index_2]].detach()
                self.x.requires_grad_(True)
                dim1 = self.x_full[:, self.base.index_1].detach().cpu().numpy()
                dim2 = self.x_full[:, self.base.index_2].detach().cpu().numpy()
                self.base.min_project = np.min([np.min(dim1), np.min(dim2)])
                self.base.max_project = np.max([np.max(dim1), np.max(dim2)])
                # 绘制散点图
                plt.scatter(dim1, dim2, alpha=0.3, s=5)
                plt.xlabel(f'Dim {self.base.index_1}')
                plt.ylabel(f'Dim {self.base.index_2}')
                plt.show()
            else:
                self.x = self.base.get_simulated_data(self.sample_size, 
                                                    self.base.problem.noise_strength)
                # Calculate force values
                self.f = self.base.force(self.x)
                self.fx = self._calculate_divergence(self.f, self.x)
        else:
            if self.base.problem.input_dim > 2:
                self._prepare_high_dim()
            else:
                self.x = self.base._uniform_sample((self.sample_size, 
                                                self.base.problem.input_dim))
                self.pdf = torch.exp(self.base.mix.log_prob(self.x))

                # Calculate force values
                self.f = self.base.force(self.x)
                self.fx = self._calculate_divergence(self.f, self.x)

    def _prepare_high_dim(self):
        # self.x = self.base._uniform_sample((self.sample_size, 2))
        # self.pdf = torch.exp(self.base.mix.log_prob(self.x))
        # self.x_full = torch.zeros((self.sample_size, self.condition_sample_size, self.base.problem.input_dim)).to(self.base.device)
        # self.f_full = torch.zeros((self.sample_size, self.condition_sample_size, self.base.problem.input_dim)).to(self.base.device)
        # for i in tqdm(range(self.sample_size)):
        #     self.x_full[i, :, :] = self.base._sample_conditional(self.x[i], self.condition_sample_size)
        #     self.f_full[i, :, :] = self.base.force(self.x_full[i, :, :])
        self.x_full = self.base.mix.sample((self.sample_size,))
        self.f_full = self.base.force(self.x_full)
        self.x_full = (self.x_full - self.base.pca_mean) @ self.base.pca_w.T
        self.f_full = self.f_full @ self.base.pca_w.T
        self.x = self.x_full[:, [self.base.index_1, self.base.index_2]].detach()
        self.x.requires_grad_(True)
        
    def _calculate_divergence(self, f, x):
        """Compute ∇·f for the force field"""
        div = torch.zeros(x.size(0), device=self.base.device)
        for i in range(f.shape[1]):
            grad_i = torch.autograd.grad(
                f[:, i].sum(), x, 
                retain_graph=True, create_graph=False
            )[0][:, i]
            div += grad_i
        return div.unsqueeze(1)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        if self.base.problem.input_dim == 2:
            item = {'x': self.x[idx], 'f': self.f[idx], 'fx': self.fx[idx]}
            if self.pdf is not None:
                item['pdf'] = self.pdf[idx]
        else:
            item = {'x': self.x[idx], 'x_full': self.x_full[idx], 'f_full': self.f_full[idx]}
            if self.f is not None:
                item['f'] = self.f[idx]
                item['fx'] = self.fx[idx]
            if self.pdf is not None:
                item['pdf'] = self.pdf[idx]
        return item

    def update_pdf(self, flow_model):
        """Update density estimates using flow model"""
        if self.base.problem.input_dim == 2:
            with torch.no_grad():
                self.pdf = torch.exp(flow_model.log_prob(self.x))
        else:
            self.x_full, _ = flow_model.sample(self.sample_size)
            self.f_full = self.base.force(self.x_full)
            self.x_full = (self.x_full - self.base.pca_mean) @ self.base.pca_w.T
            self.f_full = self.f_full @ self.base.pca_w.T
            self.x = self.x_full[:, [self.base.index_1, self.base.index_2]].detach()
            self.x.requires_grad_(True)
            self.force_proj()
    def force_proj(self):
        self.f = self.f_proj(self.x)
        self.fx = self._calculate_divergence(self.f, self.x)

class FlowMLEDataset(Dataset):
    """Dataset for Flow maximum likelihood estimation"""
    def __init__(self, base_dataset, sample_size):
        self.base = base_dataset
        self.sample_size = sample_size
        self.data = None
        self._prepare()
    def _prepare(self):

        self.data = self.base.get_simulated_data(
            self.sample_size, 
            self.base.problem.noise_strength
        )

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # if self.base.problem.input_dim == 2:
        return {'x': self.data[idx], 'mode': 'mle'}
        # else:
        #     return {'x': self.data[idx, [self.base.index_1, self.base.index_2]], 'mode': 'mle'}

class FlowConstraintDataset(Dataset):
    """Dataset for Flow physical constraint training"""
    def __init__(self, base_dataset, sample_size):
        self.base = base_dataset
        self.sample_size = sample_size
        self.x = None
        self.targets = None
        self._prepare()

    def _prepare(self):
        # self.x = self.base._uniform_sample(
        #     (self.sample_size, self.base.problem.input_dim)
        # )
        samples = self.base.mix.sample((self.sample_size,))
        self.x = samples.to(self.base.device).requires_grad_(True)


    def update_targets(self, dnn):
        """Update constraint targets from DNN"""
        if self.x is None:
            self._prepare()
        with torch.no_grad():
            self.targets = dnn(self.x).detach()

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        item = {'x': self.x[idx], 'mode': 'constraint'}
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item

class HybridDataset(Dataset):
    """
    Coordinator dataset for managing multiple training components with mode awareness
    
    Attributes:
        mode (str): Current training mode, one of ['hybrid', 'dnn_only', 'flow_only']
        dnn (Dataset|None): DNN training dataset
        flow_mle (Dataset|None): Flow MLE training dataset
        flow_constraint (Dataset|None): Flow constraint training dataset
    """
    def __init__(self, dnn_dataset=None, flow_mle_dataset=None, flow_constraint_dataset=None):
        """
        Initialize with optional components
        
        Args:
            dnn_dataset: Optional DNN training dataset
            flow_mle_dataset: Optional Flow MLE training dataset
            flow_constraint_dataset: Optional Flow constraint dataset
        """
        self.dnn = dnn_dataset
        self.flow_mle = flow_mle_dataset
        self.flow_constraint = flow_constraint_dataset
        self.mode = self._detect_mode()

    def _detect_mode(self):
        """Automatically detect training mode based on available components"""
        has_dnn = self.dnn is not None
        has_flow = self.flow_mle is not None
        
        if has_dnn and has_flow:
            return 'hybrid'
        if has_dnn:
            return 'dnn_only'
        if has_flow:
            return 'flow_only'
        raise ValueError("Invalid dataset configuration - must have at least one component")

    def get_loader(self, component, batch_size, shuffle=True):
        """
        Get DataLoader for specific component with safety checks
        
        Args:
            component: One of ['dnn', 'flow_mle', 'flow_constraint']
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for requested component
            
        Raises:
            ValueError: If requested component is unavailable in current mode
        """
        # Validate component availability
        if component == 'dnn' and self.dnn is None:
            raise ValueError("DNN component not available in current mode")
        if component == 'flow_mle' and self.flow_mle is None:
            raise ValueError("Flow MLE component not available in current mode")
        if component == 'flow_constraint' and self.flow_constraint is None:
            raise ValueError("Flow constraint component not available in current mode")

        # Get appropriate dataset
        dataset = getattr(self, component)
        return DataLoader(dataset, batch_size, shuffle=shuffle)

    def update_dependencies(self, dnn_model=None, flow_model=None):
        """Update cross-component dependencies if components exist"""
        if flow_model is not None and self.dnn is not None:
            self.dnn.update_pdf(flow_model)
        # if dnn_model is not None and self.flow_constraint is not None:
        #     self.flow_constraint.update_targets(dnn_model)