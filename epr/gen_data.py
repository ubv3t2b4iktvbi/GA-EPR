import math
import os
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
        self.simulation_data_x = None
        #self.simulation_data_y = None

        # GMM parameters cache
        self._gmm_params = None  
        self._mix = None
        self._q0_dist = None     # Converted flow distribution

        # Optimization: pre-allocate 20% more than typical batch size
        self.base_batch_size = max(1024, int(args.base_batch_size * 1.2))

        # Store projection indices
        self.index_1 = getattr(self.problem, 'index_1', 0)
        self.index_2 = getattr(self.problem, 'index_2', 1)
        self._ed_test_done = False

    # ---------- Goodness-of-fit helpers (Energy distance) ----------
    def _energy_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Energy distance statistic (larger => more different).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, m = x.shape[0], y.shape[0]
        d_xy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2).sum()
        d_xx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2).sum()
        d_yy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2).sum()
        return (2.0 / (n * m)) * d_xy - (1.0 / (n * n)) * d_xx - (1.0 / (m * m)) * d_yy

    def _normalize_points(self, pts: np.ndarray, ref_pts: np.ndarray = None) -> np.ndarray:
        """Normalize 2D points to [0, 1] using ref_pts min/max (default: pts)."""
        pts = np.asarray(pts, dtype=np.float64)
        ref = pts if ref_pts is None else np.asarray(ref_pts, dtype=np.float64)
        x_min, x_max = ref[:, 0].min(), ref[:, 0].max()
        y_min, y_max = ref[:, 1].min(), ref[:, 1].max()
        denom_x = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        denom_y = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = (pts[:, 0] - x_min) / denom_x
        out[:, 1] = (pts[:, 1] - y_min) / denom_y
        return out

    def _sample_mix_bounded(self, mix, num_samples, max_attempts=200, y_offset=0.0):
        """Sample from mix constrained to [samp_x_min,x_max]x[samp_y_min,y_max]."""
        x_min = self.problem.samp_x_min
        x_max = self.problem.samp_x_max
        y_min = self.problem.samp_y_min
        y_max = self.problem.samp_y_max

        samples = []
        remaining = num_samples
        attempts = 0
        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            with torch.no_grad():
                batch = mix.sample((max(remaining * 2, 50),)).cpu().numpy()
            mask = (
                (batch[:, 0] >= x_min) & (batch[:, 0] <= x_max) &
                (batch[:, 1] >= y_min) & (batch[:, 1] <= y_max)
            )
            if np.any(mask):
                accepted = batch[mask]
                take = accepted[:remaining]
                samples.append(take)
                remaining -= take.shape[0]

        if remaining > 0:
            # Fallback: clip to ensure bounds
            with torch.no_grad():
                batch = mix.sample((remaining,)).cpu().numpy()
            batch[:, 0] = np.clip(batch[:, 0], x_min, x_max)
            batch[:, 1] = np.clip(batch[:, 1], y_min, y_max)
            samples.append(batch)

        out = np.concatenate(samples, axis=0)
        if y_offset != 0.0:
            out[:, 1] = out[:, 1] + y_offset
        return out

    def _energy_distance_test(self, real_points, ref_sample_size=2000, num_bootstrap=200,
                              mix=None, test_mode="fixed_x", normalize=False):
        """
        Right-tailed MC test: H0: real_points ~ self._mix.
        Returns (statistic, p_value, null_stats).
        """
        x = np.asarray(real_points, dtype=np.float64)
        ref_pts = x
        if normalize:
            x = self._normalize_points(x, ref_pts=ref_pts)
        if mix is None:
            mix = self._mix
        with torch.no_grad():
            y_ref = self._sample_mix_bounded(mix, ref_sample_size, y_offset=0.0)
        if normalize:
            y_ref = self._normalize_points(y_ref, ref_pts=ref_pts)
        stat_obs = self._energy_distance(x, y_ref)

        null_stats = []
        for _ in range(num_bootstrap):
            with torch.no_grad():
                yb = self._sample_mix_bounded(mix, ref_sample_size)
                if test_mode == "two_sample":
                    xb = self._sample_mix_bounded(mix, x.shape[0])
                else:
                    xb = x
            if normalize:
                yb = self._normalize_points(yb, ref_pts=ref_pts)
                if test_mode == "two_sample":
                    xb = self._normalize_points(xb, ref_pts=ref_pts)
            null_stats.append(self._energy_distance(xb, yb))
        null_stats = np.asarray(null_stats)
        p_val = (np.sum(null_stats >= stat_obs) + 1.0) / (num_bootstrap + 1.0)
        return stat_obs, p_val, null_stats

    def _energy_distance_similarity_test(self, real_points, ref_sample_size=2000,
                                         baseline_samples=500, num_bootstrap=200,
                                         mix=None, alpha=0.05, gamma=0.1, delta_c=0.1,
                                         normalize=False):
        """
        Similarity test with baseline noise:
          H0: T(P, P0) >= A + delta  vs  H1: T(P, P0) < A + delta
        A = Quantile_{1-gamma}(T0),  T0 = T(X0, Y0), X0~P0, Y0~P0.
        delta = c * A.
        Returns (t_obs, upper_ci, A, delta, reject_h0).
        """
        x = np.asarray(real_points, dtype=np.float64)
        ref_pts = x
        if normalize:
            x = self._normalize_points(x, ref_pts=ref_pts)
        if mix is None:
            mix = self._mix
        with torch.no_grad():
            y_ref = self._sample_mix_bounded(mix, ref_sample_size, y_offset=0.0)
        if normalize:
            y_ref = self._normalize_points(y_ref, ref_pts=ref_pts)
        t_obs = self._energy_distance(x, y_ref)

        # Step 3: baseline noise A from same-distribution samples
        baseline = []
        n = x.shape[0]
        for _ in range(baseline_samples):
            with torch.no_grad():
                x0 = mix.sample((n,)).cpu().numpy()
                y0 = mix.sample((ref_sample_size,)).cpu().numpy()
            if normalize:
                x0 = self._normalize_points(x0, ref_pts=ref_pts)
                y0 = self._normalize_points(y0, ref_pts=ref_pts)
            baseline.append(self._energy_distance(x0, y0))
        baseline = np.asarray(baseline)
        A = np.quantile(baseline, 1.0 - gamma)
        delta = delta_c * A

        # Step 4: upper CI for T(X, Y_ref) via resampling Y_ref (X fixed)
        boot_stats = []
        for _ in range(num_bootstrap):
            with torch.no_grad():
                yb = self._sample_mix_bounded(mix, ref_sample_size)
            if normalize:
                yb = self._normalize_points(yb, ref_pts=ref_pts)
            boot_stats.append(self._energy_distance(x, yb))
        boot_stats = np.asarray(boot_stats)
        upper_ci = np.quantile(boot_stats, 1.0 - alpha)
        reject_h0 = upper_ci < (A + delta)
        return t_obs, upper_ci, A, delta, reject_h0

    def _energy_distance_diff_test(self, real_points, ref_sample_size=200, num_bootstrap=200,
                                   mix=None, normalize=False):
        """
        Difference test:
          H0: X ~ P0  vs  H1: X !~ P0 (right tail).
        t_obs = T(X, Y), Y~P0; null from Xb~P0, Yb~P0.
        Returns (t_obs, p_value).
        """
        x = np.asarray(real_points, dtype=np.float64)
        ref_pts = x
        if normalize:
            x = self._normalize_points(x, ref_pts=ref_pts)
        if mix is None:
            mix = self._mix
        
        with torch.no_grad():
            y = self._sample_mix_bounded(mix, ref_sample_size)
        if normalize:
            y = self._normalize_points(y, ref_pts=ref_pts)
        t_obs = self._energy_distance(x, y)

        null_stats = []
        for _ in range(num_bootstrap):
            with torch.no_grad():
                xb = self._sample_mix_bounded(mix, x.shape[0])
                yb = self._sample_mix_bounded(mix, ref_sample_size)
            if normalize:
                xb = self._normalize_points(xb, ref_pts=ref_pts)
                yb = self._normalize_points(yb, ref_pts=ref_pts)
            null_stats.append(self._energy_distance(xb, yb))
        null_stats = np.asarray(null_stats)
        p_val = (np.sum(null_stats >= t_obs) + 1.0) / (num_bootstrap + 1.0)
        return t_obs, p_val

    def print_wsga_energy_distance_test(self, real_points=None, ref_sample_size=200, num_bootstrap=200, force_run=False):
        """
        Print energy distance test result during main.py execution.
        """
        if self.problem.input_dim != 2:
            print("[EnergyDistance] skipped: input_dim != 2")
            return
        if self._ed_test_done and not force_run:
            return
        test_mode = getattr(self.args, "energy_test_mode", "fixed_x")
        alpha = getattr(self.args, "energy_test_alpha", 0.05)
        gamma = getattr(self.args, "energy_test_gamma", 0.1)
        delta_c = getattr(self.args, "energy_test_delta_c", 0.1)
        normalize = True
        if test_mode == "similarity":
            t_obs, upper_ci, A, delta, reject_h0 = self._energy_distance_similarity_test(
                real_points,
                ref_sample_size=ref_sample_size,
                num_bootstrap=num_bootstrap,
                alpha=alpha,
                gamma=gamma,
                delta_c=delta_c,
                normalize=normalize,
            )
            verdict = "REJECT H0 (close enough)" if reject_h0 else "FAIL TO REJECT H0 (not close enough)"
            print(f"[EnergyDistance][similarity] T_obs={t_obs:.4f}, U_1-a={upper_ci:.4f}, "
                  f"A={A:.4f}, delta={delta:.4f}, alpha={alpha:.3f}, gamma={gamma:.3f} => {verdict}")
        else:
            ed_stat, ed_p, _ = self._energy_distance_test(
                real_points,
                ref_sample_size=ref_sample_size,
                num_bootstrap=num_bootstrap,
                test_mode=test_mode,
                normalize=normalize,
            )
            print(f"[EnergyDistance][{test_mode}] T_ed={ed_stat:.4f}, right-tailed p={ed_p:.4f} "
                  f"(H0: real_points ~ wsga mix)")

        if getattr(self.args, "energy_test_run_diff", True):
            t_obs_d, p_val_d = self._energy_distance_diff_test(
                real_points,
                ref_sample_size=ref_sample_size,
                num_bootstrap=num_bootstrap,
                normalize=normalize,
            )
            print(f"[EnergyDistance][diff] T_obs={t_obs_d:.4f}, right-tailed p={p_val_d:.4f} "
                  f"(H0: real_points ~ wsga mix)")
        self._ed_test_done = True

    def get_simulated_data(self, required_size, noise_strength):
        """
        Get simulated data with pre-simulation optimization:
        - Generate all needed data once on first call
        - Return subsets as needed for efficiency
        """
        # Lazy initialization of the full simulation
        # if self.simulation_data_x or self.simulation_data_y is None:
        if self.simulation_data is None:
            print(f"Pre-generating {self.max_required_size} simulation samples...")
            # self.simulation_data_x, self.simulation_data_y, self.simulation_data = self._run_simulation(
            #     self.max_required_size, 
            #     noise_strength
            # )
            self.simulation_data = self._run_simulation(
                self.max_required_size, 
                noise_strength
            )
            # self.simulation_data_x.requires_grad_(True)
            # self.simulation_data_y.requires_grad_(True)
            self.simulation_data.requires_grad_(True)
            print("Simulation complete and cached.")
        return self.simulation_data[:required_size]

    def _run_simulation(self, batch_size, noise_strength):
        """Core SDE simulation with progress tracking"""
        x = self._uniform_sample((batch_size, self.problem.input_dim))
        z_dist = MultivariateNormal(
            torch.zeros(self.problem.input_dim, device=self.device),
            torch.eye(self.problem.input_dim, device=self.device)
        ).expand((batch_size,))#为每个样本准备的噪声分布 z_dist
        #padding = 0.1 * (self.problem.x_max - self.problem.x_min)
        padding = 0
        # 使用对应维度的范围设置边界
        # self.bound_min = self.problem.sim_min - padding
        # self.bound_max = self.problem.sim_max + padding
        self.bound_min_x = self.problem.x_min - padding
        self.bound_max_x = self.problem.x_max + padding
        self.bound_min_y = self.problem.y_min - padding
        self.bound_max_y = self.problem.y_max + padding
        with tqdm(total=self.args.sim_steps, desc=f'Simulating {batch_size} samples') as pbar:
            for step in range(self.args.sim_steps):
                # x, y, sim= self._sde_step(x, z_dist, noise_strength)
                x = self._sde_step(x, z_dist, noise_strength)
                pbar.set_description(f'Simulating {batch_size} samples (Step {step+1}/{self.args.sim_steps})')  
                pbar.update(1)
        # return x , y , sim
        return x

    def _sde_step(self, x, z_dist, noise_strength):
        """Single SDE integration step with boundary handling"""
        with torch.no_grad():
            dt = self.args.sim_dt
            z = z_dist.sample()
            x_new = x + dt*self.force(x) + math.sqrt(2*noise_strength*dt)*z
        
            # Reflect boundaries using corresponding dimension ranges
            if self.problem.input_dim >= 2:
                # 第一维度使用x_min和x_max
                x_new[:, 0] = torch.where(x_new[:, 0] < self.problem.x_min, 
                                          2*self.problem.x_min - x_new[:, 0], x_new[:, 0])
                x_new[:, 0] = torch.where(x_new[:, 0] > self.problem.x_max,
                                          2*self.problem.x_max - x_new[:, 0], x_new[:, 0])
                
                # 第二维度使用y_min和y_max
                x_new[:, 1] = torch.where(x_new[:, 1] < self.problem.y_min, 
                                          2*self.problem.y_min - x_new[:, 1], x_new[:, 1])
                x_new[:, 1] = torch.where(x_new[:, 1] > self.problem.y_max,
                                          2*self.problem.y_max - x_new[:, 1], x_new[:, 1])
                
                # 其他维度使用x_min和x_max
                if self.problem.input_dim > 2:
                    for i in range(2, self.problem.input_dim):
                        x_new[:, i] = torch.where(x_new[:, i] < self.problem.x_min, 
                                                  2*self.problem.x_min - x_new[:, i], x_new[:, i])
                        x_new[:, i] = torch.where(x_new[:, i] > self.problem.x_max,
                                                  2*self.problem.x_max - x_new[:, i], x_new[:, i])
            else:
                # 单维度情况
                x_new = torch.where(x_new < self.problem.x_min, 
                                    2*self.problem.x_min - x_new, x_new)
                x_new = torch.where(x_new > self.problem.x_max,
                                    2*self.problem.x_max - x_new, x_new)
                
        return x_new.detach()
    
    def _init_grid_data(self):
        """Initialize visualization grid data"""
        if self.problem.input_dim > 2:
            X = np.linspace(self.dnn_dataset.base.min_project-1, self.dnn_dataset.base.max_project+1, 501)
            Y = np.linspace(self.dnn_dataset.base.min_project-1, self.dnn_dataset.base.max_project+1, 501)
        else:
            X = np.linspace(self.problem.x_min, self.problem.x_max, 501)
            Y = np.linspace(self.problem.y_min, self.problem.y_max, 501)
        self.x_grid, self.y_grid = np.meshgrid(X, Y)
        self.grid_tensor = torch.tensor(
            np.column_stack([self.x_grid.ravel(), self.y_grid.ravel()]), 
            dtype=torch.float32, device=self.device)

    @property
    def gmm_components(self):
        """Lazy-loaded GMM parameters with validation"""
        if self._gmm_params is None:
            self._gmm_params = self._calculate_gmm()
        return self._gmm_params
    
    def _save_viz(self, suffix=''):
        """Save visualization without relying on global attributes"""
        # 简单地保存到当前工作目录，带有一个基本的文件名
        filename = f"gmm_visualization{suffix}.png"
        plt.savefig(filename)
        plt.close()

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
                save_dir = r'D:/GA-EPR/GA-EPR/results/ToggleBasic'
                samples = self._mix.sample((10000,)).cpu().numpy()
                plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
                plt.title('2D Gaussian Mixture Model Samples')
                plt.xlabel('X1')
                plt.ylabel('X2')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"wsga_gmm_samples_2d_growth_rate_{self.force.gr}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"[Saved] 2D GMM figure saved to: {save_path}")

                self._init_grid_data()
                landscape = self._mix.log_prob(self.grid_tensor).cpu().numpy().reshape(501, 501)
                landscape = landscape*self.problem.noise_strength*(-1)
                
                #坐标伸缩
                # 创建映射后的x_grid，将原始范围[self.problem.x_min, self.problem.x_max]映射到[30, 120]
                original_x_range = self.problem.x_max - self.problem.x_min
                target_x_min, target_x_max = self.problem.x_min, self.problem.x_max
                #target_x_min, target_x_max = 100, 310
                #target_x_min, target_x_max = 10, 100
                target_x_range = target_x_max - target_x_min
                # 创建映射后的y_grid
                original_y_range = self.problem.y_max - self.problem.y_min
                target_y_min, target_y_max = self.problem.y_min, self.problem.y_max
                #target_y_min, target_y_max = 0, 300
                #target_y_min, target_y_max = -20, 170
                target_y_range = target_y_max - target_y_min
                # 计算映射后的x_grid
                mapped_x_grid = ((self.x_grid - self.problem.x_min) / original_x_range) * target_x_range + target_x_min
                mapped_y_grid = ((self.y_grid - self.problem.y_min) / original_y_range) * target_y_range + target_y_min
                
                # 平移） 改
                mapped_x_grid = mapped_x_grid + 0
                mapped_y_grid = mapped_y_grid + 0
                
                ax = plt.axes()
                color_map = 'rainbow'
                surf = ax.pcolormesh(mapped_x_grid, mapped_y_grid, landscape, 
                                    cmap=color_map, shading='auto')
                ax.contour(mapped_x_grid, mapped_y_grid, landscape, 50, cmap=color_map)
                ax.set_title(f'wsga Landscape')
                ax.set_aspect('auto')
                # 显式设置坐标轴范围，确保正确显示 改
                ax.set_xlim(target_x_min + 0, target_x_max + 0)
                ax.set_ylim(target_y_min + 0, target_y_max + 0)  # 同时更新y轴范围

                real_points= [
                            (137.8, 18),
                            (219.5, 24.2),
                            (246.5, 30.1),
                            (297, 38),
                            (321.4, 48.4),
                            (298, 62.5),
                            (138.4, 10.8),
                            (184.5, 17.7),
                            (273.9, 22.4),
                            (337.7, 28.8),
                            (224.4, 67.4),
                            (208.9, 53.3),
                            (221.3, 47.6),
                            (274.3, 24.9),
                        ]
                # Energy distance test: delegate to unified printer (respects test_mode)
                if not self._ed_test_done:
                    self.print_wsga_energy_distance_test(
                        real_points=real_points,
                        force_run=True,
                    )
                
                for point_x, point_y in real_points:
                    ax.scatter(point_x, point_y, color='black', s=50, zorder=5)
                    # 在点旁边标注坐标值
                    ax.annotate(f'({point_x:.1f}, {point_y:.1f})', 
                               xy=(point_x, point_y), 
                               xytext=(3, 3),  # 减少偏移量
                               textcoords='offset points',
                               fontsize=6,      # 减小字体大小
                               alpha=0.7,
                               bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.5))  # 添加半透明背景框
                
                plt.colorbar(surf, shrink=0.5)
                
                # # 绘制模拟数据点
                # samples = self.simulation_data.cpu().numpy()
                # plt.scatter(samples[:, self.index_1], samples[:, self.index_2], s=0.1, c='k')
            
                self._save_viz(suffix=f'wsga_{self.force.gr}')
                
                # 添加俯视图
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(mapped_x_grid, mapped_y_grid, landscape, cmap='rainbow', antialiased=True)
                
                # 设置视角：仰角60度，选择较清晰的观测方位
                ax.view_init(elev=60, azim=-150)
                

                # 为每个真实数据点找到对应的z值（势能值）并加1
                for point_x, point_y in real_points:
                    # 找到最接近的网格点索引
                    x_idx = np.argmin(np.abs(mapped_x_grid[0, :] - point_x))
                    y_idx = np.argmin(np.abs(mapped_y_grid[:, 0] - point_y))
                    
                    # 获取对应位置的势能值并加1
                    z_value = landscape[y_idx, x_idx] + 100
                    
                    # 在3D图上绘制点，使用更鲜艳的颜色和更大的尺寸
                    ax.scatter(point_x, point_y, z_value, color='black', s=50, label=f'({point_x}, {point_y})')
                
                # 设置坐标轴范围
                x_range = [mapped_x_grid.min(), mapped_x_grid.max()]
                y_range = [mapped_y_grid.min(), mapped_y_grid.max()]  # 使用映射后的范围
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                
                # 设置字体属性
                ax.tick_params(axis='both', which='major', labelsize=20)
                ax.xaxis.line.set_linewidth(1.5)
                ax.yaxis.line.set_linewidth(1.5)
                ax.zaxis.line.set_linewidth(1.5)
                
                # 设置背景颜色
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                
                # 保存俯视图
                self._save_viz(suffix=f'wsga_top_{self.force.gr}')
                plt.show()
                plt.close()
                
        return self._mix
    @property
    def q0(self):
        if self._q0_dist is None:
            self._q0_dist = convert_nfdist(self.mix)
        return self._q0_dist   #这个类里有从GMM取的点和点的log_prob(z)
    
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
        self.cov_unobs = covs[:, self.unobs_idx][:, :, self.unobs_idx] #如何提取Σ_22
        
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
        # 支持不同维度使用不同的范围
        if len(size) == 2 and size[1] >= 2:
            # 对于多维情况，分别使用各自维度的范围
            samples = torch.zeros(size)
            # 第一维度使用x_min和x_max
            samples[:, 0] = torch.rand(size[0]) * (self.problem.x_max - self.problem.x_min) + self.problem.x_min
            # 第二维度使用y_min和y_max（如果存在）
            if size[1] > 1:
                samples[:, 1] = torch.rand(size[0]) * (self.problem.y_max - self.problem.y_min) + self.problem.y_min
                # 其他维度如果存在，继续使用x_min和x_max
                if size[1] > 2:
                    samples[:, 2:] = torch.rand(size[0], size[1]-2) * (self.problem.x_max - self.problem.x_min) + self.problem.x_min
            samples = samples.to(self.device)
        else:
            # 原始逻辑，所有维度使用x_min和x_max
            samples = (torch.rand(size) * (self.problem.x_max - self.problem.x_min) + self.problem.x_min).to(self.device)
        
        return Variable(samples, requires_grad=True)

#交叉训练适配器
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
                dim1 = self.x[:, self.base.index_1].detach().cpu().numpy()
                dim2 = self.x[:, self.base.index_2].detach().cpu().numpy()

                # 绘制散点图
                plt.scatter(dim1, dim2, alpha=0.3, s=5)
                plt.xlabel(f'Dim {self.base.index_1}')
                plt.ylabel(f'Dim {self.base.index_2}')
                plt.show()
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
                # # Normalize the PDF values so they sum to 1
                # self.pdf_norm = self.pdf / torch.sum(self.pdf) * self.sample_size

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

#准备优化参数 θ
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


#Flow训练完了用DNN训练，采样一批状态点x，并计算 DNN 在这些点的输出𝑓𝜃(X)作为物理约束
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
            self.targets = (self.x).detach()

    def __len__(self):
        return self.sample_size


#如果已经调用过update_targets()，后续训练时Flow 可以根据 target 计算物理约束损失
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
        if dnn_model is not None and self.flow_constraint is not None:
            self.flow_constraint.update_targets(dnn_model)
