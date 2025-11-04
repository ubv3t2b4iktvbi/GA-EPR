import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from gen_data import *
import normflows as nf

class EnergyLandscape:
    """Main class for training energy landscape models with hybrid strategies"""
    
    def __init__(self, args, problem, network, force_fn):
        """Initialize the complete training system"""
        # Core components
        self.args = args
        self.problem = problem
        self.network = network
        self.force_fn = force_fn
        self.device = args.device
        self.global_step = 0  # Unified training step counter
        
        # Store projection indices
        self.index_1 = getattr(self.problem, 'index_1', 0)
        self.index_2 = getattr(self.problem, 'index_2', 1)

        # Training infrastructure
        self._init_datasets()
        self._init_optimizers()
        self._init_hybrid_scheduler()
        
        # Support modules
        self._init_grid_data()
        self._init_logging_system()
        self._create_directories()

    def _init_datasets(self):
        """Initialize all dataset components"""
        # Base dataset for sharing common resources
        self.base_dataset = SharedBaseDataset(self.args, self.problem, self.force_fn)
        
        # Component-specific datasets
        self.dnn_dataset = None
        self.flow_mle_dataset = None
        self.flow_constraint_dataset = None

        # Dataset initialization based on training mode
        if self.args.train_mode in ['hybrid', 'dnn_only']:
            self.dnn_dataset = DNNDataset(
                self.base_dataset,
                self.problem.dnn_sample_size,
                use_simulation=(self.args.pdf_method == 'Simulate'))
            
        if self.args.train_mode in ['hybrid', 'flow_only']:
            self.flow_mle_dataset = FlowMLEDataset(
                self.base_dataset,
                self.problem.flow_sample_size)
            
        # if self.args.train_mode == 'hybrid':
        #     self.flow_constraint_dataset = FlowConstraintDataset(
        #         self.base_dataset,
        #         self.problem.flow_constraint_sample_size)

        # Hybrid dataset coordinator
        self.hybrid_dataset = HybridDataset(
            dnn_dataset=self.dnn_dataset,
            flow_mle_dataset=self.flow_mle_dataset,
            flow_constraint_dataset=self.flow_constraint_dataset)

    def _init_optimizers(self):
        """Initialize optimizers for different components"""
        self.dnn_optim = torch.optim.Adam(
            self.network.dnn.parameters(),
            lr=self.args.dnn_lr,
            weight_decay=self.args.weight_decay)
        
        self.flow_optim = torch.optim.Adam(
            self.network.flow.parameters(),
            lr=self.args.flow_lr,
            weight_decay=self.args.weight_decay)

    def _init_hybrid_scheduler(self):
        """Initialize state tracking for hybrid training"""
        self._current_dnn_steps = 0
        self._current_flow_steps = 0
        self._update_counter = 0  
        # Explicit training phase tracking
        self._training_phase = 'dnn'

    def _init_grid_data(self):
        """Initialize visualization grid data"""
        if self.problem.input_dim > 2:
            X = np.linspace(self.dnn_dataset.base.min_project-1, self.dnn_dataset.base.max_project+1, 501)
            Y = np.linspace(self.dnn_dataset.base.min_project-1, self.dnn_dataset.base.max_project+1, 501)
        else:
            X = np.linspace(self.problem.x_min, self.problem.x_max, 501)
            Y = np.linspace(self.problem.x_min, self.problem.x_max, 501)
        self.x_grid, self.y_grid = np.meshgrid(X, Y)
        self.grid_tensor = torch.tensor(
            np.column_stack([self.x_grid.ravel(), self.y_grid.ravel()]), 
            dtype=torch.float32, device=self.device)

    def _init_logging_system(self):
        """Configure complete logging infrastructure"""
        self.log_dir = os.path.join(self.args.prefix, 'logs')
        self._init_csv_loggers()

    def _init_csv_loggers(self):
        """Initialize CSV log files with headers"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training metrics log
        self.train_log_path = os.path.join(self.log_dir, 'train_metrics.csv')
        with open(self.train_log_path, 'w') as f:
            f.write('global_step,phase,dnn_loss,flow_mle_loss,flow_constraint_loss\n')
        
        # Validation metrics log
        self.val_log_path = os.path.join(self.log_dir, 'val_metrics.csv')
        with open(self.val_log_path, 'w') as f:
            f.write('global_step,phase,dnn_loss,flow_mle_loss\n')

    def _create_directories(self):
        """Create required output directories"""
        os.makedirs(os.path.join(self.log_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)

    def train(self):
        """Orchestrate complete training workflow"""
        self._execute_pretraining_phase()
        if self.args.pretrain_dnn_epochs > 0 and self.args.pretrain_flow_epochs > 0:
            self.hybrid_dataset.update_dependencies(
                dnn_model=self.network.dnn,
                flow_model=self.network.flow
            )
            print("Dependencies updated after pretraining.")
        self._execute_main_training_phase()

    def _execute_pretraining_phase(self):
        """Handle all pretraining operations"""
        if self.args.pretrain_dnn_epochs > 0:
            self._run_pretraining('dnn', self.args.pretrain_dnn_epochs)
        
        if self.args.pretrain_flow_epochs > 0:
            self._run_pretraining('flow', self.args.pretrain_flow_epochs)

    def _run_pretraining(self, model_type, num_epochs):
        """Generic pretraining workflow"""
        original_mode = self.args.train_mode
        self.args.train_mode = f'{model_type}_only'
        
        print(f"\n=== Pre-training {model_type.upper()} ===")
        for _ in range(num_epochs):
            train_metrics = self._train_epoch()
            self._validate_and_log(train_metrics, phase=f'pre-{model_type}')
            self._handle_support_tasks()
            self.global_step += 1

        self.args.train_mode = original_mode

    def _execute_main_training_phase(self):
        """Main training phase controller"""
        print("\n=== Main Training ===")
        for _ in range(self.args.num_epochs):
            train_metrics = self._train_single_step()
            self._validate_and_log(train_metrics, phase='main')
            self._handle_support_tasks()
            self.global_step += 1

    def _train_single_step(self):
        """Execute single training step based on mode"""
        if self.args.train_mode == 'hybrid':
            return self._hybrid_training_step()
        else:
            return self._standard_training_step()

    def _hybrid_training_step(self):
        """Custom logic for hybrid training steps with phase tracking"""
        if self._training_phase == 'dnn':
            # Execute DNN training
            train_metrics = self._train_component('dnn')
            self._current_dnn_steps += 1
            
            # Check if we should switch to flow training phase
            if self._current_dnn_steps >= self.args.dnn_steps_per_flow_step:
                self._training_phase = 'flow'
                print(f"[Step {self.global_step}] Switching to FLOW training phase")
        else:  # flow training phase
            # Execute Flow training
            train_metrics = self._train_component('flow')
            self._current_flow_steps += 1

            self._update_counter += 1  

            if self._update_counter % self.args.update_interval == 0:
                self.hybrid_dataset.update_dependencies(
                    dnn_model=self.network.dnn,
                    flow_model=self.network.flow
                )
                print(f"Dependencies updated after {self.args.update_interval} cycles.")
            # After completing flow training, switch back to DNN and reset counter
            self._training_phase = 'dnn'
            self._current_dnn_steps = 0
            print(f"[Step {self.global_step}] FLOW training completed. Switching back to DNN phase")

        return train_metrics

    def _train_component(self, component):
        """Generic component training"""
        return self._train_hybrid_epoch(
            train_dnn=(component == 'dnn'),
            train_flow=(component == 'flow'))

    def _standard_training_step(self):
        """Standard training for non-hybrid modes"""
        return self._train_epoch()

    def _handle_support_tasks(self):
        """Manage visualization and checkpoints"""
        if self._should_visualize():
            self._visualize()
            
        if self._should_save_checkpoint():
            self._save_checkpoint()

    def _should_visualize(self):
        """Check visualization timing"""
        return (self.global_step + 1) % self.args.viz_interval == 0

    def _should_save_checkpoint(self):
        """Check checkpoint saving timing"""
        return (self.global_step + 1) % self.args.save_interval == 0

    def _validate_and_log(self, train_metrics, phase='main'):
        """Unified validation and logging workflow"""
        val_metrics = self._validate() if self._should_validate() else {}
        self._log_metrics(train_metrics, val_metrics, phase)

    def _should_validate(self):
        """Determine validation timing"""
        return self.global_step % self.args.val_interval == 0

    def _log_metrics(self, train_metrics, val_metrics, phase):
        """Enhanced logging handler"""
        # Console logging
        self._print_console_logs(train_metrics, val_metrics, phase)
        
        # CSV logging
        self._write_csv_logs(train_metrics, val_metrics, phase)

    def _print_console_logs(self, train_metrics, val_metrics, phase):
        """Format console output"""
        train_str = f"[{phase.upper()}][STEP {self.global_step:04d}] | "
        train_str += self._format_metrics(train_metrics)
        print(train_str)

        if val_metrics:
            val_str = "VALIDATION | " + self._format_metrics(val_metrics)
            print(val_str + "\n")

    def _format_metrics(self, metrics):
        """Standardize metric formatting"""
        return ' | '.join(
            f"{k.upper()}: {v:.4e}" 
            for k, v in metrics.items() 
            if not np.isnan(v))

    def _write_csv_logs(self, train_metrics, val_metrics, phase):
        """Write metrics to CSV files"""
        # Training metrics
        with open(self.train_log_path, 'a') as f:
            f.write(f"{self.global_step},{phase},"
                    f"{train_metrics.get('dnn_loss', 'nan')},"
                    f"{train_metrics.get('flow_mle_loss', 'nan')},"
                    f"{train_metrics.get('flow_constraint_loss', 'nan')}\n")

        # Validation metrics
        if val_metrics:
            with open(self.val_log_path, 'a') as f:
                f.write(f"{self.global_step},{phase},"
                        f"{val_metrics.get('dnn_loss', 'nan')},"
                        f"{val_metrics.get('flow_mle_loss', 'nan')}\n")

    def _visualize(self, suffix=''):
        """Generate visualization plots"""
        self.network.eval()
        with torch.no_grad():
            # DNN visualization
            if self.args.train_mode in ['hybrid', 'dnn_only']:
                self._visualize_component('dnn', suffix)
            
            # Flow visualization
            if self.args.train_mode in ['hybrid', 'flow_only']:
                self._visualize_component('flow', suffix)

    def _visualize_component(self, component, suffix):
        """Visualize specific component"""
        plt.figure(figsize=(10, 10))
        model = getattr(self.network, component)
        if component == 'dnn' or self.problem.input_dim == 2:

            landscape = model(self.grid_tensor).cpu().numpy().reshape(501, 501)
        
            ax = plt.axes()
            color_map = 'rainbow'
            surf = ax.pcolormesh(self.x_grid, self.y_grid, landscape, 
                                cmap=color_map, shading='auto')
            ax.contour(self.x_grid, self.y_grid, landscape, 50, cmap=color_map)
            ax.set_title(f'{component.upper()} Landscape (Step {self.global_step})')
            ax.set_aspect('equal')
            plt.colorbar(surf, shrink=0.5)
            samples = self.base_dataset.simulation_data.cpu().numpy()
            plt.scatter(samples[:, self.index_1], samples[:, self.index_2], s=0.1, c='k')
        
            self._save_viz(suffix=f'_{component}{suffix}')
            
            # 添加俯视图，模仿ddga.py的绘图方法
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(self.x_grid, self.y_grid, landscape, cmap='viridis', antialiased=True)
            
            # 设置视角，与ddga.py一致
            ax.view_init(elev=59, azim=-29)
            
            # 设置坐标轴范围
            x_range = [self.x_grid.min(), self.x_grid.max()]
            y_range = [self.y_grid.min(), self.y_grid.max()]
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
            self._save_viz(suffix=f'_{component}{suffix}_top')
            plt.close()

    def _save_viz(self, suffix=''):
        """Save visualization with global step"""
        viz_path = os.path.join(self.log_dir, 'visualizations')
        plt.savefig(os.path.join(viz_path, 
                   f"step_{self.global_step:05d}{suffix}.png"))
        plt.close()

    def _save_checkpoint(self, suffix=''):
        """Save model checkpoint"""
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        torch.save({
            'global_step': self.global_step,
            'dnn_state_dict': self.network.dnn.state_dict(),
            'flow_state_dict': self.network.flow.state_dict(),
            'dnn_optimizer': self.dnn_optim.state_dict(),
            'flow_optimizer': self.flow_optim.state_dict(),
        }, os.path.join(ckpt_path, 
                      f"step_{self.global_step:05d}{suffix}.pt"))

    def _train_epoch(self):
        """Standard training epoch implementation"""
        self.network.train()
        metrics = {'dnn_loss': [], 'flow_mle_loss': [], 'flow_constraint_loss': []}
        
        # Mode-specific training logic
        if self.args.train_mode == 'dnn_only':
            self._train_dnn_only(metrics)
        elif self.args.train_mode == 'flow_only':
            self._train_flow_only(metrics)
            
        return {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}

    def _train_dnn_only(self, metrics):
        """DNN-only training logic"""
        dnn_loader = self.hybrid_dataset.get_loader('dnn', self.args.batch_size)
        for batch in dnn_loader:
            metrics['dnn_loss'].append(self._train_dnn_step(batch))

    def _train_flow_only(self, metrics):
        """Flow-only training logic"""
        mle_loader = self.hybrid_dataset.get_loader('flow_mle', self.args.batch_size)
        self.network.flow.update_q0(mle_loader.dataset.base.q0)
        for batch in mle_loader:
            metrics['flow_mle_loss'].append(self._train_flow_mle_step(batch))

    def _train_hybrid_epoch(self, train_dnn=True, train_flow=True):
        """Custom hybrid training implementation"""
        self.network.train()
        metrics = defaultdict(list)
        
        # DNN training
        if train_dnn:
            dnn_loader = self.hybrid_dataset.get_loader('dnn', self.args.batch_size)
            for batch in dnn_loader:
                metrics['dnn_loss'].append(self._train_dnn_step(batch))
        
        # Flow training
        # if train_flow:
        #     # MLE training
        #     mle_loader = self.hybrid_dataset.get_loader('flow_mle', self.args.batch_size)
        #     self.network.flow.update_q0(mle_loader.dataset.base.q0)
        #     for batch in mle_loader:
        #         metrics['flow_mle_loss'].append(self._train_flow_mle_step(batch))
            
        #     # Constraint training
        #     constraint_loader = self.hybrid_dataset.get_loader('flow_constraint', self.args.batch_size)
        #     for batch in constraint_loader:
        #         metrics['flow_constraint_loss'].append(
        #             self._train_flow_constraint_step(batch))
        
        # return {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}
    
        # Flow training
        if train_flow:
            mle_loader = self.hybrid_dataset.get_loader('flow_mle', self.args.batch_size)
            # constraint_loader = self.hybrid_dataset.get_loader('flow_constraint', self.args.batch_size)
            self.network.flow.update_q0(mle_loader.dataset.base.q0)
            
            # if getattr(constraint_loader.dataset, "targets", None) is None:
            # # 用当前 DNN 先生成 constraint targets
            #     self.hybrid_dataset.update_dependencies(dnn_model=self.network.dnn)
            # 初始化优化器并清零梯度
            self.flow_optim.zero_grad()
            
            # 遍历所有MLE批次并累积梯度
            for mle_batch in mle_loader:
                x_mle = mle_batch['x'].to(self.device)
                loss_mle = self.network.flow.forward_kld(x_mle)
                loss_mle.backward(retain_graph=True)  # 保留计算图以继续累积梯度
                metrics['flow_mle_loss'].append(loss_mle.item())
            
            # # 遍历所有约束批次并累积梯度
            # if self.problem.input_dim == 2:
            #     for constraint_batch in constraint_loader:
            #         x_constraint = constraint_batch['x'].to(self.device)
            #         target = constraint_batch['target'].to(self.device)
            #         pred = self.network.flow(x_constraint)
            #         loss_constraint = torch.mean(pred - target)
            #         loss_constraint.backward()  # 继续累积梯度
            #         metrics['flow_constraint_loss'].append(loss_constraint.item())
            
            # 统一梯度裁剪和参数更新
            torch.nn.utils.clip_grad_norm_(self.network.flow.parameters(), 5.0)
            self.flow_optim.step()
        
        return {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}

    def _train_dnn_step(self, batch):
        """Single DNN training step"""
        self.dnn_optim.zero_grad()
        
        x = batch['x'].to(self.device)
        f = batch['f'].to(self.device)
        fx = batch['fx'].to(self.device)
        pdf = batch.get('pdf', None)
        if pdf is not None:
            pdf = pdf.to(self.device)
            
        # Compute losses
        if self.problem.input_dim > 2:
            x_full = batch['x_full'].detach().to(self.device).squeeze(0)
            f_full = batch['f_full'].detach().to(self.device).squeeze(0)
            loss_epr = self.loss_epr_high(x_full, f_full, pdf)

        else:
            loss_epr = self.loss_epr(x, f, pdf)
        loss_hjb = self.loss_hjb(x, f, fx, pdf)
        total_loss = self.args.rho_1 * loss_epr + self.args.rho_2 * loss_hjb
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.dnn.parameters(), 5.0)
        self.dnn_optim.step()
        
        return total_loss.item()

    def _train_flow_mle_step(self, batch):
        """Single Flow MLE training step"""
        self.flow_optim.zero_grad()
        
        x = batch['x'].to(self.device)
        loss = self.network.flow.forward_kld(x)
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.flow.parameters(), 5.0)
        self.flow_optim.step()
        nf.utils.update_lipschitz(self.network.flow, 50)
        
        # Save loss value
        loss_value = loss.item()
        if not hasattr(self, 'flow_mle_losses'):
            self.flow_mle_losses = []
        self.flow_mle_losses.append(loss_value)
        
        return loss_value

    def _train_flow_constraint_step(self, batch):
        """Single Flow constraint training step"""
        self.flow_optim.zero_grad()
        
        x = batch['x'].to(self.device)
        target = batch['target'].to(self.device)
        
        # Forward pass
        pred = self.network.flow(x)
        loss = torch.mean(pred - target)
        mse = torch.mean((pred - target) ** 2)
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.flow.parameters(), 5.0)
        self.flow_optim.step()
        
        return loss.item(), mse.item()
    
    

    def _validate(self):
        """Validation step implementation"""
        self.network.eval()
        metrics = {'dnn_loss': [], 'flow_mle_loss': []}
        
        # DNN validation
        if self.args.train_mode in ['hybrid', 'dnn_only']:
            dnn_loader = self.hybrid_dataset.get_loader('dnn', self.args.batch_size, False)
            for batch in dnn_loader:
                x = batch['x'].to(self.device)
                f = batch['f'].to(self.device)
                fx = batch['fx'].to(self.device)
                pdf = batch.get('pdf', None)
                if pdf is not None:
                    pdf = pdf.to(self.device)
                if self.problem.input_dim > 2:
                    x_full = batch['x_full'].detach().to(self.device).squeeze(0)
                    f_full = batch['f_full'].detach().to(self.device).squeeze(0)
                    loss_epr = self.loss_epr_high(x_full, f_full, pdf)
                else:
                    loss_epr = self.loss_epr(x, f, pdf)
                loss_hjb = self.loss_hjb(x, f, fx, pdf)
                total_loss = self.args.rho_1 * loss_epr + self.args.rho_2 * loss_hjb
                metrics['dnn_loss'].append(total_loss.item())
        
        # Flow validation
        if self.args.train_mode in ['hybrid', 'flow_only']:
            mle_loader = self.hybrid_dataset.get_loader('flow_mle', self.args.batch_size, False)
            for batch in mle_loader:
                x = batch['x'].to(self.device)
                loss = self.network.flow.forward_kld(x)
                metrics['flow_mle_loss'].append(loss.item())
                
        return {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}

    def loss_epr(self, x, f, pdf_values=None, model=None):
        """Compute Entropy Production Rate loss"""
        model = model or self.network.dnn
        u = model(x)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        loss_terms = (f.detach() + u_x).pow(2).sum(dim=1)
        
        if pdf_values is not None:
            loss_terms *= pdf_values.detach()
            
        return loss_terms.mean()
    
    def loss_epr_high(self, x, f, pdf_values=None, model=None):
        model = model or self.network.dnn
        x = x[:, [self.index_1, self.index_2]]
        x.requires_grad_(True)
        u = model(x)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        loss_terms = (f[:, [self.index_1, self.index_2]].detach() + u_x).pow(2).sum(dim=1)
        if pdf_values is not None:
            loss_terms *= pdf_values.detach()
            
        return loss_terms.mean()

    def compute_pdf_mean(self, x):

        flow_pdf = torch.exp(self.network.flow.forward(x) / -self.problem.noise_strength)
        # Get pdf values - assuming they are stored in the base dataset
        # This requires that _prepare has been called and pdf is accessible
        if hasattr(self.dnn_dataset, 'pdf') and self.dnn_dataset is not None:
            # Interpolate or match pdf values to the current x points
            pdf_values = self.dnn_dataset.pdf_norm # This is a simplified access, might need interpolation
            # For proper implementation, we might need to interpolate pdf at points x
            mean_pdf = (flow_pdf + pdf_values[:u.shape[0]]) / 2.0
            return mean_pdf # Simple truncation for shape matching and calculate average
        else:
            raise ValueError("PDF values not available. Make sure _prepare has been called.")

    # def mean_pdf_kld(self, x):
    #     """
    #     Compute KL divergence between mean PDF and actual distribution.
        
    #     The KL divergence is computed as:
    #     KL(mean_pdf || actual_pdf) = E_mean[log(mean_pdf/actual_pdf)]
    #                                = E_mean[log(mean_pdf) - log(actual_pdf)]
        
    #     Returns:
    #         kl_div: KL divergence value
    #     """
    #     # Compute the mean PDF
    #     mean_pdf = self.compute_pdf_mean(x)
        
    #     # Get the actual PDF values from the dataset
    #     if hasattr(self.dnn_dataset, 'pdf_norm') and self.dnn_dataset.pdf_norm is not None:
    #         actual_pdf = self.dnn_dataset.pdf_norm[:mean_pdf.shape[0]]
    #     else:
    #         # If normalized PDF is not available, compute it from the mixture model
    #         actual_pdf = torch.exp(self.base_dataset.mix.log_prob(x))
    #         # Normalize the PDF values
    #         actual_pdf = actual_pdf / torch.sum(actual_pdf) * actual_pdf.shape[0]
            
    #     # Add small epsilon to prevent log(0)
    #     epsilon = 1e-10
    #     mean_pdf = mean_pdf + epsilon
    #     actual_pdf = actual_pdf + epsilon
        
    #     # Compute KL divergence: KL(P||Q) = E_P[log(P/Q)]
    #     log_mean_pdf = torch.log(mean_pdf)
    #     log_actual_pdf = torch.log(actual_pdf)
    #     kl_div = torch.mean(log_mean_pdf - log_actual_pdf)
        
    #     return kl_div

# def compute_mse(
#     self,
#     model_pdf=None,
#     model_name=None,
#     num_samples=10000,
#     bins=50,
#     qmin=0.01,
#     qmax=0.99,            # 可调用：lambda n -> (n,2) torch.Tensor 采样点   # 可调用：lambda X,Y -> Z 密度矩阵，X,Y为meshgrid
#     pdf_eps=1e-12,
# ):

    def model_mse_result(self, x):
        # Define lambda functions that return the appropriate values
        dnn_lambda = lambda X, Y: self._evaluate_on_grid(self.dnn_dataset.pdf_norm, X, Y)
        flow_lambda = lambda X, Y: torch.exp(self.network.flow.forward(
            torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=self.device)
        )).reshape(X.shape)
        mean_lambda = lambda X, Y: self._evaluate_on_grid(self.compute_pdf_mean(x), X, Y)
        wsga_lambda = lambda X, Y: self.base_dataset.mix.log_prob(
            torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=self.device)
        ).exp().reshape(X.shape)
        
        dnn_mse = compute_mse(self, dnn_lambda, 'dnn')
        flow_mse = compute_mse(self, flow_lambda, 'flow')
        mean_mse = compute_mse(self, mean_lambda, 'mean')
        wsga_mse = compute_mse(self, wsga_lambda, 'wsga')
        print(f"[Model: dnn]  MSE = {dnn_mse:.6e}")
        print(f"[Model: flow] MSE = {flow_mse:.6e}")
        print(f"[Model: mean] MSE = {mean_mse:.6e}")
        print(f"[Model: wsga] MSE = {wsga_mse:.6e}")
        return None

    def _evaluate_on_grid(self, pdf_values, X, Y):
        """
        Helper method to evaluate PDF values on a grid
        This is a simplified version - in practice you might want to implement
        proper interpolation
        """
        # For now, we'll just return a reshaped version of the PDF values
        # A more sophisticated implementation would interpolate values at (X,Y) coordinates
        return pdf_values[:X.size].reshape(X.shape)
    
    def forward_kld_mean(self, x):
        """
        Estimates forward KL divergence KL(actual || mean_pdf) in the same spirit as forward_kld:
            KL(p || q_mean) = E_p[log p(x) - log q_mean(x)] = E_p[log p(x)] - E_p[log q_mean(x)]
            
        Note: This is NOT the same as the standard forward KL which typically computes:
            KL(q || p) = E_q[log q(x) - log p(x)]
            
        The second term -E_p[log q_mean(x)] is returned here, while the first term 
        E_p[log p(x)] (the negative entropy of true distribution) is omitted.
        This value can be negative since we're missing the entropy term.

        Args:
            x: Batch sampled from the target (actual) distribution
        Returns:
            Scalar: estimate of -E_p[log q_mean(x)] (missing entropy term, can be negative)
        """
        # q_mean(x): mean PDF evaluated at data samples x
        q_mean = self.compute_pdf_mean(x)   # shape: [len(x)], should be non-negative

        # Numerical stability: avoid log(0)
        eps = 1e-12
        log_q_mean = torch.log(q_mean + eps)

        # Return - E_p[log q(x)] (constant entropy term is dropped)
        # This can be negative because the entropy term is missing
        return -torch.mean(log_q_mean)
    


    def forward_kld_wsga(self, x):
        log_q = self.base_dataset.mix.log_prob(x)
        return -torch.mean(log_q)
    
    # Add a new method to compare all four KL divergence computations
    def compare_all_forward_kld(self, x):
        """
        Compare all four forward KL divergence computation methods
        
        Args:
            x: Batch sampled from the target distribution
            
        Returns:
            Dictionary containing results from all four methods
        """
        # Method 1: Flow model's built-in forward KL
        if self.network.flow is not None:
            flow_kld = self.network.flow.forward_kld(x)
        else:
            flow_kld = None
            
        # Method 2: DNN-based forward KL
        dnn_kld = self.network.dnn.forward_kld(x)
        
        # Method 3: Mean PDF-based forward KL
        mean_kld = self.forward_kld_mean(x)
        
        # Method 4: WSGA-based forward KL
        wsga_kld = self.forward_kld_wsga(x)
        print(f"KL divergence results: flow_kld={flow_kld.item() if flow_kld is not None else None}, dnn_kld={dnn_kld.item()}, mean_kld={mean_kld.item()}, wsga_kld={wsga_kld.item()}")
        
        return {
            'flow_kld': flow_kld.item() if flow_kld is not None else None,
            'dnn_kld': dnn_kld.item(),
            'mean_kld': mean_kld.item(),
            'wsga_kld': wsga_kld.item()
        }

    def loss_hjb(self, x, f, fx, pdf_values=None, model=None):
        """Compute Hamilton-Jacobi-Bellman residual loss"""
        model = model or self.network.dnn
        u = model(x)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        
        # Compute second derivatives
        u_xx = torch.stack([
            torch.autograd.grad(u_x[:, i].sum(), x, retain_graph=True)[0][:, i]
            for i in range(2)
        ]).sum(dim=0)
        
        # Compute HJB residual
        residual = -(f.detach() * u_x).sum(dim=1) + \
                   self.problem.noise_strength * (u_xx + fx.detach()) - \
                   (u_x**2).sum(dim=1)
        
        if pdf_values is not None:
            residual *= pdf_values.detach()
            
        return residual.pow(2).mean()

class DimensionReduction(EnergyLandscape):
    """Dimension reduction class for high-dimensional problems projected to 2D"""
    
    def __init__(self, args, problem, network, force_fn):
        super().__init__(args, problem, network, force_fn)
        
        print(f"Dimension Reduction: The origin dim is {self.problem.input_dim}.")
        
        # Initialize optimizer for force network
        self.force_optim = torch.optim.Adam(
            self.network.dnn_f.parameters(), 
            lr=args.dnn_lr,
            weight_decay=args.weight_decay)
        


    def _save_checkpoint(self, suffix=''):
        """Override to include force network state"""
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        checkpoint = {
            'global_step': self.global_step,
            'dnn_state_dict': self.network.dnn.state_dict(),
            'flow_state_dict': self.network.flow.state_dict(),
            'dnn_optimizer': self.dnn_optim.state_dict(),
            'flow_optimizer': self.flow_optim.state_dict(),
        }
        
        checkpoint['dnn_f_state_dict'] = self.network.dnn_f.state_dict()
        checkpoint['force_optimizer'] = self.force_optim.state_dict()
        
        torch.save(checkpoint, os.path.join(ckpt_path, 
                  f"step_{self.global_step:05d}{suffix}.pt"))

    def train(self):
        """Override training to include force training phase"""
        # First train the force network
        self._train_force_network()
        self.hybrid_dataset.dnn.f_proj = self.network.dnn_f
        self.hybrid_dataset.dnn.force_proj()
        # Then proceed with standard training
        super().train()

    def _train_force_network(self):
        """Train the force network to learn averaged forces"""
        print("\n=== Training Force Network ===")
        
        # Get number of force training epochs
        force_epochs = getattr(self.args, 'force_epochs', 1000)
        
        for epoch in tqdm(range(force_epochs)):
            self.network.train()
            force_loss_list = []
            
            # Get training data
            force_loader = self.hybrid_dataset.get_loader('dnn', 1024)

            # Loop over batches
            for batch in force_loader:
                x_full = batch['x_full'].detach().to(self.device).squeeze(0)
                f_full = batch['f_full'].detach().to(self.device).squeeze(0)
                pdf = batch.get('pdf', None)
                if pdf is not None:
                    pdf = pdf.detach().to(self.device)
                    # Compute force loss
                    loss = pdf * self.loss_force(x_full, f_full)
                else:
                    # Compute force loss
                    loss = self.loss_force(x_full, f_full)

                # Save loss
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    force_loss_list.append(loss.item())
                    # Backpropagation
                    self.force_optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.dnn_f.parameters(), 5.0)
                    self.force_optim.step()
            
            # Logging
            if force_loss_list:
                force_loss = np.mean(force_loss_list)
                
                if epoch % self.args.viz_interval == 0:
                    print(f"Force Training Epoch {epoch}, Force Loss: {force_loss:.6f}")
                    
                    # Visualize projected force if needed
                    if hasattr(self, '_visualize_projected_force'):
                        self._visualize_projected_force(epoch)

    def loss_force(self, data_high, f_high):
        """Compute loss for force network learning"""
        # Project high-dimensional data to 2D
        data_dr = data_high[:, [self.index_1, self.index_2]]
        
        # Get predicted 2D force
        f_dr = self.network.dnn_f(data_dr)
        
        # Compute loss between predicted and true projected forces
        loss_f = ((f_dr - f_high[:, [self.index_1, self.index_2]]) ** 2).mean()
            
        return loss_f

def compute_mse(
    landscape,
    model_pdf=None,
    model_name=None,
    num_samples=10000,
    bins=50,
    qmin=0.01,
    qmax=0.99,            # 可调用：lambda n -> (n,2) torch.Tensor 采样点   # 可调用：lambda X,Y -> Z 密度矩阵，X,Y为meshgrid
    pdf_eps=1e-12,
):
    with torch.no_grad():
        # 1) 真实样本
        if landscape.base_dataset.simulation_data is not None and len(landscape.base_dataset.simulation_data) >= num_samples:
            true_samples = landscape.base_dataset.simulation_data[:num_samples]
        else:
            true_samples = landscape.base_dataset.get_simulated_data(num_samples, landscape.problem.noise_strength)

        true_np = true_samples.detach().cpu().numpy()
        if true_np.shape[1] != 2:
            # 若数据>2维，默认使用你已有的 index_1/index_2；否则要求正好2维
            i = getattr(landscape, "index_1", 0)
            j = getattr(landscape, "index_2", 1)
            true_np = true_np[:, [i, j]]
            
        # ---- 2) 用分位数确定共享网格范围（稳健抑制离群点） ----
        lo0 = np.quantile(true_np[:, 0], qmin)
        hi0 = np.quantile(true_np[:, 0], qmax)
        lo1 = np.quantile(true_np[:, 1], qmin)
        hi1 = np.quantile(true_np[:, 1], qmax)

        # 避免区间退化
        if not np.isfinite([lo0,hi0,lo1,hi1]).all() or lo0==hi0 or lo1==hi1:
            lo0, hi0 = float(true_np[:,0].min()), float(true_np[:,0].max())
            lo1, hi1 = float(true_np[:,1].min()), float(true_np[:,1].max())
        ranges = [[lo0, hi0], [lo1, hi1]]

        # 4) 真实分布 PMF（计数归一）
        tr_counts, xedges, yedges = np.histogram2d(
            true_np[:,0], true_np[:,1], bins=bins, range=ranges, density=False
        )
        tr_total = tr_counts.sum()
        if tr_total == 0:
            # 极端：范围太窄，回退到极值范围
            ranges = [[true_np[:,0].min(), true_np[:,0].max()],
                    [true_np[:,1].min(), true_np[:,1].max()]]
            tr_counts, xedges, yedges = np.histogram2d(
                true_np[:,0], true_np[:,1], bins=bins, range=ranges, density=False
            )
            tr_total = max(tr_counts.sum(), 1)
        tr_pmf = tr_counts / tr_total

        # 用 PDF → PMF：先在格心评估 pdf，然后乘以 cell area，再归一
        # 5.1 网格中心
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")  # 形状 [bins, bins]

        # 5.2 评估密度
        Z = model_pdf(X, Y)  # 期望返回 [bins, bins] 的密度矩阵
        # 修改: 确保CUDA tensor转换为numpy前先移到CPU
        if torch.is_tensor(Z):
            Z = Z.detach().cpu().numpy()
        Z = np.asarray(Z, dtype=float)
        Z = np.maximum(Z, 0.0) + pdf_eps

        # 5.3 变为 PMF：密度 * 面积，再全局归一
        dx = (xedges[1] - xedges[0])
        dy = (yedges[1] - yedges[0])
        mass = Z * dx * dy
        total_mass = np.sum(mass)
        if not np.isfinite(total_mass) or total_mass <= 0:
            # 兜底：改用非负归一
            total_mass = np.sum(Z) + 1e-12
            mass = Z / total_mass
        mo_pmf = mass / np.sum(mass)

        # 6) MSE（在 PMF 上）
        mse = float(np.mean((mo_pmf - tr_pmf) ** 2))

        print(f"[Model: {model_name}] MSE = {mse:.6e}")
        return mse
