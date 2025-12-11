import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from collections import OrderedDict
import normflows as nf

class NetworkWrapper:
    """
    Wrapper class to contain both DNN and flow models
    as required by the EnergyLandscape class
    """
    def __init__(self, args, problem):
        self.dnn = DNN(args, problem).to(args.device)
        # 根据args.use_flow决定是否创建FlowNet
        self.use_flow = getattr(args, 'use_flow', True)
        self.flow = None
        if self.use_flow:
            self.flow = FlowNet(args, problem).to(args.device)
        self.dnn_f = None
        if problem.input_dim > 2:
            self.dnn_f = DNN(args, problem,  output_dim=2).to(args.device)

    def train(self):
        self.dnn.train()
        # 只有在使用flow时才调用其train方法
        if self.flow is not None:
            self.flow.train()
        if self.dnn_f is not None:
            self.dnn_f.train()
    def eval(self):
        self.dnn.eval()
        # 只有在使用flow时才调用其eval方法
        if self.flow is not None:
            self.flow.eval()
        if self.dnn_f is not None:
            self.dnn_f.eval()



class DNN(nn.Module):
    """Neural network for potential energy modeling"""
    def __init__(self, args, problem, output_dim=1):
        super(DNN, self).__init__()
        self.scale = 1.0
        self.problem = problem
        
        # Parse hidden layer sizes
        layers = [int(x) for x in args.hidden_sizes.split(',')]
        layers.insert(0, problem.meta_dim + 2)
        layers.append(output_dim)

        self.depth = len(layers) - 1
        self.activation = nn.Tanh

        # Build network layers
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )
        # Add a final sigmoid layer for bounded output
        # layer_list.append(('', nn.Sigmoid()))

        layer_dict = OrderedDict(layer_list)
        self.layers = nn.Sequential(layer_dict)
    
    def _process_input(self, x):
        """统一处理输入数据，如果x是字典则提取'x'键的值"""
        if isinstance(x, dict):
            return x['x']
        return x

    def forward(self, x):
        """Forward pass for the network"""
        x = self._process_input(x)
        
        out = self.layers(x)
        out = out * self.scale
        return out #输出势能函数值
    def forward_kld(self, x):
        """Compute KL divergence loss (scaled by noise strength)
        
        Note: This computes -E_p[log q(x)] where q is related to the DNN output.
        The actual value can be positive or negative depending on:
        1. The scale of the DNN outputs (before scaling by noise_strength)
        2. The noise_strength parameter
        3. The distribution of input samples x
        
        This is NOT a complete KL divergence calculation. A full KL divergence
        would be E_p[log p(x) - log q(x)] where the first term (entropy of p) is missing.
        """
        x = self._process_input(x)
        out = self.forward(x) / -self.problem.noise_strength
        self.log_pdf_unnorm = out  # 直接存储对数形式

        # 使用预计算的归一化常数，而不是每次都重新计算
        if not hasattr(self, 'log_Z') or self.log_Z is None:
            self.update_normalization_constant()

        # 使用数值稳定的方法计算归一化概率的对数
        # log(pdf_norm) = log(pdf_unnorm) - log(Z)
        self.log_pdf_norm = self.log_pdf_unnorm - self.log_Z
        self.pdf_norm = torch.exp(self.log_pdf_norm)
        
        # 返回负的对数似然（考虑了归一化）
        # -E[log(pdf_norm)] = -E[log_pdf_norm]
        return -torch.mean(self.log_pdf_norm)

    def update_normalization_constant(self):
        """
        Update the normalization constant log_Z based on current model parameters.
        This should be called periodically during training to ensure accuracy,
        especially when model parameters change significantly.
        Uses numerically stable log-sum-exp computation.
        Supports both 1D and multi-dimensional cases.
        """
        with torch.no_grad():
            # 使用problem中定义的全局范围
            grid_points_per_dim = 50  # 每个维度的网格点数，避免高维时网格点过多
            device = next(self.parameters()).device  # 获取模型所在的设备
            
            # 获取输入维度
            input_dim = self.problem.input_dim
            
            if input_dim == 1:
                # 1D情况
                x_min, x_max = self.problem.x_min, self.problem.x_max
                x_grid = torch.linspace(x_min, x_max, grid_points_per_dim).to(device)
                x_grid = x_grid.reshape(-1, 1)
                
                # 在网格点上计算势能的对数形式
                out_grid = self.forward(x_grid) / -self.problem.noise_strength
                
                # 计算dx
                dx = (x_max - x_min) / (grid_points_per_dim - 1)
                log_dx = torch.log(torch.tensor(dx, device=device))
                
                # 使用数值稳定的log-sum-exp计算log(Z)
                log_Z_terms = out_grid + log_dx
                self.log_Z = torch.logsumexp(log_Z_terms, dim=0)
                
            else:
                # 多维情况
                x_min, x_max = self.problem.x_min, self.problem.x_max
                
                # 创建多维网格
                grids = []
                for i in range(input_dim):
                    grid = torch.linspace(x_min, x_max, grid_points_per_dim).to(device)
                    grids.append(grid)
                
                # 创建网格点组合
                mesh_grids = torch.meshgrid(*grids, indexing='ij')
                x_grid = torch.stack([grid.flatten() for grid in mesh_grids], dim=1)
                
                # 在网格点上计算势能的对数形式
                out_grid = self.forward(x_grid) / -self.problem.noise_strength
                
                # 计算每个维度的dx
                dx = (x_max - x_min) / (grid_points_per_dim - 1)
                # 多维情况下的体积元
                volume_element = dx ** input_dim
                log_volume_element = torch.log(torch.tensor(volume_element, device=device))
                
                # 使用数值稳定的log-sum-exp计算log(Z)
                log_Z_terms = out_grid + log_volume_element
                self.log_Z = torch.logsumexp(log_Z_terms, dim=0)


    def reset_normalization_constant(self):
        """
        Reset the normalization constant, forcing it to be recomputed on next use.
        This can be called at the beginning of each epoch or training cycle.
        """
        self.log_Z = None


class FlowNet(nn.Module):
    """Normalizing flow network for density estimation"""
    def __init__(self, args, problem):
        super(FlowNet, self).__init__()
        self.args = args
        self.problem = problem

        # Base distribution
        self.q0 = nf.distributions.DiagGaussian(
            problem.input_dim, 
            trainable=False
        ).to(args.device)
        
        # Create flow layers
        self.flows = self._create_flow()
        
        # Initialize flow model
        self.flow_model = nf.NormalizingFlow(
            self.q0, 
            self.flows
        ).to(args.device)
    
    def _process_input(self, x):
        """统一处理输入数据，如果x是字典则提取'x'键的值"""
        if isinstance(x, dict):
            return x['x']
        return x

    def _create_flow(self):
        """Create normalizing flow architecture using Residual blocks"""
        flows = []
        latent_size = self.problem.input_dim
        for i in range(self.args.flow_layers):
            net = nf.nets.LipschitzMLP([latent_size] + [self.args.flow_hidden_units] * (self.args.flow_num_blocks - 1) + [latent_size],
                                    init_zeros=True, lipschitz_const=0.9)
            flows += [nf.flows.Residual(net, reduce_memory=False)]
            flows += [nf.flows.ActNorm(latent_size)]
            
        return flows
    
    def forward(self, x):
        """Forward pass: compute negative log likelihood scaled by noise"""
        x = self._process_input(x)   
        u = self.flow_model.log_prob(x)
        return -self.problem.noise_strength * u
    
    def forward_kld(self, x):
        """Compute KL divergence for the flow model"""
        x = self._process_input(x)
            
        return self.flow_model.forward_kld(x)
    
    def log_prob(self, x):
        """Compute log probability for samples"""
        x = self._process_input(x)
        return self.flow_model.log_prob(x)
    
    def sample(self, n, batch_size=1000):
        """Generate samples from the flow model"""
        samples = self.flow_model.sample(n)
        return samples
        
    def update_q0(self, q0):
        """Update base distribution"""
        self.q0 = q0
        self.flow_model.q0 = q0

