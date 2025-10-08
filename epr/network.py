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
        # layer_list.append(('sigmoid', nn.Sigmoid()))

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
        return out
    
    def forward_kld(self, x):
        """Compute KL divergence loss (scaled by noise strength)"""
        x = self._process_input(x)
        out = self.forward(x) / -self.problem.noise_strength
        return -torch.mean(out)


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

