import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Literal
import yaml

@dataclass
class ModelArgs:    
    # PDF generation method
    pdf_method: Literal["WSGA", "Simulate", "DDGA"] = "DDGA"
    
    # Simulation parameters
    sim_dt: float = 0.01
    sim_steps: int = 50000
    # boundary_type: str = "reflect"
    
    # WSGA parameters
    wsga_delta_t: float = 0.1
    wsga_num_steps: int = 5000
    # per attractor simulation num
    rand_num: int = 500
    
    # DDGA parameters
    ddga_delta_t: float = 0.01
    ddga_num_steps: int = 200000

    base_batch_size: int = 1024
    
    # Loss parameters
    rho_1: float = 1.0
    rho_2: float = 1.0
    
    # Training parameters
    train_mode: Literal["hybrid", "dnn_only", "flow_only"] = "dnn_only"
    update_interval: int = 20
    num_epochs: int = 1000
    pretrain_dnn_epochs: int = 0
    pretrain_flow_epochs: int = 0
    dnn_steps_per_flow_step: int = 5
    batch_size: int = 1024

    
    # New parameter for validation frequency
    val_interval: int = 20
    
    # Network parameters
    hidden_sizes: str = "32,64,64,32"
    flow_hidden_units: int = 128
    flow_num_blocks: int = 3
    flow_layers: int = 8
    
    # Learning rates
    dnn_lr: float = 1e-3
    flow_lr: float = 3e-4
    weight_decay: float = 1e-5
    
    # Visualization and save parameters
    viz_interval: int = 10
    save_interval: int = 50
    prefix: str = "./2008/"
    
    # Device configuration
    use_gpu: bool = True
    device: torch.device = field(init=False)
    
    # Force field configuration
    force_type: str = 'bistable'
    force_params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set device after initialization based on use_gpu flag and availability"""
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    @classmethod
    def from_yaml(cls, filepath):
        """Create ModelArgs instance from YAML configuration file"""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        if 'device' in config:
            device_str = config.pop('device')
            if device_str == 'cuda':
                config['use_gpu'] = True
            else:
                config['use_gpu'] = False
        return cls(**config)
    

    def save_to_yaml(self, yaml_path):
        """Save ModelArgs instance to YAML configuration file"""
        # Convert to dictionary, excluding non-serializable fields
        config_dict = {}
        for key, value in self.__dict__.items():
            # Convert torch.device to string for YAML serialization
            if isinstance(value, torch.device):
                config_dict[key] = value.type  # 只保存设备类型字符串，如'cuda'或'cpu'
            else:
                config_dict[key] = value
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


@dataclass
class Problem:
    # Problem dimensions
    input_dim: int = 2
    index_1: int = 0
    index_2: int = 1
    output_dim: int = 1
    noise_strength: float = 0.01

    # Meta parameters
    meta_dim: int = 0
    
    # Domain parameters
    x_max: float = 1.5
    x_min: float = -1.5
    # Sample sizes for different components
    dnn_sample_size: int = 100000
    condition_sample_size: int = 1000
    flow_sample_size: int = 10000
    flow_constraint_sample_size: int = 10000

    @classmethod
    def from_yaml(cls, yaml_path):
        """Create Problem instance from YAML configuration file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def save_to_yaml(self, yaml_path):
        """Save Problem instance to YAML configuration file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

def set_seed_everywhere(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
