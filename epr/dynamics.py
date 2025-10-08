import torch

class Force:
    """Base class for force """
    def __call__(self, state):
        return self.force(state)
    def force(self, state):
        raise NotImplementedError("Subclasses must implement force method")

class BistableForce(Force):
    """2D bistable system force """
    def __init__(self, a=1.0):
        self.n = 4  # Hill coefficient
        self.S = 0.5  # Activation threshold
        self.S_n = self.S ** self.n
        self.k = 1.0  # Linear decay rate
        self.a = a  # Maximum activation rate
        
    def activate(self, x):
        """Activation term (Hill function)"""
        return self.a * torch.pow(x, self.n) / (self.S_n + torch.pow(x, self.n))
    
    def restrict(self, x):
        """Repression term (inverse Hill function)"""
        return self.a * self.S_n / (self.S_n + torch.pow(x, self.n))

    def force(self, x):
        """Complete force  with mutual inhibition"""
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            f = -self.k * x + self.activate(x) + torch.flip(self.restrict(x), dims=[1])
            return f.squeeze()
        else:
            f = -self.k * x + self.activate(x) + torch.flip(self.restrict(x), dims=[1])
            return f

class Stuart_Landau(Force):
    """Stuart Landau force """
    def __init__(self, lambda_=2.5, m1=-1.5, m2=1.5):
        self.lambda_ = lambda_
        self.m1 = m1
        self.m2 = m2
        
    def force(self, state):
        """Compute the drift force for the given state(s)"""
        # Extract x and y components from the state tensor
        x = state[..., 0]
        y = state[..., 1]
        
        # Compute dx1/dt components
        dx1 = (self.lambda_ * x - y + 
            self.lambda_ * self.m1 * x**3 + 
            (self.m2 - self.m1 + self.m1 * self.m2) * x**2 * y + 
            self.lambda_ * self.m1 * self.m2 * x * y**2 + 
            self.m2 * y**3)
        
        # Compute dx2/dt components
        dx2 = (x + self.lambda_ * y - 
            x**3 + 
            self.lambda_ * self.m1 * x**2 * y + 
            (self.m1 * self.m2 - self.m1 - 1) * x * y**2 + 
            self.lambda_ * self.m1 * self.m2 * y**3)
        
        # Stack results along the last dimension
        return torch.stack([dx1, dx2], dim=-1)

class Biochemical_oscillation(Force):
    """Genetic toggle switch force """
    def __init__(self):
        self.a = 0.1
        self.b = 0.1
        self.c = 100
        self.epsilon = 0.1
        self.tau_0 = 5.0
            
    def force(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        f1 = 200 * ((self.epsilon ** 2 + x1 ** 2) / (1 + x1 ** 2)) / (1 + x2) - 100 * self.a * x1
        f2 = 200 / self.tau_0 * (self.b - x2 / (1 + self.c * x1 ** 2))
        return torch.stack([f1, f2], dim=-1)

# 新增: 转录因子调控系统的 N 维漂移力场
class TranscriptionFactorForce(Force):
    """N-dimensional drift force field for transcription factor regulatory system"""
    def __init__(self, alpha=0.6, beta=15, Kd=1, n=1.5):
        self.alpha = alpha    # 基础表达率
        self.beta = beta      # 最大诱导表达率
        self.Kd = Kd          # 二聚体解离常数
        self.n = n            # Hill 系数
        
    def force(self, state):
        """
        计算给定状态下的漂移力
        输入: state - 形状为 (..., N) 的张量，其中 N 是维度数
        输出: 形状为 (..., N) 的漂移力张量
        """
        # 计算所有转录因子的总和
        S = state.sum(dim=-1, keepdim=True)  # 保持维度用于广播
        
        # 计算分母项 (添加小量防止除零错误)
        denom = self.Kd + 4 * S + torch.sqrt(self.Kd**2 + 8 * S * self.Kd + 1e-8)
        
        # 计算每个分量的二聚体浓度
        x2 = 2 * state**2 / denom
        
        # 计算 Hill 函数项
        hill = x2**self.n / (1 + x2**self.n)
        
        # 计算导数
        dx_dt = self.alpha + self.beta * hill - state
        
        return dx_dt

# 新增: 52维双稳态系统力场
class Bistable52DForce(Force):
    """52-dimensional bistable system force field"""
    
    def __init__(self, a=0.37, b=0.5, k=1.0, S=0.5, n=3, matrix_file=None, matrix_data=None):
        """
        Initialize the 52D force field
        
        Args:
            matrix_file: Path to CSV file containing the interaction matrix
            matrix_data: Direct matrix data (52x52 numpy array or tensor)
            a: Maximum activation rate
            b: Maximum repression rate  
            k: Linear decay rate
            S: Activation threshold
            n: Hill coefficient
        """
        self.n = n
        self.S = S
        self.S_n = self.S ** self.n
        self.k = k
        self.a = a
        self.b = b
        # Load the interaction matrix

        if matrix_file is not None:
            # Load from CSV file
            import pandas as pd
            data = pd.read_csv(matrix_file, header=None)
            self.matrix = torch.tensor(data.values).float()
        elif matrix_data is not None:
            # Load from direct data
            self.matrix = torch.tensor(matrix_data).float() if not isinstance(matrix_data, torch.Tensor) else matrix_data
        else:
            raise ValueError("Either matrix_file or matrix_data must be provided")
        
        # Ensure matrix is 52x52
        if self.matrix.shape != (52, 52):
            raise ValueError(f"Matrix must be 52x52, got {self.matrix.shape}")
        
        # Create activation and repression matrices
        # Transpose to match the original implementation pattern
        self.matrix_act = (self.matrix == 1).float().T
        self.matrix_res = (self.matrix == -1).float().T

    def activate(self, x):
        """Activation term (Hill function)"""
        return self.a * torch.pow(x, self.n) / (self.S_n + torch.pow(x, self.n))
    
    def restrict(self, x):
        """Repression term (inverse Hill function)"""
        return self.b * self.S_n / (self.S_n + torch.pow(x, self.n))
    
    def force(self, x):
        """
        Complete force field with matrix-defined interactions
        
        Args:
            x: Input tensor of shape (52,) or (batch_size, 52)
            
        Returns:
            Force field tensor of same shape as input
        """
        # Handle single vector input
        self.matrix_act =  self.matrix_act.to(x.device)
        self.matrix_res = self.matrix_res.to(x.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
                        
        # Compute force components
        decay_term = -self.k * x
        activation_term = torch.mm(self.activate(x), self.matrix_act)
        repression_term = torch.mm(self.restrict(x), self.matrix_res)
        
        # Total force
        f = decay_term + activation_term + repression_term
        
        # Return to original shape if needed
        if squeeze_output:
            return f.squeeze(0)
        else:
            return f

def get_force_(force_type, **kwargs):
    """Factory method to get force  by type"""
    force_ = {
        'bistable': BistableForce,
        'Stuart_Landau': Stuart_Landau,
        'Biochemical': Biochemical_oscillation,
        'transcription_factor': TranscriptionFactorForce,
        'bistable_52d': Bistable52DForce
    }
    
    if force_type not in force_:
        raise ValueError(f"Unknown force  type: {force_type}")
    
    return force_[force_type](**kwargs)