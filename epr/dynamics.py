from typing import List
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
def trasns_fitting(lambda_, pars):
        a, b, c, d, e = pars
        return lambda_ * (a + b / (c + (lambda_ / d) ** e))

class ToggleBasic(Force):
    def __init__(self, growth_rate):
        # 基本参数
        #self.k_t = 1.0
        #self.k_l = 2.0
        self.k_t = 9.0   
        self.k_l = 12.6  #L2
        self.n_t = 2.0
        self.n_l = 4.0
        self.tau_p_trc = 0.13 #L2  # Ptrc leakage, TetR leakage expression level
        self.tau_p_ltet = 0.015    # PLtetO-1 leakage, LacI leakage
        self.alphal_factor = 1.1
        self.alphat_factor = 1.1  #???
        self.gr = growth_rate   # type: float # cell growth rate
        self.protein_decay = 0.0

        #计算alpha
        # 保存拟合函数与参数到 self，供 setter 复用
        self.alpha_fuc = trasns_fitting
        self.tetR_pars = dict(pars=[26.836, 320.215, 1.0, 0.661, 4.09])
        self.lacI_pars = dict(pars=[16.609, 627.747, 1.0, 0.865, 4.635])
        
        # 计算表达率
        self.alpha_trc  = self.alphal_factor * self.alpha_fuc(self.gr, **self.tetR_pars)
        self.alpha_ltet = self.alphal_factor * self.alpha_fuc(self.gr, **self.lacI_pars)

        # 归一化（无量纲）表达水平
        self.alpha_trc_over_gr  = self.alpha_trc  / (self.gr + self.protein_decay)
        self.alpha_ltet_over_gr = self.alpha_ltet / (self.gr + self.protein_decay)
        self.tilde_k_t = self.k_t / self.alpha_trc_over_gr
        self.tilde_k_l = self.k_l / self.alpha_ltet_over_gr

        # 诱导剂等
        self.atc_conc = 0.0
        self.iptg_conc = 0.0
        self.k_atc = 1.0
        self.k_iptg = 1.0
        self.m = 1.0
        self.n = 1.0

        # steady-state 相关（保持原样）
        self.sst_laci_conc = None
        self.sst_tetr_conc = None
        self.sst_state = None
        self.bistable = False
    @property
    def growth_rate(self):
        return self.gr
    
    @growth_rate.setter
    def growth_rate(self, growth_rate):
        self.gr = float(growth_rate)
        self.alpha_trc  = self.alphal_factor * self.alpha_fuc(self.gr, **self.tetR_pars)
        self.alpha_ltet = self.alphal_factor * self.alpha_fuc(self.gr, **self.lacI_pars)
        self.alpha_trc_over_gr  = self.alpha_trc  / (self.gr + self.protein_decay)
        self.alpha_ltet_over_gr = self.alpha_ltet / (self.gr + self.protein_decay)
        self.tilde_k_t = self.k_t / self.alpha_trc_over_gr
        self.tilde_k_l = self.k_l / self.alpha_ltet_over_gr

    def set_alpha_trc(self, alpha_trc: float) -> None:
        self.alpha_trc = self.alphal_factor * float(alpha_trc)
        self.alpha_trc_over_gr = self.alpha_trc / (self.gr + self.protein_decay)

    def set_alpha_ltet(self, alpha_ltet: float) -> None:
        self.alpha_ltet = self.alphal_factor * float(alpha_ltet)
        self.alpha_ltet_over_gr = self.alpha_ltet / (self.gr + self.protein_decay)

    # 下面用张量友好的实现，支持批量
    def h_l(self, laci):
        # Ptrc 对 LacI 的抑制 Hill 函数
        return self.tau_p_trc + (1.0 - self.tau_p_trc) / (1.0 + (laci / self.k_l) ** self.n_l)

    def h_t(self, tetr):
        # PLtetO-1 对 TetR 的抑制 Hill 函数
        return self.tau_p_ltet + (1.0 - self.tau_p_ltet) / (1.0 + (tetr / self.k_t) ** self.n_t)

    def null_cline_tetr(self, laci_tot):
        laci_free = laci_tot * (1.0 + self.iptg_conc / self.k_iptg * laci_tot) ** (-self.n)
        return self.alpha_trc * self.h_l(laci_free) / (self.gr + self.protein_decay)

    def null_cline_laci(self, tetr_tot):
        tetr = tetr_tot * (1.0 + self.atc_conc / self.k_atc * tetr_tot) ** (-self.m)
        return self.alpha_ltet * self.h_t(tetr) / (self.gr + self.protein_decay)

    def force(self, y):
        """
        输入:
            y: 形状(2,)或(N,2)；按顺序 [LacI, TetR]
               可为 list / np.ndarray / torch.Tensor
        输出:
            torch.Tensor，形状与 y 相同
        """
        # 转 tensor（保持 device / dtype 一致性如果已是张量）
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float32)

        # 统一形状：(..., 2)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        laci_tot = y[..., 0]
        tetr_tot = y[..., 1]

        # # 诱导剂有效自由浓度
        tetr = tetr_tot * (1.0 + (self.atc_conc / self.k_atc) * tetr_tot) ** (-self.m) 
        laci = laci_tot * (1.0 + (self.iptg_conc / self.k_iptg) * laci_tot) ** (-self.n)
        
        #计算蛋白质浓度的时间变化率
        dev_laci = self.alpha_ltet * self.h_t(tetr) - laci_tot * (self.gr + self.protein_decay)
        dev_tetr = self.alpha_trc  * self.h_l(laci) - tetr_tot * (self.gr + self.protein_decay)
        
        dy = torch.stack([dev_laci, dev_tetr], dim=-1)  # (..., 2)
        return dy.squeeze(0) if dy.shape[0] == 1 else dy


def get_force_(force_type, **kwargs):
    """Factory method to get force  by type"""
    force_ = {
        'bistable': BistableForce,
        'Stuart_Landau': Stuart_Landau,
        'Biochemical': Biochemical_oscillation,
        'transcription_factor': TranscriptionFactorForce,
        'bistable_52d': Bistable52DForce,
        'ToggleBasic': ToggleBasic
    }
    
    if force_type not in force_:
        raise ValueError(f"Unknown force  type: {force_type}")
    
    # # 修改为显式处理 ToggleBasic 的 growth_rate 参数
    # if force_type == 'ToggleBasic' and 'growth_rate' in kwargs:
    #     return force_[force_type](growth_rate=kwargs['growth_rate'])
    # else:
    return force_[force_type](**kwargs)
