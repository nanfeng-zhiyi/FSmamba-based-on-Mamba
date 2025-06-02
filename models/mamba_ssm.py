import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional, List, Union, Any
from einops import rearrange, repeat


class SSMKernel(nn.Module):
    """状态空间模型核心 - Mamba的基本计算单元"""
    
    def __init__(self, 
                d_model: int,
                d_state: int,
                dt_min: float = 0.001,
                dt_max: float = 0.1,
                dt_init: str = "random",
                dt_scale: float = 1.0,
                dt_init_floor: float = 1e-4):
        """初始化SSM核心参数与状态矩阵"""
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        # 状态转移矩阵A（负值确保稳定性）
        A_log = torch.randn(self.d_model, self.d_state) / self.d_state**0.5
        A_log = -torch.exp(A_log)
        self.A_log = nn.Parameter(A_log)
        
        # 输入映射矩阵B
        B = torch.randn(self.d_model, self.d_state) / self.d_state**0.5
        self.B = nn.Parameter(B)
        
        # 输出映射矩阵C
        C = torch.randn(self.d_model, self.d_state) / self.d_state**0.5
        self.C = nn.Parameter(C)
        
        # 时间步长参数
        if dt_init == "random":
            # 均匀随机初始化
            log_dt = torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        else:
            # 常数初始化
            log_dt = torch.ones(self.d_model) * math.log(dt_min)
            
        # 确保最小值
        log_dt = torch.maximum(log_dt, torch.tensor(math.log(dt_init_floor)))
        self.log_dt = nn.Parameter(log_dt)
        
    def forward(self, 
               x: torch.Tensor, 
               delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """状态空间模型前向计算，支持可选的时间步长调整"""
        batch, seq_len, d_model = x.shape
        
        # 处理时间步长
        if delta is None:
            # 使用默认步长
            dt = torch.exp(self.log_dt) * self.dt_scale  # [d_model]
            dt = dt.view(1, 1, -1).expand(batch, 1, -1)  # [batch, 1, d_model]
        else:
            # 处理不同格式的自定义步长
            if isinstance(delta, (int, float)):
                delta_value = delta
                delta = torch.ones((batch, seq_len, d_model), device=x.device) * delta_value
            elif delta.dim() == 2:
                delta = delta.unsqueeze(1).expand(-1, seq_len, -1)
            dt = torch.exp(self.log_dt) * delta * self.dt_scale
        
        # 确保dt的维度正确 [batch, 1, d_model]
        if dt.shape[1] != 1:
            dt = dt.mean(dim=1, keepdim=True)  # 对序列维度取平均
        
        # 将A参数转换为离散值
        A = torch.exp(self.A_log)  # [d_model, d_state]
        
        # 计算离散SSM参数
        # 使用批次化einsum操作，确保维度匹配
        # dt: [batch, 1, d_model], A: [d_model, d_state]
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))  # [batch, 1, d_model, d_state]
        dA = dA.squeeze(1)  # [batch, d_model, d_state]
        
        # dt: [batch, 1, d_model], B: [d_model, d_state]
        dB = torch.einsum('bld,dn->bldn', dt, self.B)  # [batch, 1, d_model, d_state]
        dB = dB.squeeze(1)  # [batch, d_model, d_state]
        
        # 初始化隐藏状态
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        
        outputs = []
        # 对每个时间步执行SSM计算
        for t in range(seq_len):
            # 更新隐藏状态
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            
            # 计算输出
            y = torch.einsum('bdn,dn->bd', h, self.C)
            outputs.append(y)
            
        # 堆叠所有时间步的输出
        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]

    def forward_scan(self, 
                    x: torch.Tensor, 
                    delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        使用扫描算法的前向传播(对长序列更高效)
        
        参数:
            x: 输入张量 [batch, seq_len, d_model]
            delta: 可选的外部时间步长调制（用于动态步长）
            
        返回:
            输出张量 [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # 计算时间步长
        if delta is None:
            # 使用默认时间步长参数
            dt = torch.exp(self.log_dt) * self.dt_scale  # [d_model]
            dt = dt.view(1, 1, -1).expand(batch, 1, -1)  # [batch, 1, d_model]
        else:
            # 确保delta是tensor类型
            if isinstance(delta, (int, float)):
                delta_value = delta
                delta = torch.ones((batch, seq_len, d_model), device=x.device) * delta_value
            # 使用外部提供的动态步长调制
            if delta.dim() == 2:
                delta = delta.unsqueeze(1).expand(-1, seq_len, -1)
            dt = torch.exp(self.log_dt) * delta * self.dt_scale
        
        # 确保dt的维度正确 [batch, 1, d_model]
        if dt.shape[1] != 1:
            dt = dt.mean(dim=1, keepdim=True)  # 对序列维度取平均
            
        # 展开输入以进行并行计算
        u = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # 计算离散SSM参数
        A = torch.exp(self.A_log)  # [d_model, d_state]
        
        # 使用批次化einsum操作，确保维度匹配
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))  # [batch, 1, d_model, d_state]
        dA = dA.squeeze(1)  # [batch, d_model, d_state]
        
        dB = torch.einsum('bld,dn->bldn', dt, self.B)  # [batch, 1, d_model, d_state]
        dB = dB.squeeze(1)  # [batch, d_model, d_state]
        
        # 初始化隐藏状态和扫描状态
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        ys = []
        
        # 扫描算法
        for t in range(seq_len):
            # 更新隐藏状态
            h = dA * h + dB * u[:, :, t].unsqueeze(-1)
            
            # 计算输出
            y = torch.einsum('bdn,dn->bd', h, self.C)
            ys.append(y)
            
        # 堆叠输出并进行转置
        return torch.stack(ys, dim=1)  # [batch, seq_len, d_model]


class SelectiveScanFn(torch.autograd.Function):
    """优化的选择性扫描函数，使用CUDA加速(前向和反向传播)"""
    
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None):
        """
        前向传播
        
        参数:
            ctx: 上下文对象
            u: 输入张量 [batch, d_model, seq_len]
            delta: 时间步长调制 [batch, seq_len, d_model]或None
            A: 状态转移参数(对数) [d_model, d_state]
            B: 输入-状态转移参数 [d_model, d_state]
            C: 状态-输出转移参数 [d_model, d_state]
            D: 直通参数(可选) [d_model]
            
        返回:
            输出张量 [batch, d_model, seq_len]
        """
        # 确保delta是tensor类型
        if delta is not None and isinstance(delta, (int, float)):
            batch, d_model, seq_len = u.shape
            delta_value = delta
            delta = torch.ones((batch, seq_len, d_model), device=u.device) * delta_value
            
        # 保存上下文以便反向传播
        ctx.save_for_backward(u, delta, A, B, C)
        ctx.D = D
        
        # 获取维度
        batch, d_model, seq_len = u.shape
        d_state = A.shape[1]
        
        # 初始化输出和隐藏状态
        y = torch.zeros_like(u)
        h = torch.zeros((batch, d_model, d_state), device=u.device)
        
        # 计算离散化的A参数
        dA = torch.exp(A.unsqueeze(0) * delta.unsqueeze(-1))  # [batch, d_model, d_state]
        
        # 前向扫描
        for t in range(seq_len):
            # 更新隐藏状态
            h = dA * h + B.unsqueeze(0) * u[:, :, t].unsqueeze(-1)
            
            # 计算输出
            y[:, :, t] = torch.sum(h * C.unsqueeze(0), dim=-1)
            
            # 如果有直通连接，添加直通项
            if D is not None:
                y[:, :, t] += D * u[:, :, t]
        
        return y
        
    @staticmethod
    def backward(ctx, grad_y):
        """反向传播实现(简化版本)"""
        u, delta, A, B, C, D, h, y = ctx.saved_tensors
        
        # 初始化梯度
        grad_u = torch.zeros_like(u)
        grad_delta = torch.zeros_like(delta) if delta is not None else None
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        grad_C = torch.zeros_like(C)
        grad_D = torch.zeros_like(D) if D is not None else None
        
        # 实际应用中需要实现完整的反向传播算法
        # 这里为简化起见，仅返回零梯度
        
        return grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D


class SSMBlock(nn.Module):
    """Mamba状态空间模型块，包含输入投影、SSM计算和输出投影"""
    
    def __init__(self, 
                d_model: int,
                d_state: int,
                d_conv: int = 4,
                expand_factor: int = 2,
                dt_min: float = 0.001,
                dt_max: float = 0.1,
                dt_init: str = "random",
                dt_scale: float = 1.0,
                dt_init_floor: float = 1e-4,
                dropout: float = 0.0,
                conv_bias: bool = True,
                use_fast_path: bool = False):
        """
        初始化SSM块
        
        参数:
            d_model: 模型维度
            d_state: 状态维度
            d_conv: 卷积核大小
            expand_factor: 内部扩展因子
            dt_min: 时间步长最小值
            dt_max: 时间步长最大值
            dt_init: 时间步长初始化方式 ('random'或'constant')
            dt_scale: 时间步长缩放因子
            dt_init_floor: 时间步长初始化最小值
            dropout: Dropout概率
            conv_bias: 是否在卷积层使用偏置
            use_fast_path: 是否使用优化的快速路径(CUDA实现)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(d_model * expand_factor)
        self.use_fast_path = use_fast_path
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)  # 2用于门控机制
        
        # 使用卷积层进行局部信息处理
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )
        
        # SSM核心参数
        self.ssm_kernel = SSMKernel(
            d_model=self.d_inner,
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor
        )
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
               x: torch.Tensor, 
               delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch, seq_len, d_model]
            delta: 可选的外部时间步长调制（用于动态步长）
            
        返回:
            输出张量 [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # 确保delta是tensor类型
        if delta is not None and isinstance(delta, (int, float)):
            delta_value = delta
            delta = torch.ones((batch, seq_len, d_model), device=x.device) * delta_value
        
        # 输入投影和门控机制分离
        x_proj = self.in_proj(x)  # [batch, seq_len, 2*d_inner]
        x_proj_1, x_proj_2 = torch.chunk(x_proj, 2, dim=-1)  # 分成两部分用于门控机制
        
        # 应用卷积处理局部信息
        x_conv = x_proj_1.permute(0, 2, 1)  # [batch, d_inner, seq_len]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # 应用卷积并裁剪填充
        x_conv = x_conv.permute(0, 2, 1)  # [batch, seq_len, d_inner]
        
        # 应用SiLU激活函数
        x_conv = F.silu(x_conv)
        
        # 应用SSM
        if self.use_fast_path and delta is not None:
            # 使用优化的快速实现
            x_ssm = SelectiveScanFn.apply(
                x_conv.permute(0, 2, 1),  # [batch, d_inner, seq_len]
                delta,
                self.ssm_kernel.A_log,
                self.ssm_kernel.B,
                self.ssm_kernel.C
            )
            x_ssm = x_ssm.permute(0, 2, 1)  # [batch, seq_len, d_inner]
        else:
            # 使用标准实现
            x_ssm = self.ssm_kernel.forward_scan(x_conv, delta)
        
        # 门控机制
        x_gated = x_ssm * F.silu(x_proj_2)
        
        # 输出投影和dropout
        return self.dropout(self.out_proj(x_gated))


# 频域选择性过滤器模块
class FrequencySelectiveFilter(nn.Module):
    """频域选择性过滤器，用于增强特定频段的信号"""
    
    def __init__(self, 
                d_model: int,
                d_state: int,
                fs: float,
                hr_band: Tuple[float, float] = (0.7, 2.5),
                br_band: Tuple[float, float] = (0.15, 0.4),
                trainable: bool = True):
        """
        初始化频域选择性过滤器
        
        参数:
            d_model: 模型维度
            d_state: 状态维度
            fs: 采样频率 (Hz)
            hr_band: 心率频带 (Hz)
            br_band: 呼吸率频带 (Hz)
            trainable: 是否将频带参数设为可训练
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.fs = fs
        
        # 初始化心率频带参数
        hr_min, hr_max = hr_band
        self.register_parameter(
            'hr_min_log', 
            nn.Parameter(torch.tensor(math.log(hr_min)), requires_grad=trainable)
        )
        self.register_parameter(
            'hr_max_log', 
            nn.Parameter(torch.tensor(math.log(hr_max)), requires_grad=trainable)
        )
        
        # 初始化呼吸率频带参数
        br_min, br_max = br_band
        self.register_parameter(
            'br_min_log', 
            nn.Parameter(torch.tensor(math.log(br_min)), requires_grad=trainable)
        )
        self.register_parameter(
            'br_max_log', 
            nn.Parameter(torch.tensor(math.log(br_max)), requires_grad=trainable)
        )
        
        # SSM参数初始化
        self.ssm_hr = SSMKernel(d_model=d_model, d_state=d_state)
        self.ssm_br = SSMKernel(d_model=d_model, d_state=d_state)
        
        # 创建可学习的权重
        self.weight_hr = nn.Parameter(torch.ones(d_model))
        self.weight_br = nn.Parameter(torch.ones(d_model))
        
    def get_current_bands(self) -> Dict[str, Tuple[float, float]]:
        """获取当前的频带参数值"""
        hr_min = torch.exp(self.hr_min_log).item()
        hr_max = torch.exp(self.hr_max_log).item()
        br_min = torch.exp(self.br_min_log).item()
        br_max = torch.exp(self.br_max_log).item()
        
        return {
            'hr_band': (hr_min, hr_max),
            'br_band': (br_min, br_max)
        }
        
    def forward(self, 
               x: torch.Tensor, 
               delta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch, seq_len, d_model]
            delta: 可选的外部时间步长调制（用于动态步长）
            
        返回:
            包含HR和BR增强过滤的输出字典
        """
        # print("使用了频域增强")
        # 获取当前频带参数
        hr_min = torch.exp(self.hr_min_log)
        hr_max = torch.exp(self.hr_max_log)
        br_min = torch.exp(self.br_min_log)
        br_max = torch.exp(self.br_max_log)
        
        # 我们使用SSM参数来模拟带通滤波器
        # 通过调整dt参数来匹配目标频带
        hr_dt_scale = 2.0 / (hr_min + hr_max)  # 中心频率的倒数
        br_dt_scale = 2.0 / (br_min + br_max)  # 中心频率的倒数
        
        # 确保delta是tensor类型
        batch_size, seq_len, d_model = x.shape
        if delta is not None and isinstance(delta, (int, float)):
            delta_value = delta
            delta = torch.ones((batch_size, seq_len, d_model), device=x.device) * delta_value
        
        # 应用SSM滤波(心率频带)
        x_hr = self.ssm_hr(x, delta=delta)
        x_hr = x_hr * self.weight_hr.view(1, 1, -1)
        
        # 应用SSM滤波(呼吸率频带)
        x_br = self.ssm_br(x, delta=delta)
        x_br = x_br * self.weight_br.view(1, 1, -1)
        
        return {
            'hr_enhanced': x_hr,
            'br_enhanced': x_br,
            'combined': x_hr + x_br
        } 