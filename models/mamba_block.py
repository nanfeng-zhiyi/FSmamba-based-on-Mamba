import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union, Any
import math

from .mamba_ssm import SSMBlock, FrequencySelectiveFilter


class LayerNorm(nn.Module):
    """LayerNorm实现，带有可选的偏置"""
    def __init__(self, d_model: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            y = y * self.weight + self.bias
        else:
            y = y * self.weight
        return y


class MambaBlock(nn.Module):
    """Mamba块 - 带残差连接的SSM处理单元"""
    
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
                bias: bool = True,
                use_fast_path: bool = False):
        """构建基本Mamba块组件与时间步长参数"""
        super().__init__()
        
        # 层正则化
        self.norm = LayerNorm(d_model, bias=bias)
        
        # SSM主体处理单元
        self.ssm = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            dropout=dropout,
            conv_bias=bias,
            use_fast_path=use_fast_path
        )
        
    def forward(self, 
               x: torch.Tensor, 
               delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算Mamba块输出，支持可选的时间步长调整"""
        # 将标量类型的delta转换为张量
        batch_size, seq_len, d_model = x.shape
        if delta is not None and isinstance(delta, (int, float)):
            delta_value = delta
            delta = torch.ones((batch_size, seq_len, d_model), device=x.device) * delta_value
            
        # 应用层规范化
        z = self.norm(x)
        
        # 应用SSM处理
        output = self.ssm(z, delta)
        
        # 残差连接
        return output + x


class FrequencyEnhancedMambaBlock(nn.Module):
    """具有频域增强的Mamba块"""
    
    def __init__(self,
                d_model: int,
                d_state: int,
                d_conv: int = 4,
                expand_factor: int = 2,
                fs: float = 30.0,
                hr_band: Tuple[float, float] = (0.7, 2.5),
                br_band: Tuple[float, float] = (0.15, 0.4),
                dt_min: float = 0.001,
                dt_max: float = 0.1,
                dt_init: str = "random",
                dt_scale: float = 1.0,
                dt_init_floor: float = 1e-4,
                dropout: float = 0.0,
                bias: bool = True,
                use_fast_path: bool = False,
                filter_d_state: Optional[int] = None):
        """
        初始化带频域增强的Mamba块
        
        参数:
            d_model: 模型维度
            d_state: 状态维度
            d_conv: 卷积核大小
            expand_factor: 内部扩展因子
            fs: 采样频率 (Hz)
            hr_band: 心率频带 (Hz)
            br_band: 呼吸率频带 (Hz)
            dt_min: 时间步长最小值
            dt_max: 时间步长最大值
            dt_init: 时间步长初始化方式
            dt_scale: 时间步长缩放因子
            dt_init_floor: 时间步长初始化最小值
            dropout: Dropout概率
            bias: 是否使用偏置
            use_fast_path: 是否使用CUDA优化路径
            filter_d_state: 频域增强滤波器的状态维度
        """
        super().__init__()
        
        # 规范化层
        self.norm = LayerNorm(d_model, bias=bias)
        
        # SSM块
        self.ssm = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            dropout=dropout,
            conv_bias=bias,
            use_fast_path=use_fast_path
        )
        
        # 频域增强滤波器 - 确保默认使用固定值16，与训练时一致
        filter_d = filter_d_state if filter_d_state is not None else 16  # 默认使用固定值16，而不是d_state//2
        self.freq_filter = FrequencySelectiveFilter(
            d_model=d_model,
            d_state=filter_d,
            fs=fs,
            hr_band=hr_band,
            br_band=br_band,
            trainable=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
               x: torch.Tensor, 
               delta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch, seq_len, d_model]
            delta: 可选的时间步长调制
            
        返回:
            包含处理结果的字典
        """
        # 确保delta是tensor类型
        batch_size, seq_len, d_model = x.shape
        if delta is not None and isinstance(delta, (int, float)):
            delta_value = delta
            delta = torch.ones((batch_size, seq_len, d_model), device=x.device) * delta_value
            
        # 应用层规范化
        z = self.norm(x)
        
        # 应用SSM处理
        ssm_output = self.ssm(z, delta)
        
        # 应用频域增强滤波
        freq_outputs = self.freq_filter(z, delta)
        
        # 将SSM输出与频域增强输出连接
        combined = torch.cat([ssm_output, freq_outputs['combined']], dim=-1)
        
        # 融合连接的特征
        fused = self.fusion(combined)
        
        # 残差连接
        output = fused + x
        
        return {
            'output': output,
            'hr_enhanced': freq_outputs['hr_enhanced'],
            'br_enhanced': freq_outputs['br_enhanced']
        }


class CrossAttention(nn.Module):
    """自注意力和跨模态注意力机制的实现"""
    
    def __init__(self,
                query_dim: int,
                key_dim: int = None,
                num_heads: int = 8,
                head_dim: int = 64,
                dropout: float = 0.1):
        """
        初始化注意力模块
        
        参数:
            query_dim: 查询维度
            key_dim: 键值维度，如果为None则等于query_dim
            num_heads: 注意力头数
            head_dim: 每个注意力头的维度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        
        # 如果未指定键值维度，则使用查询维度
        if key_dim is None:
            key_dim = query_dim
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        
        # 线性投影层
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(key_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, query_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 相对位置编码
        self.max_pos_encoding = 512
        self.rel_pos_emb = nn.Parameter(torch.zeros(self.max_pos_encoding, self.max_pos_encoding))
        nn.init.trunc_normal_(self.rel_pos_emb, std=0.02)
    
    def _reshape_for_multihead_attention(self, x, batch_size, seq_len, target_dim):
        """将输入张量重塑为多头注意力格式"""
        # 确保输入具有正确的维度 [batch, seq_len, dim]
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        elif len(x.shape) == 2:
            if x.shape[0] == batch_size:
                x = x.unsqueeze(1)  # [batch, 1, dim]
            else:
                x = x.unsqueeze(0)  # [1, seq, dim]
        
        # 如果序列长度为1，但形状不匹配
        if x.shape[1] == 1 and seq_len > 1:
            x = x.expand(-1, seq_len, -1)
            
        # 确保最后一维是目标维度
        if x.shape[-1] != target_dim:
            x = self.to_out(x)  # 临时修复，确保维度正确
            
        return x


    def _prepare_qkv(self, query, key, value):
        """准备查询、键和值张量"""
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # 确保所有输入都有正确的维度
        if len(query.shape) != 3:
            query = query.unsqueeze(1)
        if len(key.shape) != 3:
            key = key.unsqueeze(1)
        if len(value.shape) != 3:
            value = value.unsqueeze(1)
            
        # 确保所有输入都设置了requires_grad
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        
        return query, key, value, batch_size, seq_len_q, seq_len_k
    
    def get_rel_pos_bias(self, seq_len: int) -> torch.Tensor:
        """获取相对位置编码"""
        # 如果序列长度超过最大位置编码长度，则截断
        if seq_len > self.max_pos_encoding:
            return self.rel_pos_emb[:seq_len, :seq_len]
        
        # 否则使用完整的相对位置编码
        return self.rel_pos_emb[:seq_len, :seq_len]
    
    def forward(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 查询张量 [batch, seq_len_q, query_dim] 或其他维度
            key: 键张量 [batch, seq_len_k, key_dim] 或其他维度
            value: 值张量 [batch, seq_len_k, key_dim] 或其他维度
            mask: 注意力掩码 [batch, seq_len_q, seq_len_k]
            
        返回:
            注意力输出 [batch, seq_len_q, query_dim]
        """
        # 准备查询、键和值
        query, key, value, batch_size, seq_len_q, seq_len_k = self._prepare_qkv(query, key, value)
        
        # 投影到多头注意力空间
        q = self.to_q(query)  # [batch, seq_len_q, inner_dim]
        k = self.to_k(key)    # [batch, seq_len_k, inner_dim]
        v = self.to_v(value)  # [batch, seq_len_k, inner_dim]
        
        # 确保维度正确
        inner_dim = q.shape[-1]
        if inner_dim % self.num_heads != 0:
            raise ValueError(f"特征维度 {inner_dim} 必须能被头数 {self.num_heads} 整除")
        
        head_dim = inner_dim // self.num_heads
        
        # 确保张量形状正确并重塑为多头格式
        q = q.reshape(batch_size, seq_len_q, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len_k, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len_k, self.num_heads, head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 添加相对位置编码
        rel_pos_bias = self.get_rel_pos_bias(max(seq_len_q, seq_len_k))
        if seq_len_q != seq_len_k:
            rel_pos_bias = rel_pos_bias[:seq_len_q, :seq_len_k]
        rel_pos_bias = rel_pos_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_k]
        attn_scores = attn_scores + rel_pos_bias
        
        # 应用掩码(如果提供)
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1, -1)
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出
        out = torch.matmul(attn_weights, v)
        
        # 重塑回原始维度
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len_q, -1)
        
        # 最终线性投影
        out = self.to_out(out)
        
        return out


class MultiModalFusionBlock(nn.Module):
    """多模态特征融合模块。将RGB和NIR特征融合为单一表示。"""
    
    def __init__(self, 
                fusion_dim: int,
                feature_dim: int = None,  # 兼容旧版本
                rgb_dim: int = None, 
                nir_dim: int = None,
                num_heads: int = 4,
                dropout: float = 0.1):
        """
        初始化多模态融合模块
        
        参数:
            fusion_dim: 融合后的特征维度
            feature_dim: 兼容旧版本，同时设置RGB和NIR维度
            rgb_dim: RGB特征维度
            nir_dim: NIR特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        # 参数兼容性处理
        if feature_dim is not None:
            # 如果提供了feature_dim，则同时设置rgb_dim和nir_dim
            rgb_dim = feature_dim
            nir_dim = feature_dim
            print(f"[信息] 使用feature_dim={feature_dim}初始化融合模块")
        
        # 参数验证
        if rgb_dim is None or nir_dim is None:
            raise ValueError("必须提供rgb_dim和nir_dim参数，或者提供feature_dim参数")
        
        # 记录配置参数
        self.rgb_dim = rgb_dim
        self.nir_dim = nir_dim
        self.fusion_dim = fusion_dim
        
        # 特征投影层 - 静态初始化（移除动态创建机制）
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        self.nir_proj = nn.Linear(nir_dim, fusion_dim)
        
        # 记录初始化维度（用于参考）
        self.init_rgb_dim = rgb_dim
        self.init_nir_dim = nir_dim
        
        # LayerNorm层
        self.rgb_norm = LayerNorm(fusion_dim)
        self.nir_norm = LayerNorm(fusion_dim)
        self.fusion_norm = LayerNorm(fusion_dim)
        
        # 跨模态注意力
        self.rgb_to_nir_attention = CrossAttention(
            query_dim=fusion_dim,
            key_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.nir_to_rgb_attention = CrossAttention(
            query_dim=fusion_dim,
            key_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # dropout层
        self.fusion_dropout = nn.Dropout(dropout)
        
        # 残差连接的门控机制
        self.rgb_gate = nn.Parameter(torch.ones(1))
        self.nir_gate = nn.Parameter(torch.ones(1))
    
    # 已移除_update_projections方法，改为使用静态初始化的投影层
        
    def _attention(self, query, key, value, name="attention"):
        """
        计算跨模态注意力
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            name: 使用的注意力模块名称
            
        Returns:
            注意力输出张量
        """
        # 添加批次维度以与CrossAttention兼容
        q_batch = query.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        k_batch = key.unsqueeze(1)    # [batch_size, 1, fusion_dim]
        v_batch = value.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        
        # 应用注意力
        if name == "rgb_to_nir":
            out = self.rgb_to_nir_attention(q_batch, k_batch, v_batch)
        else:  # nir_to_rgb
            out = self.nir_to_rgb_attention(q_batch, k_batch, v_batch)
            
        # 移除序列维度
        out = out.squeeze(1)  # [batch_size, fusion_dim]
        return out
    
    def forward(self, rgb_features, nir_features):
        """
        融合RGB和NIR特征
        
        参数:
            rgb_features: RGB特征 [batch_size, seq_len, rgb_dim] 或 [batch_size, rgb_dim]
            nir_features: NIR特征 [batch_size, seq_len, nir_dim] 或 [batch_size, nir_dim]
            
        返回:
            融合后的特征 [batch_size, seq_len, fusion_dim] 或 [batch_size, 1, fusion_dim]
        """
        # 如果任一输入为None，返回另一个
        if rgb_features is None:
            return nir_features
        if nir_features is None:
            return rgb_features
        
        device = rgb_features.device
        
        # 处理极端维度情况 (如 [1, 1, 1, 20, 256])
        if len(rgb_features.shape) > 3:
            # 减少不必要的维度
            while len(rgb_features.shape) > 3 and rgb_features.shape[0] == 1:
                rgb_features = rgb_features.squeeze(0)
            # 确保最多是3D
            if len(rgb_features.shape) > 3:
                shape = rgb_features.shape
                rgb_features = rgb_features.reshape(shape[0], shape[-2], shape[-1])
                
        if len(nir_features.shape) > 3:
            # 减少不必要的维度
            while len(nir_features.shape) > 3 and nir_features.shape[0] == 1:
                nir_features = nir_features.squeeze(0)
            # 确保最多是3D
            if len(nir_features.shape) > 3:
                shape = nir_features.shape
                nir_features = nir_features.reshape(shape[0], shape[-2], shape[-1])
        
        # 0. 检查输入张量的维度并进行处理
        rgb_is_3d = len(rgb_features.shape) == 3
        nir_is_3d = len(nir_features.shape) == 3
        
        # 是否有序列维度
        has_seq_dim = rgb_is_3d or nir_is_3d
        
        # 保存原始形状以便重塑
        orig_rgb_shape = rgb_features.shape
        orig_nir_shape = nir_features.shape
        
        # 确保两者都是2D或3D
        if rgb_is_3d and not nir_is_3d:
            # NIR是2D [batch, dim]，需要扩展到3D [batch, 1, dim]
            nir_features = nir_features.unsqueeze(1)
            nir_is_3d = True
        elif nir_is_3d and not rgb_is_3d:
            # RGB是2D [batch, dim]，需要扩展到3D [batch, 1, dim]
            rgb_features = rgb_features.unsqueeze(1)
            rgb_is_3d = True
        
        # 如果两者都是3D，确保序列长度一致
        if rgb_is_3d and nir_is_3d:
            if rgb_features.shape[1] != nir_features.shape[1]:
                # 找到最短序列长度
                min_seq_len = min(rgb_features.shape[1], nir_features.shape[1])
                rgb_features = rgb_features[:, :min_seq_len]
                nir_features = nir_features[:, :min_seq_len]
        
        # 如果是3D张量 [batch, seq_len, dim]，展平为2D [batch*seq_len, dim]
        if rgb_is_3d:
            batch_size, seq_len, rgb_dim = rgb_features.shape
            rgb_features = rgb_features.reshape(-1, rgb_dim)
        
        if nir_is_3d:
            batch_size, seq_len, nir_dim = nir_features.shape
            nir_features = nir_features.reshape(-1, nir_dim)
        
        # 使用静态投影层，确保输入维度与初始化时一致
        if rgb_features.shape[-1] != self.init_rgb_dim:
            # 调整输入维度以匹配初始化维度
            rgb_features = torch.nn.functional.pad(
                rgb_features, 
                (0, self.init_rgb_dim - rgb_features.shape[-1])
            ) if rgb_features.shape[-1] < self.init_rgb_dim else rgb_features[:, :self.init_rgb_dim]
        
        if nir_features.shape[-1] != self.init_nir_dim:
            # 调整输入维度以匹配初始化维度
            nir_features = torch.nn.functional.pad(
                nir_features, 
                (0, self.init_nir_dim - nir_features.shape[-1])
            ) if nir_features.shape[-1] < self.init_nir_dim else nir_features[:, :self.init_nir_dim]
        
        # 投影特征到共同维度
        rgb_proj = self.rgb_proj(rgb_features)
        nir_proj = self.nir_proj(nir_features)
        
        # 应用简单的加权融合
        rgb_weight = 0.6  # 偏向RGB的权重
        nir_weight = 0.4
        
        # 应用权重
        rgb_weighted = rgb_proj * rgb_weight
        nir_weighted = nir_proj * nir_weight
        
        # 将加权特征直接相加，而不是拼接
        fused_features = rgb_weighted + nir_weighted
        
        # 使用线性层进行最终投影，确保维度符合期望
        fused_dropout = self.fusion_dropout(fused_features)
        result = self.fusion_norm(fused_dropout)
        
        # 确保返回的是3D张量，因为Mamba编码器期望输入是[batch_size, seq_len, d_model]
        if has_seq_dim:
            # 如果原始输入是3D的，将结果重塑回原始形状
            result = result.reshape(batch_size, seq_len, self.fusion_dim)
        else:
            # 如果原始输入是2D的，添加一个序列维度
            result = result.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        
        return result


class MambaEncoder(nn.Module):
    """基于Mamba的编码器，用于序列特征提取"""
    
    def __init__(self,
            d_model: int,
            d_state: int,
            depth: int,
            fs: float = 30.0,
            hr_band: Tuple[float, float] = (0.7, 2.5),
            br_band: Tuple[float, float] = (0.15, 0.4),
            dropout: float = 0.1,
            use_frequency_enhanced: bool = True,
            use_fast_path: bool = False,
            filter_d_state: Optional[int] = None):
        """
        初始化Mamba编码器
        
        参数:
            d_model: 模型维度
            d_state: 状态维度
            depth: 编码器层数
            fs: 采样频率 (Hz)
            hr_band: 心率频带 (Hz)
            br_band: 呼吸率频带 (Hz)
            dropout: Dropout比率
            use_frequency_enhanced: 是否使用频域增强
            use_fast_path: 是否使用CUDA优化路径
            filter_d_state: 可选的滤波器状态维度
        """
        super().__init__()
        
        self.d_model = d_model
        self.depth = depth
        self.use_frequency_enhanced = use_frequency_enhanced
        
        # 构建编码器层
        self.layers = nn.ModuleList()
        for i in range(depth):
            # 使用频域增强版本的Mamba块
            if use_frequency_enhanced and i > 0:  # 第一层使用标准Mamba，后续层使用频域增强版本
                # 优先使用filter_d_state，如果没有则使用固定值16，确保与训练时一致
                # 根据训练日志，需要确保此值为16而不是32
                filter_d = filter_d_state if filter_d_state is not None else 16  # 强制使用确定值16，而不是d_state//2
                self.layers.append(FrequencyEnhancedMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    fs=fs,
                    hr_band=hr_band,
                    br_band=br_band,
                    dropout=dropout,
                    use_fast_path=use_fast_path,
                    filter_d_state=filter_d  # 显式传递滤波器d_state参数
                ))
            else:
                self.layers.append(MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    use_fast_path=use_fast_path
                ))
        
        # 最终层规范化
        self.norm = LayerNorm(d_model)
        
    def forward(self, 
               x: torch.Tensor, 
               delta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch, seq_len, d_model]
            delta: 可选的时间步长调制
            
        返回:
            包含编码器输出和频域增强特征的字典
        """
        # 确保delta是tensor类型
        batch_size, seq_len, d_model = x.shape
        if delta is not None and isinstance(delta, (int, float)):
            delta_value = delta
            delta = torch.ones((batch_size, seq_len, d_model), device=x.device) * delta_value
            
        # 初始化频域增强特征存储
        hr_features = []
        br_features = []
        
        # 应用编码器层
        current = x
        for i, layer in enumerate(self.layers):
            # 应用层处理
            if isinstance(layer, FrequencyEnhancedMambaBlock):
                layer_output = layer(current, delta)
                current = layer_output['output']
                
                # 收集频域增强特征
                hr_features.append(layer_output['hr_enhanced'])
                br_features.append(layer_output['br_enhanced'])
            else:
                current = layer(current, delta)
        
        # 应用最终规范化
        output = self.norm(current)
        
        # 准备返回结果
        result = {'output': output}
        
        # 如果有频域增强特征，添加到结果
        if hr_features and br_features:
            # 平均各层的频域增强特征
            hr_enhanced = torch.stack(hr_features).mean(dim=0)
            br_enhanced = torch.stack(br_features).mean(dim=0)
            
            result.update({
                'hr_enhanced': hr_enhanced,
                'br_enhanced': br_enhanced
            })
            
        return result 