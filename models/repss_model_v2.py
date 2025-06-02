import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import torch.utils.checkpoint as checkpoint
import math
import logging

# 创建或获取logger
logger = logging.getLogger(__name__)

# 从mamba_ssm和FeatureExactor的导入
from .mamba_ssm import SSMBlock, FrequencySelectiveFilter
from .FeatureExactor import DualPathFeatureExtractor

@dataclass
class MambaModelConfig:
    """RePSS模型配置参数类"""
    def __init__(self):
        # Mamba网络核心参数
        self.d_model = 256          # 隐藏层维度
        self.n_layers = 6           # Mamba层数
        self.d_state = 128          # 状态空间维度
        self.expand = 2             # 扩展比例
        self.dropout = 0.1          # Dropout率
        
        # SSM相关参数
        self.dt_min = 0.01          # 时间步长范围和初始值
        self.dt_max = 0.1
        self.dt_init = 0.02
        
        # 多模态特征参数
        self.rgb_feature_dim = 256   # RGB和NIR特征维度
        self.nir_feature_dim = 256
        self.use_rgb = True          # 模态使用开关
        self.use_nir = True
        
        # 内存优化相关
        self.use_checkpointing = True   # 梯度检查点
        self.use_mixed_precision = True  # 混合精度训练
        self.max_batch_size = 2     # 限制批次大小节省内存
        
        # 信号处理参数
        self.FS = 30.0              # 采样率(Hz)
        self.HR_BAND = (0.7, 2.5)   # 心率和呼吸率频带(Hz)
        self.BR_BAND = (0.15, 0.4)
        self.verbose = False        # 日志详细程度
        self.USE_FREQUENCY_ENHANCED = True  # 频域增强
        self.USE_FAST_PATH = False


class SignalOptimizer(nn.Module):
    """信号优化器 - 结合心率特征与基准值进行优化"""
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 心率基准值处理网络
        self.base_hr_processor = nn.Sequential( 
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # 特征处理网络
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 特征和心率基准值融合
        self.fusion_layer = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # 深度特征提取网络
        self.shared_layers = nn.ModuleList()
        for i in range(num_layers):
            self.shared_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ))
        
        # 信号优化输出头
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 方向预测层 - 专门预测需要向上调整还是向下调整
        self.direction_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1是基准心率
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # 输出范围[-1,1]，正值表示需要增加，负值表示需要减少
        )
        
        # 输出放大因子，将输出范围限制到[-2.5, 2.5] BPM
        # 显著降低输出范围，提高预测稳定性，防止过大偏差
        self.output_scale = 2.5  # 从12.0降低到2.5
        
        # 方向校正因子
        self.direction_scale = 1.0
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数，确保网络初始输出接近零"""
        # 特别处理最后一层的偏置，使其初始输出接近零
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的标准差初始化权重
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 特别处理方向预测器的最后一层，使其初始偏向零预测
        if hasattr(self.direction_predictor[-2], 'bias'):
            nn.init.zeros_(self.direction_predictor[-2].bias)
        
        # 特别处理信号头的最后一层
        if hasattr(self.signal_head[-1], 'bias'):
            nn.init.zeros_(self.signal_head[-1].bias)
    
    def forward(self, features: torch.Tensor, base_hr: torch.Tensor) -> torch.Tensor:
        """
        前向传播，同时考虑特征和基准心率
        
        Args:
            features: 编码器的心率增强特征 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
            base_hr: 解码器输出的基准心率 [batch_size]
            
        Returns:
            心率偏置值 [batch_size, 1]
        """
        # 添加防御性检查
        if base_hr is None:
            logger.warning("SignalOptimizer接收到的base_hr为None")
            if features is None:
                logger.warning("特征也为None，返回零调整值")
                return torch.tensor(0.0, device=self.feature_processor[0].weight.device)
            device = features.device
            batch_size = features.shape[0] if len(features.shape) > 1 else 1
            return torch.zeros(batch_size, 1, device=device)
        
        batch_size = base_hr.shape[0]
        
        # 确保特征维度正确 [batch_size, feature_dim]
        if len(features.shape) == 3:
            # 如果是序列特征，取最后一个时间步的特征
            features = features[:, -1, :]
        
        # 处理心率基准值 [batch_size, hidden_dim//4]
        base_hr_expanded = base_hr.view(batch_size, 1)
        base_hr_features = self.base_hr_processor(base_hr_expanded)
        
        # 处理特征 [batch_size, hidden_dim]
        processed_features = self.feature_processor(features)
        
        # 融合特征和心率基准值 [batch_size, hidden_dim]
        combined = torch.cat([processed_features, base_hr_features], dim=1)
        fused = F.silu(self.fusion_layer(combined))
        
        # 应用共享特征提取网络，使用残差连接
        x = fused
        for layer in self.shared_layers:
            x = x + layer(x)  # 残差连接
        
        # 预测调整值 [batch_size, 1]
        delta_magnitude = self.signal_head(x)
        
        # 预测调整方向 [batch_size, 1]
        direction_input = torch.cat([x, base_hr_expanded], dim=1)
        adjustment_direction = self.direction_predictor(direction_input)
        
        # 估计真实心率与基准心率的差异
        # 使用真实心率估计器的输出和方向一起确定最终调整
        # 方向: -1表示减少心率，+1表示增加心率
        signed_delta = delta_magnitude * torch.sign(adjustment_direction) * self.direction_scale
        
        # 应用缩放和区间限制
        # 使用sigmoid进行平滑剪裁，确保输出在合理范围内
        scaled_delta = torch.tanh(signed_delta) * self.output_scale
        
        return scaled_delta


class RePSSModelV2(nn.Module):
    """RePSS模型V2：基于多模态融合的信号优化系统"""
    
    def __init__(self, config: MambaModelConfig):
        """
        初始化RePSS模型V2
        
        参数:
            config: 模型配置
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.verbose = getattr(config, 'verbose', False)
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', True)
        
        # 添加日志记录器
        import logging
        self.logger = logging.getLogger(__name__)
        self.max_batch_size = getattr(config, 'max_batch_size', 2)  # 从配置中获取最大批次大小，默认为2
        
        # 记录配置参数
        if self.verbose:
            print(f"初始化RePSS模型V2，配置: {config}")
        
        # RGB特征提取器
        if config.use_rgb:
            # 使用预训练特征提取器
            use_pretrained = getattr(config, 'use_pretrained_extractor', True)
            print(f"[INFO] 特征提取器初始化: {'使用预训练模型' if use_pretrained else '随机初始化'}")
            
            if use_pretrained:
                # 导入预训练模型
                try:
                    from torchvision.models import resnet18, ResNet18_Weights
                    print("[INFO] 使用ResNet18作为特征提取器骨干网络")
                    backbone_available = True
                except ImportError:
                    print("[WARNING] 无法导入预训练模型，将使用随机初始化")
                    backbone_available = False
            else:
                backbone_available = False
                
            # 创建特征提取器
            self.rgb_feature_extractor = DualPathFeatureExtractor(
                in_channels=3,  # RGB: 3通道
                feature_dim=config.rgb_feature_dim,
                fusion_dim=config.d_model,
                dropout=config.dropout,
                use_pretrained=use_pretrained and backbone_available
            )
            
            # 强制初始化RGB特征提取器的关键层
            # 首先直接搜索特征提取器的具体层并初始化
            spatial_extractor = getattr(self.rgb_feature_extractor, 'spatial_extractor', None)
            if spatial_extractor is not None:
                if hasattr(spatial_extractor, 'conv1'):
                    nn.init.kaiming_normal_(spatial_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
                    if spatial_extractor.conv1.bias is not None:
                        nn.init.constant_(spatial_extractor.conv1.bias, 0.01)
                if hasattr(spatial_extractor, 'bn1'):
                    nn.init.constant_(spatial_extractor.bn1.weight, 1.0)
                    nn.init.constant_(spatial_extractor.bn1.bias, 0.01)
            
            stmap_extractor = getattr(self.rgb_feature_extractor, 'stmap_extractor', None)
            if stmap_extractor is not None:
                if hasattr(stmap_extractor, 'pre_conv'):
                    nn.init.kaiming_normal_(stmap_extractor.pre_conv.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(stmap_extractor.pre_conv.bias, 0.01)
                if hasattr(stmap_extractor, 'pre_norm'):
                    nn.init.constant_(stmap_extractor.pre_norm.weight, 1.0)
                    nn.init.constant_(stmap_extractor.pre_norm.bias, 0.01)
                    
                # 初始化ST-Map卷积层
                conv_layers = ['stmap_conv1', 'stmap_conv2', 'stmap_conv3', 'stmap_conv4']
                norm_layers = ['stmap_norm1', 'stmap_norm2', 'stmap_norm3', 'stmap_norm4']
                
                for conv_name in conv_layers:
                    if hasattr(stmap_extractor, conv_name):
                        conv_layer = getattr(stmap_extractor, conv_name)
                        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
                        nn.init.constant_(conv_layer.bias, 0.01)
                        
                for norm_name in norm_layers:
                    if hasattr(stmap_extractor, norm_name):
                        norm_layer = getattr(stmap_extractor, norm_name)
                        nn.init.constant_(norm_layer.weight, 1.0)
                        nn.init.constant_(norm_layer.bias, 0.01)
            
            # 额外的递归初始化
            for m in self.rgb_feature_extractor.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    # 由于kaimang_normal_初始化时可能有小概率生成为零的权重，所以添加一个小的灰度因子
                    if torch.all(m.weight == 0):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # 确保权重不为零
                        if torch.all(m.weight == 0):
                            m.weight.data = m.weight.data + torch.randn_like(m.weight.data) * 0.01
                    if m.bias is not None and torch.all(m.bias == 0):
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    if torch.all(m.weight == 0):
                        nn.init.constant_(m.weight, 1.0)
                    if torch.all(m.bias == 0):
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.Linear):
                    if torch.all(m.weight == 0):
                        nn.init.normal_(m.weight, 0, 0.01)
                        # 确保权重不为零
                        if torch.all(m.weight == 0):
                            m.weight.data = m.weight.data + torch.randn_like(m.weight.data) * 0.01
                    if m.bias is not None and torch.all(m.bias == 0):
                        nn.init.constant_(m.bias, 0.01)
                    
            # 再次检查还存不存在权重为零的参数
            zero_params = 0
            total_params = 0
            for name, param in self.rgb_feature_extractor.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    if torch.all(param == 0):
                        zero_params += 1
                        print(f"[WARNING] 参数 {name} 仍然全为零！")
                        # 强制设置一个小的差异值
                        param.data = torch.randn_like(param.data) * 0.01
            
            print(f"[INFO] 已强制初始化RGB特征提取器权重, 共{total_params}个参数, 剩余{zero_params}个全零参数")
        
        if config.use_nir:
            self.nir_feature_extractor = DualPathFeatureExtractor(
                in_channels=1,  # NIR通常是单通道
                feature_dim=config.nir_feature_dim,
                fusion_dim=config.nir_feature_dim,
                dropout=config.dropout
            )
            
            # 强制初始化NIR特征提取器的关键层
            # 首先直接搜索特征提取器的具体层并初始化
            spatial_extractor = getattr(self.nir_feature_extractor, 'spatial_extractor', None)
            if spatial_extractor is not None:
                if hasattr(spatial_extractor, 'conv1'):
                    nn.init.kaiming_normal_(spatial_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
                    if spatial_extractor.conv1.bias is not None:
                        nn.init.constant_(spatial_extractor.conv1.bias, 0.01)
                if hasattr(spatial_extractor, 'bn1'):
                    nn.init.constant_(spatial_extractor.bn1.weight, 1.0)
                    nn.init.constant_(spatial_extractor.bn1.bias, 0.01)
            
            stmap_extractor = getattr(self.nir_feature_extractor, 'stmap_extractor', None)
            if stmap_extractor is not None:
                if hasattr(stmap_extractor, 'pre_conv'):
                    nn.init.kaiming_normal_(stmap_extractor.pre_conv.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(stmap_extractor.pre_conv.bias, 0.01)
                if hasattr(stmap_extractor, 'pre_norm'):
                    nn.init.constant_(stmap_extractor.pre_norm.weight, 1.0)
                    nn.init.constant_(stmap_extractor.pre_norm.bias, 0.01)
                    
                # 初始化ST-Map卷积层
                conv_layers = ['stmap_conv1', 'stmap_conv2', 'stmap_conv3', 'stmap_conv4']
                norm_layers = ['stmap_norm1', 'stmap_norm2', 'stmap_norm3', 'stmap_norm4']
                
                for conv_name in conv_layers:
                    if hasattr(stmap_extractor, conv_name):
                        conv_layer = getattr(stmap_extractor, conv_name)
                        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
                        nn.init.constant_(conv_layer.bias, 0.01)
                        
                for norm_name in norm_layers:
                    if hasattr(stmap_extractor, norm_name):
                        norm_layer = getattr(stmap_extractor, norm_name)
                        nn.init.constant_(norm_layer.weight, 1.0)
                        nn.init.constant_(norm_layer.bias, 0.01)
            
            # 额外的递归初始化
            for m in self.nir_feature_extractor.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    # 由于kaimang_normal_初始化时可能有小概率生成为零的权重，所以添加一个小的灰度因子
                    if torch.all(m.weight == 0):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # 确保权重不为零
                        if torch.all(m.weight == 0):
                            m.weight.data = m.weight.data + torch.randn_like(m.weight.data) * 0.01
                    if m.bias is not None and torch.all(m.bias == 0):
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    if torch.all(m.weight == 0):
                        nn.init.constant_(m.weight, 1.0)
                    if torch.all(m.bias == 0):
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.Linear):
                    if torch.all(m.weight == 0):
                        nn.init.normal_(m.weight, 0, 0.01)
                        # 确保权重不为零
                        if torch.all(m.weight == 0):
                            m.weight.data = m.weight.data + torch.randn_like(m.weight.data) * 0.01
                    if m.bias is not None and torch.all(m.bias == 0):
                        nn.init.constant_(m.bias, 0.01)
                    
            # 再次检查还存不存在权重为零的参数
            zero_params = 0
            total_params = 0
            for name, param in self.nir_feature_extractor.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    if torch.all(param == 0):
                        zero_params += 1
                        print(f"[WARNING] 参数 {name} 仍然全为零！")
                        # 强制设置一个小的差异值
                        param.data = torch.randn_like(param.data) * 0.01
            
            print(f"[INFO] 已强制初始化NIR特征提取器权重, 共{total_params}个参数, 剩余{zero_params}个全零参数")
        
        # 多模态融合模块 - 延迟导入以避免循环导入问题
        from .mamba_block import MultiModalFusionBlock
        self.fusion_block = MultiModalFusionBlock(
            rgb_dim=config.rgb_feature_dim,
            nir_dim=config.nir_feature_dim,
            fusion_dim=config.d_model,  # 使用统一的维度作为融合输出维度
            num_heads=getattr(config, 'NUM_HEADS', 4),
            dropout=config.dropout
        ) if config.use_rgb and config.use_nir else None
        
        # 特征投影层 - 用于单模态情况下将特征投影到正确维度
        self.rgb_projector = nn.Sequential(
            nn.Linear(config.rgb_feature_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(config.dropout)
        )
        
        self.nir_projector = nn.Sequential(
            nn.Linear(config.nir_feature_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(config.dropout)
        )
        
        # Mamba编码器 - 延迟导入以避免循环导入问题
        from .mamba_block import MambaEncoder
        self.mamba_encoder = MambaEncoder(
            d_model=config.d_model,  # 使用融合后的维度作为编码器输入维度
            d_state=config.d_state,  # 状态维度
            depth=getattr(config, 'depth', config.n_layers),  # 编码器深度，默认使用n_layers
            fs=getattr(config, 'fs', 30.0),  # 采样频率，默认30Hz
            hr_band=getattr(config, 'hr_band', (0.7, 2.5)),  # 心率频带，默认0.7-2.5Hz
            br_band=getattr(config, 'br_band', (0.15, 0.4)),  # 呼吸率频带，默认0.15-0.4Hz
            dropout=config.dropout,  # Dropout比率
            use_frequency_enhanced=getattr(config, 'use_frequency_enhanced', True),  # 是否使用频域增强
            use_fast_path=getattr(config, 'use_fast_path', False),  # 是否使用CUDA优化路径
            filter_d_state=getattr(config, 'filter_d_state', None)  # 可选的滤波器状态维度
        )
        
        # 基准心率预测器 - 使用分类解码器
        from .baseline_decoder import ClassificationHRDecoder, SpO2Decoder
        self.baseline_hr_predictor = ClassificationHRDecoder(config)
        
        # 基准呼吸率预测器
        self.baseline_br_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.LayerNorm(config.d_model // 2),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)  # 输出单个值作为基准呼吸率
        )
        
        # 血氧饱和度预测器
        self.spo2_predictor = SpO2Decoder(config)
        
        # 信号优化器 - 增强版本同时考虑心率增强特征和基准心率
        self.signal_optimizer = SignalOptimizer(
            feature_dim=config.d_model,
            hidden_dim=config.d_model // 2,
            num_layers=3  # 增加一层以增强处理能力
        )
        
        # 频域增强滤波器
        self.freq_filter = FrequencySelectiveFilter(
            d_model=config.d_model,
            d_state=config.d_state // 2,
            fs=config.FS,
            hr_band=config.HR_BAND,
            br_band=config.BR_BAND,
            trainable=True
        )
        
    def _forward_internal(self, rgb_frames=None, nir_frames=None, training: bool = False) -> Dict[str, torch.Tensor]:
        """
        内部前向传播实现，由forward方法调用
        
        Args:
            rgb_frames: RGB帧张量
            nir_frames: NIR帧张量（可选）
            training: 是否处于训练模式
            
        Returns:
            Dict: 包含预测的心率和呼吸率的字典
        """
        device = next(self.parameters()).device
        
        # 确定批次大小
        batch_size = rgb_frames.shape[0] if rgb_frames is not None else nir_frames.shape[0]
        
        # 确保数据在正确的设备上
        if rgb_frames is not None:
            rgb_frames = rgb_frames.to(device)
        if nir_frames is not None:
            nir_frames = nir_frames.to(device)
            
        # 确保NIR帧的格式正确
        if nir_frames is not None:
            # 如果是(batch, seq, H, W)，添加通道维度
            if len(nir_frames.shape) == 4:
                nir_frames = nir_frames.unsqueeze(2)  # 变为(batch, seq, 1, H, W)
            
            # 检查通道维度是否正确
            if nir_frames.shape[2] != 1 and len(nir_frames.shape) == 5:
                # 可能是(batch, seq, H, W, 1)
                if nir_frames.shape[4] == 1:
                    nir_frames = nir_frames.permute(0, 1, 4, 2, 3)  # 变为(batch, seq, 1, H, W)
                    
        # 添加标记来避免重复初始化
        self._weights_initialized = True
        
        # 检查是否有真实心率作为输入（用于监督学习）
        heart_rate_gt = None
        
        # 提取特征
        with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
            # 使用RGB特征提取器
            rgb_features = None
            nir_features = None
            
            # 明确记录当前处理的模态类型
            modality_type = None
            if rgb_frames is not None and nir_frames is not None:
                modality_type = "dual_modal"
            elif rgb_frames is not None:
                modality_type = "rgb_only"
            elif nir_frames is not None:
                modality_type = "nir_only"
            
            # 1. 处理RGB特征
            if rgb_frames is not None:
                rgb_features = self.rgb_feature_extractor(rgb_frames)
            
            # 2. 处理NIR特征
            if nir_frames is not None:
                nir_features = self.nir_feature_extractor(nir_frames)
            
            # 3. 特征融合处理
            fused_features = None
            
            # 双模态情况：执行融合
            if modality_type == "dual_modal" and rgb_features is not None and nir_features is not None:
                fused_features = self.fusion_block(rgb_features, nir_features)
            elif modality_type == "rgb_only":
                # 单模态RGB：直接使用RGB特征
                fused_features = rgb_features
            elif modality_type == "nir_only":
                # 单模态NIR：直接使用NIR特征
                fused_features = nir_features
            heart_rate_gt = None
            
            # 提取特征
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                # 使用RGB特征提取器
                rgb_features = None
                nir_features = None
                
                # 明确记录当前处理的模态类型
                modality_type = None
                if rgb_frames is not None and nir_frames is not None:
                    modality_type = "dual_modal"
                    if hasattr(self, 'logger'):
                        self.logger.info("检测到双模态数据 (RGB + NIR)")
                elif rgb_frames is not None:
                    modality_type = "rgb_only"
                    if hasattr(self, 'logger'):
                        self.logger.info("检测到单模态RGB数据")
                elif nir_frames is not None:
                    modality_type = "nir_only"
                    if hasattr(self, 'logger'):
                        self.logger.info("检测到单模态NIR数据")
                
                # 1. 处理RGB特征
                if rgb_frames is not None:
                    try:
                        # 调试输出
                        # print(f"[DEBUG] RGB帧输入的形状: {rgb_frames.shape}, 值范围: {torch.min(rgb_frames).item()} ~ {torch.max(rgb_frames).item()}")
                        
                        # 调用RGB特征提取器
                        # print(f"[DEBUG] 调用RGB特征提取器开始...")
                        rgb_features = self.rgb_feature_extractor(rgb_frames)
                        # print(f"[DEBUG] 调用RGB特征提取器完成")
                        
                        # 检查RGB特征有效性
                        if rgb_features is not None:
                            zero_percentage = (rgb_features == 0).float().mean().item() * 100
                            # print(f"[DEBUG] RGB特征形状: {rgb_features.shape}, 零元素比例: {zero_percentage:.2f}%, 均值: {torch.mean(rgb_features).item():.6f}, 标准差: {torch.std(rgb_features).item():.6f}")
                            
                        else:
                            # 单模态RGB模式下返回默认输出
                            if modality_type == "rgb_only":
                                print(f"[ERROR] 单模态RGB下特征提取失败，返回默认输出")
                                return self._create_default_output(batch_size, device)
                    except Exception as e:
                        print(f"[ERROR] RGB特征提取失败: {e}")
                        traceback.print_exc()
                        # 单模态RGB下返回默认输出
                        if modality_type == "rgb_only":
                            return self._create_default_output(batch_size, device)
                
                # 2. 处理NIR特征
                if nir_frames is not None:
                    try:
                        # 调试输出
                        # print(f"[DEBUG] NIR帧输入的形状: {nir_frames.shape}, 值范围: {torch.min(nir_frames).item()} ~ {torch.max(nir_frames).item()}")
                        
                        # 检查NIR特征提取器
                        if hasattr(self, 'nir_feature_extractor'):
                            # print(f"[DEBUG] NIR特征提取器类型: {type(self.nir_feature_extractor).__name__}")
                            # 检查权重
                            spatial_extractor = getattr(self.nir_feature_extractor, 'spatial_extractor', None)
                            if spatial_extractor is not None:
                                spatial_params = list(spatial_extractor.parameters())
                                if spatial_params:
                                    spatial_weight_sum = sum(p.abs().sum().item() for p in spatial_params if p.requires_grad)
                                    # print(f"[DEBUG] NIR空间特征提取器有效权重总和: {spatial_weight_sum:.6f}")
                        
                        # 调用NIR特征提取器
                        # print(f"[DEBUG] 调用NIR特征提取器开始...")
                        nir_features = self.nir_feature_extractor(nir_frames)
                        # print(f"[DEBUG] 调用NIR特征提取器完成")
                        
                        # 检查NIR特征有效性
                        if nir_features is not None:
                            zero_percentage = (nir_features == 0).float().mean().item() * 100
                            # print(f"[DEBUG] NIR特征形状: {nir_features.shape}, 零元素比例: {zero_percentage:.2f}%, 均值: {torch.mean(nir_features).item():.6f}, 标准差: {torch.std(nir_features).item():.6f}")
                            
                            # 如果NIR特征全为零，尝试生成简单特征
                            if torch.all(nir_features == 0).item():
                                # print(f"[WARNING] NIR特征全为零，尝试手动修复")
                                batch_size, seq_len = nir_frames.shape[0], nir_frames.shape[1]
                                nir_features = torch.mean(nir_frames, dim=[3, 4])  # 平均池化
                                if nir_features.shape[2] != self.config.nir_feature_dim:
                                    nir_features = nn.functional.interpolate(
                                        nir_features.transpose(1, 2),
                                        size=self.config.nir_feature_dim,
                                        mode='linear'
                                    ).transpose(1, 2)
                                # print(f"[DEBUG] 手动生成的NIR特征形状: {nir_features.shape}")
                        else:
                            # 单模态NIR模式下返回默认输出
                            if modality_type == "nir_only":
                                print(f"[ERROR] 单模态NIR下特征提取失败，返回默认输出")
                                return self._create_default_output(batch_size, device)
                    except Exception as e:
                        print(f"[ERROR] NIR特征提取失败: {e}")
                        traceback.print_exc()
                        # 单模态NIR下返回默认输出
                        if modality_type == "nir_only":
                            return self._create_default_output(batch_size, device)
                
                # 3. 特征融合处理
                fused_features = None
                
                # 双模态情况：执行融合
                if modality_type == "dual_modal" and rgb_features is not None and nir_features is not None:
                    try:
                        # 检查形状匹配
                        rgb_shape = rgb_features.shape
                        nir_shape = nir_features.shape
                        
                        # 调整形状不匹配的情况
                        if rgb_shape[1:] != nir_shape[1:]:
                            print(f"[WARNING] RGB与NIR特征形状不匹配: RGB={rgb_shape}, NIR={nir_shape}")
                            
                            # 序列长度调整
                            if rgb_shape[1] != nir_shape[1]:
                                min_seq_len = min(rgb_shape[1], nir_shape[1])
                                rgb_features = rgb_features[:, :min_seq_len, :]
                                nir_features = nir_features[:, :min_seq_len, :]
                                print(f"[INFO] 调整序列长度为: {min_seq_len}")
                            
                            # 特征维度调整
                            if rgb_shape[2] != nir_shape[2]:
                                print(f"[INFO] 调整NIR特征维度: {nir_shape[2]} -> {rgb_shape[2]}")
                                # 使用线性层调整
                                nir_features_reshaped = nir_features.reshape(-1, nir_shape[2])
                                adapter_name = f"nir_adapter_{nir_shape[2]}_{rgb_shape[2]}"
                                if not hasattr(self, adapter_name):
                                    setattr(self, adapter_name, nn.Linear(nir_shape[2], rgb_shape[2]).to(device))
                                    adapter = getattr(self, adapter_name)
                                    nn.init.xavier_uniform_(adapter.weight)
                                    nn.init.zeros_(adapter.bias)
                                
                                adapter = getattr(self, adapter_name)
                                nir_features_adapted = adapter(nir_features_reshaped)
                                nir_features = nir_features_adapted.reshape(nir_shape[0], nir_shape[1], rgb_shape[2])
                        
                        # 执行融合
                        if hasattr(self, 'logger'):
                            self.logger.info("执行多模态融合")
                        fused_features = self.fusion_block(rgb_features, nir_features)
                        # print(f"[DEBUG] 融合特征形状: {fused_features.shape}, 均值: {torch.mean(fused_features).item():.6f}, 标准差: {torch.std(fused_features).item():.6f}")
                        
                        # 检查融合结果有效性
                        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
                            # print(f"[ERROR] 融合特征包含NaN或Inf，退回使用RGB特征")
                            fused_features = rgb_features
                            modality_type = "rgb_only"
                    except Exception as e:
                        print(f"[ERROR] 特征融合失败: {e}")
                        traceback.print_exc()
                        # 融合失败时退回使用RGB特征
                        fused_features = rgb_features
                        modality_type = "rgb_only"
                elif modality_type == "rgb_only":
                    # 单模态RGB：直接使用RGB特征
                    fused_features = rgb_features
                    if hasattr(self, 'logger'):
                        self.logger.info("单模态RGB处理: 直接使用RGB特征")
                    # print(f"[DEBUG] RGB单模态特征形状: {fused_features.shape}")
                    
                    # 确保维度与Mamba编码器期望维度匹配
                    if fused_features.shape[2] != self.config.d_model:
                        # print(f"[INFO] 调整RGB特征维度以匹配编码器: {fused_features.shape[2]} -> {self.config.d_model}")
                        # 创建或使用投影器
                        if not hasattr(self, 'rgb_dim_adapter') or self.rgb_dim_adapter.in_features != fused_features.shape[2]:
                            self.rgb_dim_adapter = nn.Linear(fused_features.shape[2], self.config.d_model).to(device)
                            nn.init.xavier_uniform_(self.rgb_dim_adapter.weight)
                            nn.init.zeros_(self.rgb_dim_adapter.bias)
                        
                        # 应用投影
                        fused_features_flat = fused_features.reshape(-1, fused_features.shape[2])
                        fused_features_projected = self.rgb_dim_adapter(fused_features_flat)
                        fused_features = fused_features_projected.reshape(fused_features.shape[0], fused_features.shape[1], self.config.d_model)
                        # print(f"[DEBUG] 调整后的RGB特征形状: {fused_features.shape}")
                        
                elif modality_type == "nir_only":
                    # 单模态NIR：直接使用NIR特征
                    fused_features = nir_features
                    if hasattr(self, 'logger'):
                        # self.logger.info("单模态NIR处理: 直接使用NIR特征")
                        pass
                    # print(f"[DEBUG] NIR单模态特征形状: {fused_features.shape}")
                    
                    # 确保维度与Mamba编码器期望维度匹配
                    if fused_features.shape[2] != self.config.d_model:
                        # print(f"[INFO] 调整NIR特征维度以匹配编码器: {fused_features.shape[2]} -> {self.config.d_model}")
                        # 创建或使用投影器
                        if not hasattr(self, 'nir_dim_adapter') or self.nir_dim_adapter.in_features != fused_features.shape[2]:
                            self.nir_dim_adapter = nn.Linear(fused_features.shape[2], self.config.d_model).to(device)
                            nn.init.xavier_uniform_(self.nir_dim_adapter.weight)
                            nn.init.zeros_(self.nir_dim_adapter.bias)
                        
                        # 应用投影
                        fused_features_flat = fused_features.reshape(-1, fused_features.shape[2])
                        fused_features_projected = self.nir_dim_adapter(fused_features_flat)
                        fused_features = fused_features_projected.reshape(fused_features.shape[0], fused_features.shape[1], self.config.d_model)
                        # print(f"[DEBUG] 调整后的NIR特征形状: {fused_features.shape}")
                
                # 最终特征有效性检查
                if fused_features is None:
                    # print(f"[ERROR] 最终特征为None，返回默认输出")
                    return self._create_default_output(batch_size, device)
                
                # 检查特征是否全为零
                if torch.all(fused_features == 0).item():
                    # print(f"[ERROR] 最终特征全为零，返回默认输出")
                    return self._create_default_output(batch_size, device)
                
                # 检查特征是否包含NaN或Inf
                if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
                    # print(f"[ERROR] 最终特征包含NaN或Inf值，返回默认输出")
                    return self._create_default_output(batch_size, device)
                
                # print(f"[DEBUG] 融合特征非零，形状: {fused_features.shape}, 均值: {torch.mean(fused_features).item()}, 标准差: {torch.std(fused_features).item()}")
                
                # 使用Mamba编码器处理融合特征
                try:
                    encoded_features = self.mamba_encoder(fused_features)
                except Exception as e:
                    print(f"[ERROR] Mamba编码失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return self._create_default_output(batch_size, device)
                
                # 处理编码器输出 - 可能是字典或张量
                if isinstance(encoded_features, dict):
                    # 如果是字典，获取'output'键对应的张量
                    encoded_output = encoded_features['output']
                    # 保存增强特征以供后续使用
                    hr_enhanced = encoded_features.get('hr_enhanced', None)
                    br_enhanced = encoded_features.get('br_enhanced', None)
                else:
                    # 如果直接是张量，直接使用
                    encoded_output = encoded_features
                    hr_enhanced = None
                    br_enhanced = None
                
                # 使用分类解码器预测基准心率和不确定性
                # 传入完整的特征字典，包含原始编码器输出和频域增强特征
                # 确保在验证模式下也正确调用解码器
                try:
                    # 添加详细日志，记录编码器输出结构
                    if hasattr(self, 'logger'):
                        self.logger.info(f"编码器输出类型: {type(encoded_output)}")
                        if isinstance(encoded_output, dict):
                            self.logger.info(f"编码器输出字典元素: {encoded_output.keys()}")
                        elif isinstance(encoded_output, torch.Tensor):
                            self.logger.info(f"编码器输出张量尺寸: {encoded_output.shape}")
                    
                    # 传入multi_task=True，启用分类预测
                    decoder_output = self.baseline_hr_predictor(encoded_output, multi_task=True)
                    
                    if hasattr(self, 'logger'):
                        self.logger.info(f"解码器输出类型: {type(decoder_output)}")
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"调用解码器出错: {e}")
                        self.logger.error(traceback.format_exc())
                    # 出错时创建一个默认字典
                    decoder_output = {'regression': torch.ones(batch_size, device=device) * 75.0}
                
                # ClassificationHRDecoder返回字典，包含回归值和分类logits
                # 处理解码器输出并提取基准心率信息
                try:
                    if isinstance(decoder_output, dict):
                        # 如果输出是字典格式 - 使用全局回归而非加权回归
                        base_hr = decoder_output.get('regression_global', decoder_output.get('regression', None))
                        hr_uncertainty = decoder_output.get('uncertainty', None)
                        hr_logits = decoder_output.get('logits', None)
                        
                        # 记录使用的是哪种回归结果
                        if hasattr(self, 'logger') and 'regression_global' in decoder_output:
                            self.logger.info(f"使用全局回归(regression_global)作为心率预测")
                        
                        # 详细记录解码器输出情况
                        if hasattr(self, 'logger'):
                            self.logger.info(f"解码器输出字典内容: {list(decoder_output.keys())}")
                            if base_hr is not None:
                                self.logger.info(f"基准心率: {base_hr.shape if hasattr(base_hr, 'shape') else 'None'}")
                            
                    elif isinstance(decoder_output, tuple) and len(decoder_output) == 2:
                        # 如果输出是元组(hr, uncertainty)
                        base_hr, hr_uncertainty = decoder_output
                        if hasattr(self, 'logger'):
                            self.logger.info(f"从元组中提取基准心率: {base_hr.shape if hasattr(base_hr, 'shape') else 'None'}")
                    else:
                        # 如果是其他类型，假设是心率值
                        base_hr = decoder_output
                        hr_uncertainty = torch.ones_like(base_hr) * 0.1  # 设置默认不确定性
                        if hasattr(self, 'logger'):
                            self.logger.info(f"将其他类型结果作为基准心率: {type(decoder_output)}")
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"处理解码器输出时出错: {e}")
                        self.logger.error(traceback.format_exc())
                    base_hr = torch.ones(batch_size, device=device) * 75.0
                    hr_uncertainty = torch.ones_like(base_hr) * 5.0
                    hr_logits = None
                
                # self.logger.info(f"结果{base_hr}")
                
                # 保存不确定性和活动状态概率以便在输出中使用
                self._hr_uncertainty = hr_uncertainty
                
                # 使用原有方法预测基准呼吸率
                global_features = encoded_output.mean(dim=1)
                base_br_raw = self.baseline_br_predictor(global_features)  # [batch_size, 1]
                
                # 将呼吸率预测值调整到合理范围
                # 呼吸率范围：12-20 bpm（静息状态下的正常范围）
                base_br = 12.0 + torch.sigmoid(base_br_raw) * 8.0  # 映射到12-20的范围
                base_br = base_br.squeeze(-1)  # 移除最后一个维度，变成[batch_size]
                
                # 预测血氧饱和度
                try:
                    # 使用SpO2解码器预测血氧值
                    spo2, spo2_uncertainty = self.spo2_predictor(encoded_output)
                    if hasattr(self, 'logger'):
                        self.logger.info(f"血氧饱和度预测结果: {spo2.mean().item():.2f}%, "
                                       f"不确定性: {spo2_uncertainty.mean().item():.4f}")
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"血氧饱和度预测出错: {e}")
                        self.logger.error(traceback.format_exc())
                    # 出错时使用默认值: 97%是健康成人的典型平均值
                    spo2 = torch.ones(batch_size, device=device) * 97.0
                    spo2_uncertainty = torch.ones(batch_size, device=device) * 0.5
                
                # 将预测值存储为类属性，确保它们可以在输出中使用
                self._base_hr = base_hr
                self._base_br = base_br
                self._spo2 = spo2
                self._spo2_uncertainty = spo2_uncertainty
                
                with torch.amp.autocast(device_type='cuda', enabled=self.use_mixed_precision):
                    # 使用信号优化器预测心率和呼吸率的偏置值
                    # 使用编码器输出的心率增强特征和解码器输出的基准心率
                    hr_enhanced_features = None
                    if isinstance(encoded_output, dict) and 'hr_enhanced' in encoded_output:
                        hr_enhanced_features = encoded_output['hr_enhanced']
                    else:
                        # 如果没有心率增强特征，使用原始编码器输出
                        hr_enhanced_features = encoded_output
                        
                    # 检查特征是否非常小或接近零，如果是，添加小的噪声
                    if torch.mean(torch.abs(hr_enhanced_features)).item() < 1e-6:
                        hr_enhanced_features = hr_enhanced_features + torch.randn_like(hr_enhanced_features) * 0.01
                    
                    # 添加try-catch来处理信号优化器可能的错误
                    try:
                        # 验证输入是否有效
                        if base_hr is not None and hr_enhanced_features is not None:
                            signal_delta_raw = self.signal_optimizer(hr_enhanced_features, base_hr)
                            # 信号优化器现在返回的是 [batch_size, 1] 形状的偏置值
                            signal_delta = signal_delta_raw
                        else:
                            logger.warning("基准心率或增强特征为None，使用零修正值")
                            signal_delta = torch.zeros(batch_size, 1, device=device)
                    except Exception as e:
                        logger.error(f"信号优化器执行出错: {str(e)}")
                        # 出错时使用零修正值
                        signal_delta = torch.zeros(batch_size, 1, device=device)
                    
                    # 确保信号偏置有正确的维度 [batch_size, 2]
                    if signal_delta.shape[1] == 1:
                        # 如果只有一个特征（心率偏置），为呼吸率创建零偏置
                        zero_br_delta = torch.zeros_like(signal_delta)
                        signal_delta = torch.cat([signal_delta, zero_br_delta], dim=1)
                    
                    # 增强的基准心率和呼吸率处理
                    # 首先检查基准心率是否有效
                    has_valid_base_hr = False
                    
                    # 检查base_hr的有效性
                    if base_hr is not None:
                        try:
                            # 检查是否是张量并且不包含任何NaN或Inf
                            if isinstance(base_hr, torch.Tensor) and not torch.isnan(base_hr).any() and not torch.isinf(base_hr).any():
                                has_valid_base_hr = True
                        except Exception as e:
                            logger.warning(f"检查基准心率有效性时出错: {e}")
                    
                    # 如果基准心率无效，使用默认值
                    if not has_valid_base_hr:
                        logger.warning("基准心率无效，使用健康默认值75 BPM")
                        base_hr = torch.ones((batch_size,), dtype=torch.float32, device=device) * 75.0
                    
                    # 对呼吸率进行类似处理
                    has_valid_base_br = False
                    if base_br is not None:
                        try:
                            if isinstance(base_br, torch.Tensor) and not torch.isnan(base_br).any() and not torch.isinf(base_br).any():
                                has_valid_base_br = True
                        except Exception as e:
                            logger.warning(f"检查基准呼吸率有效性时出错: {e}")
                    
                    if not has_valid_base_br:
                        logger.warning("基准呼吸率无效，使用健康默认值16 BPM")
                        base_br = torch.ones((batch_size,), dtype=torch.float32, device=device) * 16.0
                    
                    # 现在可以安全地进行运算
                    heart_rate = base_hr.unsqueeze(-1) + signal_delta[:, 0:1]  
                    respiration_rate = base_br.unsqueeze(-1) + signal_delta[:, 1:2]
                # self.logger.info(f"最终的心率偏置{signal_delta}")
                
                # 构建结果字典 - 确保所有必需的键都存在
                result_dict = {
                    'hr': heart_rate,                   # 兼容旧键名
                    'heart_rate': heart_rate,          # 兼容新键名
                    'br': respiration_rate,            # 兼容旧键名
                    'respiration_rate': respiration_rate,  # 兼容新键名
                    'signal_delta': signal_delta,
                    'base_hr': base_hr,                # 确保基准心率被添加到输出中
                    'base_br': base_br,                # 确保基准呼吸率被添加到输出中
                    'uncertainty': hr_uncertainty,      # 添加不确定性
                    'hr_uncertainty': hr_uncertainty,   # 兼容旧键名
                    'logits': hr_logits,               # 添加分类logits以支持多任务学习
                    'regression': base_hr,              # 添加回归预测值以支持多任务学习
                    'spo2': spo2,                      # 血氧饱和度预测值
                    'spo2_uncertainty': spo2_uncertainty # 血氧饱和度预测的不确定性
                }
                
                # 最后再次检查结果字典中是否包含hr键
                if 'hr' not in result_dict:
                    self.logger.error("HR键在结果字典中丢失，添加默认值")
                    result_dict['hr'] = torch.ones(batch_size, 1, device=device) * 80.0
                
                # 添加标记表示这不是默认输出
                result_dict['_is_default_output'] = False
                
                return result_dict
    

    
    def forward(self, inputs, training: bool = False):
        """模型前向计算入口点，支持字典和张量输入"""
        # 从输入数据中提取RGB和NIR帧
        if isinstance(inputs, dict):
            # 支持rgb/rgb_frames和nir/nir_frames键名
            rgb_frames = inputs.get('rgb', inputs.get('rgb_frames', None))
            nir_frames = inputs.get('nir', inputs.get('nir_frames', None))
        else:
            # 非字典输入则当作RGB帧处理
            rgb_frames = inputs
            nir_frames = None
        
        # 执行内部计算并格式化结果
        outputs = self._forward_internal(rgb_frames, nir_frames, training)
        
        # 转换为字典格式并确保全为PyTorch张量
        if not isinstance(outputs, dict):
            outputs = {'heart_rate': outputs}
            
        for key, value in outputs.items():
            if value is not None and not isinstance(value, torch.Tensor):
                device = next(self.parameters()).device
                outputs[key] = torch.tensor(value, dtype=torch.float32, device=device)
        
        return outputs
    

    
    def _initialize_extractor_weights(self):
        """
        特征提取器权重初始化/重置方法，确保所有权重非零
        在前向传播时，如果发现全零权重，将调用此方法重置权重
        """
        print("[INFO] 正在重新初始化特征提取器权重...")
        
        # 1. RGB特征提取器初始化
        if hasattr(self, "rgb_feature_extractor"):
            print("[INFO] 初始化RGB特征提取器权重...")
            # 特征提取器的核心层直接初始化
            extractor = self.rgb_feature_extractor
            
            # 初始化空间特征提取器
            if hasattr(extractor, "spatial_extractor"):
                spatial = extractor.spatial_extractor
                for name, module in spatial.named_modules():
                    if isinstance(module, nn.Conv2d):
                        print(f"  初始化RGB空间特征层: {name}")
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.01)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1.0)
                        nn.init.constant_(module.bias, 0.01)
            
            # 初始化ST-Map特征提取器
            if hasattr(extractor, "stmap_extractor"):
                stmap = extractor.stmap_extractor
                for name, module in stmap.named_modules():
                    if isinstance(module, nn.Conv2d):
                        print(f"  初始化RGB ST-Map特征层: {name}")
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.01)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1.0)
                        nn.init.constant_(module.bias, 0.01)
        
        # 2. NIR特征提取器初始化
        if hasattr(self, "nir_feature_extractor"):
            print("[INFO] 初始化NIR特征提取器权重...")
            extractor = self.nir_feature_extractor
            
            # 初始化空间特征提取器
            if hasattr(extractor, "spatial_extractor"):
                spatial = extractor.spatial_extractor
                for name, module in spatial.named_modules():
                    if isinstance(module, nn.Conv2d):
                        print(f"  初始化NIR空间特征层: {name}")
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.01)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1.0)
                        nn.init.constant_(module.bias, 0.01)
            
            # 初始化ST-Map特征提取器
            if hasattr(extractor, "stmap_extractor"):
                stmap = extractor.stmap_extractor
                for name, module in stmap.named_modules():
                    if isinstance(module, nn.Conv2d):
                        print(f"  初始化NIR ST-Map特征层: {name}")
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.01)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1.0)
                        nn.init.constant_(module.bias, 0.01)
            
            device = next(self.parameters()).device
            result_dict = self._create_default_output(1, device)
            
            # 确保心率键名一致性
            if 'heart_rate' in inputs:
                default_hr = torch.ones((1,), dtype=torch.float32, device=device) * 75.0
                result_dict['heart_rate'] = default_hr
            
            return result_dict