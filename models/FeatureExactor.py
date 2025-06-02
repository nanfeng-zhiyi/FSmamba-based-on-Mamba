#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双路径特征提取器

结合空间和时空特征(STMap)的提取模块，为RePSSModelV2提供特征输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Union, Any


# 添加全局调试变量控制打印详细程度
DEBUG_FEATURE_EXTRACTOR = False  # 关闭调试打印以快速训练

# 记录内存使用情况的函数
def print_gpu_memory(msg=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        cached = torch.cuda.memory_reserved() / (1024 * 1024)
        free, total = torch.cuda.mem_get_info()
        free = free / (1024 * 1024 * 1024)
        total = total / (1024 * 1024 * 1024)
        print(f"GPU内存 [{msg}] - 已分配: {allocated:.2f}MB | 已缓存: {cached:.2f}MB | 可用: {free:.2f}GB / {total:.2f}GB")
    else:
        print(f"GPU内存 [{msg}] - CUDA不可用")

class STMapFeatureExtractor(nn.Module):
    """时空图特征提取器 - 捕获信号时域变化"""
    
    def __init__(self, in_channels: int, feature_dim: int, dropout: float = 0.1):
        """RGB(3通道)或NIR(1通道)输入的特征提取器"""
        super().__init__()
        self.feature_dim = feature_dim
        
        # 预处理层
        self.pre_conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pre_norm = nn.BatchNorm2d(16)
        
        # STMap特征提取网络 - 4层卷积结构
        self.stmap_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.stmap_norm1 = nn.BatchNorm2d(32)
        self.stmap_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.stmap_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.stmap_norm2 = nn.BatchNorm2d(64)
        self.stmap_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.stmap_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.stmap_norm3 = nn.BatchNorm2d(128)
        
        self.stmap_conv4 = nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1)
        self.stmap_norm4 = nn.BatchNorm2d(feature_dim)
        
        # 池化和正则化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将视频帧序列转换为STMap特征 [B, D, S, W]"""
        original_device = x.device
        
        # 创建备用特征（当输入有问题时使用）
        batch_size, seq_len, channels, height, width = x.shape
        base_feature = torch.ones((batch_size, 16, seq_len, width), device=original_device, dtype=x.dtype) * 0.5
        noise_feature = torch.randn((batch_size, 16, seq_len, width), device=original_device, dtype=x.dtype) * 0.01
        fallback_feature = base_feature + noise_feature
        
        # 处理单帧输入
        if len(x.shape) == 4:  # [B, C, H, W]
            batch_size, channels, height, width = x.shape
            seq_len = 1
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
        else:
            batch_size, seq_len, channels, height, width = x.shape
            
            # 预处理帧序列
            buffer_device = x.device
            
            # 预分配缓冲区
            processed_frames = []
            processed_buffer = torch.zeros((batch_size, 16, seq_len, height, width), 
                                          device=buffer_device, 
                                          dtype=torch.float16 if x.dtype == torch.float32 else x.dtype)
            use_buffer = True
            
            # 处理每一帧
            for i in range(seq_len):
                # 提取单帧
                frame = x[:, i]  # [batch_size, channels, height, width]
                
                # 执行卷积操作
                processed = self.pre_conv(frame)  # 首先执行卷积
                processed = self.pre_norm(processed)  # 然后归一化
                processed = self.act(processed)  # 最后激活
                
                # 将处理后的帧存到缓冲区中
                processed_buffer[:, :, i] = processed.to(dtype=processed_buffer.dtype)
            
            # 打印堆叠前的内存情况
            if DEBUG_FEATURE_EXTRACTOR:
                if use_buffer:
                    print(f"[STMap] 使用预分配缓冲区方式，缓冲区形状: {processed_buffer.shape}, 设备: {processed_buffer.device}")
                else:
                    print(f"[STMap] 使用列表方式，处理帧列表长度: {len(processed_frames)}帧")
                print_gpu_memory("帧堆叠前")
            
            # 根据存储方式选择不同的处理路径
            if use_buffer:
                # 直接在缓冲区所在的设备上计算均值
                stmap = torch.mean(processed_buffer, dim=3)  # [batch_size, 16, seq_len, width]
                # 确保结果在目标设备上
                stmap = stmap.to(x.device)
            else:
                # 堆叠帧并计算均值
                processed_stack = torch.stack(processed_frames, dim=2)  # [batch_size, 16, seq_len, height, width]
                stmap = torch.mean(processed_stack, dim=3)  # [batch_size, 16, seq_len, width]
                
                # 确保结果在目标设备上
                stmap = stmap.to(x.device)
                        
    
    def generate_stmap(self, x: torch.Tensor) -> torch.Tensor:
        """生成时空图 (STMap)
        
        参数:
            x: 输入序列 [batch_size, seq_len, channels, height, width]
        
        返回:
            STMap特征 [batch_size, channels, seq_len, width]
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 应用预处理卷积进行特征提取
        # 首先将帧序列重塑为 [batch_size*seq_len, channels, height, width]
        reshaped_frames = x.view(batch_size * seq_len, channels, height, width)
        processed_frames = self.act(self.pre_norm(self.pre_conv(reshaped_frames)))
        
        # 将处理后的帧重塑回 [batch_size, seq_len, 16, H', W']
        h_out = processed_frames.shape[2]
        w_out = processed_frames.shape[3]
        processed_frames = processed_frames.view(batch_size, seq_len, 16, h_out, w_out)
        
        # 生成STMap，汇总高度维度生成时空图
        stmap = torch.mean(processed_frames, dim=3)  # [batch_size, seq_len, 16, W']
        
        # 调整维度顺序为 [batch_size, 16, seq_len, width]
        stmap = stmap.permute(0, 2, 1, 3)
        
        return stmap
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, channels, height, width]
            
        返回:
            特征向量 [batch_size, feature_dim]
        """
        # 生成STMap
        stmap = self.generate_stmap(x)  # [batch_size, 16, seq_len, width]
        
        # 应用STMap特征提取网络
        x = self.act(self.stmap_norm1(self.stmap_conv1(stmap)))
        x = self.stmap_pool1(x)
        
        x = self.act(self.stmap_norm2(self.stmap_conv2(x)))
        x = self.stmap_pool2(x)
        
        x = self.act(self.stmap_norm3(self.stmap_conv3(x)))
        x = self.act(self.stmap_norm4(self.stmap_conv4(x)))
        
        # 全局池化得到特征向量
        x = self.global_pool(x)  # [batch_size, feature_dim, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, feature_dim]
        
        x = self.dropout(x)
        
        return x


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


class PretrainedSpatialExtractor(nn.Module):
    """预训练空间特征提取器"""
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        
    def forward(self, x):
        # ResNet预期输入尺寸为[B, 3, H, W]
        features = self.backbone(x)  # 得到[B, 512, H/32, W/32]
        features = self.adapter(features)  # 转换为[B, feature_dim, H/32, W/32]
        # 全局平均池化得到特征向量
        features = torch.mean(features, dim=(2, 3))  # [B, feature_dim]
        return features


class SpatialFeatureExtractor(nn.Module):
    """空间特征提取器：提取单帧或帧序列的空间特征"""
    def __init__(self, in_channels: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_dim)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(-1, C, H, W)
            x = self.act(self.bn1(self.conv1(x)))
            x = self.pool(x).view(B, T, -1)
        else:
            x = self.act(self.bn1(self.conv1(x)))
            x = self.pool(x).view(x.size(0), -1)
        return x


class MultiModalFusionBlock(nn.Module):
    """多模态特征融合模块，简化版"""
    
    def __init__(self, feature_dim: int, fusion_dim: int, dropout: float = 0.1):
        """
        初始化特征融合模块
        
        参数:
            feature_dim: 输入特征维度
            fusion_dim: 融合后的特征维度
            dropout: Dropout比率
        """
        super().__init__()
        
        # 特征投影层
        self.spatial_proj = nn.Linear(feature_dim, fusion_dim)
        self.stmap_proj = nn.Linear(feature_dim, fusion_dim)
        
        # LayerNorm层
        self.spatial_norm = LayerNorm(fusion_dim)
        self.stmap_norm = LayerNorm(fusion_dim)
        self.fusion_norm = LayerNorm(fusion_dim)
        
        # 注意力权重
        self.spatial_attention = nn.Parameter(torch.ones(1))
        self.stmap_attention = nn.Parameter(torch.ones(1))
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spatial_features, stmap_features):
        """
        前向传播
        
        参数:
            spatial_features: 空间特征 [batch_size, seq_len?, feature_dim] 或 [batch_size, feature_dim]
            stmap_features: STMap特征 [batch_size, feature_dim]
            
        返回:
            融合后的特征
        """
        # 处理维度不匹配的情况
        if len(spatial_features.shape) == 3 and len(stmap_features.shape) == 2:
            # 空间特征是序列，STMap特征是单个向量
            batch_size, seq_len, _ = spatial_features.shape
            stmap_features = stmap_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 投影特征
        spatial_proj = self.spatial_proj(spatial_features)
        stmap_proj = self.stmap_proj(stmap_features)
        
        # 应用LayerNorm
        spatial_norm = self.spatial_norm(spatial_proj)
        stmap_norm = self.stmap_norm(stmap_proj)
        
        # 应用注意力权重
        spatial_weighted = spatial_norm * self.spatial_attention
        stmap_weighted = stmap_norm * self.stmap_attention
        
        # 拼接特征
        concat_features = torch.cat([spatial_weighted, stmap_weighted], dim=-1)
        
        # 应用融合层
        fused = self.fusion_layer(concat_features)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        
        # 残差连接
        output = fused + spatial_proj  # 使用空间特征作为残差基准
        
        return output


class DualPathFeatureExtractor(nn.Module):
    """双路径特征提取器
    
    结合空间特征提取和时空图(STMap)特征提取的双路径特征提取器
    支持使用预训练模型作为特征提取器骨干
    """
    
    def __init__(self, 
                 in_channels: int, 
                 feature_dim: int, 
                 fusion_dim: int,
                 dropout: float = 0.1,
                 debug: bool = False,
                 use_pretrained: bool = False):
        """
        初始化双路径特征提取器
        
        参数:
            in_channels: 输入通道数（RGB=3, NIR=1）
            feature_dim: 单个路径特征维度
            fusion_dim: 融合后的特征维度
            dropout: Dropout比率
            debug: 是否输出调试信息
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
        self.debug = debug
        self.use_pretrained = use_pretrained
        
        # 判断是否使用预训练模型作为特征提取器
        if use_pretrained:
            try:
                # 导入预训练ResNet18
                from torchvision.models import resnet18, ResNet18_Weights
                
                # 创建并初始化预训练的空间特征提取器
                print("[INFO] 正在初始化预训练ResNet18特征提取器...")
                
                # 加载预训练的ResNet18
                pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                
                # 移除最后的全连接层和平均池化层
                self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
                
        
                # 添加适配层，将ResNet18特征(512通道)转换为所需的特征维度
                resnet_out_channels = 512  # ResNet18的输出通道数
                print(f"[信息]创建特征适配器: 从{resnet_out_channels}通道转换到{feature_dim}通道")
                self.adapter = nn.Sequential(
                    nn.Conv2d(resnet_out_channels, feature_dim, kernel_size=1),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(inplace=True)
                )
                
                # 重新定义空间特征提取器为使用预训练骨干的版本
                self.spatial_extractor = PretrainedSpatialExtractor(self.backbone, self.adapter)
                print("[INFO] 成功初始化预训练特征提取器!")
            except Exception as e:
                print(f"[WARNING] 初始化预训练特征提取器失败: {e}")
                print("[INFO] 使用默认随机初始化的特征提取器")
                # 使用普通的特征提取器作为备选
                self.spatial_extractor = SpatialFeatureExtractor(
                    in_channels=in_channels,
                    feature_dim=feature_dim,
                    dropout=dropout
                )
        else:
            # 使用普通的特征提取器
            self.spatial_extractor = SpatialFeatureExtractor(
                in_channels=in_channels,
                feature_dim=feature_dim,
                dropout=dropout
            )
        
        # 创建辅助路径STMap特征提取器
        self.stmap_extractor = STMapFeatureExtractor(
            in_channels=in_channels,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # 创建特征融合模块 - 使用原始参数名以保持兼容性
        self.fusion_block = MultiModalFusionBlock(
            feature_dim=feature_dim,  # 保持使用原始参数名以兼容服务器版本
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        print(f"[信息]创建融合模块：输入维度={feature_dim}, 输出维度={fusion_dim}")
        
        if self.debug:
            print(f"初始化双路径特征提取器:")
            print(f"  输入通道数: {in_channels}")
            print(f"  单路径特征维度: {feature_dim}")
            print(f"  融合特征维度: {fusion_dim}")
    
    def forward(self, x, return_separate_features=False):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, channels, height, width]
               或[batch_size, channels, height, width](单帧形式)
            return_separate_features: 是否返回分离的特征，如果为True则返回(fusion, spatial, stmap)
            
        返回:
            融合后的特征向量或特征元组
        """
        batch_size = x.shape[0]
        
        # 判断输入是单帧还是序列
        is_sequence = len(x.shape) == 5
        
        if is_sequence:
            # 序列输入 [batch_size, seq_len, channels, height, width]
            seq_len = x.shape[1]
            
            # 1. 提取空间特征
            # 重塑为 [batch_size*seq_len, channels, height, width]
            reshaped_frames = x.reshape(batch_size * seq_len, self.in_channels, x.shape[3], x.shape[4])
            spatial_features = self.spatial_extractor(reshaped_frames)
            # 重塑回 [batch_size, seq_len, feature_dim]
            spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)
            
            # 2. 提取STMap特征 [batch_size, feature_dim]
            stmap_features = self.stmap_extractor(x)
            
            # 3. 融合特征
            # 使用特征融合模块融合特征
            fused_features = self.fusion_block(spatial_features, stmap_features)
        else:
            # 单帧输入 [batch_size, channels, height, width]
            # 这种情况下无法提取STMap特征，直接使用空间特征
            spatial_features = self.spatial_extractor(x)
            stmap_features = torch.zeros((batch_size, self.feature_dim), device=x.device)  # 空特征
            fused_features = spatial_features  # 单帧情况下直接使用空间特征
            
        if return_separate_features:
            return fused_features, spatial_features, stmap_features
        else:
            return fused_features
