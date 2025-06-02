import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 设置日志
logger = logging.getLogger(__name__)

class SincConv1d(nn.Module):
    """可学习的Sinc卷积层
    
    实现了基于sinc函数的可学习带通滤波器组，用于从时序信号中学习频谱特征
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=30.0, 
                 min_low_hz=0.5, min_band_hz=0.2, init_min_hz=0.7, init_max_hz=3.0):
        """初始化SincConv1d层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数（滤波器数量）
            kernel_size: 卷积核大小，应为奇数
            sample_rate: 采样率(Hz)
            min_low_hz: 最小低截止频率(Hz)
            min_band_hz: 最小带宽(Hz)
            init_min_hz: 初始化最小频率(Hz)，约为42 BPM
            init_max_hz: 初始化最大频率(Hz)，约为180 BPM
        """
        super(SincConv1d, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保kernel_size为奇数
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # 初始化可学习参数
        # 低截止频率: 对应滤波器组的中心频率
        self.low_hz_ = nn.Parameter(
            torch.Tensor(out_channels).uniform_(init_min_hz, init_max_hz)
        )
        
        # 带宽: 对应滤波器组的频带宽度
        self.band_hz_ = nn.Parameter(
            torch.Tensor(out_channels).uniform_(min_band_hz, init_max_hz - init_min_hz)
        )
        
        # —— 心率特定频段初始化 —— 
        # 目标：将低截止频率聚焦到 1.25–1.667 Hz (75–100 BPM)
        target_min_hz = 60.0 / 60.0    # ≈ 1.25
        target_max_hz = 100.0 / 60.0   # ≈ 1.667
        with torch.no_grad():
            low_offsets = torch.empty(out_channels).uniform_(
                target_min_hz - self.min_low_hz,
                target_max_hz - self.min_low_hz
            )
            self.low_hz_.data.copy_(low_offsets.abs())

        # 给带宽一个较宽范围，让滤波器能覆盖更多频段
        with torch.no_grad():
            self.band_hz_.data.uniform_(0.1, 0.6)
        
        # 创建窗口函数
        window = torch.hann_window(kernel_size)
        self.register_buffer('window', window)
        
        # 创建滤波器中心点
        n = (kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / sample_rate
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入特征 [batch, in_channels, sequence_length]
            
        返回:
            y: 滤波后的特征 [batch, out_channels, sequence_length]
        """
        # 确保输入形状正确
        if x.dim() != 3:
            raise RuntimeError("SincConv1d期望输入形状为 [B, C, T]")
            
        # 确保所有张量在同一设备上
        device = x.device
        n_ = self.n_.to(device)
        
        # 生成滤波器的参数: 频率范围约束
        low_hz = self.min_low_hz + torch.abs(self.low_hz_)  # 确保低截止频率 > min_low_hz
        band_hz = self.min_band_hz + torch.abs(self.band_hz_)  # 确保带宽 > min_band_hz
        high_hz = torch.clamp(low_hz + band_hz, min=self.min_low_hz, max=self.sample_rate/2)
        
        # 计算滤波器
        # 中心频率的滤波器响应: 2*low_hz*sinc(2*pi*low_hz*n)
        low = 2 * low_hz.view(-1, 1) * n_.view(1, -1)
        low_pass1 = torch.sin(low) / low
        
        # 处理分母为0的情况
        low_pass1[:, (self.kernel_size - 1) // 2] = 1.0
        
        # 高频部分的滤波器响应: 2*high_hz*sinc(2*pi*high_hz*n)
        high = 2 * high_hz.view(-1, 1) * n_.view(1, -1)
        low_pass2 = torch.sin(high) / high
        
        # 处理分母为0的情况
        low_pass2[:, (self.kernel_size - 1) // 2] = 1.0
        
        # 带通滤波器 = 高通 - 低通
        band_pass = low_pass2 - low_pass1
        
        # 应用窗口函数
        band_pass = band_pass * self.window.view(1, -1)
        
        # 归一化滤波器
        band_pass = band_pass / torch.max(torch.abs(band_pass), dim=1, keepdim=True)[0]
        
        # 规整滤波器组的形状以适应卷积操作
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        filters = filters.repeat(1, self.in_channels, 1) / self.in_channels
        
        # 将扩展的滤波器应用于输入信号
        return F.conv1d(x, filters, padding=(self.kernel_size - 1) // 2)


class Regressor(nn.Module):
    """回归器模块，用于从频谱特征预测心率"""
    
    def __init__(self, in_features, hidden_features=None):
        """初始化回归器
        
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层维度，默认为in_features//2
        """
        super(Regressor, self).__init__()
        if hidden_features is None:
            hidden_features = in_features // 2
            hidden_features = max(hidden_features, 8)  # 确保至少8个隐藏单元
            
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, 1)
        )
        
        # 显式初始化最后一层的偏置，使得初始输出落在 75-100 BPM 区间
        # 使用逆映射方法: sigmoid⁻¹((hr-30)/150) 计算对应心率的raw值
        with torch.no_grad():
            last_layer = self.net[-1]  # 获取最后一层
            # 逆映射函数，将心率转化为对应的raw值
            import math
            inv = lambda hr: math.log((hr-30)/(180-hr))
            lo = inv(75.0)    # ≈ log(45/105) ≈ -0.847
            hi = inv(100.0)   # ≈ log(70/80) ≈ -0.1335
            # 在 [-0.847, -0.1335] 区间随机初始化偏置
            last_layer.bias.uniform_(lo, hi)
    
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入特征 [B, in_features]
            
        返回:
            output: 心率预测值 [B, 1]
        """
        # 注意: 返回的是 [B, 1]，squeeze(-1) 在外面调用时进行
        return self.net(x)

class ClassificationHRDecoder(nn.Module):
    """基于分类的心率解码器
    
    将心率分为五个类别:
        - 类别 0: 极低心率 (30-60 BPM)
        - 类别 1: 低心率 (60-80 BPM)
        - 类别 2: 中等心率 (80-100 BPM)
        - 类别 3: 高心率 (100-120 BPM)
        - 类别 4: 极高心率 (120-180 BPM)
    
    每个类别内部使用专门的回归头进行精细预测
    """
    
    def __init__(self, config):
        """初始化解码器
        
        参数:
            config: 配置对象，包含d_model、hidden_dim等参数
        """
        super().__init__()
        self.d_model = getattr(config, 'd_model', 512)    # 输入特征维度
        self.hidden_dim = getattr(config, 'hidden_dim', 128)  # 隐藏层维度
        self.dropout = getattr(config, 'dropout', 0.1)
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        
        # Sinc卷积层 - 使用可学习的带通滤波器提取心率相关频段
        self.sinc_conv = SincConv1d(
            in_channels=1,            # 输入通道数为1
            out_channels=16,          # 使用16个不同的滤波器
            kernel_size=33,           # 33个时间步的卷积核
            sample_rate=30.0,         # 30Hz采样率
            init_min_hz=0.7,          # 约42 BPM
            init_max_hz=2.5           # 约180 BPM
        )
        
        # 频谱特征处理网络
        self.freq_feature_net = nn.Sequential(
            nn.Linear(16, self.hidden_dim // 2),  # 16个滤波器输出
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Dropout(self.dropout),
        )
        
        # 特征融合网络 - 组合时域和频域特征
        self.fusion_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 2, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        
        # 分类头 - 扩展到5个类别
        self.classification_head = nn.Linear(self.hidden_dim, 5)  # 5个类别
        
        # 每个类别的内部回归头 - 使用Regressor替代简单的线性层
        self.class_regression_heads = nn.ModuleList([
            Regressor(self.hidden_dim, self.hidden_dim // 4) for _ in range(5)  # 每个类别一个回归器
        ])
        
        # 使用Regressor替代简单的线性层做全局回归
        self.regression_head = Regressor(self.hidden_dim, self.hidden_dim // 2)
        
        # 5档心率阈值
        self.hr_thresholds = [30.0, 60.0, 80.0, 100.0, 120.0, 180.0]  # 各档位的边界值
    
        # 定义全局回归的最小和最大心率范围
        self.min_bpm = 30.0  # 全局回归最小心率值
        self.max_bpm = 180.0  # 全局回归最大心率值
    
        # 对于评估模式，我们需要从分类结果转换回BPM值
        # 每档的中心心率 (BPM)
        centers = torch.tensor([50.0, 70.0, 90.0, 110.0, 140.0])
        self.register_buffer('class_centers', centers)
        
        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()  # 确保不确定性为正值
        )
        
    def forward(self, features_dict, return_freq_data=False, multi_task=True):
        """前向传播
        
        参数:
            features_dict: 特征字典或张量
            return_freq_data: 是否返回频率数据(在分类模式下忽略)
            multi_task: 是否返回多任务学习所需的全部输出信息
            
        返回:
            在训练模式下(当multi_task=True时):
                logits: 分类 logits [batch, num_classes]
                regression: 心率回归预测 [batch]
                class_regressions: 每个类别的精细回归预测 [batch, num_classes]
            
            在评估模式下:
                hr: 心率估计值 [batch]
                uncertainty: 不确定性估计 [batch]
        """
        # 处理输入特征
        if isinstance(features_dict, dict):
            # 使用字典中的'output'或'hr_enhanced'字段
            if 'output' in features_dict:
                features = features_dict['output']
            elif 'hr_enhanced' in features_dict:
                features = features_dict['hr_enhanced']
            else:
                # 作为后备，检查其他可能的键
                for key in ['features', 'encoder_output']:
                    if key in features_dict:
                        features = features_dict[key]
                        break
                else:
                    # 如果没有找到匹配的键，尝试使用第一个可用的张量值
                    for key, value in features_dict.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 1:
                            features = value
                            break
                    else:
                        raise ValueError("在字典中找不到有效的特征张量")
        else:
            # 直接使用张量
            features = features_dict
            
        # 确保特征张量的维度正确
        if features.dim() == 3:
            # 时序特征 [batch, seq_len, channels]
            # 全局平均池化消除序列长度维度
            features = torch.mean(features, dim=1)  # [batch, channels]
        
        # 先提取通用特征
        batch_size = features.shape[0]
        device = features.device
        
        # 应用输入特征处理器
        hidden_features = self.feature_net(features)  # [batch, hidden_dim]
        
        # === 时域特征处理 (Sinc特征) ===
        # 使用Sinc卷积提取频谱特征
        # 创建电影多帧组成的输入信号
        t = torch.arange(0, 300, device=device).float() / 30.0  # 10秒的时间序列在0.033秒采样率(30Hz)
        
        # 生成一个标准正弦波序列
        freq = torch.ones((batch_size, 1), device=device) * 1.5  # 默认频率在1.5Hz (90 BPM)
        signal = torch.sin(2 * math.pi * freq * t.view(1, -1))  # [batch, time]
        signal = signal.unsqueeze(1)  # [batch, 1, time]
        
        # 使用Sinc卷积提取频谱特征
        freq_features = self.sinc_conv(signal)  # [batch, n_filters, time]
        
        # 全局平均池化转换为特征向量
        freq_features = torch.mean(freq_features, dim=2)  # [batch, n_filters]
        
        # 处理频谱特征
        freq_hidden = self.freq_feature_net(freq_features)  # [batch, hidden_dim//2]
        
        # === 融合特征 ===
        combined_features = torch.cat([hidden_features, freq_hidden], dim=1)  # [batch, hidden_dim + hidden_dim//2]
        fused_features = self.fusion_net(combined_features)  # [batch, hidden_dim]
        
        # === 分类与回归 ===
        # 分类头 - 5个类别
        logits = self.classification_head(fused_features)  # [batch, 5]
        softmax_probs = F.softmax(logits, dim=1)  # [batch, 5]
        
        # 实现每个类别的精细回归
        class_regressions = []
        for i, head in enumerate(self.class_regression_heads):
            # 每个类别范围
            min_hr = self.hr_thresholds[i]  # 当前档位的下限
            max_hr = self.hr_thresholds[i+1]  # 当前档位的上限
            
            # 类内回归 - 使用Regressor
            class_reg_raw = head(fused_features)  # [batch, 1]
            class_reg_raw = class_reg_raw.squeeze(-1)  # [batch]
            
            # 将预测结果映射到具体类别范围内
            class_reg = torch.sigmoid(class_reg_raw) * (max_hr - min_hr) + min_hr
            class_regressions.append(class_reg)
        
        # 将各类别的回归预测合并为一个张量 [batch, 5]
        class_regressions = torch.stack(class_regressions, dim=1)  # [batch, 5]
        
        # 对全局回归进行兼容性处理 - 使用Regressor
        regression_global = self.regression_head(fused_features)  # [batch, 1]
        regression_global = regression_global.squeeze(-1)  # [batch]
        regression_global = torch.sigmoid(regression_global) * (self.max_bpm - self.min_bpm) + self.min_bpm
        
        # 带权重的总体回归预测 - 结合全局预测和每个类别的精细回归
        weighted_regression = torch.sum(softmax_probs * class_regressions, dim=1)  # [batch]
        
        # 不确定性估计 - 无论什么模式都使用
        uncertainty = self.uncertainty_estimator(fused_features)
        
        # 直接将加权回归预测作为最终结果
        results = {
            'logits': logits,                  # [batch, num_classes] 分类预测
            'regression': weighted_regression,  # [batch] 加权后的最终回归预测
            'class_regressions': class_regressions,  # [batch, num_classes] 每个类别的回归预测
            'regression_global': regression_global,  # [batch] 全局回归预测
            'softmax_probs': softmax_probs,    # [batch, num_classes] softmax概率
            'uncertainty': uncertainty         # [batch] 不确定性估计
        }
        
        # 预测类别信息仅用于调试输出
        if batch_size > 0 and hasattr(self, 'logger') and not self.training:
            _, pred_class = torch.max(logits, dim=1)
            class_probs = F.softmax(logits, dim=1)
            self.logger.debug(f"预测类别: {pred_class[0].item()}, 类别概率: {class_probs[0].tolist()}, 回归值: {weighted_regression[0].item():.1f} BPM")
        
        # 非多任务模式下，仍然返回完整结果字典
        if not multi_task and not self.training:
            # 返回简化的结果(依然使用字典)
            return {
                'regression': weighted_regression,
                'uncertainty': uncertainty,
                'logits': logits
            }
        
        return results


def heart_rate_to_class(hr_values, num_classes=5):
    """将心率值(BPM)转换为类别标签
    
    参数:
        hr_values: 心率值，形状为 [B]
        num_classes: 类别总数，默认为5
        
    返回:
        classes: 类别标签，形状为 [B]
            0: 档位 0 (30-60 BPM)   → 中心 45 BPM
            1: 档位 1 (60-80 BPM)   → 中心 70 BPM
            2: 档位 2 (80-100 BPM)  → 中心 90 BPM
            3: 档位 3 (100-120 BPM) → 中心 110 BPM
            4: 档位 4 (120-180 BPM) → 中心 150 BPM
    """
    try:
        # 确保输入为PyTorch张量
        if not isinstance(hr_values, torch.Tensor):
            hr_values = torch.tensor(hr_values, dtype=torch.float32)
        
        # 处理异常值 - 限制输入在有效范围内
        # 对于异常心率值，限制在物理上可能的心率范围内
        hr_values = torch.clamp(hr_values, min=30.0, max=180.0)
        
        # 初始化类别标签
        # 默认为最常见的中等心率档位（档位2: 80-100 BPM）
        classes = torch.ones_like(hr_values, dtype=torch.long) * min(2, num_classes-1)  # 确保默认类别也在有效范围内
        
        # 分档位 - 不同类别数量的适配
        if num_classes >= 5:  # 原始四档分类
            classes = torch.where(hr_values < 60, torch.zeros_like(classes), classes)                      # 档位0: 30-60 BPM
            classes = torch.where((hr_values >= 60) & (hr_values < 80), torch.ones_like(classes), classes)     # 档位1: 60-80 BPM
            classes = torch.where((hr_values >= 80) & (hr_values < 100), torch.ones_like(classes)*2, classes)  # 档位2: 80-100 BPM
            classes = torch.where((hr_values >= 100) & (hr_values < 120), torch.ones_like(classes)*3, classes) # 档位3: 100-120 BPM
            classes = torch.where(hr_values >= 120, torch.ones_like(classes)*4, classes)                   # 档位4: 120-180 BPM
        elif num_classes == 4:  # 四档分类
            classes = torch.where(hr_values < 70, torch.zeros_like(classes), classes)                      # 档位0: 30-70 BPM
            classes = torch.where((hr_values >= 70) & (hr_values < 90), torch.ones_like(classes), classes)     # 档位1: 70-90 BPM
            classes = torch.where((hr_values >= 90) & (hr_values < 110), torch.ones_like(classes)*2, classes)  # 档位2: 90-110 BPM
            classes = torch.where(hr_values >= 110, torch.ones_like(classes)*3, classes)                   # 档位3: 110-180 BPM
        elif num_classes == 3:  # 三档分类
            classes = torch.where(hr_values < 75, torch.zeros_like(classes), classes)                      # 档位0: 30-75 BPM
            classes = torch.where((hr_values >= 75) & (hr_values < 100), torch.ones_like(classes), classes)    # 档位1: 75-100 BPM
            classes = torch.where(hr_values >= 100, torch.ones_like(classes)*2, classes)                   # 档位2: 100-180 BPM
        elif num_classes == 2:  # 二档分类
            classes = torch.where(hr_values < 90, torch.zeros_like(classes), classes)                      # 档位0: 30-90 BPM
            classes = torch.where(hr_values >= 90, torch.ones_like(classes), classes)                      # 档位1: 90-180 BPM
        else:  # 其他情况，使用线性插值
            # 心率范围[30, 180]BPM 线性映射到[0, num_classes-1]
            normalized = (hr_values - 30.0) / (180.0 - 30.0) * (num_classes - 1)
            classes = torch.clamp(normalized.long(), min=0, max=num_classes-1)
        
        # 最后再次确保类别在有效范围内
        classes = torch.clamp(classes, min=0, max=num_classes-1)
        
        return classes
        
    except Exception as e:
        # 日志错误
        import logging
        logging.getLogger().error(f"心率分类出错: {e}")
        # 出错时返回安全的默认类别（中间类别）
        if isinstance(hr_values, torch.Tensor):
            safe_class = torch.ones_like(hr_values, dtype=torch.long) * min(2, num_classes-1)
            return safe_class
        else:
            # 如果这也失败，返回一个标量
            return torch.tensor(min(2, num_classes-1), dtype=torch.long)


def class_to_heart_rate(class_indices, detailed=False):
    """将类别索引转换为代表性心率值
    
    参数:
        class_indices: 类别索引，形状为 [B]
        detailed: 是否返回详细的范围信息
        
    返回:
        hr_values: 代表性心率值，形状为 [B]
    """
    # 类别映射到代表性心率值
    class_centers = {
        0: 50.0,   # 档位 0 (40-60 BPM)
        1: 70.0,   # 档位 1 (60-80 BPM)
        2: 90.0,   # 档位 2 (80-100 BPM)
        3: 110.0,  # 档位 3 (100-120 BPM)
        4: 140.0   # 档位 4 (120-160 BPM)
    }
    
    # 类别范围 (用于detailed=True的情况)
    class_ranges = {
        0: "40-60 BPM (偏低)",
        1: "60-80 BPM (低)",
        2: "80-100 BPM (正常)",
        3: "100-120 BPM (高)",
        4: "120-160 BPM (偏高)"
    }
    
    # 创建结果容器
    if isinstance(class_indices, torch.Tensor):
        device = class_indices.device
        hr_values = torch.zeros_like(class_indices, dtype=torch.float32)
        
        # 填充代表性心率值
        for cls_idx, hr_val in class_centers.items():
            hr_values[class_indices == cls_idx] = hr_val
    else:
        # 处理单个数值情况
        hr_values = class_centers.get(class_indices, 80.0)
    
    if detailed:
        if isinstance(class_indices, torch.Tensor):
            ranges = [class_ranges.get(idx.item(), "未知") for idx in class_indices]
            return hr_values, ranges
        else:
            return hr_values, class_ranges.get(class_indices, "未知")
    else:
        return hr_values


class UncertaintyEstimator(nn.Module):
    """不确定性估计器，预测模型输出的不确定性"""
    
    def __init__(self, in_features, hidden_features=None):
        """初始化不确定性估计器
        l
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层维度，默认为in_features//2
        """
        super(UncertaintyEstimator, self).__init__()
        if hidden_features is None:
            hidden_features = in_features // 2
            hidden_features = max(hidden_features, 8)  # 确保至少8个隐藏单元
            
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, 1),
            nn.Softplus()  # 确保不确定性为正值
        )
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入特征 [batch, in_features]
            
        返回:
            uncertainty: 预测不确定性 [batch]
        """
        uncertainty = self.net(x).squeeze(-1)
        return uncertainty


class EnhancedBaselineBRDecoder(nn.Module):
    """呼吸率解码器 - 简单版本
    
    专为呼吸率频段(0.15-0.4Hz)优化的解码器
    """
    
    def __init__(self, config):
        """初始化呼吸率解码器
        
        参数:
            config: 配置对象，应包含d_model、hidden_dim等参数
        """
        super().__init__()
        self.d_model = getattr(config, 'd_model', 512)  # 输入特征维度
        self.hidden_dim = getattr(config, 'hidden_dim', 128)  # 隐藏层维度
        self.dropout = getattr(config, 'dropout', 0.1)
        self.fs = getattr(config, 'FS', 30.0)  # 采样率
        
        # 呼吸率频段 - 典型范围为0.15-0.4Hz (9-24 BPM)
        self.min_freq = nn.Parameter(torch.tensor([0.15]))  # ~9 BPM
        self.max_freq = nn.Parameter(torch.tensor([0.4]))   # ~24 BPM
        
        # 特征处理网络
        self.feature_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        
        # 回归头
        self.regression_head = nn.Linear(self.hidden_dim, 1)
        
        # 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, features):
        """前向传播
        
        参数:
            features: 编码器输出特征 [B, L, C]
            
        返回:
            br: 呼吸率预测值 [B]
            uncertainty: 不确定性 [B]
        """
        if isinstance(features, dict):
            if 'output' in features:
                features = features['output']
            elif 'br_enhanced' in features:
                features = features['br_enhanced']
        
        # 全局平均池化消除序列长度维度
        features = torch.mean(features, dim=1)  # [B, C]
        
        # 特征处理
        hidden = self.feature_net(features)  # [B, hidden_dim]
        
        # 回归预测
        br_raw = self.regression_head(hidden).squeeze(-1)  # [B]
        
        # 将预测限制在合理范围内 (9-24 BPM)
        br = 9.0 + torch.sigmoid(br_raw) * 15.0
        
        # 不确定性估计
        uncertainty = self.uncertainty_head(hidden).squeeze(-1)  # [B]
        
        return br, uncertainty


class SpO2Decoder(nn.Module):
    """血氧饱和度(SpO2)解码器
    
    针对血氧饱和度(通常范围为90%-100%)的专用解码器
    """
    
    def __init__(self, config):
        """初始化血氧解码器
        
        参数:
            config: 配置对象，应包含d_model、hidden_dim等参数
        """
        super().__init__()
        self.d_model = getattr(config, 'd_model', 512)  # 输入特征维度
        self.hidden_dim = getattr(config, 'hidden_dim', 128)  # 隐藏层维度
        self.dropout = getattr(config, 'dropout', 0.1)
        
        # 特征处理网络
        self.feature_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Dropout(self.dropout)
        )
        
        # 回归头 - 使用Sigmoid激活函数将输出限制在[0,1]范围内
        # 然后再映射到血氧饱和度范围[90,100]
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, features):
        """血氧饱和度预测器前向计算"""
        # 如果是序列特征，平均池化到单个特征向量
        if features.dim() == 3:
            features = features.mean(dim=1)  # [B, C]
        
        # 特征提取
        features = self.feature_net(features)  # [B, hidden_dim//2]
        
        # 血氧饱和度预测(90-100%范围)
        spo2_norm = self.regression_head(features).squeeze(-1) 
        spo2 = 90.0 + spo2_norm * 10.0
        
        # 预测不确定性
        uncertainty = self.uncertainty_head(features).squeeze(-1)
        
        return spo2, uncertainty
    
    
    