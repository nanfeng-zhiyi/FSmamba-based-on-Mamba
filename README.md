# repss
code for repss2025
# RePSS 项目架构详细说明

## 项目概述

RePSS（Remote Physiological Signal Sensing）是一个基于计算机视觉和深度学习的远程生理信号感知系统，旨在从 RGB 和 NIR 视频中提取心率和呼吸率等生理信号。该项目利用了 Mamba 模型架构和频域增强技术，实现了高精度的无接触式生理信号监测。

## 核心模型架构

### 1. RePSS 模型 (V1 和 V2)

RePSS 模型包含两个版本，基本架构如下：

#### RePSSModel（V1）

```
输入 → 特征提取 → 特征融合 → Mamba编码器 → 生理信号回归 → 输出
      ↑            ↑
     RGB          NIR
     特征         特征
```

#### RePSSModelV2（进阶版）

```
输入 → 特征提取 → 特征融合 → Mamba编码器 → 频域增强 → 信号优化器 → 输出
      ↑      ↑        ↑
     RGB    NIR     vPPG
     特征   特征    特征
```

### 2. 特征提取组件

#### FeatureExtractor（RGB/NIR 特征提取）

卷积网络结构：

- Conv2D(in_channels, 32, kernel_size=3, stride=2)
- Conv2D(32, 64, kernel_size=3, stride=2)
- Conv2D(64, 128, kernel_size=3, stride=2)
- Conv2D(128, feature_dim, kernel_size=3, stride=2)
- 全局池化 + 维度重塑

参数：

- RGB 特征维度: 64
- NIR 特征维度: 32
- Dropout 率: 0.1

#### VPPGProcessor（光电容积脉搏波处理器）

处理从面部视频中提取的 vPPG 信号：

- 卷积层: Conv1D(in_channels, 16, kernel_size=3)
- 池化层: MaxPool1d(2)
- 卷积层: Conv1D(16, 32, kernel_size=3)
- 全连接层: Linear(feature_size, feature_dim)

参数：

- vPPG 特征维度: 32
- Dropout 率: 0.1

### 3. Mamba 核心组件

#### SSMKernel（状态空间模型核心）

实现状态空间模型的核心算法：

- A 参数：控制状态保留（类似遗忘门）
- B 参数：控制输入对状态的影响
- C 参数：控制状态到输出的转换
- Δ 参数：控制时间步长

参数：

- 状态维度: 64
- 时间步长范围: [0.001, 0.1]

#### SSMBlock（Mamba 基础块）

完整的 SSM 块实现：

- 投影层：扩展输入通道
- 卷积层：捕获局部信息
- SSM 核心：处理长序列依赖
- 激活函数：SiLU
- 输出投影：压缩回原始维度

参数：

- 模型维度: 128
- 卷积核大小: 4
- 扩展因子: 2

### 4. 频域增强组件

#### FrequencySelectiveFilter（频率选择滤波器）

实现频域增强的专用滤波器：

- 心率带通滤波器（0.65-3.0Hz，对应 40-180 BPM）
- 呼吸率带通滤波器（0.16-0.50Hz，对应 10-30 BPM）
- 可学习的频率响应参数

参数：

- 采样频率: 30.0 Hz
- 心率频带: (0.65, 3.0) Hz
- 呼吸率频带: (0.16, 0.50) Hz

#### FrequencyEnhancedMambaBlock（频率增强 Mamba 块）

结合频域滤波和 Mamba 处理：

- 层归一化
- SSM 处理
- 频域滤波
- 特征融合

参数：

- 融合层：Linear(d_model \* 2, d_model)

### 5. 特征融合组件

#### MultiModalFusionBlock（多模态融合块）

融合 RGB 和 NIR 特征：

- 跨注意力：CrossAttention(rgb_dim, nir_dim)
- 自注意力：CrossAttention(rgb_dim, rgb_dim)
- 融合层：Linear([rgb+nir], fusion_dim)

参数：

- 注意力头数: 4
- 头维度: 32

#### CrossAttention（跨模态注意力）

实现多头自注意力和跨模态注意力：

- 查询/键/值投影
- 多头注意力计算
- 输出投影和残差连接

参数：

- 注意力头数: 8
- 头维度: 64
- Dropout 率: 0.1

### 6. 生理信号回归组件

#### PhysiologicalSignalRegressor（生理信号回归器）

从特征预测心率和呼吸率：

- 共享层：多层 MLP
- 心率头：预测心率（45-200 BPM）
- 呼吸率头：预测呼吸率（5-40 BPM）
- 谐波检测器：预测是否为谐波信号

参数：

- 隐藏维度: 128
- 层数: 2
- 心率校准因子: 0.98

#### SignalOptimizer（信号优化器 - V2 模型）

优化生理信号预测：

- 共享层：多层 MLP
- 信号头：预测信号调整值

参数：

- 隐藏维度: 64
- 层数: 2

### 7. 编码器组合结构

#### MambaEncoder（Mamba 编码器）

堆叠多层 Mamba 块：

- 多层 FrequencyEnhancedMambaBlock 或 MambaBlock
- 层归一化
- 输出投影

参数：

- 深度: 4
- 状态维度: 64
- Dropout 率: 0.1

## 模型配置参数

### MambaModelConfig

```python
@dataclass
class MambaModelConfig:
    input_dim: int = 3                           # 输入维度 (RGB通道数)
    rgb_feature_dim: int = 64                    # RGB特征维度
    nir_feature_dim: int = 32                    # NIR特征维度
    vppg_feature_dim: int = 32                   # vPPG特征维度
    hidden_dim: int = 128                        # 隐藏层维度
    state_dim: int = 64                          # 状态维度
    depth: int = 4                               # 编码器深度
    dropout: float = 0.1                         # Dropout比率
    hr_band: Tuple[float, float] = (0.65, 3.0)   # 心率频带 (Hz) - 40-180 BPM
    br_band: Tuple[float, float] = (0.16, 0.50)  # 呼吸率频带 (Hz) - 10-30 BPM
    fs: float = 30.0                             # 采样频率 (Hz)
    use_hrv_modulation: bool = False             # 是否使用HRV调制动态步长
    use_rgb: bool = True                         # 是否使用RGB输入
    use_nir: bool = True                         # 是否使用NIR输入
    use_vppg: bool = True                        # 是否使用vPPG输入
    hr_calibration_factor: float = 1.05          # 心率校准因子
    detect_harmonics: bool = True                # 是否检测谐波
    adaptive_filtering: bool = True              # 是否使用自适应滤波
    use_simclr: bool = False                     # 是否使用自监督对比学习
```

## 训练配置

### 数据加载参数

- 数据目录: `data/vlhr`
- 批大小: 4-16（根据 GPU 可用性自适应）
- 序列长度: 64（默认）
- 重叠率: 0.5
- 数据并行处理: 4 个工作线程

### 优化器配置

- 优化器: AdamW
- 学习率: 1e-4（带有余弦衰减）
- 权重衰减: 1e-5
- EMA 权重更新: 0.995
- 混合精度训练: 启用

### 损失函数

- 主要损失: MSE（均方误差）
- 谐波损失: 针对谐波检测的 BCE 损失
- 频谱一致性损失: 频域约束损失

## 创新技术点

1. **Mamba 状态空间模型**

   - 使用选择状态空间模型（SSM）替代传统 Transformer
   - 线性计算复杂度，高效处理长序列
   - 动态时间步长调制，适应变化的信号频率

2. **频域增强技术**

   - 专用心率和呼吸率带通滤波器
   - 频率响应参数可训练
   - 频率选择性激活，增强特定频段信号

3. **多模态融合**

   - RGB、NIR 和 vPPG 三模态融合
   - 跨模态注意力机制
   - 动态特征适配层

4. **谐波检测机制**

   - 自动检测心率谐波
   - 校正倍频误差
   - 减少预测偏差

5. **信号优化器**
   - 动态调整预测信号
   - 自适应频率滤波
   - 基于上下文的信号修正

## 数据流程

1. **数据加载**

   - 从 VIPL-HR 数据集加载 RGB/NIR 视频帧
   - 提取面部区域
   - 生成 vPPG 信号（色度法/PCA 法）

2. **预处理**

   - 面部对齐和跟踪
   - 数据增强（旋转、缩放、亮度变化）
   - 序列分割（固定长度序列）

3. **特征提取**

   - RGB 特征提取（4 层 CNN）
   - NIR 特征提取（同样结构）
   - vPPG 信号处理（1D CNN）

4. **多模态融合和编码**

   - 特征融合（拼接或注意力）
   - Mamba 编码（多层 SSM）
   - 频域增强

5. **生理信号预测**
   - 心率预测（40-180 BPM）
   - 呼吸率预测（10-30 BPM）
   - 谐波检测和校正

## 评估指标

- MAE（平均绝对误差）
- RMSE（均方根误差）
- MAPE（平均百分比误差）
- Pearson 相关系数
- Bland-Altman 分析

## 总结

RePSS 项目结合了最新的状态空间模型技术（Mamba）和传统的信号处理方法，创建了一个高效精确的远程生理信号监测系统。通过多模态融合和频域增强，该系统能够从普通 RGB 和 NIR 视频中准确提取心率和呼吸率信息，同时针对常见问题（如谐波、噪声和丢失帧）进行了专门的优化处理。
