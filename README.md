

# FSMamba：基于 Mamba 架构的频率选择性多模态远程生理信号感知系统

> **A Frequency-Selective Multimodal Remote Physiological Signal Sensing System Based on Mamba Architecture**

## 项目简介

![图片](./images/FSMamba.png)

远程心率估计算法能够直接从视频中提取生理信号，但在真实应用场景中面临三大挑战：

1. **缺乏对低频生理节律的频率选择性建模**，信号易被噪声覆盖。
2. **单一模态输入在复杂光照、遮挡或设备差异下稳定性下降**。
3. **传统时序建模在长序列条件下计算复杂度高、泛化能力有限**。

本项目提出了 **FSMamba**（Frequency-Selective Mamba），将以下思想集成到一个端到端框架中：

- **多模态融合**（RGB + NIR），利用双向跨模态注意力增强两者互补性。
- **频率选择性编码**，通过可学习分段滤波聚焦心率频段（0.7–2.5Hz）。
- **高效时序建模**，基于 Mamba 状态空间结构捕获长距离依赖。
- **多任务解码与分布对齐损失**，联合回归与区间分类，提高预测鲁棒性与准确度。

在标准 VIPL-HR 数据集上，FSMamba 在多种场景下均取得优异的 RMSE 性能，并通过消融实验验证各模块的独立贡献。

* 基于1-fold数据集划分模型下VIPL-HR数据集上的表现

![](./images/test.png)

![](./images/erro_dist.png)

* 模型消融表现

![](./images/ablation.png)

---

## 主要特性

- 🔷 **双流特征提取**  

  - 采用 STMap（时空差分）加轻量卷积编码器，从 RGB 和 NIR 视频分别抽取时空特征。
  - 双模态分支并行处理，保留各自特征分布。

- 🔷 **双向跨模态注意力**  

  ![](./images/Cross Attention.png)

  - 设计 RGB→NIR 与 NIR→RGB 两路交叉注意力，使两种模态信息动态互补。
  - 融合后用 MLP 投影得到最终联合表示，用于后续时序建模。

- 🔷 **FSMamba 编码器**  

  - 继承 Mamba 状态空间模型（SSM）的高效序列处理，具备线性时间复杂度。
  - 引入 **频率选择路径（FSFilter）**：可学习的滤波参数显式关注心率频段（0.7–2.5Hz），提升低频信号检测能力。
  - 输出包含全局时序特征与心率频段特征两部分，二者通过残差连接累积。

- 🔷 **心率解码模块**  

  - 分支一：对 FSMamba 编码器输出的全局时序特征进行 MLP 变换，提取原始回归信息。  
  - 分支二：对 FSMamba 编码器输出的频率选择特征进行 SincConv1d 滤波与 MLP 变换，强化频谱信息。  
  - 将两部分特征融合，采用 **区间分类 + 组内回归** 的混合结构预测最终心率，并运用分布对齐损失提高一致性。

- 🔷 **多任务损失设计**  

  ```text
  L_total = λ_reg * L_reg + λ_cls * L_cls + λ_dist * L_dist
  ```

* 性能比较

![](./images/compare.png)

# 模型实现概述

## 特征提取模块（Feature Extraction Module）

**目的**

- 从原始 RGB 与 NIR 视频中提取空间–时间特征，突出人脸微弱的脉搏振动信号，为后续时序建模与融合提供输入。

### 关键步骤

1. **时序差分构造（STMap）**
   - 令输入视频帧为 I₁, I₂, …, Iₜ。
   - 通过相邻帧像素差分：
     Xₜ = Iₜ₊₁ − Iₜ,   t = 1 … (T−1)
     以削弱静态背景与人脸纹理，保留由血液流动引起的亮度微小变化——这正是 rPPG 信号在视频中的来源。
2. **轻量空间编码（Spatial Encoder）**
   - 对每个时序差分图 Xₜ 输入五层轻量卷积网络：
     - 每层由 3×3 卷积 + ReLU + 批归一化 + 最大池化构成，逐层加深通道数并降低空间分辨率。
     - 最后通过全局平均池化（Global Average Pooling），输出一个形状为 T×d 的时序特征序列，其中 d 为最后卷积层通道数。
3. **双模态并行处理**
   - RGB 与 NIR 两条通路各自独立进行 STMap 生成与空间编码，保证它们在后续融合时具有相同的序列长度 T 和特征维度 d：
     X_rgb = SpatialEncoder(STMap(I_rgb)),
     X_nir = SpatialEncoder(STMap(I_nir)).

## 多模态融合机制：双向跨模态注意（Bidirectional Cross-Modal Attention）

**目的**

- 在复杂光照、遮挡或摄像条件下，RGB 与 NIR 各自带来不同优势：
  - RGB 提供高保真颜色与纹理；
  - NIR 对低光环境与肤色变化更稳定。
- 双向跨模态注意力机制可以让一条模态“查询”另一条模态，动态抓取互补信息并进行融合。

### 主要思路

1. **跨模态注意力计算**
   - 以 X_rgb, X_nir ∈ B×T×d 为输入：
     1. RGB→NIR：
        F_rgb→nir = Attention(Q = X_rgb, K = X_nir, V = X_nir)
     2. NIR→RGB：
        F_nir→rgb = Attention(Q = X_nir, K = X_rgb, V = X_rgb)
2. **残差更新**
   - 将跨模态注意力输出与原始特征相加（残差连接）：
     X_rgb′ = X_rgb + F_nir→rgb,
     X_nir′ = X_nir + F_rgb→nir.
3. **融合映射**
   - 将更新后的两路特征在通道维度拼接：
     [X_rgb′; X_nir′] ∈ B×T×2d.
   - 通过一个全连接网络或多层感知机（MLP），将通道数从 2d 压缩回 d，得到融合特征 X_fused ∈ B×T×d。
   - 最终，X_fused 既保留原始 RGB/NIR 各自信息，又有跨模态的互补融合效果。

## FSMamba 编码器：频率选择性状态空间建模（Frequency-Selective State Space Modeling）

**目的**

- 借助 Mamba（Selective State Space Models）的高效时序建模能力，捕获视频中长距离的时序依赖；
- 同时通过频率选择滤波机制专门聚焦心率频段（约 0.7–2.5 Hz），使模型对微弱脉搏振动更敏感。

### 模块架构

给定融合后的序列 X ∈ B×T×D，FSMamba 编码器由 L 层堆叠而成，每层包括两条并行路径：

1. **标准状态空间路径（SSM Path）**
   - 对 X^(ℓ) 做归一化，输入到 Mamba 状态空间块 SSM_ℓ：
     Y_ssm^(ℓ) = SSM_ℓ(LayerNorm(X^(ℓ)))
     该块利用参数化的状态转移矩阵 A,B,C,D 对序列进行线性递推，借助预先构造的卷积核一次性实现全序列的上下文聚合，时间复杂度 O(T)。
2. **频率选择性路径（FSFilter Path）**
   - 对同一个 X^(ℓ) 做归一化，输入到可学习带通滤波单元 FSFilter_ℓ：
     Y_hr^(ℓ) = FSFilter_ℓ(LayerNorm(X^(ℓ)))
   - FSFilter 会根据可学习的频段参数 f_hr,min 和 f_hr,max，显式滤掉心率频段之外的成分，仅保留 0.7–2.5 Hz 左右区域的信息。
3. **多层感知机融合与残差**
   - 将两部分输出在特征维度拼接：
     [Y_ssm^(ℓ); Y_hr^(ℓ)] ∈ B×T×(D + D′)
     其中 D′ 为频率路径输出维度。
   - 经 MLP 映射回 D 维度，并与原始 X^(ℓ) 做残差相加：
     X^(ℓ+1) = X^(ℓ) + MLP([Y_ssm^(ℓ); Y_hr^(ℓ)])
4. **最终输出**
   - 经过 L 层后，对最顶层输出再做一次归一化：
     H_output = LayerNorm(X^(L)) ∈ B×T×D.
   - 同时，将所有层的频率选择输出 {Y_hr^(ℓ)}*{ℓ=1..L−1} 做均值或加权融合，得到整体心率频率表示：
     H_hr = (1/(L−1)) \* ∑*{ℓ=1..L−1} Y_hr^(ℓ) ∈ B×T×D′.

## 心率解码器模块

**目的**

- 在 FSMamba 编码器提取到的全局时序表示 H_output 和心率频段增强表示 H_hr 之上，结合多任务策略，输出最终心率估计，兼顾精度与鲁棒性。

### 关键步骤

1. **原始特征映射**
   - 将全局时序表示 H_output 通过一组多层感知机 (MLP1) 映射到基础回归特征 F_raw。
   - 该特征主要关注时域上的整体趋势，用于粗略回归心率值。
2. **频谱特征映射**
   - 将心率频段特征 H_hr 先经另一组多层感知机 (MLP2) 映射，再通过可学习带通滤波器 (如 SincConv1d)，进一步强化心率频带信息，得到 F_freq。
   - 这样能让模型同时兼顾时域脉搏走向与频域心率峰值。
3. **特征融合与加权回归**
   - 将 F_raw 与 F_freq 在通道维度拼接后，经过融合网络 (MLP_fusion) 得到融合向量 F_fused。
   - 采用“区间分类 + 区间内回归”策略：
     1. 先将 F_fused 输入分类头，预测心率所属的离散区间，得到概率向量 {p_i}₱₌1…C。
     2. 对每个区间再用一个小型回归头 Reg_i 预测该区间内的偏移量 δᵢ。
     3. 最终取加权和：
        ŷ_final = ∑₍ᵢ₌₁..C₎ pᵢ × Regᵢ(F_fused).

## 损失函数设计（Loss Function Design）

**目的**

- 通过多目标联合优化，使模型既能精准回归心率，又能保证预测分布与真实标签分布的一致性，提升整体鲁棒性。

### 损失函数组成

L_total = λ_reg · L_reg + λ_cls · L_cls + λ_dist · L_dist

- 默认权重设为 λ_reg = 1.0，λ_cls = 1.0，λ_dist = 0.5。

### (1) 回归损失 L_reg

- 采用 Smooth-L1 损失 (Huber Loss) 对模型最终预测 ŷᵢ 与真实心率 yᵢ 计算：
  L_reg = (1/N) · ∑_{i=1..N} SmoothL1(ŷᵢ − yᵢ).
- 相较于纯 L2 损失，Smooth-L1 在误差较小时贴近 L2，在误差较大时退化为 L1，更能缓解异常值影响。

### (2) 区间分类损失 L_cls

- 将真实心率离散到 C 个预定义区间，若第 i 个样本真实所属区间为 cᵢ，预测概率为 p_{i,cᵢ}，交叉熵损失为：
  L_cls = −(1/N) · ∑*{i=1..N} log(p*{i,cᵢ}).
- 该损失能引导模型先粗略定位心率大致区间，再在回归阶段做精细拟合，提升收敛稳定性与准确度。

### (3) 分布对齐损失 L_dist

- 计算真实心率与预测心率在批次或全部样本上的统计量：
  - 真实均值 μ_y，真实标准差 σ_y；
  - 预测均值 μ_{ŷ}，预测标准差 σ_{ŷ}。
- 定义分布对齐损失：
  L_dist = (μ_{ŷ} − μ_y)² + (σ_{ŷ} − σ_y)².
- 该项惩罚预测分布与真实分布在均值与方差上的差异，从整体上保证模型对心率整体趋势与波动范围的把握。

# 使用教程

首先请将VIPL-HR数据集放置在项目目录中的data目录下，随后执行如下操作

## 项目流程概述

1. 数据准备与处理
2. 数据集分割
3. 模型训练、测试及评估

##  数据准备与处理

### 使用VIPL-HR数据集

VIPL-HR数据集包含RGB和NIR视频，需要进行处理以提取人脸ROI并创建配对数据。

```bash
python data/vipl_hr_processor.py --input /path/to/vipl_hr --output /path/to/processed_vipl
```

处理后的数据将保存在paired目录下，按受试者ID和场景ID组织。. 数据集分割

### 基于受试者的1-fold分割

```bash
python data/create_scene_split.py --data-dir /path/to/processed_vipl --output-dir /path/to/processed_vipl/splits
```

##  模型训练

准备一个配置文件（config.yaml），然后运行训练脚本：

```bash
# 使用配置文件运行训练
python training/train_multitask.py --config /path/to/config.yaml
```

**注意**：train_multitask会对于每一轮对应的对应的数据集执行对应的训练、验证、测试

## 结果输出

为了方对于数据执行可视化分析，最终的训练结果将以json格式执行保存，具体可以查看项目中的metris文件夹

## 训练示例

![](./images/example1.png)

![](./images/example2.png)



# 注意事项

* 如果您在项目文件中看到活动检测不要疑惑，活动检测是之前迭代实现的版本实现过的二级活动检测分类网络（但是由于实践后发现网络表现效果不佳因此将其去掉）
* 对于多任务来讲本项目在FSMambaEncoder输出字典中还包括其余生理指标的特征生成机制，本人将其中的心率预测分支摘除参加4rd_repss竞赛（*The 4th RePSS* – *Multimodal Fusion Learning for Remote Physiological Signal Sensing, IJCAI 2025, Montreal, Canada, August* *2025*），基于轻量化的与训练权重我取得了第六名的位次
* 项目预测的生理指标包含心率、呼吸率、血氧
