import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.baseline_decoder import heart_rate_to_class

class MultiTaskHeartRateLoss(nn.Module):
    """
    多任务心率损失函数，结合回归、分类和分布匹配三种损失
    
    包含三个主要组成部分:
    1. 回归损失: Smooth L1 (Huber) 损失对预测心率与真实心率进行惩罚
    2. 分类损失: 将心率分为低/中/高三类，用交叉熵损失进行监督
    3. 分布匹配损失: 匹配批次内预测心率与真实心率的统计特性(均值和标准差)
    """
    
    def __init__(self, regression_weight=1.0, classification_weight=1.0, 
                 distribution_weight=0.5, beta=1.0, class_weights=None):
        """
        初始化多任务心率损失函数
        
        参数:
            regression_weight: 回归损失权重
            classification_weight: 分类损失权重
            distribution_weight: 分布匹配损失权重
            beta: Smooth L1损失的beta参数，控制平滑区域的大小
            class_weights: 分类任务中各类别的权重，格式为[w_low, w_mid, w_high]
        """
        super(MultiTaskHeartRateLoss, self).__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.distribution_weight = distribution_weight
        
        # 回归损失 - Smooth L1 (Huber)
        self.regression_loss = nn.SmoothL1Loss(beta=beta)
        
        # 分类损失 - 交叉熵
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)
        self.classification_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 心率类别阈值
        self.low_threshold = 60.0   # 低于60 BPM为低心率
        self.high_threshold = 100.0  # 高于100 BPM为高心率
        
        # 用于数值稳定性
        self.eps = 1e-8
    
    def forward(self, outputs, targets):
        """
        计算多任务心率损失
        
        参数:
            outputs: 模型输出字典或元组
                如果是字典: {'logits': 分类logits, 'regression': 回归预测值}
                如果是元组: (logits, regression)
            targets: 目标心率值 [batch_size]
            
        返回:
            total_loss: 总损失
            loss_dict: 包含各损失组成部分的字典
        """
        # 解析模型输出
        if isinstance(outputs, dict):
            logits = outputs.get('logits', None)
            regression = outputs.get('regression', None)
        elif isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, regression = outputs[:2]
        else:
            raise ValueError("输出格式不正确，需要字典或元组")
        
        # 确保targets是正确的形状
        if len(targets.shape) > 1:
            targets = targets.view(-1)
        
        batch_size = targets.size(0)
        device = targets.device
        
        # 初始化损失组件
        loss_dict = {}
        
        # 1. 回归损失 - 带中心距离加权的 Smooth L1 (Huber)
        if regression is not None:
            regression = regression.view(-1)
            
            # 使用Smooth L1/Huber损失，更适合回归任务
            # 移除HR-80加权方式，使用统一权重以降低过拟合风险
            regression_loss = self.regression_loss(regression, targets)
            
            # 仍然计算RMSE作为指标，但不用于反向传播
            squared_errors = (regression - targets) ** 2
            mse = torch.mean(squared_errors)
            rmse = torch.sqrt(mse + self.eps)  # 加入self.eps避免数值不稳定
            loss_dict['rmse'] = rmse
            
            loss_dict['regression_loss'] = regression_loss
            # 保存未加权的原始MAE供参考
            abs_errors = torch.abs(regression - targets)  # 重新计算绝对误差用于指标
            loss_dict['raw_mae'] = torch.mean(abs_errors)
        else:
            regression_loss = torch.tensor(0.0, device=device)
            loss_dict['regression_loss'] = regression_loss
        
        # 2. 分类损失 - 交叉熵
        if logits is not None:
            # 从logits获取实际类别数量
            num_classes = logits.shape[1]  # 应该是5
            
            # 将目标心率转换为类别标签，明确使用num_classes参数
            hr_classes = heart_rate_to_class(targets, num_classes=num_classes)
            
            # 验证类别是否在有效范围内 (0到num_classes-1)
            valid_mask = (hr_classes >= 0) & (hr_classes < num_classes)
            if not torch.all(valid_mask):
                # 记录无效类别的信息
                invalid_indices = torch.where(~valid_mask)[0]
                invalid_hrs = targets[invalid_indices]
                invalid_classes = hr_classes[invalid_indices]
                print(f"警告: 发现无效心率类别: indices={invalid_indices.tolist()}, hrs={invalid_hrs.tolist()}, classes={invalid_classes.tolist()}")
                
                # 将无效类别替换为默认类别(中间类别)
                middle_class = min(2, num_classes - 1)  # 默认使用类别2，除非类别总数更少
                hr_classes = torch.where(valid_mask, hr_classes, torch.tensor(middle_class, device=device).to(hr_classes.dtype))
            
            # 如果类别权重在设备上，确保放在正确的设备
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(device)
                self.classification_loss = nn.CrossEntropyLoss(weight=self.class_weights)
            
            # 计算分类损失
            classification_loss = self.classification_loss(logits, hr_classes)
            loss_dict['classification_loss'] = classification_loss
            
            # 记录分类准确率
            _, predicted_classes = torch.max(logits, 1)
            accuracy = (predicted_classes == hr_classes).float().mean()
            loss_dict['classification_accuracy'] = accuracy
        else:
            classification_loss = torch.tensor(0.0, device=device)
            loss_dict['classification_loss'] = classification_loss
            loss_dict['classification_accuracy'] = torch.tensor(0.0, device=device)
        
        # 3. 分布匹配损失 - 均值和标准差MSE
        if regression is not None and batch_size > 1:
            # 计算批次统计量
            pred_mean = torch.mean(regression)
            pred_std = torch.std(regression)
            
            target_mean = torch.mean(targets)
            target_std = torch.std(targets)
            
            # 计算均值匹配损失
            mean_loss = F.mse_loss(pred_mean, target_mean)
            loss_dict['mean_matching_loss'] = mean_loss
            
            # 计算标准差匹配损失
            std_loss = F.mse_loss(pred_std, target_std)
            loss_dict['std_matching_loss'] = std_loss
            
            # 总分布匹配损失
            distribution_loss = mean_loss + std_loss
            loss_dict['distribution_loss'] = distribution_loss
        else:
            distribution_loss = torch.tensor(0.0, device=device)
            loss_dict['distribution_loss'] = distribution_loss
            loss_dict['mean_matching_loss'] = torch.tensor(0.0, device=device)
            loss_dict['std_matching_loss'] = torch.tensor(0.0, device=device)
        
        # 仅作为指标计算RMSE，不用于损失计算
        if regression is not None:
            squared_errors = (regression - targets) ** 2
            mse = torch.mean(squared_errors)
            rmse = torch.sqrt(mse + self.eps)  # 加入self.eps避免数值不稳定
            loss_dict['rmse'] = rmse  # 仅用于指标记录
        else:
            loss_dict['rmse'] = torch.tensor(0.0, device=device)
        
        # 计算加权总损失
        total_loss = (
            self.regression_weight * regression_loss +
            self.classification_weight * classification_loss +
            self.distribution_weight * distribution_loss
        )
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    
# 首先去掉分类损失

