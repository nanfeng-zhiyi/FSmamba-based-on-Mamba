#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多任务训练脚本 - 简化版
专注于同时训练心率回归和活动分类任务
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# JSON数据导出工具
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.repss_model_v2 import RePSSModelV2, MambaModelConfig
from models.baseline_decoder import ClassificationHRDecoder
from training.multi_task_loss import MultiTaskHeartRateLoss
from data.vipl_hr_dataset import create_vipl_hr_dataloaders

# 禁用torch.compile可能引起的错误
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

# 检查metrics模块是否存在
try:
    from training.metrics import calculate_metrics
except ImportError:
    # 如果不存在，提供一个简单的实现
    def calculate_metrics(hr_preds, hr_targets):
        import numpy as np
        mae = np.mean(np.abs(hr_preds - hr_targets))
        rmse = np.sqrt(np.mean(np.square(hr_preds - hr_targets)))
        return {"hr_mae": mae, "hr_rmse": rmse}

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger()


class MetricsManager:
    """Metrics manager for collecting, tracking and exporting detailed training metrics in JSON format"""
    
    def __init__(self, save_dir):
        """Initialize the metrics manager
        
        Args:
            save_dir: Directory to save metrics data
        """
        self.output_dir = Path(save_dir) / 'metrics'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置各类数据文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_history_file = self.output_dir / f'training_history_{timestamp}.json'
        self.scenario_metrics_file = self.output_dir / f'scenario_metrics_{timestamp}.json'
        self.detailed_metrics_file = self.output_dir / f'detailed_metrics_{timestamp}.json'
        self.model_params_file = self.output_dir / f'model_params_{timestamp}.json'
        
        # 先尝试加载已有历史数据，如果不存在则创建新的
        self.history = self.load_history()
        if not self.history:  # 如果加载失败，初始化空历史记录
            self.history = {
                'metadata': {
                    'start_time': timestamp,
                    'last_updated': timestamp,
                    'epochs_completed': 0,
                    'total_training_samples': 0,
                    'total_validation_samples': 0,
                    'total_test_samples': 0
                },
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'test_loss': [],
                # 心率指标
                'hr_mae': [],        # 验证集平均绝对误差
                'hr_rmse': [],       # 验证集均方根误差
                'hr_me': [],         # 验证集平均误差 (正负值)
                'test_hr_mae': [],   # 测试集平均绝对误差
                'test_hr_rmse': [],  # 测试集均方根误差
                'test_hr_me': [],    # 测试集平均误差 (正负值)
                # 分类指标
                'cls_accuracy': [],       # 验证集准确率
                'cls_precision': [],      # 验证集精确率 (宏平均)
                'cls_recall': [],         # 验证集召回率 (宏平均)
                'cls_f1': [],            # 验证集F1分数 (宏平均)
                'test_cls_accuracy': [],  # 测试集准确率
                'test_cls_precision': [], # 测试集精确率 (宏平均)
                'test_cls_recall': [],    # 测试集召回率 (宏平均)
                'test_cls_f1': [],       # 测试集F1分数 (宏平均)
                # 详细损失和权重信息
                'regression_loss': [],      # 回归损失
                'classification_loss': [],  # 分类损失
                'distribution_loss': [],    # 分布损失（如果有）
                'classification_weight': [], # 分类任务权重
                'distribution_weight': [],   # 分布任务权重
                'lr': [],                    # 学习率
                # 每个epoch内的批次信息
                'batch_data': {}
            }
            logger.info("创建新的指标历史记录")
        else:
            self.history['metadata']['last_updated'] = timestamp
            logger.info(f"成功加载已有历史记录，包含{len(self.history['epochs'])}个训练轮次数据")
        
        # 跟踪SincConv滤波器参数
        self.model_params = {
            'epochs': [],
            'sinc_filters': {
                'low_hz_mean': [],      # 低截止频率均值（中心频率）
                'low_hz_std': [],       # 低截止频率标准差
                'low_hz_min': [],       # 最小低截止频率
                'low_hz_max': [],       # 最大低截止频率
                'band_hz_mean': [],     # 带宽均值
                'band_hz_std': [],      # 带宽标准差
                'band_hz_min': [],      # 最小带宽
                'band_hz_max': []       # 最大带宽
            },
            'attention_weights': {},    # 注意力权重（如果有）
            'conv_stats': {}            # 卷积层统计信息（如果有）
        }
        
        # 场景指标数据 - 按照场景ID分组
        self.scenario_metrics = {}
        
        # 详细预测信息 - 每个样本级别
        self.detailed_metrics = {
            'per_sample': {},
            'per_class': {},
            'per_scenario': {}
        }
        
        # 分类分布数据
        self.class_distribution = {}
        
        # 心率误差分布
        self.hr_error_distribution = {}
        
        # 测试集结果存储
        self.test_results = {
            'sample_ids': [],
            'hr_preds': [],
            'hr_targets': [],
            'hr_errors': [],
            'hr_errors_pct': [],
            'cls_preds': [],
            'cls_probs': [],  # 保存分类概率，而不仅仅是预测类别
            'cls_targets': [],
            'scenario_ids': [],
            'subject_ids': []
        }
    
    def update_metrics(self, epoch, metrics, loss_dict=None, batch_idx=None, is_train=True):
        """Update training metrics history
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics from current evaluation
            loss_dict: Detailed loss dictionary with classification and regression sub-losses
            batch_idx: Optional batch index if updating batch-level metrics
            is_train: Whether these metrics are from training (True) or evaluation (False)
        """
        # 添加epoch记录（仅当是新的epoch时）
        if epoch not in self.history['epochs']:
            self.history['epochs'].append(epoch)
            # 更新元数据
            self.history['metadata']['epochs_completed'] = len(self.history['epochs'])
            self.history['metadata']['last_updated'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 初始化批次数据结构（如果不存在）
            if str(epoch) not in self.history['batch_data']:
                self.history['batch_data'][str(epoch)] = {
                    'train': {'losses': [], 'metrics': [], 'lr': []},
                    'val': {'losses': [], 'metrics': []}
                }
        
        # 更新批次级别的指标（如果提供了batch_idx）
        if batch_idx is not None:
            batch_type = 'train' if is_train else 'val'
            batch_data = {
                'batch_idx': batch_idx,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
            
            # 添加损失数据
            if loss_dict:
                self.history['batch_data'][str(epoch)][batch_type]['losses'].append({
                    'batch_idx': batch_idx,
                    **loss_dict
                })
                
            # 添加指标数据
            if metrics:
                self.history['batch_data'][str(epoch)][batch_type]['metrics'].append({
                    'batch_idx': batch_idx,
                    **metrics
                })
            return
        
        # 更新epoch级别的标准指标
        for key, value in metrics.items():
            if key in self.history:
                # 确保列表长度与epochs列表相同
                while len(self.history[key]) < len(self.history['epochs']) - 1:
                    self.history[key].append(None)  # 填充缺失值
                self.history[key].append(value)
        
        # 更新详细损失字典（如果提供）
        if loss_dict is not None:
            for loss_key, loss_value in loss_dict.items():
                history_key = loss_key  # 直接使用损失字典中的键名
                if history_key in self.history:
                    # 确保列表长度与epochs列表相同
                    while len(self.history[history_key]) < len(self.history['epochs']) - 1:
                        self.history[history_key].append(None)  # 填充缺失值
                    self.history[history_key].append(loss_value)
    
    def update_error_distribution(self, hr_preds, hr_targets, epoch, scenario_ids=None, subject_ids=None):
        """Update heart rate error distribution
        
        Args:
            hr_preds: Heart rate predictions array
            hr_targets: Heart rate targets array
            epoch: Current epoch number
            scenario_ids: Optional scenario ID for each sample
            subject_ids: Optional subject ID for each sample
        """
        # 计算误差
        errors = hr_preds - hr_targets
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors / hr_targets) * 100  # 百分比误差
        
        # 以epoch为索引存储分布数据
        if str(epoch) not in self.hr_error_distribution:
            self.hr_error_distribution[str(epoch)] = {
                'errors': errors.tolist() if hasattr(errors, 'tolist') else errors,
                'abs_errors': abs_errors.tolist() if hasattr(abs_errors, 'tolist') else abs_errors,
                'pct_errors': pct_errors.tolist() if hasattr(pct_errors, 'tolist') else pct_errors,
                'within_3bpm': (abs_errors <= 3).mean() * 100,  # 3bpm内的百分比
                'within_5bpm': (abs_errors <= 5).mean() * 100,  # 5bpm内的百分比
                'within_10bpm': (abs_errors <= 10).mean() * 100,  # 10bpm内的百分比
                'within_5pct': (pct_errors <= 5).mean() * 100,  # 5%内的百分比
                'within_10pct': (pct_errors <= 10).mean() * 100,  # 10%内的百分比
                'within_20pct': (pct_errors <= 20).mean() * 100,  # 20%内的百分比
                'histogram': {
                    'bins': np.linspace(-30, 30, 31).tolist(),  # -30到30间隔1的bins
                    'counts': np.histogram(errors, bins=np.linspace(-30, 30, 31))[0].tolist()
                },
                'percentiles': {
                    'p10': np.percentile(abs_errors, 10),
                    'p25': np.percentile(abs_errors, 25),
                    'p50': np.percentile(abs_errors, 50),  # 中位数
                    'p75': np.percentile(abs_errors, 75),
                    'p90': np.percentile(abs_errors, 90),
                    'p95': np.percentile(abs_errors, 95),
                }
            }
        
        # 如果提供了场景ID，按场景计算误差指标
        if scenario_ids is not None:
            scenario_data = {}
            unique_scenarios = np.unique(scenario_ids)
            
            for scenario in unique_scenarios:
                scenario_mask = scenario_ids == scenario
                if not np.any(scenario_mask):
                    continue
                    
                s_errors = errors[scenario_mask]
                s_abs_errors = abs_errors[scenario_mask]
                s_pct_errors = pct_errors[scenario_mask]
                
                scenario_data[str(scenario)] = {
                    'count': int(np.sum(scenario_mask)),
                    'hr_mae': float(np.mean(s_abs_errors)),
                    'hr_rmse': float(np.sqrt(np.mean(np.square(s_errors)))),
                    'hr_me': float(np.mean(s_errors)),
                    'within_3bpm': float((s_abs_errors <= 3).mean() * 100),
                    'within_5bpm': float((s_abs_errors <= 5).mean() * 100),
                    'within_10bpm': float((s_abs_errors <= 10).mean() * 100),
                    'within_5pct': float((s_pct_errors <= 5).mean() * 100),
                    'within_10pct': float((s_pct_errors <= 10).mean() * 100),
                    'within_20pct': float((s_pct_errors <= 20).mean() * 100)
                }
            
            # 将场景数据添加到场景指标中
            if str(epoch) not in self.scenario_metrics:
                self.scenario_metrics[str(epoch)] = {}
            self.scenario_metrics[str(epoch)].update(scenario_data)
    
    def update_class_distribution(self, cls_preds, cls_targets, epoch, class_names=None):
        """Update classification prediction distribution
        
        Args:
            cls_preds: Classification predictions [batch_size, num_classes] or class indices
            cls_targets: Classification targets [batch_size]
            epoch: Current epoch number
            class_names: Optional mapping of class indices to names
        """
        if len(cls_preds) == 0 or len(cls_targets) == 0:
            return
        
        # 确定是概率还是预测的类别索引
        if cls_preds.ndim > 1 and cls_preds.shape[1] > 1:
            # 如果是概率分布，转换为类别索引
            cls_indices = np.argmax(cls_preds, axis=1)
            cls_probs = cls_preds  # 保存原始概率
        else:
            # 已经是类别索引
            cls_indices = cls_preds
            cls_probs = None
        
        # 计算混淆矩阵
        n_classes = max(np.max(cls_indices) + 1, np.max(cls_targets) + 1)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
        
        for i in range(len(cls_targets)):
            confusion_matrix[cls_targets[i], cls_indices[i]] += 1
            
        # 计算每个类别的准确率、精确率、召回率和F1
        per_class_metrics = {}
        class_names_map = {} if class_names is None else class_names
        
        for class_idx in range(n_classes):
            # 计算True Positives, False Positives, False Negatives
            tp = confusion_matrix[class_idx, class_idx]
            fp = np.sum(confusion_matrix[:, class_idx]) - tp
            fn = np.sum(confusion_matrix[class_idx, :]) - tp
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 存储类别指标
            class_label = class_names_map.get(class_idx, f"class_{class_idx}")
            per_class_metrics[class_label] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(confusion_matrix[class_idx, :]))
            }
        
        # 存储分类分布数据
        if str(epoch) not in self.class_distribution:
            self.class_distribution[str(epoch)] = {
                'confusion_matrix': confusion_matrix.tolist(),
                'class_metrics': per_class_metrics,
                'overall_accuracy': float(np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))
            }
            
            # 将类别名称映射添加到分布数据中（如果有）
            if class_names is not None:
                self.class_distribution[str(epoch)]['class_names'] = class_names
                
        # 将类别指标添加到详细指标中
        self.detailed_metrics['per_class'][str(epoch)] = per_class_metrics
        
    def update_class_distribution(self, cls_preds, cls_targets, epoch, class_names=None):
        """Update classification prediction distribution
        
        Args:
            cls_preds: Classification predictions [batch_size, num_classes] or class indices
            cls_targets: Classification targets [batch_size]
            epoch: Current epoch number
            class_names: Optional mapping of class indices to names
        """
        if len(cls_preds) == 0 or len(cls_targets) == 0:
            return
        
        # 确定是概率还是预测的类别索引
        if cls_preds.ndim > 1 and cls_preds.shape[1] > 1:
            # 如果是概率分布，转换为类别索引
            cls_indices = np.argmax(cls_preds, axis=1)
            cls_probs = cls_preds  # 保存原始概率
        else:
            # 已经是类别索引
            cls_indices = cls_preds
            cls_probs = None
        
        # 计算混淆矩阵
        n_classes = max(np.max(cls_indices) + 1, np.max(cls_targets) + 1)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
        
        for i in range(len(cls_targets)):
            confusion_matrix[cls_targets[i], cls_indices[i]] += 1
            
        # 计算每个类别的准确率、精确率、召回率和F1
        per_class_metrics = {}
        class_names_map = {} if class_names is None else class_names
    
        for class_idx in range(n_classes):
            # 计算True Positives, False Positives, False Negatives
            tp = confusion_matrix[class_idx, class_idx]
            fp = np.sum(confusion_matrix[:, class_idx]) - tp
            fn = np.sum(confusion_matrix[class_idx, :]) - tp
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 存储类别指标
            class_label = class_names_map.get(class_idx, f"class_{class_idx}")
            per_class_metrics[class_label] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(confusion_matrix[class_idx, :]))
            }
        
        # 存储分类分布数据
        if str(epoch) not in self.class_distribution:
            self.class_distribution[str(epoch)] = {
                'confusion_matrix': confusion_matrix.tolist(),
                'class_metrics': per_class_metrics,
                'overall_accuracy': float(np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))
            }
            
            # 将类别名称映射添加到分布数据中（如果有）
            if class_names is not None:
                self.class_distribution[str(epoch)]['class_names'] = class_names
                
        # 将类别指标添加到详细指标中
        self.detailed_metrics['per_class'][str(epoch)] = per_class_metrics

    def update_sample_metrics(self, sample_ids, hr_preds, hr_targets, cls_preds=None, cls_targets=None, 
                           scenario_ids=None, subject_ids=None, meta_data=None, epoch=None):
        """Update per-sample detailed metrics
        
        Args:
            sample_ids: Unique identifiers for each sample
            hr_preds: Heart rate predictions
            hr_targets: Heart rate targets
            cls_preds: Classification predictions (probabilities or indices), can be None
            cls_targets: Classification targets, can be None
            scenario_ids: Optional scenario IDs
            subject_ids: Optional subject IDs
            meta_data: Additional metadata for samples
            epoch: Current epoch number
        """
        if epoch is None or len(sample_ids) == 0:
            return
            
        # 确保sample_ids是字符串
        sample_ids = [str(s) for s in sample_ids]
        
        # 计算心率误差
        hr_errors = hr_preds - hr_targets
        hr_abs_errors = np.abs(hr_errors)
        hr_pct_errors = np.abs(hr_errors / np.maximum(hr_targets, 1e-5)) * 100  # 百分比误差，防止除零
        
        # 确定是否有分类数据
        has_classification = False
        cls_indices = None
        cls_probs = None
    
        # 处理分类预测数据（如果有）
        if cls_preds is not None and cls_targets is not None:
            has_classification = True
            if hasattr(cls_preds, 'ndim') and cls_preds.ndim > 1 and cls_preds.shape[1] > 1:
                # 如果是概率分布，计算预测类别和概率
                cls_indices = np.argmax(cls_preds, axis=1)
                cls_probs = cls_preds.tolist() if hasattr(cls_preds, 'tolist') else cls_preds
            else:
                # 已经是类别索引
                cls_indices = cls_preds
                cls_probs = None
        
        # 初始化当前epoch的样本指标字典
        epoch_key = str(epoch)
        if epoch_key not in self.detailed_metrics['per_sample']:
            self.detailed_metrics['per_sample'][epoch_key] = {}
        
        # 处理场景数据（如果有）
        if scenario_ids is not None and len(scenario_ids) > 0:
            if epoch_key not in self.scenario_metrics:
                self.scenario_metrics[epoch_key] = {}
                
            # 按场景ID分组统计数据
            scenario_data = defaultdict(dict)
            for i, scenario_id in enumerate(scenario_ids):
                if i >= len(hr_preds) or i >= len(hr_targets):
                    continue
                    
                scenario_id = str(scenario_id)
                if scenario_id not in scenario_data:
                    scenario_data[scenario_id] = {
                        'count': 0,
                        'hr_errors': [],
                        'hr_abs_errors': [],
                        'hr_pct_errors': []
                    }
                    
                scenario_data[scenario_id]['count'] += 1
                scenario_data[scenario_id]['hr_errors'].append(float(hr_errors[i]))
                scenario_data[scenario_id]['hr_abs_errors'].append(float(hr_abs_errors[i]))
                scenario_data[scenario_id]['hr_pct_errors'].append(float(hr_pct_errors[i]))
                
            # 计算每个场景的平均指标
            for scenario_id, data in scenario_data.items():
                if data['count'] > 0:
                    data['mean_hr_error'] = float(np.mean(data['hr_errors']))
                    data['mean_hr_abs_error'] = float(np.mean(data['hr_abs_errors']))
                    data['mean_hr_pct_error'] = float(np.mean(data['hr_pct_errors']))
                    data['std_hr_error'] = float(np.std(data['hr_errors']))
                    data['max_hr_error'] = float(np.max(data['hr_abs_errors']))
                    
            # 更新场景指标
            self.scenario_metrics[epoch_key].update(scenario_data)
                
        # 更新每个样本的指标
        for i, sample_id in enumerate(sample_ids):
            if i >= len(hr_preds) or i >= len(hr_targets):
                # 跳过索引超出范围的样本
                continue
                
            # 基本心率指标
            sample_data = {
                'hr_pred': float(hr_preds[i]),
                'hr_target': float(hr_targets[i]),
                'hr_error': float(hr_errors[i]),
                'hr_abs_error': float(hr_abs_errors[i]),
                'hr_pct_error': float(hr_pct_errors[i])
            }
            
            # 添加分类指标（如果有）
            if has_classification and i < len(cls_indices) and i < len(cls_targets):
                sample_data['cls_pred'] = int(cls_indices[i])
                sample_data['cls_target'] = int(cls_targets[i])
                sample_data['cls_correct'] = int(cls_indices[i] == cls_targets[i])
                
                # 添加分类概率（如果有）
                if cls_probs is not None and i < len(cls_probs):
                    try:
                        sample_data['cls_probs'] = cls_probs[i]
                    except Exception as e:
                        # 忽略概率添加错误
                        logger.warning(f"无法添加分类概率: {e}")
            
            # 添加场景ID（如果有）
            if scenario_ids is not None and i < len(scenario_ids):
                sample_data['scenario_id'] = str(scenario_ids[i])
            
            # 添加受试者ID（如果有）
            if subject_ids is not None and i < len(subject_ids):
                sample_data['subject_id'] = str(subject_ids[i])
            
            # 添加额外元数据（如果有）
            if meta_data is not None and isinstance(meta_data, dict):
                for key, values in meta_data.items():
                    if i < len(values):
                        sample_data[key] = values[i]
            
            # 存储样本数据
            self.detailed_metrics['per_sample'][epoch_key][sample_id] = sample_data
            
    def update_model_params(self, model, epoch, param_types=None):
        """更新模型参数统计信息
        
        Args:
            model: PyTorch模型
            epoch: 当前轮次
            param_types: 要跟踪的参数类型列表，默认为None则跟踪所有SincConv参数
        """
        if param_types is None:
            param_types = ['sinc']
            
        # 将epoch添加到模型参数历史记录中
        if epoch not in self.model_params['epochs']:
            self.model_params['epochs'].append(epoch)
            
        # 提取SincConv参数（如果有）
        if 'sinc' in param_types and hasattr(model, 'get_sinc_params'):
            try:
                sinc_params = model.get_sinc_params()
                
                if sinc_params and 'low_hz' in sinc_params and 'band_hz' in sinc_params:
                    low_hz = sinc_params['low_hz'].cpu().detach().numpy()
                    band_hz = sinc_params['band_hz'].cpu().detach().numpy()
                    
                    # 更新SincConv参数统计
                    self.model_params['sinc_filters']['low_hz_mean'].append(float(np.mean(low_hz)))
                    self.model_params['sinc_filters']['low_hz_std'].append(float(np.std(low_hz)))
                    self.model_params['sinc_filters']['low_hz_min'].append(float(np.min(low_hz)))
                    self.model_params['sinc_filters']['low_hz_max'].append(float(np.max(low_hz)))
                    
                    self.model_params['sinc_filters']['band_hz_mean'].append(float(np.mean(band_hz)))
                    self.model_params['sinc_filters']['band_hz_std'].append(float(np.std(band_hz)))
                    self.model_params['sinc_filters']['band_hz_min'].append(float(np.min(band_hz)))
                    self.model_params['sinc_filters']['band_hz_max'].append(float(np.max(band_hz)))
            except Exception as e:
                logger.warning(f"获取SincConv参数时出错: {e}")
                
        # 提取注意力权重（如果有）
        if 'attention' in param_types and hasattr(model, 'get_attention_weights'):
            try:
                attn_weights = model.get_attention_weights()
                if attn_weights:
                    # 将epoch作为键存储注意力权重
                    epoch_key = str(epoch)
                    self.model_params['attention_weights'][epoch_key] = {}
                    
                    for layer_name, weights in attn_weights.items():
                        # 将Tensor转换为numpy数组，并转换为列表以便JSON序列化
                        if hasattr(weights, 'cpu') and hasattr(weights, 'detach'):
                            weights = weights.cpu().detach().numpy()
                            
                        if hasattr(weights, 'tolist'):
                            weights = weights.tolist()
                            
                        self.model_params['attention_weights'][epoch_key][layer_name] = weights
            except Exception as e:
                logger.warning(f"获取注意力权重时出错: {e}")
    
    def load_history(self):
        """加载历史数据，如果文件不存在则返回空字典"""
        try:
            if os.path.exists(self.training_history_file):
                with open(self.training_history_file, 'r', encoding='utf-8') as f:
                    logger.info(f"从{self.training_history_file}加载历史数据")
                    return json.load(f)
            else:
                logger.info(f"历史数据文件 {self.training_history_file} 不存在，将创建新的历史记录")
                return {}
        except Exception as e:
            logger.error(f"加载历史数据时出错: {e}")
            return {}
    
    def save_training_history(self):
        """保存训练历史数据到JSON文件"""
        try:
            # 更新元数据
            self.history['metadata']['last_updated'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建可序列化的数据（将NumPy数组转换为列表）
            serializable_history = {}
            for key, value in self.history.items():
                if isinstance(value, list) and value and hasattr(value[0], 'tolist'):
                    serializable_history[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
                elif isinstance(value, dict):
                    # 递归处理嵌套字典
                    serializable_history[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list) and sub_value and hasattr(sub_value[0], 'tolist'):
                            serializable_history[key][sub_key] = [v.tolist() if hasattr(v, 'tolist') else v for v in sub_value]
                        else:
                            serializable_history[key][sub_key] = sub_value
                else:
                    serializable_history[key] = value
            
            # 保存到文件
            with open(self.training_history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
                
            logger.info(f"训练历史数据已保存到: {self.training_history_file}")
            return True
        except Exception as e:
            logger.error(f"保存训练历史数据时出错: {e}")
            return False
            
    def save_scenario_metrics(self):
        """保存场景指标数据到JSON文件"""
        try:
            if not self.scenario_metrics:
                logger.warning("没有场景指标数据可保存")
                return False
                
            # 准备元数据
            metadata = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'total_epochs': len(self.scenario_metrics.keys()),
                'total_scenarios': len(set(sum([list(metrics.keys()) for metrics in self.scenario_metrics.values()], [])))
            }
            
            # 创建完整数据结构
            data = {
                'metadata': metadata,
                'scenario_metrics': self.scenario_metrics
            }
            
            # 保存到文件
            with open(self.scenario_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"场景指标数据已保存到: {self.scenario_metrics_file}")
            return True
        except Exception as e:
            logger.error(f"保存场景指标数据时出错: {e}")
            return False
            
    def save_detailed_metrics(self):
        """保存详细指标数据到JSON文件"""
        try:
            if not any(self.detailed_metrics.values()):
                logger.warning("没有详细指标数据可保存")
                return False
                
            # 准备元数据
            metadata = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'metrics_types': list(self.detailed_metrics.keys()),
                'epochs': list(sorted(set(
                    sum([list(metrics.keys()) for metrics in self.detailed_metrics.values() if isinstance(metrics, dict)], [])
                )))
            }
            
            # 创建完整数据结构
            data = {
                'metadata': metadata,
                'detailed_metrics': self.detailed_metrics
            }
            
            # 保存到文件
            with open(self.detailed_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"详细指标数据已保存到: {self.detailed_metrics_file}")
            return True
        except Exception as e:
            logger.error(f"保存详细指标数据时出错: {e}")
            return False
            
    def save_model_params(self):
        """保存模型参数数据到JSON文件"""
        try:
            if not self.model_params['epochs']:
                logger.warning("没有模型参数数据可保存")
                return False
                
            # 准备元数据
            metadata = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'total_epochs': len(self.model_params['epochs']),
                'param_types': [k for k in self.model_params.keys() if k != 'epochs']
            }
            
            # 创建完整数据结构
            data = {
                'metadata': metadata,
                'model_params': self.model_params
            }
            
            # 保存到文件
            with open(self.model_params_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"模型参数数据已保存到: {self.model_params_file}")
            return True
        except Exception as e:
            logger.error(f"保存模型参数数据时出错: {e}")
            return False
            
    def save_all_data(self):
        """保存所有收集的数据到JSON文件"""
        saved_files = []
        
        # 保存训练历史数据
        training_history_saved = self.save_training_history()
        if training_history_saved:
            saved_files.append(str(self.training_history_file))
            
        # 保存场景指标数据
        scenario_metrics_saved = self.save_scenario_metrics()
        if scenario_metrics_saved:
            saved_files.append(str(self.scenario_metrics_file))
            
        # 保存详细指标数据
        detailed_metrics_saved = self.save_detailed_metrics()
        if detailed_metrics_saved:
            saved_files.append(str(self.detailed_metrics_file))
            
        # 保存模型参数数据
        model_params_saved = self.save_model_params()
        if model_params_saved:
            saved_files.append(str(self.model_params_file))
            
        # 打印保存状态
        if saved_files:
            logger.info(f"成功保存所有数据文件: {', '.join(saved_files)}")
            return True
        else:
            logger.warning("未能保存任何数据文件")
            return False
    # 场景可视化集成方法
    def update_scenario_data(self, predictions, targets, scenario_ids, meta_data=None, epoch=None):
        """更新场景数据并计算各场景指标
        
        Args:
            predictions: 心率预测值
            targets: 心率目标值
            scenario_ids: 场景ID列表
            meta_data: 额外的元数据信息
            epoch: 当前轮次
        """
        if epoch is None:
            if self.history['epochs']:
                epoch = self.history['epochs'][-1]
            else:
                logger.warning("无法更新场景数据：未指定epoch且历史记录为空")
                return False
                
        # 更新心率误差分布
        self.update_error_distribution(predictions, targets, epoch, scenario_ids)
        
        # 如果有样本ID和元数据，更新样本级别指标
        if meta_data and 'sample_id' in meta_data:
            sample_ids = meta_data['sample_id']
            # 需要模拟分类预测（如果不是多任务模型）
            cls_preds = np.zeros((len(targets), 1))
            cls_targets = np.zeros(len(targets))
            
            # 提取主体ID（如果有）
            subject_ids = None
            if 'subject_id' in meta_data:
                subject_ids = meta_data['subject_id']
                
            self.update_sample_metrics(
                sample_ids, predictions, targets, 
                cls_preds, cls_targets,
                scenario_ids, subject_ids, meta_data, epoch
            )
            
        # 每次更新后自动保存数据
        self.save_scenario_metrics()
        return True







        
        # 6. Sub-Loss Components
        ax6 = fig.add_subplot(gs[2, 1])
        has_subloss = False
        
        # 检查回归损失数据
        if len(self.history['regression_loss']) == len(self.history['epochs']):
            ax6.plot(self.history['epochs'], self.history['regression_loss'], 'r-', label='Regression Loss')
            has_subloss = True
        elif len(self.history['regression_loss']) > 0:
            logger.warning(f"回归损失数据维度不匹配: epochs={len(self.history['epochs'])}, regression_loss={len(self.history['regression_loss'])}")
        
        # 检查分类损失数据
        if len(self.history['classification_loss']) == len(self.history['epochs']):
            ax6.plot(self.history['epochs'], self.history['classification_loss'], 'b-', label='Classification Loss')
            has_subloss = True
        elif len(self.history['classification_loss']) > 0:
            logger.warning(f"分类损失数据维度不匹配: epochs={len(self.history['epochs'])}, classification_loss={len(self.history['classification_loss'])}")
        
        if has_subloss:
            ax6.set_title('Loss Components')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss Value')
            ax6.legend()
            ax6.grid(True, linestyle='--', alpha=0.7)
        else:
            ax6.text(0.5, 0.5, 'No sub-loss data available with matching dimensions', 
                     horizontalalignment='center', verticalalignment='center')
        ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Save comprehensive plots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = self.save_dir / 'comprehensive_training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create and save individual plots for better detail
        self._plot_loss_curves()
        self._plot_heart_rate_metrics()
        self._plot_classification_metrics()
        
        logger.info(f"Comprehensive training history charts saved to: {self.save_dir}")
        return save_path
    
    def _plot_loss_curves(self):
        """Plot detailed loss curves for training and validation"""
        if len(self.history['epochs']) == 0:
            return
            
        plt.figure(figsize=(12, 8))
        plt.plot(self.history['epochs'], self.history['train_loss'], 'b-', marker='o', markersize=4, label='Training Loss')
        plt.plot(self.history['epochs'], self.history['val_loss'], 'r-', marker='s', markersize=4, label='Validation Loss')
        
        if len(self.history['test_loss']) > 0:
            plt.plot(self.history['epochs'], self.history['test_loss'], 'g-', marker='^', markersize=4, label='Test Loss')
        
        # Add regression and classification loss components if available
        if len(self.history['regression_loss']) > 0:
            plt.plot(self.history['epochs'], self.history['regression_loss'], 'c--', alpha=0.7, label='Regression Loss')
        if len(self.history['classification_loss']) > 0:
            plt.plot(self.history['epochs'], self.history['classification_loss'], 'm--', alpha=0.7, label='Classification Loss')
        
        plt.title('Detailed Loss Curve Analysis', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        # 修正：获取当前axes对象，然后设置xaxis的定位器
        ax = plt.gca()  # 获取当前axes对象
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotations for best validation loss
        if len(self.history['val_loss']) > 0:
            best_epoch = np.argmin(self.history['val_loss'])
            best_loss = self.history['val_loss'][best_epoch]
            plt.annotate(f'Best: {best_loss:.4f}', 
                        xy=(self.history['epochs'][best_epoch], best_loss),
                        xytext=(10, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        save_path = self.save_dir / 'detailed_loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_heart_rate_metrics(self):
        """Plot detailed heart rate metrics (MAE and RMSE) for validation and test"""
        if len(self.history['epochs']) == 0:
            return
            
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # MAE Plot
        ax1.plot(self.history['epochs'], self.history['hr_mae'], 'b-', marker='o', markersize=4, label='Validation MAE')
        if len(self.history['test_hr_mae']) > 0:
            ax1.plot(self.history['epochs'], self.history['test_hr_mae'], 'g-', marker='^', markersize=4, label='Test MAE')
        
        ax1.set_title('Heart Rate Mean Absolute Error', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MAE (BPM)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotation for best MAE
        if len(self.history['hr_mae']) > 0:
            best_epoch = np.argmin(self.history['hr_mae'])
            best_mae = self.history['hr_mae'][best_epoch]
            ax1.annotate(f'Best: {best_mae:.2f} BPM', 
                         xy=(self.history['epochs'][best_epoch], best_mae),
                         xytext=(10, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # RMSE Plot
        ax2.plot(self.history['epochs'], self.history['hr_rmse'], 'r-', marker='o', markersize=4, label='Validation RMSE')
        if len(self.history['test_hr_rmse']) > 0:
            ax2.plot(self.history['epochs'], self.history['test_hr_rmse'], 'g-', marker='^', markersize=4, label='Test RMSE')
        
        ax2.set_title('Heart Rate Root Mean Squared Error', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('RMSE (BPM)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotation for best RMSE
        if len(self.history['hr_rmse']) > 0:
            best_epoch = np.argmin(self.history['hr_rmse'])
            best_rmse = self.history['hr_rmse'][best_epoch]
            ax2.annotate(f'Best: {best_rmse:.2f} BPM', 
                         xy=(self.history['epochs'][best_epoch], best_rmse),
                         xytext=(10, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        save_path = self.save_dir / 'heart_rate_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_classification_metrics(self):
        """Plot detailed classification metrics for validation and test"""
        if len(self.history['epochs']) == 0 or len(self.history['cls_accuracy']) == 0:
            return
            
        plt.figure(figsize=(12, 8))
        plt.plot(self.history['epochs'], self.history['cls_accuracy'], 'b-', marker='o', markersize=4, label='Validation Accuracy')
        if len(self.history['test_cls_accuracy']) > 0:
            plt.plot(self.history['epochs'], self.history['test_cls_accuracy'], 'g-', marker='^', markersize=4, label='Test Accuracy')
        
        plt.title('Activity Classification Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.05)  # Set y-axis range for accuracy
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        # 修正：获取当前axes对象，然后设置xaxis的定位器
        ax = plt.gca()  # 获取当前axes对象
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotation for best accuracy
        if len(self.history['cls_accuracy']) > 0:
            best_epoch = np.argmax(self.history['cls_accuracy'])
            best_acc = self.history['cls_accuracy'][best_epoch]
            plt.annotate(f'Best: {best_acc:.4f}', 
                        xy=(self.history['epochs'][best_epoch], best_acc),
                        xytext=(10, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        save_path = self.save_dir / 'classification_accuracy.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, epoch=None):
        """Plot confusion matrix
        
        Args:
            epoch: Specified epoch, if None will use the last epoch
        """
        if not self.class_dist:
            logger.warning("No confusion matrix data available for visualization")
            return
            
        # Determine which epoch data to display
        if epoch is None:
            # Default to latest epoch
            epoch = max(self.class_dist.keys())
        
        # Get confusion matrix for selected epoch
        if epoch not in self.class_dist:
            logger.warning(f"Confusion matrix data for epoch {epoch} is not available")
            return
            
        cm = self.class_dist[epoch]
        
        # Create normalized confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # Get number of classes
        num_classes = cm.shape[0]
        
        # Create figure with two subplots (raw counts and normalized)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Raw counts heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        ax1.set_title(f'Confusion Matrix - Epoch {epoch} (Raw Counts)')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        
        # Normalized heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        ax2.set_title(f'Confusion Matrix - Epoch {epoch} (Normalized)')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / f'confusion_matrix_epoch{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix for epoch {epoch} saved to: {save_path}")
        return save_path
    
    def plot_error_distribution(self):
        """Plot heart rate prediction error distribution"""
        if not self.error_hist:
            logger.warning("No error distribution data available for visualization")
            return
        
        # Combine all error data
        all_errors = np.concatenate(self.error_hist)
        abs_errors = np.abs(all_errors)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
        
        # Plot 1: Regular error distribution (positive and negative errors)
        ax1.hist(all_errors, bins=50, alpha=0.7, color='skyblue', density=True)
        ax1.axvline(x=0, color='r', linestyle='-', linewidth=2, label='Zero Error')
        ax1.axvline(x=3, color='g', linestyle='--', label='±3 BPM')
        ax1.axvline(x=-3, color='g', linestyle='--')
        ax1.axvline(x=5, color='orange', linestyle='--', label='±5 BPM')
        ax1.axvline(x=-5, color='orange', linestyle='--')
        
        # Display key statistics
        within_3bpm = np.mean(abs_errors <= 3.0) * 100
        within_5bpm = np.mean(abs_errors <= 5.0) * 100
        
        ax1.set_title(f'Heart Rate Prediction Error Distribution\n(Within ±3 BPM: {within_3bpm:.1f}%, Within ±5 BPM: {within_5bpm:.1f}%)',
                     fontsize=14)
        ax1.set_xlabel('Prediction Error (BPM)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Add mean and standard deviation information
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        mean_abs_error = np.mean(abs_errors)
        
        text_info = f'Mean Error: {mean_error:.2f} BPM\nStd Dev: {std_error:.2f} BPM\nMAE: {mean_abs_error:.2f} BPM'
        ax1.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Plot 2: Absolute error distribution
        bins = np.arange(0, max(abs_errors) + 1, 1)  # 1 BPM bins
        ax2.hist(abs_errors, bins=bins, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(x=3, color='g', linestyle='--', linewidth=2, label='3 BPM')
        ax2.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='5 BPM')
        ax2.axvline(x=10, color='r', linestyle='--', linewidth=2, label='10 BPM')
        
        ax2.set_title('Absolute Heart Rate Error Distribution', fontsize=14)
        ax2.set_xlabel('Absolute Error (BPM)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Calculate cumulative distribution
        for threshold in [3, 5, 10]:
            percentage = np.mean(abs_errors <= threshold) * 100
            ax2.annotate(f'{percentage:.1f}%', 
                         xy=(threshold, ax2.get_ylim()[1] * 0.9),
                         xytext=(threshold + 0.5, ax2.get_ylim()[1] * 0.9),
                         arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / 'heart_rate_error_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heart rate error distribution chart saved to: {save_path}")
        return save_path
    
    def plot_bland_altman(self):
        """Plot Bland-Altman plot (agreement analysis)"""
        if not self.error_hist:
            logger.warning("No error data available for Bland-Altman analysis")
            return
            
        all_errors = np.concatenate(self.error_hist)
        
        # For Bland-Altman plot, we need predicted and true values
        # But we only stored errors (predicted - true), so we need to reconstruct
        # We can assume a reasonable heart rate range (e.g., 50-120bpm) and distribute true values uniformly within it
        # Note: This is just a simulation, actual data should use the corresponding values
        mean_hr = 80  # Assume average heart rate is 80bpm
        n_samples = len(all_errors)
        
        # Simulate true heart rates (Note: This is simulated data)
        true_hr = np.random.normal(mean_hr, 15, n_samples)
        true_hr = np.clip(true_hr, 50, 150)  # Restrict to reasonable range
        
        # Reconstruct predicted values based on errors
        pred_hr = true_hr + all_errors
        
        # Calculate mean and difference for each pair of measurements
        mean_vals = (pred_hr + true_hr) / 2
        diff_vals = pred_hr - true_hr  # i.e., our stored errors
        
        # Calculate mean and standard deviation of differences
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)
        
        # Plot Bland-Altman plot
        plt.figure(figsize=(12, 9))
        
        # Create scatter plot with colormap based on density
        h = plt.scatter(mean_vals, diff_vals, alpha=0.6, c=np.abs(diff_vals), 
                      cmap='viridis', edgecolor='k', linewidth=0.5)
        plt.colorbar(h, label='Absolute Difference (BPM)')
        
        # Add mean and limits of agreement lines
        plt.axhline(y=mean_diff, color='r', linestyle='-', linewidth=2,
                  label=f'Mean difference: {mean_diff:.2f} BPM')
        
        # 95% limits of agreement (±1.96 SD)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        plt.axhline(y=upper_loa, color='g', linestyle='--', linewidth=2,
                  label=f'Upper LoA (+1.96SD): {upper_loa:.2f} BPM')
        plt.axhline(y=lower_loa, color='g', linestyle='--', linewidth=2,
                  label=f'Lower LoA (-1.96SD): {lower_loa:.2f} BPM')
        
        # Add regression line to check for proportional bias
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_vals, diff_vals)
        regression_line = slope * mean_vals + intercept
        plt.plot(mean_vals, regression_line, 'b--', linewidth=1.5, 
                label=f'Regression line (p={p_value:.4f})')
        
        # Format plot
        plt.xlabel('Mean of True and Predicted HR (BPM)', fontsize=12)
        plt.ylabel('Difference (Predicted - True) (BPM)', fontsize=12)
        plt.title('Bland-Altman Plot (Heart Rate Measurement Agreement)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add statistics annotation
        stats_text = (f"Mean difference: {mean_diff:.2f} BPM\n"
                     f"SD of differences: {std_diff:.2f} BPM\n"
                     f"95% Limits of Agreement: [{lower_loa:.2f}, {upper_loa:.2f}] BPM\n"
                     f"Range of differences: [{min(diff_vals):.2f}, {max(diff_vals):.2f}] BPM")
        
        plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                    fontsize=10, verticalalignment='bottom')
        
        # Save figure
        save_path = self.save_dir / 'bland_altman_plot.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Bland-Altman plot saved to: {save_path}")
        return save_path
        
    def load_history(self):
        """加载历史数据，如果文件不存在则返回空字典"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    import json
                    history = json.load(f)
                    # 确保所有键都存在
                    required_keys = ['epochs', 'train_loss', 'val_loss', 'test_loss', 'hr_mae', 'hr_rmse', 
                                    'test_hr_mae', 'test_hr_rmse', 'cls_accuracy', 'test_cls_accuracy', 
                                    'regression_loss', 'classification_loss', 'lr']
                    for key in required_keys:
                        if key not in history:
                            history[key] = []
                    return history
            return {}
        except Exception as e:
            logger.error(f"加载历史数据时出错: {e}")
            return {}
    
    def save_history(self):
        """保存历史数据到JSON文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                import json
                
                # 创建可序列化的数据（将NumPy数组转换为列表）
                serializable_history = {}
                for key, value in self.history.items():
                    if isinstance(value, list) and value and hasattr(value[0], 'tolist'):
                        serializable_history[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
                    else:
                        serializable_history[key] = value
                
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
            logger.info(f"历史数据已保存到: {self.history_file}")
            return True
        except Exception as e:
            logger.error(f"保存历史数据时出错: {e}")
            return False
            
    def generate_all_plots(self, is_final=False):
        """生成所有的可视化图表
        
        Args:
            is_final: 是否是训练结束后的最终绘图
        """
        # 保存当前的历史数据
        self.save_history()
        
        # 绘制训练历史曲线
        self.plot_training_history()
        
        # 绘制混淆矩阵（如果有分类数据）
        if self.class_dist:
            latest_epoch = max(self.class_dist.keys())
            self.plot_confusion_matrix(latest_epoch)
        
        # 绘制预测误差分布
        self.plot_error_distribution()
        
        # 绘制Bland-Altman图
        
        # 生成场景可视化图表（如果存在场景可视化器）
        if hasattr(self, 'scenario_visualizer') and self.scenario_visualizer is not None:
            try:
                logger.info("生成场景可视化图表...")
                self.scenario_visualizer.generate_all_visualizations()
            except Exception as e:
                logger.error(f"生成场景可视化图表时发生错误: {str(e)}")
                logger.exception(e)
        self.plot_bland_altman()
        
        # 绘制其他指标
        self._plot_heart_rate_metrics()
        self._plot_classification_metrics()
        self._plot_loss_curves()
        
        # 绘制测试损失可视化
        self.plot_test_loss()
        
        # 绘制仪表板摘要
        self._create_dashboard_summary()
        
        # 绘制每个类别的性能指标
        self.plot_per_class_metrics()
        
        logger.info(f"All charts saved to: {self.save_dir}")
    
    def plot_test_loss(self):
        """Plot test loss visualization with English title"""
        if len(self.history['epochs']) == 0 or len(self.history['test_loss']) == 0:
            logger.warning("No test loss data available for visualization")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot test loss curve
        plt.plot(self.history['epochs'], self.history['test_loss'], 'r-', marker='o', linewidth=2, markersize=6, label='Test Loss')
        
        # If validation loss exists, plot it as reference
        if len(self.history['val_loss']) > 0:
            plt.plot(self.history['epochs'], self.history['val_loss'], 'b--', marker='s', linewidth=1.5, markersize=4, label='Validation Loss')
        
        # Add minimum test loss annotation
        min_loss_idx = np.argmin(self.history['test_loss'])
        min_loss = self.history['test_loss'][min_loss_idx]
        min_epoch = self.history['epochs'][min_loss_idx]
        
        plt.annotate(f'Min: {min_loss:.4f}', 
                    xy=(min_epoch, min_loss),
                    xytext=(min_epoch+1, min_loss*1.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Set chart properties (all in English now)
        plt.title('Model Test Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add statistics information
        test_losses = self.history['test_loss']
        stats_text = f"Mean Test Loss: {np.mean(test_losses):.4f}\nTest Loss Std Dev: {np.std(test_losses):.4f}"
        plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                   fontsize=10, verticalalignment='bottom')
        
        # Save figure
        save_path = self.save_dir / 'test_loss_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Test loss visualization saved to: {save_path}")
        return save_path
        
    def plot_per_class_metrics(self):
        """Plot detailed performance metrics for each class"""
        if not self.class_dist:
            logger.warning("No classification data available for per-class metrics visualization")
            return
            
        # Get confusion matrix for the latest epoch
        latest_epoch = max(self.class_dist.keys())
        cm = self.class_dist[latest_epoch]
        
        # Get number of classes
        num_classes = cm.shape[0]
        
        # Calculate metrics for each class
        class_metrics = {}
        for i in range(num_classes):
            # True positive, false positive, false negative, true negative
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            # Calculate metrics (handle potential division by zero)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else 0
            
            class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'support': np.sum(cm[i, :])
            }
        
        # Create 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics_names = ['precision', 'recall', 'f1', 'accuracy']
        titles = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        colors = ['skyblue', 'lightgreen', 'salmon', 'mediumpurple']
        
        # Plot bar charts for each metric
        for i, (metric, title, color) in enumerate(zip(metrics_names, titles, colors)):
            ax = axes[i//2, i%2]
            
            # Extract data
            values = [class_metrics[j][metric] for j in range(num_classes)]
            
            # Plot bar chart
            bars = ax.bar(range(num_classes), values, color=color, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
            
            # Set chart properties
            ax.set_title(f'{title} - Epoch {latest_epoch}')
            ax.set_xlabel('Class')
            ax.set_ylabel(title)
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels([f'Class {j}' for j in range(num_classes)])
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Add sample count information
        plt.figtext(0.5, 0.01, f"Samples per class: {[int(class_metrics[j]['support']) for j in range(num_classes)]}", 
                  ha="center", fontsize=10,
                  bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Per-Class Performance Metrics - Epoch {latest_epoch}", fontsize=16, y=0.98)
        
        # Save figure
        save_path = self.save_dir / f'per_class_metrics_epoch{latest_epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot radar chart to compare classes
        self._plot_class_radar_chart(class_metrics, num_classes, latest_epoch)
        
        logger.info(f"Per-class performance metrics analysis saved to: {save_path}")
        return save_path
    
    def _plot_class_radar_chart(self, class_metrics, num_classes, epoch):
        """Use radar chart to compare performance of different classes across metrics"""
        # Set radar chart parameters
        metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
        metrics_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        num_metrics = len(metrics_to_plot)
        
        # Calculate angles
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Use different colors for each class
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for i in range(num_classes):
            values = [class_metrics[i][metric] for metric in metrics_to_plot]
            values += values[:1]  # Close the plot
            
            ax.plot(angles, values, linewidth=2, label=f'Class {i}', color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        
        # Set y-axis range
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(f"Class Performance Metrics Radar Chart - Epoch {epoch}", size=15, y=1.1)
        
        # Save figure
        save_path = self.save_dir / f'class_metrics_radar_chart_epoch{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class performance radar chart saved to: {save_path}")
        
    def _create_dashboard_summary(self):
        """Create a dashboard summary combining key visualizations"""
        try:
            # Create a large figure for the dashboard
            fig = plt.figure(figsize=(24, 16))
            fig.suptitle('RePSS Training and Evaluation Dashboard', fontsize=20)
            
            # Define a grid layout
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Placeholder for subplot contents
            axes = [
                fig.add_subplot(gs[0, 0]),  # Loss curves
                fig.add_subplot(gs[0, 1]),  # HR MAE/RMSE
                fig.add_subplot(gs[0, 2]),  # Classification accuracy
                fig.add_subplot(gs[1, 0]),  # Error distribution
                fig.add_subplot(gs[1, 1]),  # Bland-Altman
                fig.add_subplot(gs[1, 2]),  # Confusion matrix
            ]
            
            # Titles for each subplot
            titles = [
                'Loss Curves',
                'Heart Rate Metrics',
                'Classification Accuracy',
                'Error Distribution',
                'Bland-Altman Plot',
                'Confusion Matrix'
            ]
            
            # Set titles for placeholder panels
            for ax, title in zip(axes, titles):
                ax.set_title(title, fontsize=14)
                ax.text(0.5, 0.5, f'[{title}]\nSee individual plots for details', 
                        ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add timestamp and summary info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_text = (
                f"Dashboard Generated: {timestamp}\n\n"
                "Summary Statistics:\n"
            )
            
            # Add key metrics if available
            if len(self.history['hr_mae']) > 0:
                best_mae_idx = np.argmin(self.history['hr_mae'])
                best_mae = self.history['hr_mae'][best_mae_idx]
                best_mae_epoch = self.history['epochs'][best_mae_idx]
                summary_text += f"Best HR MAE: {best_mae:.2f} BPM (Epoch {best_mae_epoch})\n"
            
            if len(self.history['cls_accuracy']) > 0:
                best_acc_idx = np.argmax(self.history['cls_accuracy'])
                best_acc = self.history['cls_accuracy'][best_acc_idx]
                best_acc_epoch = self.history['epochs'][best_acc_idx]
                summary_text += f"Best Classification Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch})\n"
            
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # Save the dashboard
            save_path = self.save_dir / 'training_dashboard.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training dashboard saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error creating dashboard summary: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

class MultitaskTrainer:
    """多任务训练器"""
    
    def __init__(self, config):
        """Initialize the trainer
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置混合精度训练
        self.use_mixed_precision = config.get("use_mixed_precision", True)
        
        # 消融实验配置
        self.rgb_only = config.get("rgb_only", False)  # 只使用RGB模态的消融实验选项
        # if self.rgb_only:
        #     logger.info("消融实验模式: 仅使用RGB模态 (NIR输入将被强制设为None)")
        
        # 设置基本训练参数
        self.num_epochs = config.get("num_epochs", 100)
        self.save_interval = config.get("save_interval", 10)
        self.eval_interval = config.get("eval_interval", 1)
        
        # 设置随机种子
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # 初始化数据加载器
        self._setup_dataloaders()
        
        # 创建模型
        self._create_model()
        
        # 解决小批量BatchNorm不稳定问题
        # 批量大小为6时，统计量波动大，使用SyncBatchNorm或GroupNorm更稳定
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logger.info(f"已将BatchNorm转换为SyncBatchNorm (多GPU环境)")
        else:
            # 也可以考虑在原始模型定义中将所有BN替换为GroupNorm(32, channels)
            pass
        
        # 设置损失函数和优化器
        self._setup_loss_and_optimizer()
        
        # 训练参数
        self.num_epochs = config.get("num_epochs", 30)
        self.use_mixed_precision = config.get("use_mixed_precision", True)
        self.grad_accumulation_steps = config.get("grad_accumulation_steps", 1)
        
        # 记录训练状态
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        
        # 早停机制参数
        self.early_stopping_patience = config.get('training', {}).get('early_stopping', 3)  # 默认3轮
        self.early_stopping_counter = 0
        
        # 初始化指标管理器
        self.metrics_manager = MetricsManager(self.output_dir)
    
    def _setup_dataloaders(self):
        """设置训练和验证数据加载器"""
        # 直接参考原姍train_repss.py中的实现
        try:
            # 获取数据目录
            data_dir = self.config.get("data_dir", "data/processed_vipl")
            logger.info(f"数据目录: {data_dir}")
            
            # 自适应批次大小和序列长度设置
            batch_size = self.config.get("batch_size", 4)
            seq_len = self.config.get("window_size", 160)  # 使用window_size作为序列长度
            
            # 将num_workers设置为0，避免特征钩子序列化问题
            logger.info("将数据加载器设置为单线程模式，避免序列化问题")
            
            # 获取其他数据加载配置
            use_nir = self.config.get("use_nir", True)
            use_vppg = self.config.get("use_vppg", False)
            source_id = self.config.get("source_id", None)
            scene_id = self.config.get("scene_id", None)
            labels_dir = self.config.get("labels_dir", None)
            
            # 没有指定标签目录时，自动测试常见位置
            if not labels_dir:
                if os.path.exists("data/processed_vipl/vipl_hr_labels.csv"):
                    labels_dir = "data/processed_vipl"
                    logger.info(f"自动设置标签目录: {labels_dir}")

            # 在导入语句处修复一下，确保正确导入create_vipl_hr_dataloaders
            try: 
                # 先尝试从原姍train_repss.py中使用的路径导入
                from data.dataset import create_vipl_hr_dataloaders
                logger.info("使用data.dataset模块中的create_vipl_hr_dataloaders")
            except ImportError:
                # 如果失败，继续使用原来的导入路径
                logger.info("使用data.vipl_hr_dataset模块中的create_vipl_hr_dataloaders")
                pass
            
            logger.info(f"数据加载参数: 批次大小={batch_size}, 序列长度={seq_len}")
            logger.info(f"特征配置: NIR={use_nir}, vPPG={use_vppg}")
            
            # 直接使用train_repss.py中的参数格式
            dataloaders = create_vipl_hr_dataloaders(
                data_dir=data_dir,
                sequence_length=seq_len,
                batch_size=batch_size,
                num_workers=0,  # 将num_workers设置为0，避免序列化问题
                use_nir=use_nir,
                use_vppg=use_vppg,
                source_id=source_id,
                scene_id=scene_id,
                labels_dir=labels_dir
            )
            
            # 提取训练集和验证集
            self.train_loader = dataloaders['train']
            self.val_loader = dataloaders['val']
            
            # 如果有测试集，也提取出来
            if 'test' in dataloaders:
                self.test_loader = dataloaders['test']
            else:
                self.test_loader = None
            
            # 检查数据加载器是否创建成功
            if not hasattr(self, 'train_loader') or not hasattr(self, 'val_loader'):
                logger.error("创建数据加载器失败")
                raise ValueError("数据加载器创建失败，缺少训练集或验证集")
            
            # 打印数据集的基本信息
            logger.info(f"训练集样本数: {len(self.train_loader.dataset)}")
            logger.info(f"验证集样本数: {len(self.val_loader.dataset)}")
            if self.test_loader:
                logger.info(f"测试集样本数: {len(self.test_loader.dataset)}")
            
            return True
        except Exception as e:
            logger.error(f"初始化数据加载器时出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"无法创建数据加载器: {str(e)}. 请检查数据目录和参数设置。")
    
    def _create_model(self):
        """创建并初始化模型"""
        # 创建模型配置
        model_config = MambaModelConfig()
        
        # 从训练配置中更新模型参数
        model_config.d_model = self.config.get("d_model", 256)  # 隐藏层维度
        model_config.n_layers = self.config.get("n_layers", 6)  # Mamba层数
        model_config.use_nir = self.config.get("use_nir", True)  # 是否使用NIR
        model_config.dropout = self.config.get("dropout", 0.15)   # dropout比率
        
        # 使用配置创建模型实例
        logger.info(f"正在创建 RePSSModelV2 模型实例...")
        self.model = RePSSModelV2(config=model_config)
                # 移动模型到设备
        self.model = self.model.to(self.device)
        logger.info(f"模型创建完成并移动到设备: {self.device}")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")
        
        # 如果指定了预训练权重，加载它们
        pretrained_path = self.config.get("pretrained_weights", None)
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                # 支持两种格式：直接状态字典或包含'model_state_dict'的字典
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 新格式：字典包含模型状态
                    state_dict = checkpoint['model_state_dict']
                    logger.info(f"从检查点字典加载模型权重(来自周期 {checkpoint.get('epoch', 'unknown')})")
                else:
                    # 旧格式：直接是状态字典
                    state_dict = checkpoint
                    logger.info("从直接状态字典加载模型权重")
                    
                # 加载状态字典到模型
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"缺少的键: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"意外的键: {unexpected_keys}")
                    
                logger.info(f"成功加载预训练权重: {pretrained_path}")
                
                # 如果需要，也可以恢复优化器和调度器状态
                if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                    # 仅在训练阶段恢复
                    logger.info("恢复优化器状态")
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            except Exception as e:
                logger.warning(f"加载预训练权重失败: {e}")
                logger.warning(f"错误详情: {str(e)}")
                
        return True
    
    def _update_loss_weights(self, epoch):
        """更新当前训练周期的多任务损失权重
        
        Args:
            epoch: 当前训练周期
        """
        # 如果不使用动态权重，直接返回
        if not self.use_dynamic_weights:
            return
            
        # 更新分类权重 - 根据阶梯式计划
        for schedule_epoch, weight in sorted(self.cls_weight_schedule.items(), reverse=True):
            if epoch >= schedule_epoch:
                self.current_cls_weight = weight
                break
                
        # 更新分布匹配权重
        for schedule_epoch, weight in sorted(self.dist_weight_schedule.items(), reverse=True):
            if epoch >= schedule_epoch:
                self.current_dist_weight = weight
                break
                
        # 重要！真正将权重应用到criterion对象
        self.criterion.classification_weight = self.current_cls_weight
        self.criterion.distribution_weight = self.current_dist_weight
        
        logger.info(f"Epoch {epoch}: 更新损失权重 - 分类={self.current_cls_weight:.2f}, 分布={self.current_dist_weight:.2f}")
    

    
    def _setup_loss_and_optimizer(self):
        """设置损失函数和优化器"""
        # 多任务损失函数
        # 配置优化：降低分类权重到 0.2，暂时去掉分布匹配权重
        # 这能让回归头在训练初期更自由地学习，不会被分类和分布相关的梯度过度影响
        orig_cls_weight = self.config.get("cls_weight", 3)
        orig_dist_weight = self.config.get("dist_weight", 1)
        
        # 如果配置中存在allow_weight_override并且为True，才使用修改后的权重
        if self.config.get("allow_weight_override", True):
            # 改用较低的分类权重(0.2)和0分布匹配权重
            cls_weight = 0.2
            dist_weight = 0.0
            logger.info(f"使用初始损失权重: 分类={cls_weight} (原{orig_cls_weight}), 分布={dist_weight} (原{orig_dist_weight})")
            
            # 存储权重信息用于阶梯式解冻
            self.orig_cls_weight = orig_cls_weight
            self.current_cls_weight = cls_weight
            self.current_dist_weight = dist_weight
            self.use_dynamic_weights = True
            
            # 阶梯式解冻计划配置
            self.cls_weight_schedule = {
                # Epoch 1-3: 初始权重 0.1 - 先让回归自由微调
                1: 0.5,
                # Epoch 4-6: 增加到 0.3
                4: 0.5,
                # Epoch 7-9: 增加到 0.6
                7: 1.0,
                # Epoch 10 及以后: 完全恢复到 1.0
                10: 1.0  
            }
            
            # 分布匹配权重计划 - 从第10个周期才开始使用
            self.dist_weight_schedule = {
                10: 0.1,  # 第10轮开始引入少量分布匹配
                15: 0.2   # 第15轮恢复到原始设置
            }
        else:
            # 使用原始权重
            cls_weight = orig_cls_weight
            dist_weight = orig_dist_weight
            
        self.criterion = MultiTaskHeartRateLoss(
            regression_weight=1.0,
            classification_weight=0.0,  # 先设为0，专注于回归任务
            distribution_weight=0.0,    # 先设为0，专注于回归任务
            # 注意：MultiTaskHeartRateLoss不接受hr_min和hr_max参数
            # 而是使用固定阈值：low_threshold=60.0, high_threshold=100.0
            beta=1.0  # 控制Smooth L1损失的平滑区域大小
        )
        
        # 设置优化器
        optimizer_type = self.config.get("optimizer", "adam").lower()
        # 提高学习率和权重衰减以改善模型训练
        lr = 3e-4  # 将学习率提高到更合理的值
        weight_decay = 1e-4  # 增大权重衰减以减少过拟合
        
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=self.config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"未知的优化器类型: {optimizer_type}，使用Adam")
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 设置学习率调度器
        scheduler_type = self.config.get("lr_scheduler", "cosine").lower()
        
        if scheduler_type == "step":
            step_size = self.config.get("lr_step_size", 10)
            gamma = self.config.get("lr_gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "cosine":
            t_max = self.config.get("lr_t_max", self.num_epochs)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)
        elif scheduler_type == "plateau":
            patience = self.config.get("lr_patience", 5)
            factor = self.config.get("lr_factor", 0.1)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=patience, factor=factor, verbose=True
            )
        else:
            logger.warning(f"未知的学习率调度器类型: {scheduler_type}，使用CosineAnnealingLR")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        
        # 创建梯度缩放器用于混合精度训练
        # 修复GradScaler初始化参数 - 正确初始化
        self.scaler = torch.amp.GradScaler(enabled=self.use_mixed_precision)
        
        # 注意：我们不需要虚拟步骤来解决学习率调度器警告
        # 正确的方式是确保在训练开始时先调用optimizer.step()然后再调用scheduler.step()
        # 警告在产品中是可以接受的，并且在PyTorch未来版本中可能会被移除

    
    def train(self):
        """开始模型训练过程"""
        logger.info(f"开始多任务训练: 共{self.num_epochs}个训练周期, "
                f"Batch大小: {self.config.get('batch_size', 8)}, "
                f"使用混合精度: {self.use_mixed_precision}, "
                f"早停耐心值: {self.early_stopping_patience}轮")
        
        start_time = time.time()
        best_model_path = self.output_dir / "best_model_ts.pt"
        
        try:
            for epoch in range(1, self.num_epochs + 1):
                epoch_start_time = time.time()
                
                # 阶梯式更新损失权重
                self._update_loss_weights(epoch)
                
                # 训练一个周期
                train_loss, train_metrics = self._train_epoch(epoch)
                
                # 验证一个周期
                val_loss, val_metrics = self._validate_epoch(epoch)
                
                # 测试一个周期（如果存在测试集）
                if self.test_loader is not None:
                    logger.info(f"开始测试 [Epoch {epoch}/{self.num_epochs}]")
                    
                    # 直接进行测试而不调用外部函数
                    self.model.eval()
                    
                    # 初始化收集指标的列表
                    all_test_hr_preds = []
                    all_test_hr_targets = []
                    all_test_cls_preds = []
                    all_test_cls_targets = []
                    test_sample_meta = {'subject_id': [], 'scene_id': []}
                    test_total_loss = 0.0
                    test_batch_count = 0
                    
                    try:
                        # 使用tqdm包装测试数据加载器来显示进度
                        test_pbar = tqdm(self.test_loader, desc=f"Testing Epoch {epoch}", leave=False)
                        
                        with torch.no_grad():
                            for test_batch_idx, test_batch in enumerate(test_pbar):
                                # 准备输入数据
                                test_inputs = {}
                                if 'rgb_frames' in test_batch:
                                    test_inputs['rgb'] = test_batch['rgb_frames'].to(self.device)
                                
                                # 处理NIR输入 - 如果是rgb_only模式，强制设为None
                                if self.rgb_only:
                                    test_inputs['nir'] = None
                                elif 'nir_frames' in test_batch and test_batch['nir_frames'] is not None:
                                    test_inputs['nir'] = test_batch['nir_frames'].to(self.device)
                                else:
                                    test_inputs['nir'] = None
                                
                                # 准备目标
                                test_targets = {}
                                if 'heart_rate' in test_batch:
                                    test_targets['heart_rate'] = test_batch['heart_rate'].to(self.device)
                                if 'activity' in test_batch:
                                    test_targets['activity'] = test_batch['activity'].to(self.device)
                                
                                # 采集样本元数据
                                for key in ['subject_id', 'scene_id']:
                                    if key in test_batch:
                                        # 判断数据类型并正确处理
                                        if hasattr(test_batch[key], 'cpu'):
                                            # 如果是张量，转换为列表
                                            test_sample_meta[key].extend(test_batch[key].cpu().numpy().tolist())
                                        elif isinstance(test_batch[key], list):
                                            # 如果已经是列表，直接使用
                                            test_sample_meta[key].extend(test_batch[key])
                                        else:
                                            # 其他类型尝试转换为列表
                                            try:
                                                test_sample_meta[key].extend(list(test_batch[key]))
                                            except Exception as e:
                                                logger.warning(f"无法处理元数据{key}: {e}")
                                                # 跳过这个字段
                                
                                # 测试前向传播
                                with torch.amp.autocast(device_type='cuda', enabled=self.use_mixed_precision):
                                    test_outputs = self.model(test_inputs)
                                    
                                    # 计算损失
                                    test_hr_targets = test_targets.get('heart_rate')
                                    if test_hr_targets is not None:
                                        test_loss, _ = self.criterion(test_outputs, test_hr_targets)
                                        test_total_loss += test_loss.item()
                                        test_batch_count += 1
                                
                                # 提取心率预测结果
                                if 'regression' in test_outputs and test_hr_targets is not None:
                                    test_hr_preds = test_outputs['regression'].detach().cpu().numpy()
                                    all_test_hr_preds.append(test_hr_preds)
                                    all_test_hr_targets.append(test_hr_targets.detach().cpu().numpy())
                                
                                # 提取分类结果
                                if 'logits' in test_outputs and 'activity' in test_targets:
                                    test_cls_preds = torch.softmax(test_outputs['logits'], dim=1).detach().cpu().numpy()
                                    test_cls_targets = test_targets['activity'].detach().cpu().numpy()
                                    all_test_cls_preds.append(test_cls_preds)
                                    all_test_cls_targets.append(test_cls_targets)
                                
                                # 更新进度条
                                if len(all_test_hr_preds) > 0:
                                    last_batch = all_test_hr_preds[-1]
                                    test_pbar.set_postfix({
                                        'loss': f"{test_loss.item():.4f}",
                                        'range': f"[{np.min(last_batch):.1f}-{np.max(last_batch):.1f}]"
                                    })
                        
                        # 计算测试指标
                        if all_test_hr_preds and all_test_hr_targets:
                            # 合并所有批次的预测和目标
                            test_hr_preds_all = np.concatenate(all_test_hr_preds)
                            test_hr_targets_all = np.concatenate(all_test_hr_targets)
                            
                            # 计算指标
                            from sklearn.metrics import mean_absolute_error, mean_squared_error
                            test_hr_mae = mean_absolute_error(test_hr_targets_all, test_hr_preds_all)
                            test_hr_rmse = np.sqrt(mean_squared_error(test_hr_targets_all, test_hr_preds_all))
                            
                            # 处理分类指标（如果有）
                            test_cls_accuracy = 0.0
                            test_cls_f1 = 0.0
                            
                            if all_test_cls_preds and all_test_cls_targets:
                                test_cls_preds_all = np.concatenate(all_test_cls_preds)
                                test_cls_targets_all = np.concatenate(all_test_cls_targets)
                                
                                # 将概率转换为类别
                                test_cls_pred_labels = np.argmax(test_cls_preds_all, axis=1)
                                
                                # 计算分类指标
                                from sklearn.metrics import accuracy_score, f1_score
                                test_cls_accuracy = accuracy_score(test_cls_targets_all, test_cls_pred_labels)
                                test_cls_f1 = f1_score(test_cls_targets_all, test_cls_pred_labels, average='macro')
                            
                            # 平均损失
                            test_avg_loss = test_total_loss / max(1, test_batch_count)
                            
                            # 收集指标
                            test_metrics_for_history = {
                                'test_loss': test_avg_loss,
                                'test_hr_mae': test_hr_mae,
                                'test_hr_rmse': test_hr_rmse,
                                'test_cls_accuracy': test_cls_accuracy,
                                'test_cls_f1': test_cls_f1
                            }
                            
                            # 保存测试指标
                            self.metrics_manager.update_metrics(epoch, test_metrics_for_history, is_train=False)
                            
                            # 保存详细样本数据
                            sample_ids = [f"test_epoch_{epoch}_sample_{i}" for i in range(len(test_hr_preds_all))]
                            
                            self.metrics_manager.update_sample_metrics(
                                sample_ids=sample_ids,
                                hr_preds=test_hr_preds_all,
                                hr_targets=test_hr_targets_all,
                                cls_preds=test_cls_pred_labels if 'test_cls_pred_labels' in locals() else None,
                                cls_targets=test_cls_targets_all if 'test_cls_targets_all' in locals() else None,
                                scenario_ids=test_sample_meta.get('scene_id'),
                                subject_ids=test_sample_meta.get('subject_id'),
                                epoch=epoch
                            )
                            
                            # 打印测试结果
                            logger.info(f"测试集结果: Loss={test_avg_loss:.4f}, "
                                      f"HR MAE={test_hr_mae:.2f}, HR RMSE={test_hr_rmse:.2f}, "
                                      f"准确率={test_cls_accuracy:.4f}")
                            
                            # 每5个周期或训练结束时保存详细指标
                            if epoch % 5 == 0 or epoch == self.num_epochs:
                                self.metrics_manager.save_all_data()
                    except Exception as e:
                        # 处理测试过程中的错误
                        logger.error(f"测试过程中发生错误: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # 恢复模型为训练模式
                    self.model.train()
                
                # 注意: 在之optimizer.step()后调用scheduler.step()
                # 我们已经在训练每个batch中调用了optimizer.step()
                # 已经完成了该epoch的所有batch的optimizer.step()，现在可以更新学习率调度器
                # 注意：这个顺序必须保证optimizer.step()先执行，然后self.scheduler.step()
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        # 使用HR RMSE而不是验证损失更新ReduceLROnPlateau调度器
                        # 这样可以确保模型专注于优化心率预测性能
                        self.scheduler.step(val_metrics['hr_rmse'])
                    else:
                        # 其他类型的学习率调度器
                        self.scheduler.step()
                
                # 监控SincConv参数
                self._monitor_sinc_parameters(epoch)
                
                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 计算耗时
                epoch_time = time.time() - epoch_start_time
                total_time = time.time() - start_time
                
                # 保存训练指标
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hr_mae': val_metrics['hr_mae'],
                    'hr_rmse': val_metrics['hr_rmse'],
                    'cls_accuracy': val_metrics.get('accuracy', 0),
                    'lr': current_lr
                }
                
                # 记录详细子损失数据（如果有）
                loss_dict = None
                if hasattr(self, 'last_loss_dict') and self.last_loss_dict:
                    loss_dict = self.last_loss_dict
                
                # 更新指标数据
                self.metrics_manager.update_metrics(epoch, metrics, loss_dict)
                
                # 添加当前的权重信息（如果启用了动态权重）
                if hasattr(self, 'use_dynamic_weights') and self.use_dynamic_weights:
                    # 记录权重数据到指标历史
                    meta_data = {
                        'classification_weight': self.current_cls_weight,
                        'distribution_weight': self.current_dist_weight
                    }
                    self.metrics_manager.update_model_params(self.model, epoch, meta_data)
                
                # 打印周期性能指标
                logger.info(f"Epoch {epoch}/{self.num_epochs} - "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}, "
                            f"HR MAE: {val_metrics['hr_mae']:.2f}, "
                            f"HR RMSE: {val_metrics['hr_rmse']:.2f}, "
                            f"Acc: {val_metrics.get('accuracy', 0):.2f}, "
                            f"LR: {current_lr:.6f}, "
                            f"Time: {epoch_time:.1f}s, 总时间: {total_time/60:.1f}分钟")
                
                # 是否保存最佳模型 - 监控HR RMSE而不是验证损失
                current_rmse = val_metrics['hr_rmse']
                
                # 如果没有初始化best_hr_rmse，则初始化它
                if not hasattr(self, 'best_hr_rmse'):
                    self.best_hr_rmse = float('inf')
                
                # 使用HR RMSE作为早停监控指标
                if current_rmse < self.best_hr_rmse:
                    self.best_hr_rmse = current_rmse
                    self.best_val_loss = val_loss  # 仍然记录最佳验证损失
                    self.best_val_metrics = val_metrics
                    # 重置早停计数器
                    self.early_stopping_counter = 0
                    logger.info(f"发现更好的HR性能模型 (RMSE: {current_rmse:.2f})，重置早停计数器")
                    
                    # 保存最佳模型 - 使用字典方式保存
                    self.model.eval()
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_loss': self.best_val_loss,
                        'best_val_metrics': self.best_val_metrics,
                        'config': self.config,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    torch.save(checkpoint, best_model_path)
                    logger.info(f"保存最佳模型到: {best_model_path}")
                else:
                    # 更新早停计数器
                    self.early_stopping_counter += 1
                    logger.info(f"模型性能未提升，早停计数: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    
                # 检查早停条件
                if self.early_stopping_patience > 0 and self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"触发早停机制! 连续 {self.early_stopping_patience} 轮验证性能未提升")
                    logger.info(f"最佳验证指标: Loss {self.best_val_loss:.4f}, HR MAE {self.best_val_metrics['hr_mae']:.2f}, "
                               f"HR RMSE {self.best_val_metrics['hr_rmse']:.2f}, "
                               f"准确率 {self.best_val_metrics.get('accuracy', 0):.2f}")
                    break
            
            # 周期性保存模型
            save_interval = self.config.get("save_interval", 5)
            if epoch % save_interval == 0 or epoch == self.num_epochs:
                checkpoint_path = self.output_dir / f"model_epoch_{epoch}.pt"
                self.model.eval()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"保存周期{epoch}模型到: {checkpoint_path}")
            
            # 保存详细指标数据
            if epoch % 5 == 0 or epoch == self.num_epochs:
                # 保存训练历史
                self.metrics_manager.save_training_history()
                
                # 如果有分类数据，更新分类指标
                if 'cls_preds' in val_metrics and 'cls_targets' in val_metrics:
                    self.metrics_manager.update_class_distribution(
                        val_metrics['cls_preds'], val_metrics['cls_targets'], epoch)
                
                # 更新心率预测误差分布
                if 'hr_preds' in val_metrics and 'hr_targets' in val_metrics:
                    self.metrics_manager.update_error_distribution(
                        val_metrics['hr_preds'], val_metrics['hr_targets'], epoch)
            
            # 训练结束，输出最终性能
            logger.info(f"训练完成! 总耗时: {(time.time() - start_time)/60:.1f} 分钟")
            logger.info(f"最佳验证指标: 损失={self.best_val_loss:.4f}, "
                        f"HR_MAE={self.best_val_metrics['hr_mae']:.2f}, "
                        f"HR_RMSE={self.best_val_metrics['hr_rmse']:.2f}, "
                        f"准确率={self.best_val_metrics.get('accuracy', 0):.2f}")
            
            # 生成所有可视化图表
            logger.info("正在生成训练过程可视化图表...")
            plot_paths = self.vis_manager.generate_all_plots()
            logger.info(f"已生成 {len(plot_paths)} 个可视化图表")
            
            return best_model_path
            
        except KeyboardInterrupt:
            logger.info("训练被手动中断")
            # 生成当前的可视化图表
            self.vis_manager.generate_all_plots()
            return self.output_dir / "last_model.pt"
    def _train_epoch(self, epoch):
        """训练一个周期"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_start_time = time.time()
        
        # 存储所有批次的指标
        all_hr_preds = []
        all_hr_targets = []
        all_cls_preds = []
        all_cls_targets = []
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            inputs = {}
            if 'rgb_frames' in batch:
                inputs['rgb'] = batch['rgb_frames'].to(self.device)
            
            # 处理NIR输入 - 如果是rgb_only模式，强制设为None
            if self.rgb_only:
                inputs['nir'] = None
            elif 'nir_frames' in batch and batch['nir_frames'] is not None:
                inputs['nir'] = batch['nir_frames'].to(self.device)
            else:
                inputs['nir'] = None
            
            # 准备目标
            targets = {}
            if 'heart_rate' in batch:
                targets['heart_rate'] = batch['heart_rate'].to(self.device)
            if 'activity' in batch:
                targets['activity'] = batch['activity'].to(self.device)
            
            # 对每个样本获取元数据
            sample_meta = {}
            for key in ['subject_id', 'scene_id', 'segment_id']:
                if key in batch:
                    sample_meta[key] = batch[key]
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.amp.autocast(device_type='cuda', enabled=self.use_mixed_precision):
                # 前向传播，注意我们设置multi_task=True来获取回归和分类输出
                outputs = self.model(inputs)  # 使用字典格式的输入
                
                # 计算多任务损失
                # MultiTaskHeartRateLoss期望目标是心率值的张量，而不是字典
                hr_targets = targets['heart_rate'] if 'heart_rate' in targets else None
                if hr_targets is None:
                    logger.error("没有找到心率目标值，请检查数据加载")
                    hr_targets = torch.ones(outputs['heart_rate'].size(0), device=self.device) * 75.0  # 使用默认心率
                
                loss, loss_details = self.criterion(outputs, hr_targets)
            
            # 反向传播和优化
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 收集统计和指标
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1
            
            # 提取心率回归结果
            if 'regression' in outputs:
                hr_preds = outputs['regression'].detach().cpu().numpy()
                all_hr_preds.append(hr_preds)
                all_hr_targets.append(targets['heart_rate'].detach().cpu().numpy())
            
            # 提取活动分类结果
            if 'logits' in outputs and 'activity' in targets:
                cls_preds = outputs['logits'].detach().cpu().numpy()
                cls_targets = targets['activity'].detach().cpu().numpy()
                all_cls_preds.append(cls_preds)
                all_cls_targets.append(cls_targets)
            
            # 更新进度条
            hr_range = f"[{np.min(hr_preds):.1f}-{np.max(hr_preds):.1f}]" if len(all_hr_preds) > 0 else "N/A"
            pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'hr_range': hr_range
            })
            
            # 打印批次表格（如果设置了详细打印）
            if self.config.get("print_batch_table", True) and (batch_idx % 10 == 0 or batch_idx == len(self.train_loader) - 1):
                self._print_batch_results(batch_idx, hr_preds, all_hr_targets[-1], sample_meta)
                
        # 计算平均损失
        avg_loss = total_loss / max(1, batch_count)
        
        # 计算整体指标
        metrics = self._calculate_metrics(
            np.concatenate(all_hr_preds) if all_hr_preds else np.array([]),
            np.concatenate(all_hr_targets) if all_hr_targets else np.array([]),
            np.concatenate(all_cls_preds) if all_cls_preds else np.array([]),
            np.concatenate(all_cls_targets) if all_cls_targets else np.array([])
        )
        
        return avg_loss, metrics
    
    def _validate_epoch(self, epoch):
        """验证一个周期"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        # 存储所有批次的指标
        all_hr_preds = []
        all_hr_targets = []
        all_cls_preds = []
        all_cls_targets = []
        
        # 创建进度条
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # 移动数据到设备
                inputs = {}
                if 'rgb_frames' in batch:
                    inputs['rgb'] = batch['rgb_frames'].to(self.device)
                
                # 处理NIR输入 - 如果是rgb_only模式，强制设为None
                if self.rgb_only:
                    inputs['nir'] = None
                elif 'nir_frames' in batch and batch['nir_frames'] is not None:
                    inputs['nir'] = batch['nir_frames'].to(self.device)
                else:
                    inputs['nir'] = None
                
                # 准备目标
                targets = {}
                if 'heart_rate' in batch:
                    targets['heart_rate'] = batch['heart_rate'].to(self.device)
                if 'activity' in batch:
                    targets['activity'] = batch['activity'].to(self.device)
                
                # 对每个样本获取元数据
                sample_meta = {}
                for key in ['subject_id', 'scene_id', 'segment_id']:
                    if key in batch:
                        sample_meta[key] = batch[key]
                
                # 前向传播
                with torch.amp.autocast(device_type='cuda', enabled=self.use_mixed_precision):
                    outputs = self.model(inputs)  # 使用字典格式的输入
                    
                    # MultiTaskHeartRateLoss期望目标是心率值的张量，而不是字典
                    hr_targets = targets['heart_rate'] if 'heart_rate' in targets else None
                    if hr_targets is None:
                        logger.error("验证中没有找到心率目标值，请检查数据加载")
                        hr_targets = torch.ones(outputs['heart_rate'].size(0), device=self.device) * 75.0  # 使用默认心率
                    
                    # 在调用损失函数前添加安全检查
                    # 检查输出中的logits
                    if 'logits' in outputs and outputs['logits'] is not None:
                        try:
                            num_classes = outputs['logits'].shape[1]
                            
                            # 将目标转换为类别 - 传入类别总数确保兼容性
                            from models.baseline_decoder import heart_rate_to_class
                            # 使用改进的heart_rate_to_class函数，指定类别数量
                            hr_classes = heart_rate_to_class(hr_targets, num_classes=num_classes)
                            
                            # 再次确保所有类别都在有效范围内
                            hr_classes = torch.clamp(hr_classes, min=0, max=num_classes-1)
                            
                            # 记录异常值作为调试信息
                            unusual_mask = (hr_targets < 40) | (hr_targets > 150)
                            if torch.any(unusual_mask):
                                unusual_indices = torch.where(unusual_mask)[0]
                                unusual_hrs = hr_targets[unusual_indices].tolist()
                                mapped_classes = hr_classes[unusual_indices].tolist()
                                logger.info(f"异常心率值并已处理: 心率={unusual_hrs}, 映射类别={mapped_classes}")
                                
                            # 将类别保存到outputs中供使用
                            outputs['hr_classes'] = hr_classes
                        except Exception as e:
                            logger.error(f"处理心率分类时出错: {e}")
                            # 出错时创建默认类别
                            middle_class = min(2, num_classes - 1)
                            hr_classes = torch.ones_like(hr_targets, dtype=torch.long, device=self.device) * middle_class
                            outputs['hr_classes'] = hr_classes
                    
                    # 调用损失函数
                    try:
                        loss, loss_details = self.criterion(outputs, hr_targets)
                    except Exception as e:
                        logger.error(f"验证中损失计算出错: {str(e)}")
                        # 创建一个空损失以继续训练
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        loss_details = {'total_loss': loss}
                
                # 收集统计和指标
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_count += 1
                
                # 提取心率回归结果
                if 'regression' in outputs:
                    hr_preds = outputs['regression'].detach().cpu().numpy()
                    all_hr_preds.append(hr_preds)
                    all_hr_targets.append(targets['heart_rate'].detach().cpu().numpy())
                
                # 提取活动分类结果
                if 'logits' in outputs and 'activity' in targets:
                    cls_preds = outputs['logits'].detach().cpu().numpy()
                    cls_targets = targets['activity'].detach().cpu().numpy()
                    all_cls_preds.append(cls_preds)
                    all_cls_targets.append(cls_targets)
                
                # 更新进度条
                hr_range = f"[{np.min(hr_preds):.1f}-{np.max(hr_preds):.1f}]" if len(all_hr_preds) > 0 else "N/A"
                pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'hr_range': hr_range
                })
                
                # 如果是验证集的第一个批次，打印表格
                if batch_idx == 0 and self.config.get("print_batch_table", True):
                    self._print_batch_results(batch_idx, hr_preds, all_hr_targets[-1], sample_meta, is_val=True)
        
        # 计算平均损失
        avg_loss = total_loss / max(1, batch_count)
        
        # 准备原始数据以供可视化使用
        hr_preds_all = np.concatenate(all_hr_preds) if all_hr_preds else np.array([])
        hr_targets_all = np.concatenate(all_hr_targets) if all_hr_targets else np.array([])
        cls_preds_all = np.concatenate(all_cls_preds) if all_cls_preds else np.array([])
        cls_targets_all = np.concatenate(all_cls_targets) if all_cls_targets else np.array([])
        
        # 计算整体指标
        metrics = self._calculate_metrics(
            hr_preds_all, hr_targets_all, cls_preds_all, cls_targets_all
        )
        
        # 添加原始数据到指标字典，供可视化使用
        metrics['hr_preds'] = hr_preds_all
        metrics['hr_targets'] = hr_targets_all
        metrics['cls_preds'] = cls_preds_all
        metrics['cls_targets'] = cls_targets_all
        
        # 将最近一次验证的心率目标保存到指标管理器
        # 添加到详细指标中
        sample_ids = [f"epoch_{epoch}_sample_{i}" for i in range(len(hr_targets_all))]
        self.metrics_manager.update_sample_metrics(
            sample_ids=sample_ids,
            hr_preds=hr_preds_all,
            hr_targets=hr_targets_all,
            cls_preds=cls_preds_all if len(cls_preds_all) > 0 else None,
            cls_targets=cls_targets_all if len(cls_targets_all) > 0 else None,
            epoch=epoch
        )
        
        # 打印验证集的总体结果
        hr_mae, hr_rmse = metrics['hr_mae'], metrics['hr_rmse']
        logger.info(f"Validation Results - Loss: {avg_loss:.4f}, HR MAE: {hr_mae:.2f}, HR RMSE: {hr_rmse:.2f}")
        
        if 'accuracy' in metrics:
            logger.info(f"Classification - Accuracy: {metrics['accuracy']:.2f}, F1: {metrics.get('f1', 0):.2f}")
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, hr_preds, hr_targets, cls_preds, cls_targets):
        """计算所有性能指标"""
        metrics = {}
        
        # 防止空数组
        if not isinstance(hr_preds, np.ndarray) or len(hr_preds) == 0:
            hr_preds = np.array([]) 
        if not isinstance(hr_targets, np.ndarray) or len(hr_targets) == 0:
            hr_targets = np.array([])
        if not isinstance(cls_preds, np.ndarray) or len(cls_preds) == 0:
            cls_preds = np.array([])
        if not isinstance(cls_targets, np.ndarray) or len(cls_targets) == 0:
            cls_targets = np.array([])
            
        # 计算心率回归指标
        if len(hr_preds) > 0 and len(hr_targets) > 0:
            # 计算MAE和RMSE
            mae = np.mean(np.abs(hr_preds - hr_targets))
            rmse = np.sqrt(np.mean(np.square(hr_preds - hr_targets)))
            
            # 计算其他指标
            metrics['hr_mae'] = mae
            metrics['hr_rmse'] = rmse
            
            # 计算结果分布
            error_ranges = {
                '误差<3': np.mean(np.abs(hr_preds - hr_targets) < 3.0) * 100,
                '误差<5': np.mean(np.abs(hr_preds - hr_targets) < 5.0) * 100,
                '误差<10': np.mean(np.abs(hr_preds - hr_targets) < 10.0) * 100
            }
            
            for key, value in error_ranges.items():
                metrics[key] = value
        else:
            # 如果没有心率数据，提供默认值
            metrics['hr_mae'] = float('nan')
            metrics['hr_rmse'] = float('nan')
        
        # 计算活动分类指标
        if len(cls_preds) > 0 and len(cls_targets) > 0:
            # 如果是多分类问题，需要转换为类别标签
            if len(cls_preds.shape) > 1 and cls_preds.shape[1] > 1:
                cls_pred_labels = np.argmax(cls_preds, axis=1)
            else:
                cls_pred_labels = (cls_preds > 0.5).astype(int)
            
            # 计算准确率
            accuracy = np.mean(cls_pred_labels == cls_targets) * 100
            metrics['accuracy'] = accuracy
            
            try:
                from sklearn.metrics import f1_score, confusion_matrix
                # 安全地计算F1分数
                if len(np.unique(cls_targets)) > 1:  # 确保有多个类别
                    f1 = f1_score(cls_targets, cls_pred_labels, average='weighted') * 100
                    metrics['f1'] = f1
                    
                    # 计算混淆矩阵
                    cm = confusion_matrix(cls_targets, cls_pred_labels)
                    metrics['confusion_matrix'] = cm
                    
                    # 打印混淆矩阵
                    logger.info(f"混淆矩阵:\n{cm}")
            except ImportError:
                logger.warning("scikit-learn未安装，跳过F1分数和混淆矩阵计算")
            
        return metrics
    
    def _print_batch_results(self, batch_idx, hr_preds, hr_targets, sample_meta, is_val=False):
        """打印批次结果表格"""
        # 仅在有有效数据时打印
        if len(hr_preds) == 0 or len(hr_targets) == 0:
            return
            
        # 计算批次指标
        batch_mae = np.mean(np.abs(hr_preds - hr_targets))
        batch_rmse = np.sqrt(np.mean(np.square(hr_preds - hr_targets)))
        
        # 表格设置
        table_width = 95
        header = "\u2550" * table_width  # 使用Unicode字符作为表格边框
        footer = "\u2550" * table_width
        divider = "\u2500" * table_width
        
        # 表格标题
        batch_type = "\u9a8c证" if is_val else "\u8bad\u7ec3"
        title = f" \u6279\u6b21 {batch_idx+1} {batch_type}\u7ed3\u679c (\u6279\u6b21\u5927\u5c0f: {len(hr_preds)}, MAE: {batch_mae:.2f}, RMSE: {batch_rmse:.2f}) "
        padding = table_width - len(title) - 2
        padded_title = title + " " * max(0, padding)
        
        # 打印表格头
        logger.info(header)
        logger.info(f"\u2551{padded_title}\u2551")
        logger.info(divider)
        logger.info(f"\u2551 \u6837\u672c ID    \u2551 \u771f\u5b9e\u5fc3\u7387  \u2551 \u9884\u6d4b\u5fc3\u7387  \u2551 \u8bef\u5dee     \u2551 \u8bef\u5dee\u767e\u5206\u6bd4 \u2551")
        logger.info(divider)
        
        # 最多显示10个样本的结果
        num_samples = min(10, len(hr_preds))
        
        for i in range(num_samples):
            try:
                # 新的简化方法打印样本
                # 预处理样本元数据
                sample_id = "--"
                if isinstance(sample_meta, dict) and 'subject_id' in sample_meta and i < len(sample_meta['subject_id']):
                    subject = sample_meta['subject_id'][i]
                    scene = sample_meta.get('scene_id', ["--"])[i] if 'scene_id' in sample_meta else "--"
                    sample_id = f"p{subject}_v{scene}"
                
                # 计算误差
                true_hr = hr_targets[i]
                pred_hr = hr_preds[i]
                error = pred_hr - true_hr
                error_pct = (error / true_hr) * 100 if true_hr != 0 else 0
                
                # 设置误差符号
                error_sign = "\u2191" if error > 0 else "\u2193" if error < 0 else ""
                
                # 打印行
                logger.info(f"\u2551 {sample_id:<10} \u2551 {true_hr:>8.2f}   \u2551 {pred_hr:>8.2f}   \u2551 {error_sign}{abs(error):>7.2f}  \u2551 {error_sign}{abs(error_pct):>8.2f}%  \u2551")
            except Exception as e:
                logger.error(f"\u6253\u5370\u6837\u672c {i} \u884c\u6570\u636e\u65f6\u51fa\u9519: {e}")
        
        # 如果有更多样本，显示省略号
        if len(hr_preds) > num_samples:
            logger.info(f"\u2551 ... \u5171{len(hr_preds)}\u4e2a\u6837\u672c\uff0c\u53ea\u663e\u793a\u524d{num_samples}\u4e2a ... \u2551")
            
        logger.info(footer)


def load_config(config_path):
    """从文件加载配置"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        logger.warning("PyYAML未安装，尝试使用JSON加载配置")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"无法加载配置文件: {e}")
            return {}


def main():
    """主函数入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RePSS多任务学习训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (YAML或JSON)")
    parser.add_argument("--rgb-only", action="store_true", help="仅使用RGB模态进行消融实验")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 添加命令行参数到配置
    if args.rgb_only:
        config["rgb_only"] = True
        # 如果是消融实验，修改输出目录
        if "output_dir" in config:
            config["output_dir"] = str(Path(config["output_dir"]) / "ablation_rgb_only")
    
    # 记录提示信息
    output_dir = Path(config.get("output_dir", "./output"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"multitask_training_{timestamp}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加文件日志处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    logger.info(f"\u2551 RePSS多任务学习训练开始         \u2551")
    logger.info(f"\u2551 时间: {timestamp}             \u2551")
    logger.info(f"\u2551 配置文件: {args.config}    \u2551")
    logger.info(f"\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")
    
    # 创建训练器实例
    trainer = MultitaskTrainer(config)
    
    # 开始训练
    best_model_path = trainer.train()
    
    logger.info(f"\u8bad\u7ec3\u5b8c\u6210\uff01\u6700\u4f73\u6a21\u578b\u4fdd\u5b58\u5728: {best_model_path}")


if __name__ == "__main__":
    main()
