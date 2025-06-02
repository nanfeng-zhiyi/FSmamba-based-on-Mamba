#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VIPL-HR数据集加载模块
直接从处理过的数据和wave.csv文件中加载vPPG信号
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.signal import butter, filtfilt
from typing import Dict, List, Tuple, Union, Optional, Any
import traceback

# 导入自定义数据增强模块
try:
    from data.augmentations import apply_augmentations
    HAS_AUGMENTATIONS = True
except ImportError:
    HAS_AUGMENTATIONS = False
    logging.warning("无法导入数据增强模块，将不使用增强功能")

# 设置日志
logger = logging.getLogger(__name__)

def _normalize_path(path):
    """
    规范化路径字符串
    """
    try:
        return Path(path).resolve()
    except Exception:
        return None

class PhysiologicalSignalDataset(Dataset):
    """生理信号数据集的基类"""
    
    def __init__(self):
        """初始化数据集"""
        super().__init__()
        self.sequence_length = 75  # 默认序列长度
        self.filter_low = 0.7  # 低通滤波器临界频率
        self.filter_high = 3.5  # 高通滤波器临界频率
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
    
    def __len__(self):
        """返回数据集长度"""
        return 0
    
    def __getitem__(self, idx):
        """获取数据集项"""
        pass

class VIPLHRDataset(PhysiologicalSignalDataset):
    """VIPL-HR数据集
    
    数据集结构：
    1. 处理后的数据目录:
    data_dir/
        └── paired/
            ├── p1/
            │   ├── v1/
            │   │   └── processed_data.npz
            │   └── v2/
            │       └── processed_data.npz
            └── ...
    
    2. 交叉验证分割结构:
    data_dir/
        └── subject_5fold_split/
            ├── fold_0/
            │   ├── dataset_split.json  # 简化格式
            │   ├── train.csv
            │   ├── val.csv
            │   ├── test.csv
            │   └── ...
            └── ...
    """
    
    def __init__(self, data_dir, mode='train', sequence_length=75, overlap=0.5, 
                 transform=None, use_nir=True, use_vppg=False, 
                 use_simclr=False, source_id=None, scene_id=None,
                 labels_dir=None, skip_missing_subjects=True, device=None,
                 subject_filter=None, use_time_jitter=True, use_color_augment=True,
                 split_path=None, fold_idx=None):
        """初始化VIPL-HR数据集
        
        参数:
            data_dir: 数据目录
            mode: 模式，'train', 'val', 或 'test'
            sequence_length: 序列长度
            overlap: 帧序列重叠比例
            transform: 数据增强转换
            use_nir: 是否使用NIR
            use_vppg: 是否使用vPPG
            use_simclr: 是否使用SimCLR
            source_id: 指定的数据源ID
            scene_id: 指定的场景ID
            labels_dir: 标签目录
            skip_missing_subjects: 是否跳过缺失主体
            device: 设备
            subject_filter: 受试者ID过滤列表，只加载包含的受试者ID
            use_time_jitter: 是否使用时域抖动增强
            use_color_augment: 是否使用颜色增强
            split_path: 数据集分割文件夹路径 (e.g., 'subject_5fold_split/fold_0')
            fold_idx: 数据集分割折索引 (e.g., 0-4)，自动使用对应的fold_N子目录
        """
        logger.info(f"初始化VIPL-HR数据集，模式: {mode}, 数据目录: {data_dir}")
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.transform = transform
        self.use_nir = use_nir
        self.use_vppg = use_vppg
        self.use_simclr = use_simclr
        self.source_id = source_id
        self.scene_id = scene_id
        self.labels_dir = labels_dir
        self.skip_missing_subjects = skip_missing_subjects
        self.subject_filter = subject_filter  # 受试者过滤
        self.use_time_jitter = use_time_jitter and mode == 'train'  # 只在训练时使用时域抖动
        self.use_color_augment = use_color_augment and mode == 'train'  # 只在训练时使用颜色增强
        
        # 处理分割路径
        self.split_path = split_path
        if fold_idx is not None and split_path is None:
            # 如果指定了fold_idx但没有split_path，假设使用默认目录
            self.split_path = f"subject_5fold_split/fold_{fold_idx}"
            logger.info(f"使用默认分割路径: {self.split_path}")
        
        # 设置设备
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 日志记录
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"数据目录路径: {os.path.abspath(str(self.data_dir))}")
        if self.split_path:
            logger.info(f"使用分割路径: {self.split_path}")
        
        # 加载元数据
        logger.info(f"开始加载{mode}元数据，数据目录: {data_dir}")
        self.samples = self._load_metadata()
        
        # 打印样本数
        self.valid_samples = [s for s in self.samples]
        
        # 根据数据是否有效过滤样本
        if len(self.valid_samples) > 0:
            logger.info(f"加载了{len(self.valid_samples)}个元数据项")
        else:
            logger.error(f"{mode}数据集为空，无法找到任何符合条件的数据")
            # 记录更多信息以便调试
            logger.error(f"请检查以下情况:")
            logger.error(f"1. 数据目录: {self.data_dir}，该目录是否存在: {self.data_dir.exists()}")
            logger.error(f"2. source_id筛选条件: {self.source_id}")
            logger.error(f"3. scene_id筛选条件: {self.scene_id}")
            logger.error(f"4. 请检查数据目录结构，确保存在paired_data.npz文件")
            # 数据集为空时创建虚拟样本
            self._create_dummy_samples()
            logger.warning(f"创建了{len(self.valid_samples)}个虚拟样本供演示使用")
        
        # 分配样本索引，考虑重叠
        self._assign_sample_indices()
    
    def __len__(self):
        """返回数据集中样本数量"""
        if hasattr(self, 'sample_indices') and self.sample_indices:
            return len(self.sample_indices)
        elif hasattr(self, 'valid_samples') and self.valid_samples:
            return len(self.valid_samples)
        return 0
        
    def __getitem__(self, idx):
        """返回指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本数据字典，包含图像帧、HR等信息
        """
        try:
            # 检查索引是否有效
            if not hasattr(self, 'valid_samples') or not self.valid_samples:
                logger.warning(f"数据集无有效样本，无法获取索引{idx}，返回默认样本")
                return self._create_default_sample()
                
            # 如果有sample_indices，使用它来获取真实索引
            if hasattr(self, 'sample_indices') and self.sample_indices:
                if idx >= len(self.sample_indices):
                    logger.error(f"索引{idx}超出有效范围{len(self.sample_indices)}，返回默认样本")
                    return self._create_default_sample()
                real_idx = self.sample_indices[idx]
            else:
                if idx >= len(self.valid_samples):
                    logger.error(f"索引{idx}超出有效范围{len(self.valid_samples)}，返回默认样本")
                    return self._create_default_sample()
                real_idx = idx
            
            # 获取对应的样本信息
            sample_info = self.valid_samples[real_idx] if hasattr(self, 'valid_samples') else None
                
            # 获取样本数据，确保即使加载失败也不返回None
            if hasattr(self, '_get_paired_item'):
                # 优先使用配对数据加载方法
                sample = self._get_paired_item(real_idx)
                if sample is None:
                    logger.warning(f"配对数据加载失败，使用默认样本代替，索引: {idx}")
                    return self._create_default_sample(sample_info)
            elif hasattr(self, '_fallback_getitem'):
                # 使用回退方法
                sample = self._fallback_getitem(real_idx)
                if sample is None:
                    logger.warning(f"回退方法加载失败，使用默认样本代替，索引: {idx}")
                    return self._create_default_sample(sample_info)
            else:
                # 如果以上方法都不可用，使用_create_default_sample
                logger.warning(f"无法找到合适的加载方法，使用默认样本，索引: {idx}")
                return self._create_default_sample(sample_info)
                
            return sample
        except Exception as e:
            logger.error(f"获取样本时出错，索引: {idx}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_default_sample()
    
    def _load_metadata(self):
        """加载数据集元数据"""
        metadata = []
        
        # 确保numpy可用
        import numpy as np
        
        # 检查是否存在paired目录
        paired_dir = self.data_dir / 'paired'
        if paired_dir.exists() and paired_dir.is_dir():
            logger.info(f"检测到paired目录，将使用专用的样本加载方法")
            # 使用专门的方法处理processed_vipl结构
            samples = self._load_samples_from_processed()
            if samples:
                logger.info(f"成功从processed_vipl结构加载样本: {len(samples)}个")
                return samples
            else:
                logger.warning("从processed_vipl结构加载样本失败，尝试其他方法")
                
        # 查找已处理的数据文件
        data_files = []
        
        try:
            # 查找已处理的数据文件
            processed_dir = self.data_dir
            logger.info(f"正在搜索数据目录: {processed_dir}")
            
            if processed_dir.exists():
                logger.info(f"数据目录存在，开始查找.npz文件")
                print(f"当前工作目录: {os.getcwd()}")
                print(f"数据目录: {processed_dir.absolute()}")
                print(f"数据目录内容: {[f.name for f in processed_dir.iterdir() if f.is_file()]}")
                print(f"数据目录子文件夹: {[d.name for d in processed_dir.iterdir() if d.is_dir()]}")

                # 首先检查是否存在数据集分割文件
                dataset_split_file = self.data_dir / 'dataset_split.json'
                logger.info(f"尝试查找数据集分割文件: {dataset_split_file}")
                dataset_split = None
                if dataset_split_file.exists():
                    logger.info(f"数据集分割文件存在: {dataset_split_file}")
                    try:
                        with open(dataset_split_file, 'r') as f:
                            dataset_split = json.load(f)
                        logger.info(f"从{dataset_split_file}加载了数据集分割信息")
                        
                        # 显示每个集合的条目数
                        for key, value in dataset_split.items():
                            logger.info(f"分割文件中 {key} 集合有 {len(value)} 个条目")
                        
                        # 如果有当前模式的分割数据，尝试加载
                        if self.mode in dataset_split and dataset_split[self.mode]:
                            mode_dirs = dataset_split[self.mode]
                            logger.info(f"在数据集分割中找到{self.mode}模式的{len(mode_dirs)}个条目")
                            logger.info(f"分割条目示例: {mode_dirs[:3]}")
                            
                            # 从dataset_split中获取条目
                            for entry in mode_dirs:
                                # 可能的数据路径
                                found = False
                                
                                # 尝试不同的路径组合
                                potential_paths = [
                                    processed_dir / entry,
                                    processed_dir / 'processed' / entry,
                                    processed_dir / entry.split('_')[0] / entry.split('_')[1] / entry.split('_')[2]
                                ]
                                
                                # 检查每个可能的路径
                                for potential_path in potential_paths:
                                    if potential_path.exists() and potential_path.is_dir():
                                        npz_file = potential_path / 'processed_data.npz'
                                        if npz_file.exists():
                                            logger.info(f"找到匹配的目录和NPZ文件: {npz_file}")
                                            data_files.append(potential_path)
                                            found = True
                                            break
                                
                                if not found:
                                    # 尝试直接在所有文件夹中搜索包含此条目部分的路径
                                    parts = entry.split('_')
                                    if len(parts) >= 3:
                                        subject_id, scene_id, source_id = parts[0], parts[1], parts[2]
                                        logger.debug(f"尝试查找包含 {subject_id}, {scene_id}, {source_id} 的路径")
                                        
                                        # 搜索所有可能的目录
                                        for root, dirs, files in os.walk(processed_dir):
                                            root_path = Path(root)
                                            root_str = str(root_path).lower()
                                            
                                            if (subject_id.lower() in root_str and 
                                                scene_id.lower() in root_str):
                                                # 检查是否有NPZ文件
                                                for file in files:
                                                    if file.endswith('.npz'):
                                                        logger.info(f"找到部分匹配的文件: {root_path / file}")
                                                        data_files.append(root_path)
                                                        found = True
                                                        break
                                            
                                            if found:
                                                break
                            
                            logger.info(f"从数据集分割文件中找到 {len(data_files)} 个匹配的数据文件")
                    except Exception as e:
                        logger.error(f"处理数据集分割文件时出错: {e}")
                        logger.error(traceback.format_exc())
                        dataset_split = None
                else:
                    logger.warning(f"数据集分割文件不存在: {dataset_split_file}，尝试其他可能位置")
                    # 尝试其他可能的位置
                    alt_split_file = processed_dir / 'dataset_split.json'
                    if alt_split_file.exists():
                        logger.info(f"在根目录找到数据集分割文件: {alt_split_file}")
                        try:
                            with open(alt_split_file, 'r') as f:
                                dataset_split = json.load(f)
                            logger.info(f"从{alt_split_file}加载了数据集分割信息")
                        except Exception as e:
                            logger.error(f"处理数据集分割文件时出错: {e}")
                
                # 如果没有从数据集分割中加载到文件，使用常规方法查找
                if not data_files:
                    logger.info(f"未从数据集分割中找到文件，开始使用常规方法查找npz文件")
                    # 增加查找深度，确保能找到processed_data.npz文件
                    processed_dirs_checked = set()  # 记录已检查的目录
                    
                    # 先查找目标目录的直接子目录
                    logger.info(f"查找主目录中的处理数据: {processed_dir}")
                    try:
                        for item in os.listdir(processed_dir):
                            item_path = processed_dir / item
                            if item_path.is_dir():
                                logger.debug(f"检查目录: {item_path}")
                                processed_dirs_checked.add(str(item_path))
                                
                                # 检查此目录下是否有npz文件
                                for npz_file in item_path.glob("*.npz"):
                                    if npz_file.name == "processed_data.npz" or "processed" in npz_file.name.lower():
                                        logger.info(f"找到npz文件: {npz_file}")
                                        data_files.append(npz_file.parent)
                    except Exception as e:
                        logger.warning(f"列出目录内容时出错: {e}")
                    
                    # 使用walk递归查找
                    logger.info(f"开始递归查找processed_data.npz文件")
                    for root, dirs, files in os.walk(processed_dir):
                        for file in files:
                            if file.endswith('.npz'):
                                npz_path = Path(root) / file
                                if npz_path.exists():
                                    data_files.append(npz_path.parent)
                                    logger.info(f"找到数据文件: {npz_path}")
                    
                    # 特别搜索dataset_split.json文件中指定的目录
                    logger.info(f"尝试针对性搜索分割文件中的目录")
                    if dataset_split and self.mode in dataset_split:
                        for entry in dataset_split[self.mode]:
                            # 构建可能的路径
                            possible_paths = [
                                processed_dir / entry,
                                processed_dir / "processed" / entry,
                                processed_dir / entry.split("_")[0] / entry.split("_")[1]
                            ]
                            
                            for path in possible_paths:
                                if path.exists() and path not in processed_dirs_checked:
                                    logger.debug(f"检查分割条目对应的目录: {path}")
                                    processed_dirs_checked.add(str(path))
                                    
                                    # 查找此目录中的npz文件
                                    for npz_file in path.glob("**/*.npz"):
                                        if npz_file.exists():
                                            logger.info(f"找到分割条目对应的npz文件: {npz_file}")
                                            data_files.append(npz_file.parent)
                    
                    # 如果仍然没找到，创建一组虚拟数据
                    if not data_files:
                        logger.warning(f"未能找到任何数据文件，将尝试创建虚拟数据用于演示")
                        dummy_dir = processed_dir / "dummy_data"
                        dummy_dir.mkdir(exist_ok=True)
                        data_files.append(dummy_dir)
                        
                        # 打印已扫描的目录，帮助诊断
                        logger.warning(f"已扫描的目录: {list(processed_dirs_checked)}")
                        
                    if not data_files:
                        # 如果没有找到npz文件，尝试在processed子目录查找
                        logger.warning(f"在主目录未找到.npz文件，检查是否存在processed子目录")
            else:
                logger.warning(f"处理目录不存在: {processed_dir}")
        except Exception as e:
            logger.error(f"查找处理过的数据文件时出错: {e}")
            logger.error(traceback.format_exc())  # 添加更详细的错误追踪信息
        
        # 加载标签数据
        labels_df = None
        if self.labels_dir:
            try:
                # 尝试加载标签文件
                label_file = self.labels_dir / "vipl_hr_labels.csv"
                if label_file.exists():
                    labels_df = pd.read_csv(label_file)
                    logger.info(f"从{label_file}加载了{len(labels_df)}个标签")
                else:
                    # 尝试其他可能的位置
                    logger.warning(f"标签文件不存在: {label_file}")
                    
                    # 尝试查找其他标签文件
                    for file in self.labels_dir.glob("*.csv"):
                        if "label" in file.name.lower():
                            try:
                                labels_df = pd.read_csv(file)
                                logger.info(f"从{file}加载了{len(labels_df)}个标签")
                                break
                            except Exception as e:
                                logger.warning(f"无法加载标签文件{file}: {e}")
            except Exception as e:
                logger.error(f"加载标签数据时出错: {e}")
                logger.error(traceback.format_exc())
        
        # 如果没有找到标签文件，尝试创建虚拟标签
        if labels_df is None:
            logger.warning("未找到标签文件，将创建虚拟标签")
            # 为每个处理过的数据文件创建一个虚拟标签
            labels_data = []
            for idx, data_path in enumerate(data_files):
                subject_id = data_path.parent.name
                scene_id = data_path.name if data_path.name != subject_id else "default"
                
                labels_data.append({
                    'subject_id': subject_id,
                    'scene_id': scene_id,
                    'heart_rate': 75.0 + idx,  # 默认心率
                    'respiration_rate': 16.0 + idx/5.0,  # 默认呼吸率
                    'spo2': 98.0,  # 默认
                    'source_id': 'unknown'
                })
            
            if labels_data:
                labels_df = pd.DataFrame(labels_data)
                logger.info(f"创建了{len(labels_df)}个虚拟标签")
        
        # 处理数据
        train_ratio = 0.72
        val_ratio = 0.08
        test_ratio = 0.20
        
        processed_files = []
        logger.info(f"开始筛选数据文件，共有{len(data_files)}个待处理文件")
        logger.info(f"筛选条件: source_id={self.source_id}, scene_id={self.scene_id}")
        
        # 如果没有找到任何文件，尝试根据数据集分割文件创建虚拟数据结构
        if not data_files and dataset_split and self.mode in dataset_split:
            logger.warning(f"未找到真实数据文件，将根据分割文件创建虚拟数据结构")
            # 从分割文件创建虚拟目录
            for entry in dataset_split[self.mode][:5]:  # 只使用前5个条目避免创建过多
                parts = entry.split('_')
                if len(parts) >= 3:
                    subject_id, scene_id, source_id = parts[0], parts[1], parts[2]
                    virtual_dir = processed_dir / "virtual" / entry
                    virtual_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"创建虚拟数据目录: {virtual_dir}")
                    data_files.append(virtual_dir)
            
            logger.info(f"创建了{len(data_files)}个虚拟数据目录")
        
        for data_path in data_files:
            # 检查是否符合source_id和scene_id筛选条件
            # 改进路径匹配，提取目录名称中的信息
            try:
                # 尝试从路径中提取信息
                dir_name = data_path.name
                
                # 检查是否是p1_v1_source1格式
                if '_' in dir_name and dir_name.count('_') >= 2:
                    parts = dir_name.split('_')
                    subject_id = parts[0]  # p1
                    scene_id = parts[1]    # v1
                    source_id = parts[2]   # source1
                    logger.debug(f"从目录名'{dir_name}'提取到: subject_id={subject_id}, scene_id={scene_id}, source_id={source_id}")
                else:
                    # 回退到旧方法，使用路径分隔符
                    parts = str(data_path).split(os.sep)
                    subject_id = parts[-2] if len(parts) >= 2 else "unknown"
                    scene_id = parts[-1] if len(parts) >= 1 else "unknown"
                    source_id = "source1"  # 默认
                    logger.debug(f"从路径分隔符提取到: subject_id={subject_id}, scene_id={scene_id}")
                
                # 判断是否符合筛选条件
                if self.source_id and source_id and self.source_id.lower() != source_id.lower():
                    logger.debug(f"数据路径不符合source_id条件: 要求={self.source_id}, 实际={source_id}")
                    continue
                
                if self.scene_id and scene_id and self.scene_id.lower() != scene_id.lower():
                    logger.debug(f"数据路径不符合scene_id条件: 要求={self.scene_id}, 实际={scene_id}")
                    continue
            except Exception as e:
                logger.warning(f"处理路径时出错: {e}, 路径={data_path}")
                
            # 检查是否存在处理过的数据文件
            npz_file = data_path / 'processed_data.npz'
            if not npz_file.exists():
                logger.debug(f"NPZ文件不存在: {npz_file}")
                
                # 增加更多可能的路径尝试
                alternative_paths = [
                    data_path / 'processed_data.npz',
                    data_path / 'processed' / 'processed_data.npz',
                    Path(str(data_path) + '.npz'),
                    Path(str(data_path) + '_processed.npz')
                ]
                
                logger.debug(f"尝试查找替代路径...")
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        logger.info(f"找到替代NPZ文件: {alt_path}")
                        npz_file = alt_path
                        break
                
                if not npz_file.exists():
                    # 打印目录内容，帮助诊断
                    if data_path.exists() and data_path.is_dir():
                        files_in_dir = list(data_path.glob("*"))
                        if files_in_dir:
                            logger.debug(f"目录{data_path}包含文件: {[f.name for f in files_in_dir]}")
                        else:
                            logger.debug(f"目录{data_path}是空的")
                    else:
                        logger.debug(f"目录{data_path}不存在或不是目录")
                    
                    # 创建虚拟NPZ文件用于演示
                    if "virtual" in str(data_path):
                        import numpy as np
                        # 创建一个简单的NPZ文件用于演示
                        dummy_data = {
                            'frames': np.random.rand(100, 128, 128, 3) * 255,  # 100帧虚拟数据
                            'metadata': {'subject_id': dir_name.split('_')[0] if '_' in dir_name else 'p1',
                                        'scene_id': scene_id or 'v1',
                                        'source_id': source_id or 'source1'}
                        }
                        np.savez(data_path / 'processed_data.npz', **dummy_data)
                        logger.info(f"创建了虚拟NPZ文件: {data_path / 'processed_data.npz'}")
                        npz_file = data_path / 'processed_data.npz'
                    else:
                        continue
                        
            logger.debug(f"发现符合条件的数据文件: {npz_file}")
            
            try:
                # 检查wave.csv文件
                wave_file = data_path / 'wave.csv'
                has_wave_csv = wave_file.exists()
                
                # 检查processed_data.npz是否包含'vppg'
                has_vppg_in_npz = False
                try:
                    with np.load(npz_file, allow_pickle=True) as data:
                        has_vppg_in_npz = 'vppg' in data
                        if has_vppg_in_npz:
                            logger.info(f"文件{npz_file}包含vPPG数据")
                        else:
                            logger.info(f"文件{npz_file}不包含vPPG数据，但将尝试从其他数据派生vPPG信号")
                except Exception as e:
                    logger.warning(f"无法检查{npz_file}是否包含vPPG数据: {e}")
                    has_vppg_in_npz = False
                
                # 即使没有vPPG数据，也保留该数据项，我们将在__getitem__中处理
                # 如果use_vppg=True但没有vPPG数据，我们将尝试从RGB帧中提取vPPG信号
                if self.use_vppg and not (has_wave_csv or has_vppg_in_npz):
                    logger.info(f"数据项{data_path}没有vPPG数据，但将保留并尝试从RGB帧中提取")
                
                # 从文件路径提取主体ID和场景ID
                try:
                    # 获取目录名
                    dir_name = data_path.name
                    
                    # 检查是否是p1_v1_source1格式
                    if '_' in dir_name and dir_name.count('_') >= 2:
                        parts = dir_name.split('_')
                        subject_id = parts[0]  # p1
                        scene_id = parts[1]    # v1
                        source_id = parts[2]   # source1
                        logger.debug(f"从目录名'{dir_name}'提取到: subject_id={subject_id}, scene_id={scene_id}, source_id={source_id}")
                    else:
                        # 回退到旧方法，使用路径分隔符
                        parts = str(data_path).split(os.sep)
                        subject_id = parts[-2] if len(parts) >= 2 else "unknown"
                        scene_id = parts[-1] if len(parts) >= 1 else "unknown"
                        source_id = "source1"  # 默认
                        logger.debug(f"从路径分隔符提取到: subject_id={subject_id}, scene_id={scene_id}")
                except Exception as e:
                    logger.warning(f"从路径提取ID时出错: {e}")
                    subject_id = "unknown"
                    scene_id = "unknown"
                    source_id = "unknown"
                
                # 尝试从标签数据中查找心率和血氧
                heart_rate = 75.0  # 默认
                respiration_rate = 16.0  # 默认呼吸率
                spo2 = 98.0  # 默认
                
                if labels_df is not None:
                    # 尝试匹配主体ID和场景ID
                    matches = labels_df[
                        (labels_df['subject_id'].astype(str) == subject_id) & 
                        (labels_df['scene_id'].astype(str) == scene_id)
                    ]
                    
                    if not matches.empty:
                        for idx, row in matches.iterrows():
                            heart_rate = float(row.get('heart_rate', 75.0))
                            respiration_rate = float(row.get('respiration_rate', 16.0))
                            spo2 = float(row.get('spo2', 98.0))
                            if 'source_id' in row:
                                source_id = str(row['source_id'])
                            processed_files.append({
                                'data_path': data_path,
                                'subject_id': subject_id,
                                'scene_id': scene_id,
                                'source_id': source_id,
                                'heart_rate': heart_rate,
                                'respiration_rate': respiration_rate,
                                'spo2': spo2,
                                'has_wave_csv': has_wave_csv,
                                'has_vppg_in_npz': has_vppg_in_npz
                            })
                    else:
                        # 如果没有匹配，使用默认值
                        processed_files.append({
                            'data_path': data_path,
                            'subject_id': subject_id,
                            'scene_id': scene_id,
                            'source_id': source_id,
                            'heart_rate': heart_rate,
                            'respiration_rate': respiration_rate,
                            'spo2': spo2,
                            'has_wave_csv': has_wave_csv,
                            'has_vppg_in_npz': has_vppg_in_npz
                        })
                else:
                    # 如果没有标签数据，使用默认值
                    processed_files.append({
                        'data_path': data_path,
                        'subject_id': subject_id,
                        'scene_id': scene_id,
                        'source_id': source_id,
                        'heart_rate': heart_rate,
                        'respiration_rate': respiration_rate,
                        'spo2': spo2,
                        'has_wave_csv': has_wave_csv,
                        'has_vppg_in_npz': has_vppg_in_npz
                    })
                
            except Exception as e:
                logger.error(f"处理数据项时出错: {data_path}, 错误: {e}")
        
        # 按主体ID排序，以便正确分割
        processed_files.sort(key=lambda x: x['subject_id'])
        
        unique_subjects = sorted(list(set([item['subject_id'] for item in processed_files])))
        total_subjects = len(unique_subjects)
        logger.info(f"筛选后的有效subject数量: {total_subjects}")
        if total_subjects > 0:
            logger.info(f"有效subjects列表: {unique_subjects}")
        
        if total_subjects == 0:
            logger.error("没有找到任何符合条件的数据，详细原因:")
            logger.error(f"1. 数据目录: {self.data_dir}，该目录是否存在: {self.data_dir.exists()}")
            logger.error(f"2. source_id筛选条件: {self.source_id}")
            logger.error(f"3. scene_id筛选条件: {self.scene_id}")
            logger.error(f"4. 请检查数据目录结构，确保存在paired_data.npz文件")
            
            # 创建虚拟数据
            logger.warning("没有找到合适的数据，将创建虚拟数据集用于演示")
            for i in range(10):  # 创建10个虚拟样本
                metadata.append({
                    'data_path': self.data_dir,  # 使用当前目录
                    'subject_id': f'dummy_p{i+1}',
                    'scene_id': f'v{(i % 3) + 1}',  # v1, v2, v3循环
                    'source_id': 'dummy_source',
                    'start_idx': 0,
                    'end_idx': self.sequence_length,
                    'heart_rate': 75.0 + i,
                    'respiration_rate': 16.0 + i/5.0,
                    'spo2': 98.0,
                    'has_wave_csv': False,
                    'has_vppg_in_npz': False,
                    'is_virtual': True  # 标记为虚拟数据
                })
            logger.info(f"已创建{len(metadata)}个虚拟数据条目")
            return metadata
        
        # 尝试查找和加载dataset_split.json文件
        split_data = None
        logger.info("======= 开始尝试查找和加载dataset_split.json文件 =======")
        try:
            # 尝试在多个可能的位置查找dataset_split.json
            possible_split_paths = [
                Path("data/vlhr/processed/dataset_split.json")
            ]
            
            # 过滤掉None值
            possible_split_paths = [p for p in possible_split_paths if p is not None]
            
            # 输出查找信息
            logger.info(f"查找分割文件，尝试以下路径:")
            for i, sp in enumerate(possible_split_paths):
                logger.info(f"  {i+1}. {sp} ({'存在' if sp.exists() else '不存在'})")
            
            # 查找第一个存在的分割文件
            split_path_found = False
            for split_path in possible_split_paths:
                if split_path.exists():
                    logger.info(f"找到分割文件: {split_path}")
                    with open(split_path, 'r') as f:
                        split_data = json.load(f)
                    
                    # 打印分割文件内容概要，帮助调试
                    logger.info("分割文件内容概要:")
                    for mode_name, dirs in split_data.items():
                        logger.info(f"  {mode_name}: {len(dirs)}个目录")
                        if dirs:
                            logger.info(f"    示例: {dirs[:3]}")
                    
                    split_path_found = True
                    break
            
            if not split_path_found:
                logger.warning("未找到任何dataset_split.json文件")
            
            if split_data:
                logger.info(f"成功加载分割文件，包含以下集合: {list(split_data.keys())}")
                
                # 使用split_data分割数据集，而不是按比例分割
                selected_files = []
                
                # 获取当前模式的数据目录
                if self.mode in split_data:
                    mode_dirs = split_data[self.mode]
                    logger.info(f"{self.mode}集包含{len(mode_dirs)}个目录")
                    logger.info(f"分割文件中{self.mode}集的部分目录: {mode_dirs[:5]}")
                    
                    for item in processed_files:
                        # 构建目录名格式，与split_data中的匹配
                        data_path = item['data_path']
                        dir_name = f"{item['subject_id']}_{item['scene_id']}_{item['source_id']}"
                        
                        # 检查目录名是否在分割数据中
                        if dir_name in mode_dirs:
                            logger.debug(f"匹配到目录名: {dir_name}")  # 改为debug级别，减少输出
                            selected_files.append(item)
                        # 也检查路径的最后一部分是否匹配
                        elif data_path.name in mode_dirs:
                            logger.debug(f"匹配到路径名: {data_path.name}")  # 改为debug级别，减少输出
                            selected_files.append(item)
                        # 检查路径本身是否匹配任何目录
                        elif any(mode_dir in str(data_path) for mode_dir in mode_dirs):
                            logger.info(f"匹配到路径: {data_path}")
                            selected_files.append(item)
                    
                    logger.info(f"根据分割文件，为{self.mode}模式选择了{len(selected_files)}个文件")
                    
                    # 如果在split_data中找到了数据，则使用这些数据，否则继续使用基于比例的分割
                    if selected_files:
                        logger.info(f"使用分割文件中的数据进行{self.mode}集创建")
                        # 为每个选定的文件生成序列元数据
                        for item in selected_files:
                            data_path = item['data_path']
                            try:
                                npz_file = data_path / 'paired_data.npz'
                                with np.load(npz_file, allow_pickle=True, mmap_mode='r') as data:
                                    if 'frames' in data:
                                        num_frames = len(data['frames'])
                                        
                                        if num_frames < self.sequence_length:
                                            logger.debug(f"[静默处理] {npz_file.name} 帧数不足: {num_frames} < {self.sequence_length}")
                                            # 在元数据中添加单个序列
                                            metadata.append({
                                                'data_path': data_path,
                                                'start_idx': 0,
                                                'end_idx': min(num_frames, self.sequence_length),
                                                'subject_id': item['subject_id'],
                                                'scene_id': item['scene_id'],
                                                'source_id': item['source_id'],
                                                'heart_rate': item['heart_rate'],
                                                'respiration_rate': item['respiration_rate'],
                                                'spo2': item['spo2'],
                                                'has_wave_csv': item['has_wave_csv'],
                                                'has_vppg_in_npz': item['has_vppg_in_npz']
                                            })
                                        else:
                                            # 计算步长
                                            step_size = int(self.sequence_length * (1 - self.overlap))
                                            if step_size <= 0:
                                                step_size = 1
                                            
                                            # 使用滑动窗口生成序列
                                            for i in range(0, num_frames - self.sequence_length + 1, step_size):
                                                metadata.append({
                                                    'data_path': data_path,
                                                    'start_idx': i,
                                                    'end_idx': i + self.sequence_length,
                                                    'subject_id': item['subject_id'],
                                                    'scene_id': item['scene_id'],
                                                    'source_id': item['source_id'],
                                                    'heart_rate': item['heart_rate'],
                                                    'respiration_rate': item['respiration_rate'],
                                                    'spo2': item['spo2'],
                                                    'has_wave_csv': item['has_wave_csv'],
                                                    'has_vppg_in_npz': item['has_vppg_in_npz']
                                                })
                                    else:
                                        logger.warning(f"NPZ文件中没有frames数据: {npz_file}")
                            except Exception as e:
                                logger.error(f"处理NPZ文件时出错: {npz_file}, 错误: {e}")
                        
                        logger.info(f"{self.mode}集元数据加载完成，共{len(metadata)}个序列")
                        return metadata
                else:
                    logger.warning(f"分割文件中没有{self.mode}模式的数据，将使用基于比例的分割")
            else:
                logger.warning("未找到有效的分割文件，将使用基于比例的分割")
        except Exception as e:
            logger.error(f"加载分割文件时出错: {e}")
            logger.error(traceback.format_exc())
            logger.warning("将使用基于比例的分割")
        
        # 如果没有找到或加载分割文件，使用基于比例的分割
        train_end = int(total_subjects * train_ratio)
        val_end = int(total_subjects * (train_ratio + val_ratio))
        
        train_subjects = unique_subjects[:train_end]
        val_subjects = unique_subjects[train_end:val_end]
        test_subjects = unique_subjects[val_end:]
        
        # 根据模式选择相应的主体
        selected_subjects = []
        if self.mode == 'train':
            selected_subjects = train_subjects
        elif self.mode == 'val':
            selected_subjects = val_subjects
        elif self.mode == 'test':
            selected_subjects = test_subjects
        
        # 过滤出所选主体的数据
        selected_files = [item for item in processed_files if item['subject_id'] in selected_subjects]
        
        if not selected_files:
            logger.warning(f"在{self.mode}集中没有找到任何数据")
            return []
        
        # 为每个处理过的文件生成序列元数据
        for item in selected_files:
            data_path = item['data_path']
            # 加载NPZ文件以获取帧数
            try:
                npz_file = data_path / 'paired_data.npz'
                with np.load(npz_file, allow_pickle=True) as data:
                    if 'frames' in data:
                        num_frames = len(data['frames'])
                        
                        if num_frames < self.sequence_length:
                            logger.debug(f"[静默处理] {npz_file.name} 帧数不足: {num_frames} < {self.sequence_length}")
                            # 在元数据中添加单个序列
                            metadata.append({
                                'data_path': data_path,
                                'start_idx': 0,
                                'end_idx': min(num_frames, self.sequence_length),
                                'subject_id': item['subject_id'],
                                'scene_id': item['scene_id'],
                                'source_id': item['source_id'],
                                'heart_rate': item['heart_rate'],
                                'respiration_rate': item['respiration_rate'],
                                'spo2': item['spo2'],
                                'has_wave_csv': item['has_wave_csv'],
                                'has_vppg_in_npz': item['has_vppg_in_npz']
                            })
                        else:
                            # 计算步长
                            step_size = int(self.sequence_length * (1 - self.overlap))
                            if step_size <= 0:
                                step_size = 1
                            
                            # 使用滑动窗口生成序列
                            for i in range(0, num_frames - self.sequence_length + 1, step_size):
                                metadata.append({
                                    'data_path': data_path,
                                    'start_idx': i,
                                    'end_idx': i + self.sequence_length,
                                    'subject_id': item['subject_id'],
                                    'scene_id': item['scene_id'],
                                    'source_id': item['source_id'],
                                    'heart_rate': item['heart_rate'],
                                    'respiration_rate': item['respiration_rate'],
                                    'spo2': item['spo2'],
                                    'has_wave_csv': item['has_wave_csv'],
                                    'has_vppg_in_npz': item['has_vppg_in_npz']
                                })
                    else:
                        logger.warning(f"NPZ文件中没有frames数据: {npz_file}")
            except Exception as e:
                logger.error(f"处理NPZ文件时出错: {npz_file}, 错误: {e}")
        
        logger.info(f"{self.mode}集元数据加载完成，共{len(metadata)}个序列")
        return metadata
    
    def _assign_sample_indices(self):
        """
        为有效样本分配索引，用于批量处理和交叉验证
        """
        if not hasattr(self, 'valid_samples') or not self.valid_samples:
            logger.warning(f"{self.mode}模式无有效样本，跳过索引分配")
            return
            
        # 初始化索引列表
        self.sample_indices = list(range(len(self.valid_samples)))
        
        # 如果需要随机打乱，可以在此处实现
        if self.mode == 'train' and hasattr(self, 'shuffle') and self.shuffle:
            import random
            random.shuffle(self.sample_indices)
            
        logger.info(f"为{self.mode}模式分配了{len(self.sample_indices)}个样本索引")

    def _create_dummy_samples(self):
        """创建虚拟样本用于演示
        
        当没有找到有效数据时使用此方法创建虚拟样本
        """
        logger.warning(f"为{self.mode}模式创建虚拟演示数据集")
        size = 20 if self.mode == 'train' else 10  # 训练集20个样本，验证/测试集10个样本
        
        self.valid_samples = []
        for i in range(size):
            # 创建虚拟样本信息
            sample = {
                'data_path': str(self.data_dir),  # 使用当前目录
                'subject_id': f'dummy_p{i+1}',
                'scene_id': f'v{(i % 3) + 1}',  # v1, v2, v3循环
                'source_id': 'dummy_source',
                'is_paired': False,
                'is_dummy': True,  # 标记为虚拟样本
                'heart_rate': 75.0 + i,  # 随机心率
                'respiration_rate': 16.0 + i/5.0,  # 随机呼吸率
                'spo2': 98.0  # 固定的SpO2值
            }
            self.valid_samples.append(sample)
        
        logger.info(f"已创建{self.mode}虚拟数据集，共{len(self.valid_samples)}个样本")

    def _create_default_item(self):
        """创建默认数据项 (兼容旧API)"""
        return self._create_default_sample()
        
    def _create_default_sample(self, sample_info=None):
        """创建默认样本，防止返回None导致批处理失败
        
        Args:
            sample_info: 可选的样本信息，用于保留原始样本的一些元数据
            
        Returns:
            默认样本字典，保证包含所有必要的键值对
        """
        # 提取样本信息中的有用元数据
        subject_id = 'default'
        scene_id = 'default'
        source_id = 'default'
        
        if isinstance(sample_info, dict):
            subject_id = sample_info.get('subject_id', 'default')
            scene_id = sample_info.get('scene_id', 'default')
            source_id = sample_info.get('source_id', 'default')
        
        # 创建一个有效但内容为空的样本结构
        seq_len = self.sequence_length  # 使用类设置的序列长度
        sample = {
            'rgb_frames': torch.zeros((seq_len, 3, 128, 128)),
            'heart_rate': torch.tensor([70.0]),  # 默认心率值
            'breathing_rate': torch.tensor([15.0]),  # 默认呼吸率
            'subject_id': subject_id,
            'scene_id': scene_id,
            'source_id': source_id,
            'is_default': True  # 标记这是一个默认样本
        }
        
        # 根据配置添加额外字段
        if self.use_nir:
            sample['nir_frames'] = torch.zeros((seq_len, 1, 128, 128))
            
        if self.use_vppg:
            sample['vppg'] = torch.zeros((seq_len, 1))
        
        return sample
    
    def _fallback_getitem(self, index):
        """原始的获取数据项方法，用作回退策略
        
        当配对数据不可用或加载失败时使用此方法
{{ ... }}
        """
        sample_info = self.samples[index]
        
        # 确定数据路径
        subject_id = sample_info['subject_id']
        scene_id = sample_info['scene_id']
        
        # 检查是否有paired目录并查找配对数据
        paired_dir = self.data_dir / "paired" / subject_id / scene_id
        paired_data_path = paired_dir / "paired_data.npz"
        
        if paired_data_path.exists():
            # 使用配对数据
            try:
                data = np.load(paired_data_path, allow_pickle=True)
                
                # 检查原始数据中的单模态标志
                original_is_single_modality = bool(data.get('single_modality', False))
                original_modality = str(data.get('modality', 'both')) if original_is_single_modality else 'both'
                
                # 获取RGB和NIR帧数据
                rgb_frames = data['rgb_frames']
                nir_frames = data['nir_frames'] if 'nir_frames' in data else None
                
                # 应用数据增强
                if self.mode == 'train' and HAS_AUGMENTATIONS:
                    # 对训练集应用时域抖动和颜色增强
                    rgb_frames, nir_frames = apply_augmentations(
                        rgb_frames, 
                        nir_frames, 
                        use_time_jitter=self.use_time_jitter,
                        use_color_augment=self.use_color_augment
                    )
                
                # 强制多模态判断 - 当use_nir=True且存在有效NIR帧时，强制将数据视为多模态
                is_single_modality = original_is_single_modality
                modality = original_modality
                
                # 检查NIR帧是否有效（非None、非空、非全零）
                has_valid_nir = False
                if nir_frames is not None and nir_frames.size > 1 and nir_frames.shape[0] > 1:
                    # 检查是否至少有部分非零像素（仅检查部分帧以提高性能）
                    sample_frames = nir_frames[:min(10, nir_frames.shape[0])]
                    if np.any(sample_frames > 5):  # 允许一些接近黑色但非零的像素
                        has_valid_nir = True
                
                # 当有效NIR数据存在且use_nir=True时，强制设置为多模态
                if has_valid_nir and self.use_nir:
                    # 更新状态为多模态
                    if is_single_modality and modality == 'rgb':
                        # logger.info(f"强制多模态: 将标记为单模态(rgb)的数据转换为多模态, 受试者={subject_id}, 场景={scene_id}")
                        is_single_modality = False
                        modality = 'both'
                    elif not is_single_modality:
                        # logger.info(f"确认多模态: 数据已正确标记为多模态, 受试者={subject_id}, 场景={scene_id}")
                        pass
                    else:
                        # logger.info(f"保持不变: 数据标记为单模态({modality}), 受试者={subject_id}, 场景={scene_id}")
                        pass
                else:
                    # 保持原始标记，记录日志
                    # logger.info(f"样本数据: 类型=单模态({modality}), 受试者={subject_id}, 场景={scene_id}, NIR有效={has_valid_nir}")
                    pass
                # 处理单模态数据
                if is_single_modality:
                    if modality == 'rgb':
                        # 只有RGB数据，需要创建合适大小的NIR占位符
                        if nir_frames is None or nir_frames.size == 1 or nir_frames.shape[0] == 1:
                            # 创建与RGB匹配大小的NIR帧（单通道灰度图）
                            nir_frames = np.zeros((rgb_frames.shape[0], rgb_frames.shape[1], rgb_frames.shape[2], 1), dtype=np.uint8)
                            # logger.debug(f"创建NIR占位符: shape={nir_frames.shape}")
                    elif modality == 'nir':
                        # 只有NIR数据，需要创建合适大小的RGB占位符
                        if rgb_frames is None or rgb_frames.size == 1 or rgb_frames.shape[0] == 1:
                            # 创建与NIR匹配大小的RGB帧（三通道图像）
                            rgb_frames = np.zeros((nir_frames.shape[0], nir_frames.shape[1], nir_frames.shape[2], 3), dtype=np.uint8)
                            # logger.debug(f"创建RGB占位符: shape={rgb_frames.shape}")
                
                # 记录多模态/单模态统计信息（仅调试用）
                if not hasattr(self, '_modality_stats'):
                    self._modality_stats = {'single_rgb': 0, 'single_nir': 0, 'multi': 0, 'forced_multi': 0}
                
                if is_single_modality:
                    if modality == 'rgb':
                        self._modality_stats['single_rgb'] += 1
                    elif modality == 'nir':
                        self._modality_stats['single_nir'] += 1
                else:
                    if original_is_single_modality:
                        self._modality_stats['forced_multi'] += 1  # 被强制转换为多模态
                    else:
                        self._modality_stats['multi'] += 1  # 原本就是多模态
                
                # 每100个样本打印一次统计信息
                stats_sum = sum(self._modality_stats.values())
                if stats_sum > 0 and stats_sum % 100 == 0:
                    total = stats_sum
                    # logger.info(f"模态统计: 单模态RGB={self._modality_stats['single_rgb']} ({self._modality_stats['single_rgb']/total*100:.1f}%), "
                    #            f"单模态NIR={self._modality_stats['single_nir']} ({self._modality_stats['single_nir']/total*100:.1f}%), "
                    #            f"多模态={self._modality_stats['multi']} ({self._modality_stats['multi']/total*100:.1f}%), "
                    #            f"强制多模态={self._modality_stats['forced_multi']} ({self._modality_stats['forced_multi']/total*100:.1f}%)")
                
                
                heart_rate = float(data.get('heart_rate', 0))
                spo2 = float(data.get('spo2', 0))
                
                # 加载元数据
                metadata_path = paired_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        'subject_id': subject_id,
                        'scene_id': scene_id,
                        'single_modality': is_single_modality,
                        'modality': modality if is_single_modality else 'both'
                    }
                
            except Exception as e:
                logger.error(f"加载配对数据失败: {paired_data_path}, 错误: {e}")
                # 如果配对数据加载失败，回退到原始方式加载
                return self._fallback_getitem(index)
        else:
            # 如果没有找到配对数据，使用原始方式加载
            return self._fallback_getitem(index)
        
        # 处理序列长度
        seq_len = self.sequence_length
        total_frames = len(rgb_frames)
        
        if total_frames <= seq_len:
            # 帧数不足，使用循环填充
            indices = np.arange(total_frames).tolist()
            # 循环填充到所需长度
            while len(indices) < seq_len:
                indices.extend(np.arange(min(total_frames, seq_len - len(indices))).tolist())
        else:
            # 帧数足够，均匀采样
            indices = np.linspace(0, total_frames - 1, seq_len, dtype=int).tolist()
        
        # 提取所需帧
        rgb_frames = rgb_frames[indices]
        # 安全地提取NIR帧 - 检查是否为空数组
        if nir_frames is not None and len(nir_frames) > 0:
            nir_frames = nir_frames[indices]
        else:
            # 如果NIR帧为空，则设置为None而不创建全零帧
            nir_frames = None
            logger.info(f"NIR帧为空，设置为None，模型将使用单模态RGB处理")
        
        # 处理RGB帧
        if rgb_frames.ndim == 3:  # 如果是单张图像
            rgb_frames = np.stack([rgb_frames] * seq_len)
        
        # 将数据转换为张量
        # 检查RGB帧的形状并转置如果需要
        if rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3:  # [seq_len, height, width, channels]格式
            # 需要进行转置，从[seq_len, height, width, channels]变为[seq_len, channels, height, width]
            # logger.info(f"RGB帧转置前的形状: {rgb_frames.shape}")
            # 交换通道轮到前面
            rgb_frames = np.transpose(rgb_frames, (0, 3, 1, 2))
            # logger.info(f"RGB帧转置后的形状: {rgb_frames.shape}")
        elif rgb_frames.ndim == 3 and rgb_frames.shape[-1] == 3:  # [height, width, channels]格式
            # 单帧情况下的转置
            rgb_frames = np.transpose(rgb_frames, (2, 0, 1))
            
        # 转换为张量
        rgb_tensor = torch.from_numpy(rgb_frames).float() / 255.0
        
        # 创建样本字典
        sample = {
            'rgb_frames': rgb_tensor,
            'heart_rate': heart_rate,
            'spo2': spo2,
            'subject_id': subject_id,
            'scene_id': scene_id,
            'sample_id': f"{subject_id}_{scene_id}",
            'is_virtual': sample_info.get('is_virtual', False),
        }
        
        # 添加NIR帧 - 增强了对None值的处理
        if self.use_nir and nir_frames is not None:
            # 处理NIR帧
            if nir_frames.ndim == 3:  # 如果是单张图像
                nir_frames = np.stack([nir_frames] * seq_len)
            
            # 检查NIR帧的形状并转置如果需要
            if nir_frames.ndim == 4 and nir_frames.shape[-1] == 1:  # [seq_len, height, width, channels]格式
                # 需要进行转置，从[seq_len, height, width, channels]变为[seq_len, channels, height, width]
                nir_frames = np.transpose(nir_frames, (0, 3, 1, 2))
            elif nir_frames.ndim == 3 and nir_frames.shape[-1] == 1:  # [height, width, channels]格式
                # 单帧情况下的转置
                nir_frames = np.transpose(nir_frames, (2, 0, 1))
            
            # 处理并添加到样本中
            nir_tensor = self._process_nir_frames(nir_frames)
            sample['nir_frames'] = nir_tensor
        elif self.use_nir:
            # NIR帧为None，将样本标记为单模态
            sample['nir_frames'] = None
            logger.info(f"跳过NIR帧处理，样本将使用单模态RGB")
        
        # 添加标签（如果有）
        if 'activity_label' in sample_info:
            sample['activity_label'] = sample_info['activity_label']
        
        # 提取vPPG信号（如果需要）
        if self.use_vppg:
            vppg = self._extract_vppg_from_frames(rgb_frames)
            sample['vppg'] = vppg
        
        return sample

    def _process_nir_frames(self, nir_frames):
        """处理NIR帧数据
        
        Args:
            nir_frames: NIR帧数据，可能是3通道或1通道，或者None
            
        Returns:
            处理后的NIR张量或None
        """
        # 先检查NIR帧是否为None
        if nir_frames is None:
            return None
            
        # 检查帧是否已经是张量
        if isinstance(nir_frames, torch.Tensor):
            nir_tensor = nir_frames
        else:
            # 检查NIR帧的形状，确保正确处理不同维度的数据
            if nir_frames.ndim == 4:  # (frames, height, width, channels)
                if nir_frames.shape[3] == 3:  # 如果是3通道
                    # 转换为灰度
                    logger.debug("将3通道NIR帧转换为灰度")
                    frames_count = nir_frames.shape[0]
                    height = nir_frames.shape[1]
                    width = nir_frames.shape[2]
                    gray_frames = np.zeros((frames_count, height, width, 1), dtype=nir_frames.dtype)
                    for i in range(frames_count):
                        # 0.299 R + 0.587 G + 0.114 B 的灰度转换
                        gray = 0.299 * nir_frames[i, :, :, 0] + 0.587 * nir_frames[i, :, :, 1] + 0.114 * nir_frames[i, :, :, 2]
                        gray_frames[i, :, :, 0] = gray
                    nir_frames = gray_frames
                elif nir_frames.shape[3] == 1:
                    # 已经是1通道，无需转换
                    pass
                else:
                    # logger.warning(f"意外的NIR帧通道数: {nir_frames.shape[3]}")  # 注释掉冗余日志
                    pass  # 保留一个空操作以维持正确的缩进
            elif nir_frames.ndim == 3:  # (height, width, channels) 或 (frames, height, width)
                if nir_frames.shape[2] == 3:  # 如果第三维是3，假设是单帧三通道图像
                    # 转换为灰度
                    gray = 0.299 * nir_frames[:, :, 0] + 0.587 * nir_frames[:, :, 1] + 0.114 * nir_frames[:, :, 2]
                    nir_frames = gray.reshape(nir_frames.shape[0], nir_frames.shape[1], 1)
                else:
                    # 假设是多帧单通道，调整形状
                    nir_frames = nir_frames.reshape(nir_frames.shape[0], nir_frames.shape[1], nir_frames.shape[2], 1)
            
            # 转换为张量，确保通道顺序正确 (N, C, H, W)
            nir_tensor = torch.from_numpy(nir_frames).float() / 255.0
            
            # 调整维度顺序 (N, H, W, C) -> (N, C, H, W)
            if nir_tensor.ndim == 4 and nir_tensor.shape[3] == 1:
                nir_tensor = nir_tensor.permute(0, 3, 1, 2)
            elif nir_tensor.ndim == 3 and nir_tensor.shape[2] == 1:
                nir_tensor = nir_tensor.permute(0, 2, 1)
        
        return nir_tensor

    def _load_samples_from_processed(self):
        """从处理好的数据目录加载样本信息"""
        logger = logging.getLogger(__name__)
        # logger.info(f"开始从处理过的数据目录加载样本: {self.data_dir}")
        
        
        # 首先尝试从dataset_split.json加载分割信息
        split_file = self.data_dir / 'dataset_split.json'
        # logger.info(f"尝试加载数据集分割文件: {split_file} (存在: {split_file.exists()})")

        if split_file.exists():
            try:
                with open(split_file, 'r', encoding='utf-8') as f:
                    split_data = json.load(f)
                
                # 记录数据集分割信息
                # for mode, entries in split_data.items():
                #     logger.info(f"分割文件中 {mode} 模式有 {len(entries)} 个条目")
                
                # 根据mode选择对应的分割
                if self.mode in split_data:
                    subjects_info = split_data[self.mode]
                    # logger.info(f"从split文件加载了{len(subjects_info)}条{self.mode}分割的记录")
                    
                    if len(subjects_info) > 0:
                        # logger.info(f"第一个条目示例: {subjects_info[0]}")
            
                        # 构建样本列表
                        samples = []
                    
                        # 统计加载情况
                        total_count = 0
                        success_count = 0
                        
                        # 处理每个样本条目
                        for subject_info in subjects_info:
                            total_count += 1
                            try:
                                subject_id = subject_info.get('subject_id')
                                scene_id = subject_info.get('scene_id')
                                dir_path = subject_info.get('dir_path', '')
                                
                                # 记录当前处理的样本信息
                                # logger.info(f"处理样本: subject_id={subject_id}, scene_id={scene_id}, dir_path={dir_path}")
                            
                                # 尝试多种路径构建方式，优先使用简单直接的方式
                                paired_data_paths = []
                                
                                # 1. 最直接的路径 - 基于subject_id和scene_id
                                if subject_id and scene_id:
                                    direct_path = self.data_dir / 'paired' / subject_id / scene_id / 'paired_data.npz'
                                    paired_data_paths.append((direct_path, f"直接路径: {direct_path}"))
                                
                                # 2. 基于dir_path的路径
                                if dir_path:
                                    # 标准化路径分隔符
                                    norm_path = dir_path.replace('\\', '/')
                                    
                                    # 尝试提取相对路径
                                    if 'paired/' in norm_path:
                                        rel_path = norm_path.split('paired/')[1] if len(norm_path.split('paired/')) > 1 else norm_path
                                        path_from_rel = self.data_dir / 'paired' / rel_path / 'paired_data.npz'
                                        paired_data_paths.append((path_from_rel, f"相对路径: {path_from_rel}"))
                                    
                                    # 尝试从项目根目录构建
                                    if norm_path.startswith('data/'):
                                        path_from_root = Path(os.path.join(os.getcwd(), norm_path)) / 'paired_data.npz'
                                        paired_data_paths.append((path_from_root, f"根目录路径: {path_from_root}"))
                                    
                                    # 尝试直接添加到数据目录
                                    path_from_data_dir = self.data_dir / norm_path / 'paired_data.npz'
                                    paired_data_paths.append((path_from_data_dir, f"数据目录路径: {path_from_data_dir}"))
                                
                                # 尝试所有可能的路径
                                found_path = None
                                for path, desc in paired_data_paths:
                                    if path.exists():
                                        # logger.info(f"找到有效的数据路径: {desc}")
                                        found_path = path
                                        break
                                
                                # 处理找到的路径
                                if found_path is not None:
                                    # 成功找到数据路径，创建样本
                                    parent_dir = found_path.parent
                                    parts = list(parent_dir.parts)
                                    
                                    # 确保有subject_id和scene_id
                                    resolved_subject_id = subject_id or (parts[-2] if len(parts) >= 2 else "unknown")
                                    resolved_scene_id = scene_id or (parts[-1] if len(parts) >= 1 else "unknown")
                                    
                                    sample = {
                                        'data_path': str(found_path),
                                        'subject_id': resolved_subject_id,
                                        'scene_id': resolved_scene_id,
                                        'is_paired': True,
                                        'source_id': 'paired'
                                    }
                                    
                                    # logger.info(f"成功添加样本: subject_id={resolved_subject_id}, scene_id={resolved_scene_id}, path={found_path}")
                                    samples.append(sample)
                                    success_count += 1
                                else:
                                    logger.warning(f"未找到有效的数据路径: subject_id={subject_id}, scene_id={scene_id}, dir_path={dir_path}")
                                    # 记录所有尝试过的路径
                                    for _, desc in paired_data_paths:
                                        logger.warning(f"  尝试路径: {desc}")
                            
                            except Exception as e:
                                logger.error(f"处理样本时出错: {str(e)}")
                                import traceback
                                logger.error(f"异常堆栈: {traceback.format_exc()}")
                        
                        # 处理完所有样本后的报告
                        logger.info(f"样本加载完成: 总共{total_count}条记录，成功加载{success_count}个样本")
                        if not samples:
                            logger.warning(f"未能加载任何样本!")
                            logger.warning(f"尝试备用加载方法...")
                        else:
                            return samples
                else:
                    logger.warning(f"在分割文件中未找到{self.mode}模式的数据")
            except Exception as e:
                logger.error(f"加载数据集分割文件失败: {str(e)}")
                import traceback
                logger.error(f"异常堆栈: {traceback.format_exc()}")

        # 如果无法从split文件加载或未找到样本，尝试扫描paired目录
        logger.warning("尝试扫描paired目录以查找样本...")
        paired_dir = self.data_dir / "paired"
        
        if paired_dir.exists():
            # logger.info(f"找到paired目录: {paired_dir}")
            samples = []
            all_samples = []
        
            # 扫描所有主体
            subject_dirs = list(paired_dir.glob('*'))
            # logger.info(f"找到{len(subject_dirs)}个subject目录")
            
            for subject_dir in subject_dirs:
                if not subject_dir.is_dir():
                    continue
            
                subject_id = subject_dir.name
            
                # 扫描所有场景
                scene_dirs = list(subject_dir.glob('*'))
                # logger.info(f"在subject {subject_id}中找到{len(scene_dirs)}个scene目录")
                
                for scene_dir in scene_dirs:
                    if not scene_dir.is_dir():
                        continue
                
                    scene_id = scene_dir.name
                    paired_data_path = scene_dir / 'paired_data.npz'
                
                    if paired_data_path.exists():
                        sample = {
                            'data_path': str(paired_data_path),
                            'subject_id': subject_id,
                            'scene_id': scene_id,
                            'is_paired': True,
                            'source_id': 'paired'
                        }
                        all_samples.append(sample)
            
            # logger.info(f"通过目录扫描找到{len(all_samples)}个样本")
            
            # 简单地将样本分为训练集、验证集和测试集
            if all_samples:
                # 按照subject_id划分，确保同一个subject的数据在同一个集合中
                unique_subjects = sorted(list(set([s['subject_id'] for s in all_samples])))
                num_subjects = len(unique_subjects)
                # logger.info(f"找到{num_subjects}个唯一的subject_id")
            
                # 不同模式选择不同的subject
                if self.mode == 'train':
                    train_subjects = unique_subjects[:int(num_subjects * 0.72)]  # 训练集72%
                    samples = [s for s in all_samples if s['subject_id'] in train_subjects]
                    # logger.info(f"训练集有{len(samples)}个样本，涉及{len(train_subjects)}个subject")
                elif self.mode == 'val':
                    val_subjects = unique_subjects[int(num_subjects * 0.72):int(num_subjects * 0.80)]  # 验证集8%
                    samples = [s for s in all_samples if s['subject_id'] in val_subjects]
                    # logger.info(f"验证集有{len(samples)}个样本，涉及{len(val_subjects)}个subject")
                else:  # test
                    test_subjects = unique_subjects[int(num_subjects * 0.80):]  # 测试集20%
                    samples = [s for s in all_samples if s['subject_id'] in test_subjects]
                    # logger.info(f"测试集有{len(samples)}个样本，涉及{len(test_subjects)}个subject")
                
                if samples:
                    # logger.info(f"成功通过目录扫描为{self.mode}集加载了{len(samples)}个样本")
                    return samples
        
        # 如果以上所有方法都失败，返回空列表
        logger.error(f"无法加载样本，返回空列表")
        return []


def custom_collate_fn(batch):
    """自定义批处理合并函数，确保维度正确"""
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if not batch:
        logger.warning("批次中没有有效样本，返回默认空批次结构")
        # 返回一个最小的有效批次结构，而不是返回None
        # 这样可以防止训练循环中的NoneType错误
        return {
            'rgb_frames': torch.zeros((1, 75, 3, 128, 128)),
            'nir_frames': torch.zeros((1, 75, 1, 128, 128)),
            'vppg': torch.zeros((1, 75, 1)),
            'heart_rate': torch.tensor([70.0]),  # 默认心率值
            'breathing_rate': torch.tensor([15.0]),  # 默认呼吸率
            'subject_id': ['default'],
            'scene_id': ['default'],
            'source_id': ['default']
        }
    
    # 检查是否有GPU上的张量，这将影响pin_memory操作
    has_cuda_tensors = False
    for item in batch:
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                has_cuda_tensors = True
                break
        if has_cuda_tensors:
            break
    
    # 合并字典
    result = {}
    for key in batch[0].keys():
        if key in ['rgb_frames', 'nir_frames']:
            # 确保帧数据格式为[B, T, C, H, W]
            try:
                # 特殊处理NIR帧 - 处理None值的情况
                if key == 'nir_frames':
                    # 提取所有非空的NIR帧
                    valid_tensors = [item[key] for item in batch if key in item and item[key] is not None]
                    
                    # 检查是否所有NIR帧都是None
                    if not valid_tensors:
                        logger.info("批次中所有NIR帧都是None，设置整个批次的NIR帧为None")
                        result[key] = None
                        continue
                    
                    # 检查是否有一些样本的NIR帧是None
                    if len(valid_tensors) < len(batch):
                        logger.warning("批次中包含混合的NIR帧（部分为None），设置整个批次的NIR帧为None")
                        result[key] = None
                        continue
                    
                    # 所有NIR帧都有效，常规处理
                    tensors = valid_tensors
                else:
                    # 常规处理其它类型的帧
                    tensors = [item[key] for item in batch if key in item]
                
                if tensors:
                    # 检查张量的维度
                    if len(tensors[0].shape) == 4:  # [T, C, H, W]
                        try:
                            # 堆叠为[B, T, C, H, W]
                            stacked = torch.stack(tensors, dim=0)
                            result[key] = stacked
                        except Exception as e:
                            logger.warning(f"尝试堆叠{key}时出错: {e}，将尝试手动堆叠")
                            # 尝试手动调整并堆叠
                            if has_cuda_tensors:
                                cpu_tensors = []
                                for t in tensors:
                                    if t.is_cuda:
                                        cpu_tensors.append(t.cpu().unsqueeze(0))
                                    else:
                                        cpu_tensors.append(t.unsqueeze(0))
                                result[key] = torch.cat(cpu_tensors, dim=0)
                            else:
                                result[key] = torch.cat([t.unsqueeze(0) for t in tensors], dim=0)
                    else:
                        # 如果维度不对，记录警告并尝试手动堆叠
                        logger.warning(f"意外的{key}维度: {tensors[0].shape}")
                        # 尝试手动调整并堆叠
                        if has_cuda_tensors:
                            cpu_tensors = []
                            for t in tensors:
                                if t.is_cuda:
                                    cpu_tensors.append(t.cpu().unsqueeze(0))
                                else:
                                    cpu_tensors.append(t.unsqueeze(0))
                            result[key] = torch.cat(cpu_tensors, dim=0)
                        else:
                            result[key] = torch.cat([t.unsqueeze(0) for t in tensors], dim=0)
            except Exception as e:
                logger.error(f"在合并{key}时出错: {e}")
                # 创建空张量
                result[key] = torch.zeros((len(batch), 1, 3 if key == 'rgb_frames' else 1, 128, 128))
        elif key == 'vppg':
            # 处理vPPG信号
            try:
                tensors = [item[key] for item in batch if key in item]
                if tensors:
                    # 检查维度并标准化
                    standardized_tensors = []
                    for t in tensors:
                        # 将CUDA张量移回CPU
                        if t.is_cuda:
                            t = t.cpu()
                        
                        # 确保是2D张量 [T, 1]
                        if len(t.shape) == 1:  # 如果是[T]，则添加通道维度
                            t = t.unsqueeze(1)
                        elif len(t.shape) > 2:  # 如果维度超过2，尝试调整
                            # 假设最后一个维度是通道，其他是时间步
                            t = t.view(t.shape[0], -1)
                        
                        standardized_tensors.append(t)
                    
                    # 如果所有张量的时间步长不同，需要调整
                    seq_lengths = [t.shape[0] for t in standardized_tensors]
                    if len(set(seq_lengths)) > 1:
                        # 找到最短的序列长度
                        min_length = min(seq_lengths)
                        # 截断所有序列到这个长度
                        standardized_tensors = [t[:min_length] for t in standardized_tensors]
                    
                    # 堆叠张量为[B, T, 1]
                    result[key] = torch.stack(standardized_tensors, dim=0)
                    
                    # 记录关于vPPG信号的信息
                    logger.debug(f"合并后的vPPG信号形状: {result[key].shape}")
            except Exception as e:
                logger.error(f"在合并vPPG信号时出错: {e}")
                # 如果出错，创建一个默认的空vPPG张量
                # 使用第一个样本的序列长度作为默认长度
                seq_len = batch[0].get('rgb_frames', torch.zeros(75, 3, 128, 128)).shape[0]
                result[key] = torch.zeros((len(batch), seq_len, 1))
        elif isinstance(batch[0][key], torch.Tensor):
            # 其他张量数据
            try:
                tensors = [item[key] for item in batch if key in item]
                if tensors:
                    # 将CUDA张量移回CPU
                    if has_cuda_tensors:
                        cpu_tensors = []
                        for t in tensors:
                            if t.is_cuda:
                                cpu_tensors.append(t.cpu())
                            else:
                                cpu_tensors.append(t)
                        result[key] = torch.stack(cpu_tensors, dim=0)
                    else:
                        result[key] = torch.stack(tensors, dim=0)
            except Exception as e:
                logger.error(f"在合并{key}时出错: {e}")
        else:
            # 非张量数据
            result[key] = [item[key] for item in batch if key in item]
    
    # 添加特殊处理：确保元数据中包含每个样本的心率值
    # logger.info(f"[DEBUG] 开始处理心率值到元数据的传递。元数据结构: {result.keys()}")
    
    if 'heart_rate' in result:
        # logger.info(f"[DEBUG] heart_rate键类型: {type(result['heart_rate'])}")
        
        if isinstance(result['heart_rate'], torch.Tensor):
            # 获取每个样本的心率值，并确保其在元数据中可用
            # logger.info(f"[DEBUG] heart_rate张量形状: {result['heart_rate'].shape}")
            
            hr_tensor = result['heart_rate']
            # 处理张量以获取值列表
            if len(hr_tensor.shape) > 1 and hr_tensor.shape[1] > 0:
                # 多维张量，需要压缩
                hr_values = hr_tensor.squeeze().tolist()
                # logger.info(f"[DEBUG] 多维张量处理后的hr_values类型: {type(hr_values)}")
            else:
                # 一维张量
                hr_values = hr_tensor.tolist()
                # logger.info(f"[DEBUG] 一维张量的hr_values类型: {type(hr_values)}")
            
            # 确保是列表
            if not isinstance(hr_values, list):
                hr_values = [hr_values]  # 处理批次大小为1的情况
                # logger.info(f"[DEBUG] 转换为列表后的hr_values: {hr_values}")
            
            # 记录具体的心率值
            # logger.info(f"[DEBUG] 提取的心率值列表: {hr_values}")
                
            # 确保元数据中包含heart_rate列表
            if 'subject_id' in result and 'scene_id' in result:
                # 添加心率值到元数据中，确保每个样本有自己的心率值
                result['heart_rate_list'] = hr_values
                # logger.info(f"[DEBUG] 已添加heart_rate_list到元数据中，值: {hr_values}, 类型: {type(result['heart_rate_list'])}")
                
                # 添加一个标志来追踪是否正确传递
                result['hr_list_added'] = True
                
                # 确认heart_rate_list是否已添加到元数据中
                # logger.info(f"[DEBUG] 当前元数据的所有键: {result.keys()}")
        else:
            logger.info(f"[DEBUG] heart_rate不是张量，而是{type(result['heart_rate'])}，值: {result['heart_rate']}")
    else:
        logger.info(f"[DEBUG] 元数据中没有找到heart_rate键")

    
    return result

def custom_collate_fn(batch):
    """
    自定义collate函数，用于组合多个样本为一个批次
    
    Args:
        batch: 需要合并的样本列表
        
    Returns:
        合并后的批次字典
    """
    # 检查批次是否为空
    if len(batch) == 0:
        return None
    
    # 过滤掉None样本
    valid_samples = [sample for sample in batch if sample is not None]
    
    # 如果过滤后没有有效样本，返回None
    if len(valid_samples) == 0:
        return None
    
    # 将多个样本字典合并为一个字典，每个键对应一个批次张量
    result = {}
    
    # 获取第一个有效样本的所有键
    keys = valid_samples[0].keys()
    
    # 需要转换为张量的数值字段
    numeric_keys = ['heart_rate', 'respiration_rate', 'spo2', 'activity_label']
    
    for key in keys:
        # 对于每个键，收集所有有效样本的对应值
        values = [sample[key] for sample in valid_samples if key in sample]
        
        # 跳过空列表
        if not values:
            continue
            
        # 特殊处理NIR帧 - 处理None值
        if key == 'nir_frames':
            # 检查是否有None值
            has_none = any(v is None for v in values)
            if has_none:
                # 如果有None值，则整个批次设置为None
                logger.info(f"批次中NIR帧包含None值，设置整个批次的NIR帧为None")
                result[key] = None
                continue
                
        # 如果值是张量，使用torch.stack合并
        if isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values)
            except Exception as e:
                # 如果不能stack，可能是形状不一致，使用其他方法处理
                logger.warning(f"无法stack键 {key} 的值: {e}")
                # 如果是nir_frames且堆叠失败，设置为None
                if key == 'nir_frames':
                    logger.info(f"无法堆叠NIR帧，设置为None使用单模态RGB")
                    result[key] = None
                    continue
                # 尝试转换为张量
                try:
                    result[key] = torch.tensor(values)
                except:
                    # 如果仍然不能转换，保持为列表
                    result[key] = values
        elif key in numeric_keys:
            # 如果是数值类型字段，转换为张量
            try:
                # 确保值是数值类型
                numeric_values = []
                for v in values:
                    if isinstance(v, (int, float)):
                        numeric_values.append(v)
                    elif isinstance(v, torch.Tensor):
                        numeric_values.append(v.item())
                    elif isinstance(v, (list, tuple)) and len(v) == 1:
                        numeric_values.append(v[0])
                    else:
                        numeric_values.append(0.0)  # 默认值
                
                result[key] = torch.tensor(numeric_values, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"将{key}转换为张量时出错: {e}")
                # 如果出错，创建一个全零张量
                result[key] = torch.zeros(len(values), dtype=torch.float32)
        elif key == 'subject_id' or key == 'scene_id' or key == 'source_id' or key == 'video_id':
            # 对于字符串类型的元数据，直接保留为列表
            result[key] = values
        else:
            # 其他类型，尝试转换为张量
            try:
                result[key] = torch.tensor(values)
            except Exception:
                # 如果不能转换，保持为列表
                result[key] = values
    
    return result

def create_vipl_hr_dataloaders(
    data_dir: str,
    batch_size: int,
    sequence_length: int,
    use_nir: bool = True,
    use_vppg: bool = False,
    use_simclr: bool = False,
    source_id: int = None,
    scene_id: int = None,
    labels_dir: str = None,
    num_workers: int = 4,
    skip_missing_subjects: bool = True,
    use_gpu: bool = True
):
    """创建VIPL-HR数据集加载器

    参数:
        data_dir: 数据目录
        batch_size: 批大小
        sequence_length: 序列长度
        use_nir: 是否使用NIR
        use_vppg: 是否使用vPPG
        use_simclr: 是否使用SimCLR
        source_id: 指定的数据源ID
        scene_id: 指定的场景ID
        labels_dir: 标签目录
        num_workers: 数据加载线程数
        skip_missing_subjects: 是否跳过缺失主体
        use_gpu: 是否使用GPU

    返回:
        数据加载器字典
    """
    # 检查数据目录是否存在
    import os
    import logging
    import platform
    logger = logging.getLogger(__name__)
    
    logger.info(f"创建VIPL-HR数据加载器，参数:\n"
                f"- 数据目录: {data_dir}\n"
                f"- 批大小: {batch_size}\n"
                f"- 序列长度: {sequence_length}\n"
                f"- 使用NIR: {use_nir}\n"
                f"- 使用vPPG: {use_vppg}\n"
                f"- 使用SimCLR: {use_simclr}\n"
                f"- 源ID: {source_id}\n"
                f"- 场景ID: {scene_id}\n"
                f"- 标签目录: {labels_dir}\n"
                f"- 工作线程: {num_workers}\n"
                f"- 跳过缺失主体: {skip_missing_subjects}\n"
                f"- 使用GPU: {use_gpu}")
    
    # 获取绝对路径
    abs_data_dir = os.path.abspath(data_dir)
    logger.info(f"数据目录绝对路径: {abs_data_dir}")
    
    if not os.path.exists(abs_data_dir):
        logger.error(f"数据目录不存在: {abs_data_dir}")
        raise FileNotFoundError(f"数据目录不存在: {abs_data_dir}")
    
    # 列出数据目录内容，帮助调试
    try:
        logger.info(f"数据目录内容: {os.listdir(abs_data_dir)}")
    except Exception as e:
        logger.error(f"无法列出数据目录内容: {e}")
    
    # 检查子目录
    expected_subdirs = ['train', 'val', 'test']
    missing_subdirs = [subdir for subdir in expected_subdirs 
                      if not os.path.exists(os.path.join(abs_data_dir, subdir))]
    if missing_subdirs:
        logger.warning(f"缺少子目录: {missing_subdirs}")
    
    # 检查平台，Windows环境下调整工作线程
    if platform.system() == "Windows" and num_workers > 0:
        logger.info(f"Windows环境检测到，工作线程数从{num_workers}调整为0")
        num_workers = 0
    
    # 检查GPU可用性
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"数据加载器使用设备: {device}")
    
    # 由于我们使用了自定义的collate_fn，它会处理将CUDA张量移回CPU，
    # 因此这里将pin_memory关闭，避免当数据已经在GPU上时出错
    use_pin_memory = False  # 不使用pin_memory
    logger.info(f"使用pin_memory: {use_pin_memory}")
    print('hello world')
    
    # 使用配置中指定的批大小，不再自动调整
    logger.info(f"使用指定的批大小: {batch_size}")
    
    # 确保data_dir是Path对象
    try:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning(f"数据目录不存在: {data_dir}")
    except Exception as e:
        logger.error(f"无法将data_dir转换为Path对象: {e}")
    
    # 检查是否存在数据集分割文件
    dataset_split_file = data_dir / 'dataset_split.json'
    dataset_split = None
    if dataset_split_file.exists():
        try:
            with open(dataset_split_file, 'r') as f:
                dataset_split = json.load(f)
        except Exception as e:
            logger.error(f"加载数据集分割文件失败: {e}")
            dataset_split = None
    else:
        logger.warning(f"数据集分割文件不存在: {dataset_split_file}，将使用默认分割比例")
    
    # 如果没有指定labels_dir，尝试找到标签文件
    if not labels_dir:
        # 尝试多个可能的位置查找标签文件
        possible_label_paths = [
            Path('data/vlhr/processed/vipl_hr_labels.csv'),
            Path('data/processed_vipl/vipl_hr_labels.csv'),
            data_dir / 'vipl_hr_labels.csv',  # 直接查找数据目录下的标签文件
        ]
        
        # 过滤None值
        possible_label_paths = [p for p in possible_label_paths if p is not None]
        
        # 输出查找信息
        logger.info(f"查找标签文件，尝试以下路径:")
        for i, lp in enumerate(possible_label_paths):
            logger.info(f"  {i+1}. {lp} ({'存在' if lp.exists() else '不存在'})")
        
        # 查找第一个存在的标签文件
        for label_path in possible_label_paths:
            if label_path.exists():
                labels_dir = label_path.parent
                logger.info(f"发现标签文件: {label_path}")
                break
    
    # 确保labels_dir是Path对象或None
    if labels_dir:
        try:
            labels_dir = Path(labels_dir)
            if not labels_dir.exists():
                logger.warning(f"标签目录不存在: {labels_dir}")
                labels_dir = None
        except Exception as e:
            logger.error(f"无法将labels_dir转换为Path对象: {e}")
            labels_dir = None
    
    # 创建数据集
    try:
        datasets = {}
        modes = ['train', 'val', 'test']
        
        any_dataset_created = False
        
        for mode in modes:
            try:
                logger.info(f"开始创建{mode}数据集，使用数据目录: {data_dir}")
                dataset = VIPLHRDataset(
                    data_dir=data_dir,
                    mode=mode,
                    sequence_length=sequence_length,
                    overlap=0.5 if mode == 'train' else 0.0,
                    transform=None,
                    use_nir=use_nir,
                    use_vppg=use_vppg,
                    use_simclr=use_simclr and mode == 'train',
                    source_id=source_id,
                    scene_id=scene_id,
                    labels_dir=labels_dir,
                    skip_missing_subjects=skip_missing_subjects,
                    device=device
                )
                
                if len(dataset) > 0:
                    datasets[mode] = dataset
                    logger.info(f"{mode}数据集创建成功，共{len(dataset)}个样本")
                    any_dataset_created = True
                else:
                    # 如果没有找到任何数据，输出更多调试信息
                    logger.error(f"{mode}数据集为空，无法找到任何符合条件的数据")
                    logger.error(f"请检查以下情况:")
                    logger.error(f"1. 数据目录: {data_dir}，该目录是否存在: {data_dir.exists()}")
                    logger.error(f"2. source_id筛选条件: {source_id}")
                    logger.error(f"3. scene_id筛选条件: {scene_id}")
                    logger.error(f"4. 请检查数据目录结构，确保存在paired_data.npz文件")
                    
                    # 数据集为空，抛出异常
                    logger.error(f"{mode}数据集为空，无法完成训练")
                    raise FileNotFoundError(f"无法找到{mode}数据集，请检查数据路径和过滤条件")
            except Exception as e:
                logger.error(f"创建{mode}数据集时出错: {e}")
                logger.error(traceback.format_exc())
                
                # 出现异常，记录错误并抛出异常
                logger.error(f"创建{mode}数据集时出错: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"创建{mode}数据集时失败: {str(e)}")
        
        # 检查是否有创建成功的数据集
        if not any_dataset_created:
            logger.error("所有数据集创建失败，终止训练!")
            raise RuntimeError("无法加载任何数据集，请检查数据路径和过滤条件")
    
        # 创建数据加载器
        dataloaders = {}
        for mode, dataset in datasets.items():
            dataloaders[mode] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(mode == 'train'),
                num_workers=num_workers,
                pin_memory=use_pin_memory and not use_gpu,  # 只在CPU模式下使用pin_memory
                drop_last=(mode == 'train'),
                collate_fn=custom_collate_fn
            )
            logger.info(f"已创建{mode}数据加载器，批大小: {batch_size}")
        
        return dataloaders
    except Exception as e:
        logger.error(f"创建数据加载器时出错: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"无法创建数据加载器: {str(e)}")

# 如果直接运行此模块
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.WARNING)
    
    # 测试数据加载器
    data_dir = "data/vlhr"
    dataloaders = create_vipl_hr_dataloaders(data_dir, batch_size=4, use_vppg=True)
    
    # 打印数据集信息
    for mode, dataloader in dataloaders.items():
        logger.info(f"{mode}数据集大小: {len(dataloader.dataset)}")
        # 尝试获取一个批次
        try:
            batch = next(iter(dataloader))
            if batch:
                logger.info(f"{mode}批次示例:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape}, {value.dtype}")
                        logger.info(f"  {key}: {type(value)}")
        except Exception as e:
            logger.error(f"获取{mode}批次时出错: {e}") 