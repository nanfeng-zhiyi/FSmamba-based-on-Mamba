#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于个体（subject）的数据集分割工具

按照「将不同受试者划分到不同集合」的方式创建可复现数据集分割：
- 训练集：72%的受试者及其所有场景
- 验证集：8%的受试者及其所有场景
- 测试集：20%的受试者及其所有场景

确保同一个受试者的所有数据都划分到同一个集合中，即subject-disjoint的数据分割方案，避免数据泄漏。
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
import logging
import argparse
import random

# 设置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_statistics(data_dir):
    """
    分析数据集的统计特性，包括有多少受试者和每个受试者有多少场景
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        subject_scenes: 包含每个受试者场景信息的字典
    """
    # 获取paired目录下的所有npz文件
    paired_dir = Path(data_dir) / "paired"
    if not paired_dir.exists():
        logger.error(f"未找到paired目录: {paired_dir}")
        return {}
    
    # 分析目录结构
    subject_scenes = {}
    for subject_path in paired_dir.glob("*"):
        if not subject_path.is_dir():
            continue
            
        subject_id = subject_path.name
        scenes = []
        
        # 获取该受试者的所有场景
        for scene_path in subject_path.glob("*"):
            if not scene_path.is_dir():
                continue
                
            scene_id = scene_path.name
            paired_data_path = scene_path / "paired_data.npz"
            
            # 确认存在paired_data.npz文件
            if paired_data_path.exists():
                scenes.append(scene_id)
                
        if scenes:
            subject_scenes[subject_id] = scenes
    
    return subject_scenes

def create_subject_based_split(data_dir, seed=42, output_file=None):
    """
    创建基于个体的数据集分割
    
    Args:
        data_dir: 数据目录路径
        seed: 随机种子
        output_file: 输出文件路径，如果为None将使用默认路径
        
    Returns:
        分割后的数据集字典
    """
    # 固定随机种子确保可复现性
    random.seed(seed)
    np.random.seed(seed)
    
    # 获取所有受试者及其场景
    subject_scenes = analyze_dataset_statistics(data_dir)
    
    if not subject_scenes:
        logger.error("未找到有效的数据，无法创建分割")
        return None
        
    # 创建分割方案 - 每个条目都是一个字典而不是字符串
    split = {"train": [], "val": [], "test": []}
    
    # 获取所有受试者ID并随机打乱
    subject_ids = list(subject_scenes.keys())
    logger.info(f"找到{len(subject_ids)}个不同的受试者")
    
    # 随机打乱受试者列表
    np.random.shuffle(subject_ids)
    
    # 按照72:8:20的比例划分受试者
    total_subjects = len(subject_ids)
    train_size = int(total_subjects * 0.72)  # 72%的受试者给训练集
    val_size = int(total_subjects * 0.08)    # 8%的受试者给验证集
    # 剩余的都给测试集
    
    # 划分受试者
    train_subjects = subject_ids[:train_size]
    val_subjects = subject_ids[train_size:train_size+val_size]
    test_subjects = subject_ids[train_size+val_size:]
    
    logger.info(f"训练集: {len(train_subjects)}个体 ({len(train_subjects)/total_subjects*100:.1f}%)")
    logger.info(f"验证集: {len(val_subjects)}个体 ({len(val_subjects)/total_subjects*100:.1f}%)")
    logger.info(f"测试集: {len(test_subjects)}个体 ({len(test_subjects)/total_subjects*100:.1f}%)")
    
    # 记录统计信息
    stats = {"train_subjects": len(train_subjects), "val_subjects": len(val_subjects), "test_subjects": len(test_subjects),
             "train_scenes": 0, "val_scenes": 0, "test_scenes": 0}
    
    # 将受试者的所有场景加入对应的集合
    for subject_id, scenes in subject_scenes.items():
        target_split = None
        
        # 确定该受试者属于哪个集合
        if subject_id in train_subjects:
            target_split = "train"
            stats["train_scenes"] += len(scenes)
        elif subject_id in val_subjects:
            target_split = "val"
            stats["val_scenes"] += len(scenes)
        elif subject_id in test_subjects:
            target_split = "test"
            stats["test_scenes"] += len(scenes)
        else:
            logger.warning(f"受试者 {subject_id} 未分配到任何集合")
            continue
        
        # 将该受试者的所有场景加入对应的集合
        for scene in scenes:
            split[target_split].append({
                "subject_id": subject_id,
                "scene_id": scene,
                "path": f"paired/{subject_id}/{scene}"
            })
    
    # 记录统计信息
    logger.info(f"总共 {stats['train_subjects'] + stats['val_subjects'] + stats['test_subjects']} 个受试者")
    logger.info(f"训练集: {stats['train_scenes']} 个场景 ({stats['train_scenes'] / (stats['train_scenes'] + stats['val_scenes'] + stats['test_scenes']) * 100:.1f}%)")
    logger.info(f"验证集: {stats['val_scenes']} 个场景 ({stats['val_scenes'] / (stats['train_scenes'] + stats['val_scenes'] + stats['test_scenes']) * 100:.1f}%)")
    logger.info(f"测试集: {stats['test_scenes']} 个场景 ({stats['test_scenes'] / (stats['train_scenes'] + stats['val_scenes'] + stats['test_scenes']) * 100:.1f}%)")
    
    # 保存到文件
    if output_file is None:
        output_file = os.path.join(data_dir, "dataset_split.json")
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已将场景分割保存到: {output_file}")
    return split

def convert_to_scene_source_format(data_dir, split, output_file=None):
    """
    将字典格式的分割转换为full_id格式（包含source信息）
    
    Args:
        data_dir: 数据目录路径
        split: 原始分割字典
        output_file: 输出文件路径
        
    Returns:
        转换后的分割字典
    """
    logger.info("分割文件已经是字典格式，无需转换为full_id格式")
    
    # 如果仍需要保存一份legacy格式的文件：
    if output_file is None:
        output_file = os.path.join(data_dir, "dataset_split_full.json")
    
    # 创建一个兼容的旧格式（仅用于兼容旧代码）
    legacy_split = {"train": [], "val": [], "test": []}
    
    # 处理每个数据集
    for dataset_name in ["train", "val", "test"]:
        for item in split[dataset_name]:
            subject_id = item["subject_id"]
            scene_id = item["scene_id"]
            # 默认使用"s1"作为source_id
            full_id = f"{subject_id}_{scene_id}_s1"
            legacy_split[dataset_name].append(full_id)
    
    # 保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(legacy_split, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已将兼容格式的场景分割保存到: {output_file} (仅用于兼容旧代码)")
    return split

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="创建基于个体的数据集分割")
    parser.add_argument("--data_dir", type=str, default="data/processed_vipl", help="数据目录路径 [默认: data/processed_vipl]")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--full_format", action="store_true", help="同时生成完整格式的分割文件")
    
    args = parser.parse_args()
    
    logger.info(f"使用数据目录: {args.data_dir}")
    
    # 创建基于个体的分割
    split = create_subject_based_split(args.data_dir, args.seed, args.output)
    
    # 如果需要，转换为完整格式
    if args.full_format and split is not None:
        convert_to_scene_source_format(args.data_dir, split)

if __name__ == "__main__":
    main()
