#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIPL-HR数据集处理模块
专门用于处理VIPL-HR数据集的特定目录结构和数据格式
"""

import os
import re  # 添加re模块导入
import cv2
import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("vipl_hr_processor.log", encoding='utf-8', mode='w'),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class SimpleROIExtractor:
    """简化版ROI提取器，使用OpenCV级联分类器"""
    
    def __init__(self, face_detection_confidence=0.5):
        self.confidence = face_detection_confidence
        # 使用OpenCV内置的人脸检测器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.tracking_bbox = None
        logger.info("初始化简化版ROI提取器")
    
    def extract_face_roi(self, frame, frame_count=0, video_name="unknown", visualize=False, vis_dir=None):
        """从图像中提取面部ROI"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 如果检测到人脸
        if len(faces) > 0:
            # 使用第一个检测到的人脸
            x, y, w, h = faces[0]
            
            # 扩展ROI
            padding = int(w * 0.3)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # 提取ROI
            roi = frame[y1:y2, x1:x2]
            bbox = [x1, y1, x2-x1, y2-y1]
            self.tracking_bbox = bbox
            
            # 创建简单的关键点（仅用于保持API兼容性）
            landmarks = np.zeros((468, 3))
            
            return roi, bbox, "opencv", landmarks
        
        # 如果没有检测到人脸，但有跟踪框
        elif self.tracking_bbox is not None:
            x, y, w, h = self.tracking_bbox
            roi = frame[y:y+h, x:x+w]
            landmarks = np.zeros((468, 3))
            return roi, self.tracking_bbox, "tracking", landmarks
        
        # 如果没有检测到人脸，也没有跟踪框
        else:
            # 返回整个图像
            h, w = frame.shape[:2]
            landmarks = np.zeros((468, 3))
            return frame, [0, 0, w, h], "full_frame", landmarks

class ROIProcessor:
    """简化版的ROI提取器
    
    用于从视频帧中提取感兴趣区域(ROI)，通常是面部区域
    """
    def __init__(self, target_size=(128, 128)):
        """初始化ROI处理器
        
        参数:
            target_size: 输出ROI的目标大小 (宽, 高)
        """
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info(f"初始化ROI处理器，目标大小: {target_size}")

    def extract_roi(self, frame):
        """从帧中提取ROI
        
        参数:
            frame: 待处理的视频帧
            
        返回:
            处理后的ROI区域，如果未检测到则返回调整大小的原始帧
        """
        # 转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 如果检测到人脸，截取最大的人脸区域
        if len(faces) > 0:
            # 找到最大的人脸
            max_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = max_face
            
            # 添加边距（如果可能）
            padding = min(w, h) // 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # 截取ROI
            roi = frame[y:y+h, x:x+w]
        else:
            # 未检测到人脸，使用整个帧
            roi = frame
        
        # 调整大小
        return cv2.resize(roi, self.target_size)

class NIRProcessor:
    """NIR(近红外)视频处理器
    
    专门用于处理NIR视频，提取ROI
    """
    def __init__(self, target_size=(128, 128)):
        """初始化NIR处理器
        
        参数:
            target_size: 输出ROI的目标大小 (宽, 高)
        """
        self.target_size = target_size
        logger.info(f"初始化NIR处理器，目标大小: {target_size}")
    
    def extract_roi(self, frame):
        """从NIR帧中提取ROI
        
        对于NIR视频，我们使用简单的阈值和轮廓检测来尝试识别面部区域
        
        参数:
            frame: 待处理的NIR视频帧
            
        返回:
            处理后的ROI区域，如果未检测到则返回调整大小的原始帧
        """
        # 如果是彩色帧，转换为灰度
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 应用高斯模糊来减少噪音
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓，通常是面部区域
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # 如果轮廓太小，可能不是面部，使用整个帧
            if w * h < 0.01 * gray.shape[0] * gray.shape[1]:
                roi = frame
            else:
                # 添加边距（如果可能）
                padding = min(w, h) // 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                # 截取ROI
                roi = frame[y:y+h, x:x+w]
        else:
            # 未检测到轮廓，使用整个帧
            roi = frame
        
        # 调整大小
        return cv2.resize(roi, self.target_size)

class VIPLHRProcessor:
    """VIPL-HR数据集处理器"""
    
    def __init__(self, 
                input_dir: str = None,
                output_dir: str = None, 
                config_path: str = None,
                debug=False):
        """
        初始化VIPL-HR数据集处理器
        
        参数:
            input_dir: VIPL-HR数据集根目录，如果为None则从配置文件读取
            output_dir: 处理后数据输出目录，如果为None则从配置文件读取
            config_path: 配置文件路径
            debug: 是否开启调试模式
        """
        # 先加载配置(如果提供)
        self.config = {}
        if config_path:
            self._load_config(config_path)
            
        # 设置目录路径，优先使用命令行参数，其次使用配置文件
        if input_dir is None and 'input_dir' in self.config:
            input_dir = self.config.get('input_dir')
            logger.info(f"从配置文件加载输入目录: {input_dir}")
            
        if output_dir is None and 'output_dir' in self.config:
            output_dir = self.config.get('output_dir')
            logger.info(f"从配置文件加载输出目录: {output_dir}")
            
        # 确保目录路径有效
        if input_dir is None:
            input_dir = "data/vlhr"
            logger.warning(f"未指定输入目录，使用默认值: {input_dir}")
            
        if output_dir is None:
            output_dir = "data/vlhr/processed"
            logger.warning(f"未指定输出目录，使用默认值: {output_dir}")
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化ROI提取器
        self.roi_extractor = ROIProcessor()
        
        # 数据集信息
        self.dataset_info = {
            "subjects": [],
            "scenes": [],
            "sources": [],
            "videos": []
        }
        
        # 处理进度跟踪
        self.processed_count = 0
        self.total_count = 0
        
        # 调试模式
        self.debug = debug
    
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {config_path}")
            
            # 打印部分关键配置以便调试
            input_dir = self.config.get('input_dir', '未指定')
            output_dir = self.config.get('output_dir', '未指定')
            logger.info(f"配置文件中的输入目录: {input_dir}")
            logger.info(f"配置文件中的输出目录: {output_dir}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.config = {}
    
    def scan_dataset(self):
        """扫描VIPL-HR数据集，收集所有受试者、场景和视频信息"""
        logger.info(f"扫描VIPL-HR数据集: {self.input_dir}")
        
        if not self.input_dir.exists():
            logger.error(f"VIPL-HR数据集目录不存在: {self.input_dir}")
            return False
        
        try:
            logger.info("扫描目录内容...")
            
            # 用于存储视频信息的列表
            videos = []
            
            # 遍历主目录（主题）
            for subject_dir in os.listdir(self.input_dir):
                subject_path = os.path.join(self.input_dir, subject_dir)
                if not os.path.isdir(subject_path):
                    continue
                
                logger.info(f"发现目录: {subject_dir}")
                subject_scenes = os.listdir(subject_path)
                logger.info(f"  - 包含 {len(subject_scenes)} 个子项目")
                
                # 遍历场景目录
                for scene_dir in subject_scenes:
                    scene_path = os.path.join(subject_path, scene_dir)
                    if not os.path.isdir(scene_path):
                        continue
                    
                    logger.info(f"  - {scene_dir}")
                    
                    # 遍历源目录
                    for source_dir in os.listdir(scene_path):
                        source_path = os.path.join(scene_path, source_dir)
                        if not os.path.isdir(source_path):
                            continue
                        
                        # 判断是否为NIR数据
                        is_nir = source_dir.lower() == 'source4'
                        
                        # 查找视频文件
                        video_path = os.path.join(source_path, 'video.avi')
                        if not os.path.exists(video_path):
                            continue
                        
                        # 查找GT心率和血氧文件
                        hr_file = os.path.join(source_path, 'gt_HR.csv')
                        spo2_file = os.path.join(source_path, 'gt_SpO2.csv')
                        wave_file = os.path.join(source_path, 'wave.csv')
                        time_file = os.path.join(source_path, 'time.txt')
                        
                        # 读取心率和血氧数据
                        hr_values = self._safe_read_csv(Path(hr_file), default_value=None)
                        spo2_values = self._safe_read_csv(Path(spo2_file), default_value=None)
                        wave_values = self._extract_wave_from_csv(Path(wave_file))
                        time_values = self._extract_time_from_file(Path(time_file))
                        
                        # 计算平均值
                        if hr_values is not None and len(hr_values) > 0:
                            hr_mean = float(np.mean(hr_values))
                        else:
                            hr_mean = 75.0  # 默认值
                        
                        if spo2_values is not None and len(spo2_values) > 0:
                            spo2_mean = float(np.mean(spo2_values))
                        else:
                            spo2_mean = 98.0  # 默认值
                        
                        # 根据源判断是RGB还是NIR
                        modality = "NIR" if is_nir else "RGB"
                        logger.debug(f"发现{modality}视频: {subject_dir}/{scene_dir}/{source_dir}")
                        
                        # 添加视频信息
                        video_info = {
                            "subject_id": subject_dir,
                            "scene_id": scene_dir,
                            "source_id": source_dir,
                            "video_path": video_path,  # 添加视频文件路径
                            "hr_path": hr_file,
                            "spo2_path": spo2_file,
                            "wave_path": wave_file if os.path.exists(wave_file) else None,
                            "time_path": time_file if os.path.exists(time_file) else None,
                            "is_nir": is_nir,
                            "heart_rate": hr_mean,
                            "spo2": spo2_mean
                        }
                        
                        videos.append(video_info)
            
            # 保存数据集信息
            self.dataset_info = {
                "videos": videos,
                "total_count": len(videos),
                "rgb_count": sum(1 for v in videos if not v["is_nir"]),
                "nir_count": sum(1 for v in videos if v["is_nir"]),
                "subject_count": len(set(v["subject_id"] for v in videos)),
                "scene_count": len(set(f"{v['subject_id']}_{v['scene_id']}" for v in videos))
            }
            
            logger.info(f"扫描结果: {self.dataset_info['subject_count']} 个受试者, "
                     f"{self.dataset_info['scene_count']} 个场景, "
                     f"{len(set(v['source_id'] for v in videos))} 个源设备, "
                     f"共 {len(videos)} 个视频")
            
            return True
            
        except Exception as e:
            logger.error(f"扫描数据集时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_video(self, video_info, sample_rate=5, target_size=(128, 128), max_frames=None):
        """处理单个视频并提取ROI，保存到NPZ文件"""
        # 提取视频信息
        subject_id = video_info.get('subject_id', 'unknown')
        scene_id = video_info.get('scene_id', 'unknown')
        source_id = video_info.get('source_id', 'unknown')
        is_nir = video_info.get('is_nir', False)
        
        # 构建输出名称和目录
        output_name = f"{subject_id}_{scene_id}_{source_id}"
        output_dir = self.output_dir / output_name
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取视频路径
        video_path = video_info.get('video_path', '')
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False
        
        try:
            logger.info(f"开始处理视频: {output_name} ({'NIR' if is_nir else 'RGB'})")
            
            # 打开视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return False
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                logger.error(f"视频没有帧: {video_path}")
                cap.release()
                return False
            
            logger.info(f"视频信息: FPS={fps}, 总帧数={total_frames}")
            
            # 计算采样间隔和帧计数
            frame_interval = int(fps / sample_rate)
            frame_interval = max(1, frame_interval)  # 确保至少为1
            
            # 处理帧限制
            if max_frames is not None and max_frames > 0:
                frames_to_process = min(total_frames, max_frames * frame_interval)
            else:
                frames_to_process = total_frames
            
            # 准备ROI提取器
            if is_nir:
                # NIR数据使用不同的预处理
                processor = NIRProcessor(target_size=target_size)
            else:
                # RGB数据使用标准人脸检测
                processor = ROIProcessor(target_size=target_size)
            
            # 逐帧处理视频
            roi_frames = []
            frame_metadata = []
            current_frame = 0
            sampled_count = 0
            
            while current_frame < frames_to_process:
                # 设置当前帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"无法读取第 {current_frame} 帧，可能已到达视频末尾")
                    break
                
                # 提取ROI
                roi = processor.extract_roi(frame)
                if roi is not None:
                    # 保存ROI帧
                    roi_frames.append(roi)
                    
                    # 收集帧元数据
                    frame_meta = {
                        "frame_index": current_frame,
                        "timestamp": current_frame / fps if fps > 0 else 0,
                        "heart_rate": video_info.get("heart_rate", 75.0),
                        "spo2": video_info.get("spo2", 98.0)
                    }
                    frame_metadata.append(frame_meta)
                    
                    sampled_count += 1
                
                # 更新帧位置
                current_frame += frame_interval
            
            # 释放视频资源
            cap.release()
            
            if len(roi_frames) == 0:
                logger.error(f"未能从视频中提取任何有效ROI: {video_path}")
                return False
            
            logger.info(f"成功从视频中提取 {len(roi_frames)} 个ROI，采样率 {sample_rate} fps")
            
            # 合并为数组
            roi_array = np.array(roi_frames)
            
            # 保存处理后的数据
            npz_path = output_dir / f"{output_name}_data.npz"
            np.savez(
                npz_path,
                frames=roi_array,
                metadata=frame_metadata,
                fps=fps,
                sample_rate=sample_rate
            )
            
            # 保存元数据
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "subject_id": subject_id,
                    "scene_id": scene_id,
                    "source_id": source_id,
                    "is_nir": is_nir,
                    "fps": float(fps),
                    "original_frame_count": int(total_frames),
                    "processed_frame_count": len(roi_array),
                    "sample_rate": sample_rate,
                    "heart_rate": video_info.get("heart_rate", 75.0),
                    "spo2": video_info.get("spo2", 98.0)
                }, f, indent=2)
            
            logger.info(f"视频处理完成: {output_name}")
            return True
            
        except Exception as e:
            logger.error(f"处理视频时出错: {video_path}, 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def create_labels_file(self):
        """创建标签文件"""
        if not self.dataset_info["videos"]:
            logger.error("没有可用的视频信息，请先扫描数据集")
            return False
        
        # 创建标签CSV文件
        labels_file = self.output_dir / "vipl_hr_labels.csv"
        
        # 同时创建活动标签文件 - 为活动分类网络准备的格式
        activity_labels_dir = Path(self.output_dir.parent, "activity_labels")
        os.makedirs(activity_labels_dir, exist_ok=True)
        activity_labels_file = activity_labels_dir / "activity_labels.csv"
        
        logger.info(f"创建标签文件: {labels_file}")
        logger.info(f"创建活动标签文件: {activity_labels_file}")
        
        # 准备标签数据
        labels_data = []
        activity_labels_data = []
        
        valid_samples_count = 0
        for video_info in self.dataset_info["videos"]:
            # 检查是否是测试数据
            if video_info.get("is_test_data", False):
                logger.info("跳过测试数据，为真实数据创建标签")
                continue
                
            # 使用默认值，如果没有找到实际值
            heart_rate = video_info.get("heart_rate", 75.0)
            spo2 = video_info.get("spo2", 98.0)
            
            # 为基础的生理参数标签添加数据
            labels_data.append({
                "subject_id": video_info["subject_id"],
                "scene_id": video_info["scene_id"],
                "source_id": video_info["source_id"],
                "video_path": video_info["video_path"],
                "heart_rate": heart_rate,
                "spo2": spo2,
                "is_nir": 1 if video_info["is_nir"] else 0
            })
            
            # 构建活动分类所需的数据路径
            subject_id = video_info['subject_id'].replace('p', '')
            scene_id = video_info['scene_id'].replace('v', '')
            source_id = video_info['source_id'].replace('source', '')
            
            # 检查多种可能的数据文件路径

            # 1. 检查常规处理目录
            video_dir = f"p{subject_id}_v{scene_id}_source{source_id}"
            processed_path = self.output_dir / video_dir
            processed_data_paths = [
                processed_path / f"{video_dir}_data.npz",  # 新格式
                processed_path / "processed_data.npz",     # 旧格式
                processed_path / f"{video_dir}.npz"        # 另一种可能的格式
            ]
            
            # 2. 检查paired目录
            paired_path = self.output_dir / "paired" / f"p{subject_id}" / f"v{scene_id}"
            if paired_path.exists():
                processed_data_paths.append(paired_path / "paired_data.npz")
            
            # 如果任何一个路径存在对应的文件，则添加到活动标签中
            file_exists = any(path.exists() for path in processed_data_paths)
            
            if file_exists:
                # 基于心率确定活动状态
                # 心率 < 80 为静息，80-100 为中等活动，>100 为剧烈运动
                activity_label = "静息"  # 默认为静息
                if heart_rate > 100:
                    activity_label = "剧烈运动"
                elif heart_rate >= 80:
                    activity_label = "中等活动"
                
                # 为活动分类添加标签数据
                activity_labels_data.append({
                    "video_dir": video_dir,  # 使用目录名作为标识
                    "subject_id": video_info["subject_id"],
                    "video_id": video_info["scene_id"],
                    "source_id": video_info["source_id"],
                    "hr": heart_rate,  # 保留原始心率值
                    "activity_label": activity_label,  # 添加活动标签字段
                    "data_path": next((str(p) for p in processed_data_paths if p.exists()), "")  # 添加实际数据文件路径
                })
                
                valid_samples_count += 1
                
                # 可选：打印找到的有效文件路径，便于调试
                if self.debug:
                    for path in processed_data_paths:
                        if path.exists():
                            logger.debug(f"找到有效数据文件: {path}")
        
        # 如果没有找到任何真实数据，添加一个测试标签
        if not labels_data:
            logger.warning("没有找到真实数据标签，添加测试标签")
            labels_data.append({
                "subject_id": "test_subject",
                "scene_id": "v1",
                "source_id": "source1",
                "video_path": "test_video.avi",
                "heart_rate": 75.0,
                "spo2": 98.0,
                "is_nir": 0
            })
        
        # 写入基础标签CSV
        pd.DataFrame(labels_data).to_csv(labels_file, index=False)
        logger.info(f"标签文件已创建: {labels_file}，包含 {len(labels_data)} 条记录")
        
        # 写入活动标签CSV
        activity_df = pd.DataFrame(activity_labels_data)
        activity_df.to_csv(activity_labels_file, index=False)
        logger.info(f"活动标签文件已创建: {activity_labels_file}，包含 {len(activity_labels_data)} 条有效记录")
        
        # 检查类别分布
        if activity_labels_data:
            class_counts = activity_df['activity_label'].value_counts().to_dict()
            logger.info(f"活动标签类别分布: {class_counts}")
            
            # 警告类别不平衡问题
            min_class = min(class_counts.values())
            max_class = max(class_counts.values())
            if min_class / max_class < 0.2:  # 如果最小类别不到最大类别的20%
                logger.warning(f"类别严重不平衡! 最小类别样本数: {min_class}, 最大类别样本数: {max_class}")
                logger.warning("考虑采用过采样、欠采样或类权重平衡等技术处理类别不平衡问题")
        
        return True
    
    def create_dataset_split(self, train_ratio=0.6, val_ratio=0.2):
        """创建数据集分割（训练/验证/测试集）
        
        改进的数据集分割方法，确保：
        1. 所有数据集（训练/验证/测试）都有样本
        2. 不再按受试者ID分组，而是按单个样本划分，确保更均衡的分布
        3. 确保验证集不为空
        
        参数:
            train_ratio: 训练集比例，默认0.6
            val_ratio: 验证集比例，默认0.2
            （测试集比例 = 1 - train_ratio - val_ratio）
        """
        logger.info("创建数据集分割...")
        
        # 设置随机种子以便复现
        np.random.seed(42)
        
        # 获取已处理的目录列表
        processed_dirs = []
        paired_dir = self.output_dir / "paired"
        
        # 检查paired目录是否存在
        if paired_dir.exists():
            # 获取所有subject目录
            subject_dirs = [d for d in paired_dir.iterdir() if d.is_dir()]
            for subject_dir in subject_dirs:
                scene_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
                for scene_dir in scene_dirs:
                    if (scene_dir / "paired_data.npz").exists():
                        processed_dirs.append({
                            "subject_id": subject_dir.name,
                            "scene_id": scene_dir.name,
                            "dir_path": str(scene_dir).replace("\\", "/")
                        })
        else:
            # 如果没有paired目录，检查常规处理目录
            for dir_path in self.output_dir.iterdir():
                if dir_path.is_dir() and not dir_path.name.startswith('.'):
                    parts = dir_path.name.split('_')
                    if len(parts) >= 2:
                        subject_id = parts[0]
                        scene_id = parts[1]
                        
                        # 检查是否有处理好的数据文件
                        data_files = list(dir_path.glob("*_data.npz"))
                        if data_files:
                            processed_dirs.append({
                                "subject_id": subject_id,
                                "scene_id": scene_id,
                                "dir_path": str(dir_path).replace("\\", "/")
                            })
        
        logger.info(f"找到 {len(processed_dirs)} 个已处理的目录")
        
        if not processed_dirs:
            logger.warning("未找到任何已处理的目录，将使用测试数据")
            # 创建测试分割
            split = {
                "train": [{"subject_id": "test_subject_1", "scene_id": "v1"}],
                "val": [{"subject_id": "test_subject_2", "scene_id": "v1"}],
                "test": [{"subject_id": "test_subject_3", "scene_id": "v1"}]
            }
        else:
            # 随机打乱所有样本（不再按受试者ID分组）
            np.random.shuffle(processed_dirs)
            n_samples = len(processed_dirs)
            
            # 计算分割点
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # 确保验证集和测试集至少有1个样本
            if n_val == 0 and n_train > 1:
                n_train -= 1
                n_val = 1
            
            # 创建分割
            split = {
                "train": processed_dirs[:n_train],
                "val": processed_dirs[n_train:n_train + n_val],
                "test": processed_dirs[n_train + n_val:]
            }
            
            logger.info(f"数据集分割: 训练集 {len(split['train'])} 项, 验证集 {len(split['val'])} 项, 测试集 {len(split['test'])} 项")
        
        # 检查分割是否合理（确保验证集和测试集不为空）
        if len(split['val']) == 0 and len(split['train']) > 1:
            logger.warning("验证集为空，从训练集中分配样本到验证集")
            # 从训练集中取20%作为验证集
            n_move = max(1, int(len(split['train']) * 0.2))
            split['val'] = split['train'][:n_move]
            split['train'] = split['train'][n_move:]
        
        if len(split['test']) == 0 and (len(split['train']) > 1 or len(split['val']) > 1):
            logger.warning("测试集为空，分配样本到测试集")
            if len(split['train']) > 1:
                split['test'] = [split['train'].pop(0)]
            elif len(split['val']) > 1:
                split['test'] = [split['val'].pop(0)]
        
        # 保存分割文件
        split_file = self.output_dir / "dataset_split.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集分割: 训练集 {len(split['train'])} 项, 验证集 {len(split['val'])} 项, 测试集 {len(split['test'])} 项")
        logger.info(f"数据集分割文件已创建: {split_file}")
        return True
    
    def process_all_videos(self, sample_rate=5, target_size=(128, 128), max_frames=None):
        """处理所有视频，提取ROI并保存

        Args:
            sample_rate: 采样率，每秒多少帧
            target_size: ROI目标大小
            max_frames: 最大帧数，None表示不限制
        
        Returns:
            bool: 是否成功
        """
        # 允许单模态处理
        pair_processing = False  # 修改：默认不强制配对处理
        flexible_pairing = True  # 修改：添加灵活配对选项
        
        if not pair_processing and not flexible_pairing:
            logger.info("未启用配对处理模式，将分别处理所有RGB和NIR视频")
        elif pair_processing:
            logger.info("启用强制配对处理模式，仅处理有RGB和NIR配对的视频")
        else:
            logger.info("启用灵活配对处理模式，优先处理配对视频，同时允许处理单模态视频")
        
        if not self.dataset_info["videos"]:
            logger.error("数据集中没有视频信息")
            return False
        
        # 按主体、场景分组
        scene_videos = defaultdict(list)
        for video_info in self.dataset_info['videos']:
            scene_key = f"{video_info['subject_id']}_{video_info['scene_id']}"
            scene_videos[scene_key].append(video_info)
        
        processed_count = 0
        skipped_count = 0
        single_modality_count = 0  # 修改：添加单模态计数
        
        for scene_key, videos in scene_videos.items():
            # 分离RGB和NIR视频
            rgb_videos = [v for v in videos if not v.get('is_nir', False)]
            nir_videos = [v for v in videos if v.get('is_nir', False)]
            
            # 检查是否有配对
            has_pair = len(rgb_videos) > 0 and len(nir_videos) > 0
            
            if pair_processing and not has_pair:
                # 强制配对模式下，跳过未配对的场景
                logger.warning(f"场景 {scene_key} 缺少RGB或NIR视频，跳过处理")
                skipped_count += 1
                continue
            
            if has_pair:
                # 处理配对视频
                for rgb_info in rgb_videos:
                    rgb_status = self.process_video(
                        rgb_info, sample_rate, target_size, max_frames
                    )
                    if not rgb_status:
                        logger.error(f"处理RGB视频失败: {rgb_info.get('video_path', '未知路径')}")
                
                for nir_info in nir_videos:
                    nir_status = self.process_video(
                        nir_info, sample_rate, target_size, max_frames
                    )
                    if not nir_status:
                        logger.error(f"处理NIR视频失败: {nir_info.get('video_path', '未知路径')}")
                
                processed_count += 1
            elif flexible_pairing:
                # 灵活配对模式下，处理单模态视频
                if len(rgb_videos) > 0:
                    for rgb_info in rgb_videos:
                        rgb_status = self.process_video(
                            rgb_info, sample_rate, target_size, max_frames
                        )
                        if not rgb_status:
                            logger.error(f"处理RGB视频失败: {rgb_info.get('video_path', '未知路径')}")
                    single_modality_count += 1
                
                if len(nir_videos) > 0:
                    for nir_info in nir_videos:
                        nir_status = self.process_video(
                            nir_info, sample_rate, target_size, max_frames
                        )
                        if not nir_status:
                            logger.error(f"处理NIR视频失败: {nir_info.get('video_path', '未知路径')}")
                    single_modality_count += 1
            else:
                skipped_count += 1
                
        logger.info(f"视频处理完成。总共处理了 {processed_count} 个配对场景，{single_modality_count} 个单模态场景，跳过 {skipped_count} 个场景")
        return True
    
    def merge_rgb_nir_data(self):
        """合并处理完的RGB和NIR数据
        
        将对应的RGB和NIR数据合并到同一个文件中，存放在 <output_dir>/paired/<subject_id>/<scene_id> 目录下
        
        Returns:
            bool: 是否成功
        """
        # 创建配对数据目录
        paired_dir = self.output_dir / "paired"
        paired_dir.mkdir(exist_ok=True)
        
        # 查找所有已处理的RGB和NIR数据
        processed_data = defaultdict(dict)
        
        for dirpath, dirnames, filenames in os.walk(self.output_dir):
            for filename in filenames:
                if not filename.endswith('_data.npz'):
                    continue
                
                filepath = Path(dirpath) / filename
                # 解析文件名获取主体ID、场景ID和源ID
                match = re.search(r'p(\d+)_v(\d+)_source(\d+)_data\.npz', filename)
                if not match:
                    continue
                
                subject_id, scene_id, source_id = match.groups()
                scene_key = f"p{subject_id}_v{scene_id}"
                
                # 判断是RGB还是NIR
                is_nir = False
                for video_info in self.dataset_info['videos']:
                    if (video_info['subject_id'] == f"p{subject_id}" and 
                        video_info['scene_id'] == f"v{scene_id}" and 
                        video_info['source_id'] == f"source{source_id}"):
                        is_nir = video_info.get('is_nir', False)
                        break
                
                if is_nir:
                    processed_data[scene_key]['nir'] = filepath
                else:
                    processed_data[scene_key]['rgb'] = filepath
        
        # 合并数据
        success_count = 0
        skip_count = 0
        single_modality_count = 0  # 修改：添加单模态处理计数

        for scene_key, data_files in processed_data.items():
            # 提取主体ID和场景ID
            match = re.match(r'p(\d+)_v(\d+)', scene_key)
            if not match:
                continue
            
            subject_id, scene_id = match.groups()
            
            # 创建主体目录
            subject_dir = paired_dir / f"p{subject_id}"
            subject_dir.mkdir(exist_ok=True)
            
            # 创建场景目录
            scene_dir = subject_dir / f"v{scene_id}"
            scene_dir.mkdir(exist_ok=True)
            
            # 输出文件路径
            output_file = scene_dir / "paired_data.npz"
            
            has_rgb = 'rgb' in data_files
            has_nir = 'nir' in data_files
            
            # 检查是否有RGB和NIR数据
            if has_rgb and has_nir:
                # 加载RGB数据
                rgb_path = data_files['rgb']
                try:
                    rgb_data = np.load(rgb_path, allow_pickle=True)
                    rgb_frames = rgb_data['frames']
                    rgb_metadata = rgb_data['metadata']
                except Exception as e:
                    logger.error(f"加载RGB数据失败: {rgb_path}, 错误: {e}")
                    skip_count += 1
                    continue
                
                # 加载NIR数据
                nir_path = data_files['nir']
                try:
                    nir_data = np.load(nir_path, allow_pickle=True)
                    nir_frames = nir_data['frames']
                    nir_metadata = nir_data['metadata']
                except Exception as e:
                    logger.error(f"加载NIR数据失败: {nir_path}, 错误: {e}")
                    skip_count += 1
                    continue
                
                # 确保帧数相同，如不同则取最小值
                min_frames = min(len(rgb_frames), len(nir_frames))
                if len(rgb_frames) != len(nir_frames):
                    logger.warning(f"RGB和NIR帧数不一致: RGB={len(rgb_frames)}, NIR={len(nir_frames)}，截取为{min_frames}帧")
                    rgb_frames = rgb_frames[:min_frames]
                    nir_frames = nir_frames[:min_frames]
                    
                    # 调整元数据
                    rgb_metadata = rgb_metadata[:min_frames]
                    nir_metadata = nir_metadata[:min_frames]
                
                # 合并元数据：平均心率和SpO2
                hr_values_rgb = [meta.get('heart_rate', 0) for meta in rgb_metadata]
                hr_values_nir = [meta.get('heart_rate', 0) for meta in nir_metadata]
                spo2_values_rgb = [meta.get('spo2', 0) for meta in rgb_metadata]
                spo2_values_nir = [meta.get('spo2', 0) for meta in nir_metadata]
                
                # 过滤掉0值（缺失值）
                hr_values_rgb = [v for v in hr_values_rgb if v > 0]
                hr_values_nir = [v for v in hr_values_nir if v > 0]
                spo2_values_rgb = [v for v in spo2_values_rgb if v > 0]
                spo2_values_nir = [v for v in spo2_values_nir if v > 0]
                
                # 计算平均值，如果没有有效值则为0
                hr_mean_rgb = np.mean(hr_values_rgb) if hr_values_rgb else 0
                hr_mean_nir = np.mean(hr_values_nir) if hr_values_nir else 0
                spo2_mean_rgb = np.mean(spo2_values_rgb) if spo2_values_rgb else 0
                spo2_mean_nir = np.mean(spo2_values_nir) if spo2_values_nir else 0
                
                # 取两个值的平均作为最终值
                paired_hr = (hr_mean_rgb + hr_mean_nir) / 2 if (hr_mean_rgb > 0 and hr_mean_nir > 0) else (hr_mean_rgb or hr_mean_nir)
                paired_spo2 = (spo2_mean_rgb + spo2_mean_nir) / 2 if (spo2_mean_rgb > 0 and spo2_mean_nir > 0) else (spo2_mean_rgb or spo2_mean_nir)
                
                # 保存合并数据
                np.savez(
                    output_file,
                    rgb_frames=rgb_frames,
                    nir_frames=nir_frames,
                    heart_rate=paired_hr,
                    spo2=paired_spo2
                )
                
                # 保存元数据
                with open(scene_dir / "metadata.json", 'w') as f:
                    json.dump({
                        'subject_id': f"p{subject_id}",
                        'scene_id': f"v{scene_id}",
                        'frame_count': min_frames,
                        'heart_rate': paired_hr,
                        'spo2': paired_spo2,
                        'rgb_source': Path(rgb_path).name,
                        'nir_source': Path(nir_path).name
                    }, f, indent=2)
                
                success_count += 1
            
            # 处理单模态数据 - 将单一模态数据保存到paired目录，但标记为单模态
            elif has_rgb or has_nir:
                modality = 'rgb' if has_rgb else 'nir'
                
                try:
                    # 加载数据
                    data_path = data_files[modality]
                    data = np.load(data_path, allow_pickle=True)
                    frames = data['frames']
                    metadata = data['metadata']
                    
                    # 提取元数据
                    hr_values = [meta.get('heart_rate', 0) for meta in metadata]
                    spo2_values = [meta.get('spo2', 0) for meta in metadata]
                    
                    # 过滤掉0值
                    hr_values = [v for v in hr_values if v > 0]
                    spo2_values = [v for v in spo2_values if v > 0]
                    
                    # 计算平均值
                    hr_mean = np.mean(hr_values) if hr_values else 0
                    spo2_mean = np.mean(spo2_values) if spo2_values else 0
                    
                    # 保存数据 - 创建空的另一模态数据
                    if modality == 'rgb':
                        dummy_nir = np.zeros((1, 1, 1), dtype=np.uint8)  # 创建一个1x1x1的空数组作为占位符
                        np.savez(
                            output_file,
                            rgb_frames=frames,
                            nir_frames=dummy_nir,  # 空NIR帧
                            heart_rate=hr_mean,
                            spo2=spo2_mean,
                            single_modality=True,
                            modality=modality
                        )
                    else:  # nir
                        dummy_rgb = np.zeros((1, 1, 1), dtype=np.uint8)  # 创建一个1x1x1的空数组作为占位符
                        np.savez(
                            output_file,
                            rgb_frames=dummy_rgb,  # 空RGB帧
                            nir_frames=frames,
                            heart_rate=hr_mean,
                            spo2=spo2_mean,
                            single_modality=True,
                            modality=modality
                        )
                    
                    # 保存元数据
                    with open(scene_dir / "metadata.json", 'w') as f:
                        json.dump({
                            'subject_id': f"p{subject_id}",
                            'scene_id': f"v{scene_id}",
                            'frame_count': len(frames),
                            'heart_rate': hr_mean,
                            'spo2': spo2_mean,
                            f'{modality}_source': Path(data_path).name,
                            'single_modality': True,
                            'modality': modality
                        }, f, indent=2)
                    
                    single_modality_count += 1
                    
                except Exception as e:
                    logger.error(f"处理单模态数据失败: {data_files}, 错误: {e}")
                    skip_count += 1
            else:
                skip_count += 1
        
        logger.info(f"数据合并完成。成功合并: {success_count}, 单模态: {single_modality_count}, 跳过: {skip_count}")
        
        # 返回结果
        if success_count > 0 or single_modality_count > 0:
            return True
        else:
            logger.error("合并数据失败")
            return False
    
    def run_pipeline(self, sample_rate=5, target_size=(128, 128), max_frames=None, 
                train_ratio=0.7, val_ratio=0.15):
        """运行完整的数据处理流程"""
        try:
            # 1. 扫描数据集
            logger.info("第1步：扫描数据集")
            if not self.scan_dataset():
                logger.error("扫描数据集失败")
                return False
            
            # 2. 处理视频
            logger.info("第2步：处理视频")
            if not self.process_all_videos(sample_rate, target_size, max_frames):
                logger.error("处理视频失败")
                return False
            
            # 3. 合并RGB和NIR数据
            logger.info("第3步：合并RGB和NIR数据")
            if not self.merge_rgb_nir_data():
                logger.warning("合并数据有部分失败，但会继续处理")  # 修改：降级为warning，允许继续处理
            
            # 4. 创建标签文件
            logger.info("第4步：创建标签文件")
            if not self.create_labels_file():
                logger.error("创建标签文件失败")
                return False
            
            # 5. 创建数据集分割
            logger.info("第5步：创建数据集分割")
            if not self.create_dataset_split(train_ratio, val_ratio):
                logger.error("创建数据集分割失败")
                return False
            
            logger.info("数据处理流程完成！")
            return True
            
        except Exception as e:
            logger.error(f"处理流程出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _safe_read_csv(self, file_path, default_value=None, column_index=0):
        """安全读取CSV文件，处理各种可能的错误
        
        参数:
            file_path: CSV文件路径
            default_value: 如果读取失败，返回的默认值
            column_index: 要读取的列索引
            
        返回:
            numpy数组或默认值
        """
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return default_value
            
        try:
            # 首先尝试检查文件内容，确定是否有标题行
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                has_header = False
                # 检查第一行是否包含非数字内容
                if first_line:
                    parts = first_line.split(',')
                    if len(parts) > column_index:
                        try:
                            float(parts[column_index])
                        except ValueError:
                            # 如果无法转换为浮点数，说明可能是标题行
                            has_header = True
                            logger.info(f"检测到CSV文件有标题行: {file_path}")
            
            # 读取CSV文件，根据是否有标题行决定是否跳过第一行
            if has_header:
                df = pd.read_csv(file_path, header=0)
                
                # 如果列名是数字索引，需要正确处理列索引
                if len(df.columns) > column_index:
                    # 使用列名获取数据
                    values = df.iloc[:, column_index].astype(float).values
                else:
                    logger.warning(f"CSV文件列数不足: {file_path}，列数={len(df.columns)}，请求列索引={column_index}")
                    return default_value
            else:
                # 无标题行，直接读取
                df = pd.read_csv(file_path, header=None)
                
                # 检查列数是否足够
                if df.shape[1] <= column_index:
                    logger.warning(f"CSV文件列数不足: {file_path}，列数={df.shape[1]}，请求列索引={column_index}")
                    return default_value
                    
                values = df.iloc[:, column_index].astype(float).values
            
            # 检查数据是否为空
            if df.empty or len(values) == 0:
                logger.warning(f"CSV文件为空或不包含有效数据: {file_path}")
                return default_value
                
            return values
            
        except Exception as e:
            logger.warning(f"读取CSV文件失败 {file_path}: {e}")
            # 尝试另一种读取方式，读取后跳过第一行
            try:
                # 直接使用numpy读取，跳过第一行
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                if data.ndim > 1 and data.shape[1] > column_index:
                    return data[:, column_index]
                elif data.ndim == 1 and column_index == 0:
                    return data
                else:
                    logger.warning(f"无法使用备选方法读取CSV数据: {file_path}")
                    return default_value
            except Exception as e2:
                import traceback
                logger.debug(f"备选读取方法也失败: {e2}")
                logger.debug(traceback.format_exc())
                return default_value

    def _extract_wave_from_csv(self, wave_file):
        """从CSV文件中提取波形数据
        
        参数:
            wave_file: 波形CSV文件路径
            
        返回:
            波形数据或None
        """
        if not wave_file.exists():
            logger.warning(f"波形文件不存在: {wave_file}")
            return None
            
        try:
            # 首先尝试使用安全读取函数
            wave_values = self._safe_read_csv(wave_file, default_value=None)
            if wave_values is not None and len(wave_values) > 0:
                logger.info(f"成功读取波形数据，长度: {len(wave_values)}")
                return wave_values
                
            # 如果失败，尝试不同的读取方式
            logger.warning(f"标准方法读取波形失败，尝试替代方法: {wave_file}")
            
            # 尝试直接读取所有内容
            try:
                with open(wave_file, 'r') as f:
                    lines = f.readlines()
                
                # 跳过可能的标题行
                data_lines = []
                for line in lines[1:] if len(lines) > 1 else lines:
                    try:
                        # 尝试转换为浮点数
                        value = float(line.strip())
                        data_lines.append(value)
                    except ValueError:
                        # 忽略无法转换的行
                        continue
                
                if data_lines:
                    wave_values = np.array(data_lines)
                    logger.info(f"使用替代方法成功读取波形数据，长度: {len(wave_values)}")
                    return wave_values
                else:
                    logger.warning(f"无法从文件中提取有效的波形数据: {wave_file}")
                    return None
            except Exception as e:
                logger.warning(f"替代方法读取波形失败: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"读取波形文件时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _extract_time_from_file(self, time_file):
        """从时间文件中提取时间数据
        
        参数:
            time_file: 时间文件路径
            
        返回:
            时间数据数组或None
        """
        if not time_file.exists():
            logger.warning(f"时间文件不存在: {time_file}")
            return None
            
        try:
            # 尝试读取时间文件
            with open(time_file, 'r') as f:
                lines = f.readlines()
            
            # 提取时间值
            time_values = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 尝试转换为浮点数
                    value = float(line)
                    time_values.append(value)
                except ValueError:
                    # 如果第一行是标题，忽略它
                    if len(time_values) == 0 and (line.lower() == 'time' or not line[0].isdigit()):
                        logger.info(f"跳过时间文件中的标题行: {line}")
                        continue
                    else:
                        logger.warning(f"无法解析时间值: {line}")
            
            if time_values:
                return np.array(time_values)
            else:
                logger.warning(f"时间文件中没有有效数据: {time_file}")
                return None
                
        except Exception as e:
            logger.warning(f"读取时间文件时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

def main():
    """主函数"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="VIPL-HR数据集处理工具")
    parser.add_argument("--input_dir", type=str, default="data/vlhr", help="VIPL-HR数据集根目录，默认data/vlhr")
    parser.add_argument("--output_dir", type=str, default="data/processed_vipl", help="处理后数据输出目录，默认data/vlhr/processed")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--sample_rate", type=int, default=5, help="视频帧采样率")
    parser.add_argument("--target_size", type=int, default=128, help="目标ROI大小")
    parser.add_argument("--max_frames", type=int, help="每个视频最大处理帧数")
    
    args = parser.parse_args()
    
    logger.info("启动VIPL-HR数据集处理...")
    # 加载配置（可选）
    config = {}
    if args.config:
        logger.info(f"使用配置文件: {args.config}")
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"成功加载配置文件: {args.config}")
                if 'input_dir' in config:
                    logger.info(f"配置文件中的输入目录: {config['input_dir']}")
                if 'output_dir' in config:
                    logger.info(f"配置文件中的输出目录: {config['output_dir']}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，将使用默认或命令行参数")
    
    # 获取目录路径，优先使用命令行参数
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # 如果未通过命令行指定，使用配置文件中的值
    if input_dir is None and 'input_dir' in config:
        input_dir = config['input_dir']
        logger.info(f"从配置文件加载输入目录: {input_dir}")
    
    if output_dir is None and 'output_dir' in config:
        output_dir = config['output_dir']
        logger.info(f"从配置文件加载输出目录: {output_dir}")
    
    # 显示最终使用的路径
    logger.info(f"使用输入目录: {input_dir}")
    logger.info(f"使用输出目录: {output_dir}")
    logger.info(f"采样率: {args.sample_rate}")
    logger.info(f"目标大小: {args.target_size}x{args.target_size}")
    
    # 创建处理器
    processor = VIPLHRProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=args.config,
        debug=False
    )
    
    # 运行处理流程
    success = processor.run_pipeline(
        sample_rate=args.sample_rate,
        target_size=(args.target_size, args.target_size),
        max_frames=args.max_frames
    )
    
    if success:
        logger.info("VIPL-HR数据集处理完成")
        return 0
    else:
        logger.error("VIPL-HR数据集处理失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 