#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统监控标签页模块
显示系统静态信息和实时监控数据
"""

import platform
import time
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QScrollArea, QGridLayout, 
    QSizePolicy, QFrame, QTabWidget,
    QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPalette, QColor, QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
# 添加中文字体支持
import matplotlib.font_manager as fm
# 检查是否有中文字体
chinese_fonts = [f.name for f in fm.fontManager.ttflist if '黑体' in f.name or '宋体' in f.name or 'Microsoft YaHei' in f.name]
# 设置默认字体
if chinese_fonts:
    matplotlib.rcParams['font.family'] = chinese_fonts[0]
else:
    # 如果没有找到中文字体，就使用无衬线字体
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from ...utils.ollama_client import OllamaClient
from ...utils.system_monitor import SystemMonitor
from ...utils.gpuInfo import GPUInfoCollector


class MetricProgressBar(QWidget):
    """指标进度条显示类"""
    
    def __init__(self, parent=None, title="", color="#3498db"):
        """
        初始化进度条指标显示
        
        参数:
            parent: 父级部件
            title: 指标标题
            color: 进度条颜色
        """
        super().__init__(parent)
        
        # 设置布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        
        # 添加标题标签
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.title_label)
        
        # 创建进度条和值标签横向布局
        self.bar_layout = QHBoxLayout()
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 5px;
            }}
        """)
        self.bar_layout.addWidget(self.progress_bar, 4)  # 进度条占4份空间
        
        # 添加值标签
        self.value_label = QLabel("0%")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.value_label.setMinimumWidth(50)  # 设置最小宽度确保数值显示完整
        self.bar_layout.addWidget(self.value_label, 1)  # 值标签占1份空间
        
        # 添加进度条布局到主布局
        self.layout.addLayout(self.bar_layout)
        
        # 存储当前值
        self.current_value = 0
    
    def update_value(self, value: float):
        """
        更新进度条和显示值
        
        参数:
            value: 新的指标值
        """
        # 更新当前值
        self.current_value = value
        
        # 更新进度条
        self.progress_bar.setValue(int(value))
        
        # 更新值标签
        self.value_label.setText(f"{value:.1f}%")


class SystemMonitorTab(QWidget):
    """系统监控标签页类"""
    
    def __init__(self, ollama_client: OllamaClient, system_monitor: SystemMonitor):
        """
        初始化系统监控标签页
        
        参数:
            ollama_client: Ollama API客户端
            system_monitor: 系统监控器
        """
        super().__init__()
        
        # 保存引用
        self.client = ollama_client
        self.monitor = system_monitor
        
        # 初始化GPU信息收集器
        self.gpu_collector = GPUInfoCollector()
        
        # 初始化UI
        self._init_ui()
        
        # 设置定时器更新监控数据
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_metrics)
        
        # 初始化静态系统信息
        self._update_static_info()
        
        # 应用启动时就开始实时监控
        self._update_metrics()
        self.update_timer.start(1000)  # 每秒更新一次
    
    def _init_ui(self):
        """初始化UI元素"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # 创建顶部静态信息区域
        info_group = QGroupBox("系统信息")
        self.gpu_info_layout = QGridLayout(info_group)
        self.gpu_info_layout.setColumnStretch(1, 1)  # 让第二列自动拉伸
        self.gpu_info_layout.setColumnStretch(3, 1)  # 让第四列自动拉伸
        
        # 系统信息标签
        self.os_label = QLabel("操作系统: 加载中...")
        self.os_label.setWordWrap(True)
        
        self.cpu_label = QLabel("CPU: 加载中...")
        self.cpu_label.setWordWrap(True)
        
        self.ram_label = QLabel("内存: 加载中...")
        
        self.gpu_labels = []  # 存储GPU标签的列表
        
        self.python_label = QLabel("Python版本: 加载中...")
        self.ollama_label = QLabel("Ollama版本: 加载中...")
        self.network_label = QLabel("实时网速: 上传 0 KB/s | 下载 0 KB/s")
        
        # 使用网格布局排列标签
        self.gpu_info_layout.addWidget(self.os_label, 0, 0, 1, 4)  # 操作系统占整行
        
        # CPU和内存信息
        self.gpu_info_layout.addWidget(self.cpu_label, 1, 0, 1, 2)  # CPU信息占左半部分
        self.gpu_info_layout.addWidget(self.ram_label, 1, 2, 1, 2)  # 内存信息占右半部分
        
        # 预留GPU信息的位置 - 会在_update_static_info中动态添加
        self.gpu_info_row = 2  # 从第3行开始添加GPU信息
        
        # 注意：网速信息将移动到GPU后面，在_update_static_info中处理
        
        # Python和Ollama版本
        self.gpu_info_layout.addWidget(self.python_label, 100, 0, 1, 2)  # 使用较大的行号确保在最后
        self.gpu_info_layout.addWidget(self.ollama_label, 100, 2, 1, 2)
        
        # 添加到主布局
        main_layout.addWidget(info_group)
        
        # 创建实时监控区域
        metrics_group = QGroupBox("实时监控")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # 创建所有监控指标的容器
        self.all_metrics_widget = QWidget()
        all_metrics_layout = QVBoxLayout(self.all_metrics_widget)
        all_metrics_layout.setSpacing(15)
        
        # CPU使用率指标
        cpu_group = QGroupBox("CPU使用率")
        cpu_layout = QVBoxLayout(cpu_group)
        self.cpu_metric = MetricProgressBar(self, "CPU使用率", "#e74c3c")  # 红色
        cpu_layout.addWidget(self.cpu_metric)
        all_metrics_layout.addWidget(cpu_group)
        
        # 内存使用率指标
        memory_group = QGroupBox("内存使用率")
        memory_layout = QVBoxLayout(memory_group)
        self.memory_metric = MetricProgressBar(self, "内存使用率", "#9b59b6")  # 紫色
        memory_layout.addWidget(self.memory_metric)
        all_metrics_layout.addWidget(memory_group)
        
        # 创建GPU监控区域
        self.gpu_metrics_layout = QVBoxLayout()
        self.gpu_placeholder = QLabel("未检测到GPU或GPU监控不可用")
        self.gpu_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gpu_metrics_layout.addWidget(self.gpu_placeholder)
        all_metrics_layout.addLayout(self.gpu_metrics_layout)
        
        # 设置占位的拉伸因子
        all_metrics_layout.addStretch(1)
        
        # 添加所有指标布局到指标组
        metrics_layout.addWidget(self.all_metrics_widget)
        
        # 添加监控区域到主布局
        main_layout.addWidget(metrics_group, 1)  # 给监控区域更多空间
        
        # 设置布局
        self.setLayout(main_layout)
    
    def custom_round(self, x):
        """显存数值取整函数"""
        integer_part = int(x)
        decimal_part = x - integer_part
        return integer_part + (1 if decimal_part > 0.5 else 0)

    def _update_static_info(self):
        """更新静态系统信息"""
        # 获取系统信息
        info = self.monitor.get_system_info()
        
        # 更新标签
        self.os_label.setText(f"操作系统: {info.os_name} {info.os_version}")
        self.cpu_label.setText(f"CPU: {info.cpu_brand} ({info.cpu_cores}核)")
        self.ram_label.setText(f"内存: {info.memory_total:.2f} GB")
        
        # 清空旧的GPU标签
        for label in self.gpu_labels:
            label.setParent(None)
        self.gpu_labels = []
        
        # 获取GPU信息，包括NVIDIA和非NVIDIA显卡
        nvidia_gpus = []
        non_nvidia_gpus = []
        
        # 首先检查系统监控器中的NVIDIA GPU信息
        for i, gpu in enumerate(info.gpu_info):
            if gpu.get("driver", "").upper() == "NVIDIA":
                nvidia_gpus.append(gpu)
            else:
                non_nvidia_gpus.append(gpu)
        
        # 如果没有找到NVIDIA GPU，使用GPUInfoCollector收集非NVIDIA GPU信息
        if not nvidia_gpus and not non_nvidia_gpus:
            non_nvidia_gpu_info = self.gpu_collector.collect()
            for gpu in non_nvidia_gpu_info:
                non_nvidia_gpus.append({
                    "name": gpu["name"],
                    "memory_total": gpu["total"] / 1024,  # 转换为GB
                    "dedicated": gpu.get("dedicated", 0) / 1024,  # 转换为GB
                    "shared": gpu.get("shared", 0) / 1024,  # 转换为GB
                    "driver": "非NVIDIA"
                })
        
        # 添加GPU信息
        current_row = self.gpu_info_row
        
        # 计算总的GPU数量
        total_gpus = len(nvidia_gpus) + len(non_nvidia_gpus)
        
        # 添加NVIDIA GPU信息
        for i, gpu in enumerate(nvidia_gpus):
            # 对于NVIDIA显卡，专用显存等于总显存，共享显存为0
            dedicated_memory = self.custom_round(gpu.get('memory_total', 0))
            shared_memory = 0
            
            # 根据GPU总数决定显示格式
            if total_gpus == 1:
                label_text = f"GPU: {gpu['name']} ({dedicated_memory} GB+{shared_memory} GB)"
            else:
                label_text = f"GPU #{i+1}: {gpu['name']} ({dedicated_memory} GB+{shared_memory} GB)"
                
            gpu_label = QLabel(label_text)
            gpu_label.setWordWrap(True)
            self.gpu_info_layout.addWidget(gpu_label, current_row, 0, 1, 4)
            self.gpu_labels.append(gpu_label)
            current_row += 1
        
        # 添加非NVIDIA GPU信息
        for i, gpu in enumerate(non_nvidia_gpus):
            # 获取专用显存和共享显存
            dedicated_memory = self.custom_round(gpu.get('dedicated', 0))
            shared_memory = self.custom_round(gpu.get('shared', 0))
            
            # 根据GPU总数决定显示格式
            if total_gpus == 1:
                label_text = f"GPU: {gpu['name']} ({dedicated_memory} GB+{shared_memory} GB) [非NVIDIA]"
            else:
                label_text = f"GPU #{len(nvidia_gpus)+i+1}: {gpu['name']} ({dedicated_memory} GB+{shared_memory} GB) [非NVIDIA]"
                
            gpu_label = QLabel(label_text)
            gpu_label.setWordWrap(True)
            self.gpu_info_layout.addWidget(gpu_label, current_row, 0, 1, 4)
            self.gpu_labels.append(gpu_label)
            current_row += 1
        
        # 更新网络和版本信息的位置
        current_row += 1
        self.gpu_info_layout.addWidget(self.network_label, current_row, 0, 1, 4)
        
        # Python和Ollama版本
        self.python_label.setText(f"Python版本: {info.python_version}")
        self.ollama_label.setText(f"Ollama版本: {info.ollama_version}")
        
        # 设置GPU监控
        gpu_vendors = []
        for gpu in nvidia_gpus:
            gpu_vendors.append("NVIDIA")
        for gpu in non_nvidia_gpus:
            gpu_vendors.append("NON_NVIDIA")
        
        # 设置GPU监控所需的信息
        self.total_gpus = total_gpus  # 保存GPU总数，供其他方法使用
        self._setup_gpu_monitoring(info.gpu_info, gpu_vendors, total_gpus)
    
    def _setup_gpu_monitoring(self, gpu_info: List[Dict[str, Any]], gpu_vendors: List[str] = None, total_gpus: int = 0):
        """
        设置GPU监控

        参数:
            gpu_info: GPU信息列表
            gpu_vendors: GPU供应商列表
            total_gpus: GPU总数
        """
        # 清除旧的GPU监控布局
        self._clear_layout(self.gpu_metrics_layout)
        
        # 如果没有GPU，显示提示
        if not gpu_info:
            self.gpu_placeholder.setText("未检测到GPU")
            self.gpu_metrics_layout.addWidget(self.gpu_placeholder)
            return
        
        # 创建GPU分组和监控指标
        self.gpu_metrics = []
        
        for i, gpu in enumerate(gpu_info):
            # 检查是否为NVIDIA GPU
            is_nvidia = False
            if gpu_vendors and i < len(gpu_vendors):
                is_nvidia = gpu_vendors[i] == "NVIDIA"
            else:
                is_nvidia = gpu.get("driver", "").upper() == "NVIDIA"
            
            # 根据GPU总数决定显示格式
            if total_gpus == 1:
                group_title = f"GPU: {gpu.get('name', '未知')}"
            else:
                group_title = f"GPU #{i+1}: {gpu.get('name', '未知')}"
                
            gpu_group = QGroupBox(group_title)
            gpu_layout = QVBoxLayout(gpu_group)
            
            if is_nvidia:
                # 对于NVIDIA显卡，显示GPU使用率和显存使用率
                gpu_util_metric = MetricProgressBar(self, "GPU使用率", "#2ecc71")  # 绿色
                gpu_memory_metric = MetricProgressBar(self, "显存使用率", "#3498db")  # 蓝色
                gpu_layout.addWidget(gpu_util_metric)
                gpu_layout.addWidget(gpu_memory_metric)
                self.gpu_metrics.append({"utilization": gpu_util_metric, "memory": gpu_memory_metric})
            else:
                # 对于非NVIDIA显卡，显示提示信息
                non_nvidia_label = QLabel("非NVIDIA显卡不支持动态监控")
                non_nvidia_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                non_nvidia_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                gpu_layout.addWidget(non_nvidia_label)
                
                # 显示静态信息
                if "name" in gpu:
                    gpu_name_label = QLabel(f"显卡型号: {gpu['name']}")
                    gpu_layout.addWidget(gpu_name_label)
                
                # 修改显存显示格式为"专用xx GB+共享xx GB"
                dedicated_memory = self.custom_round(gpu.get("dedicated", 0))
                shared_memory = self.custom_round(gpu.get("shared", 0))
                memory_label = QLabel(f"显存大小: 专用{dedicated_memory} GB + 共享{shared_memory} GB")
                gpu_layout.addWidget(memory_label)
                
                self.gpu_metrics.append({"static_only": True})
            
            self.gpu_metrics_layout.addWidget(gpu_group)
        
        # 移除原有的占位符
        self.gpu_placeholder.setParent(None)
    
    def _clear_layout(self, layout):
        """清除布局中的所有部件"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self._clear_layout(item.layout())
    
    def _update_metrics(self):
        """更新监控指标"""
        # 获取最新指标
        metrics = self.monitor.get_metrics()
        
        # 更新CPU使用率
        self.cpu_metric.update_value(metrics.cpu_percent)
        
        # 更新内存使用率
        self.memory_metric.update_value(metrics.memory_percent)
        
        # 更新GPU指标
        for i, gpu_metric in enumerate(self.gpu_metrics):
            if i < len(metrics.gpu_metrics) and "static_only" not in gpu_metric:
                gpu_data = metrics.gpu_metrics[i]
                gpu_metric["utilization"].update_value(gpu_data.get("gpu_utilization", 0))
                
                # 计算显存使用率
                memory_total = gpu_data.get("memory_total", 0)
                memory_used = gpu_data.get("memory_used", 0)
                memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                
                gpu_metric["memory"].update_value(memory_percent)
        
        # 更新网络IO信息
        self.network_label.setText(f"实时网速: 上传 {metrics.network_sent:.1f} KB/s | 下载 {metrics.network_recv:.1f} KB/s")
    
    def on_tab_selected(self):
        """标签页被选中时调用"""
        # 在标签页被选中时，无需重新启动定时器，因为我们已经在初始化时启动了
        pass
    
    def on_server_changed(self):
        """服务器连接更改时调用"""
        # 更新Ollama版本信息
        ollama_version = self.client.get_version()
        self.ollama_label.setText(f"Ollama版本: {ollama_version}")
    
    def on_close(self):
        """应用程序关闭时调用"""
        # 停止定时器
        self.update_timer.stop() 