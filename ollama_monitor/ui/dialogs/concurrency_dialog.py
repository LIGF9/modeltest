#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama并发设置对话框模块
提供用于设置Ollama并发参数的对话框界面
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QPushButton, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt

from ...utils.concurrency_settings import OllamaConcurrencySettings


class ConcurrencySettingsDialog(QDialog):
    """Ollama并发设置对话框类"""
    
    def __init__(self, parent=None, concurrency_settings: OllamaConcurrencySettings = None, ollama_client=None):
        """
        初始化对话框
        
        参数:
            parent: 父窗口
            concurrency_settings: 并发设置对象
            ollama_client: OllamaClient实例，用于获取最新设置
        """
        super().__init__(parent)
        
        # 设置标题和模态
        self.setWindowTitle("Ollama并发设置")
        self.setModal(True)
        
        # 设置大小
        self.resize(400, 200)
        
        # 保存并发设置对象和客户端
        self.concurrency_settings = concurrency_settings
        self.ollama_client = ollama_client
        
        # 初始化UI
        self._init_ui()
        
        # 从服务器获取最新设置
        self._get_latest_settings()
        
        # 加载当前设置
        self._load_settings()
    
    def _init_ui(self):
        """初始化界面元素"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建设置分组框
        settings_group = QGroupBox("并发参数设置")
        form_layout = QFormLayout(settings_group)
        
        # 并发用户数控件
        self.users_spinbox = QSpinBox()
        self.users_spinbox.setMinimum(1)
        self.users_spinbox.setMaximum(100)
        self.users_spinbox.setToolTip("设置单模型最大并发用户数")
        form_layout.addRow("单模型最大并发用户数:", self.users_spinbox)
        
        # 内存中模型数控件
        self.models_spinbox = QSpinBox()
        self.models_spinbox.setMinimum(1)
        self.models_spinbox.setMaximum(10)
        self.models_spinbox.setToolTip("设置内存中最大模型数")
        form_layout.addRow("内存中最大模型数:", self.models_spinbox)
        
        # 添加设置分组到主布局
        main_layout.addWidget(settings_group)
        
        # 说明标签
        description_label = QLabel(
            "注意: 设置较大的并发值可能导致系统资源消耗过大，请根据您的硬件配置合理设置。"
        )
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(description_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 确认按钮
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self._save_settings)
        
        # 取消按钮
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        
        # 添加按钮到按钮布局
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        # 添加按钮布局到主布局
        main_layout.addLayout(button_layout)
    
    def _get_latest_settings(self):
        """尝试从服务器获取最新设置"""
        if self.ollama_client and self.concurrency_settings:
            try:
                config = self.ollama_client.get_current_config()
                if config["parallel_enabled"]:
                    self.concurrency_settings.max_concurrent_users = config["num_parallel"] or self.concurrency_settings.max_concurrent_users
                    self.concurrency_settings.max_in_memory_models = config["max_models"] or self.concurrency_settings.max_in_memory_models
            except Exception as e:
                print(f"获取最新并发设置失败: {str(e)}")
    
    def _load_settings(self):
        """加载当前设置到控件"""
        if self.concurrency_settings:
            self.users_spinbox.setValue(self.concurrency_settings.max_concurrent_users)
            self.models_spinbox.setValue(self.concurrency_settings.max_in_memory_models)
    
    def _save_settings(self):
        """保存设置到并发设置对象"""
        if self.concurrency_settings:
            self.concurrency_settings.max_concurrent_users = self.users_spinbox.value()
            self.concurrency_settings.max_in_memory_models = self.models_spinbox.value()
            self.accept() 