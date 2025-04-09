#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能测试标签页模块
提供模型性能测试和评估功能
"""

import time
import random
import traceback  # 添加 traceback 模块导入
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QGroupBox, QGridLayout, 
    QComboBox, QSpinBox, QProgressBar, QFileDialog,
    QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDialog, QLineEdit
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QThread, 
    QTimer, QUrl, Qt
)
from PyQt6.QtGui import QFont, QDesktopServices, QColor, QMouseEvent, QCursor

from ...utils.ollama_client import OllamaClient
from ...utils.system_monitor import SystemMonitor
from ...utils.performance_tester import PerformanceTester, TestPrompt, PerformanceReport, TestResult

# 添加一个可点击的标签类
class ClickableLabel(QLabel):
    """可点击的标签，用于显示帮助信息"""
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))  # 设置鼠标样式为手型
        self.setStyleSheet("color: #3498db; font-weight: bold;")
        
    def mousePressEvent(self, event: QMouseEvent):
        """处理鼠标点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.showHelpInfo()
        super().mousePressEvent(event)
    
    def showHelpInfo(self):
        """显示并发设置的详细帮助信息"""
        QMessageBox.information(self, "并发设置说明",
            "并发用户数设置说明：\n\n"
            "1. 此设置控制同时向Ollama发送请求的用户数量\n\n"
            "2. 支持要求：\n"
            "   • 需要Ollama 0.1.33及以上版本\n"
            "   • 需要环境变量支持或systemd配置支持\n\n"
            "3. 多用户并发效果：\n"
            "   • 单个用户的响应速度(tokens/s)通常会降低\n"
            "   • 但系统整体吞吐量可能会提高\n"
            "   • 适合模拟多用户同时访问的场景\n\n"
            "4. 程序会自动检测您的Ollama是否支持多用户并发：\n"
            "   • 如不支持，将自动以单用户模式运行\n"
            "   • 如支持但未配置，将临时配置为您设置的数值\n"
            "   • 测试结束后会恢复原始设置"
        )

class CustomPromptDialog(QDialog):
    """自定义测试提示词对话框"""
    
    def __init__(self, parent=None):
        """
        初始化对话框
        
        参数:
            parent: 父级部件
        """
        super().__init__(parent)
        
        self.setWindowTitle("添加自定义提示词")
        self.resize(500, 350)
        
        # 主布局
        layout = QVBoxLayout(self)
        
        # 名称输入
        name_layout = QHBoxLayout()
        name_label = QLabel("名称:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入提示词名称...")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # 类别输入
        category_layout = QHBoxLayout()
        category_label = QLabel("类别:")
        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(["文本处理", "知识问答", "编程", "创意", "推理", "复杂任务", "自定义"])
        category_layout.addWidget(category_label)
        category_layout.addWidget(self.category_combo)
        layout.addLayout(category_layout)
        
        # 内容输入
        content_label = QLabel("提示词内容:")
        self.content_text = QTextEdit()
        self.content_text.setPlaceholderText("输入提示词内容...")
        layout.addWidget(content_label)
        layout.addWidget(self.content_text)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 取消和保存按钮
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        
        save_button = QPushButton("保存")
        save_button.clicked.connect(self._on_save)
        save_button.setDefault(True)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
    
    def _on_save(self):
        """保存按钮点击处理"""
        # 验证输入
        name = self.name_input.text().strip()
        category = self.category_combo.currentText().strip()
        content = self.content_text.toPlainText().strip()
        
        if not name:
            QMessageBox.warning(self, "输入错误", "请输入提示词名称")
            return
        
        if not content:
            QMessageBox.warning(self, "输入错误", "请输入提示词内容")
            return
        
        # 接受对话框
        self.accept()
    
    def get_prompt(self) -> TestPrompt:
        """
        获取创建的提示词
        
        返回:
            测试提示词对象
        """
        name = self.name_input.text().strip()
        category = self.category_combo.currentText().strip() or "自定义"
        content = self.content_text.toPlainText().strip()
        
        return TestPrompt(name, content, category)

class ReportPreviewDialog(QDialog):
    """报告预览对话框"""
    
    def __init__(self, parent=None, html_content=None, text_content=None):
        """
        初始化报告预览对话框
        
        参数:
            parent: 父级窗口
            html_content: HTML格式的报告内容
            text_content: 文本格式的报告内容
        """
        super().__init__(parent)
        
        self.html_content = html_content
        self.text_content = text_content
        
        self.setWindowTitle("报告预览")
        self.resize(900, 700)
        
        # 主布局
        layout = QVBoxLayout(self)
        
        # 创建标签页切换不同格式的预览
        self.tab_widget = QTabWidget()
        
        # HTML预览标签页
        if html_content:
            html_tab = QWidget()
            html_layout = QVBoxLayout(html_tab)
            
            self.html_preview = QTextEdit()
            self.html_preview.setReadOnly(True)
            self.html_preview.setHtml(html_content)
            
            html_layout.addWidget(self.html_preview)
            self.tab_widget.addTab(html_tab, "HTML预览")
        
        # 文本预览标签页
        if text_content:
            text_tab = QWidget()
            text_layout = QVBoxLayout(text_tab)
            
            self.text_preview = QTextEdit()
            self.text_preview.setReadOnly(True)
            self.text_preview.setPlainText(text_content)
            self.text_preview.setFont(QFont("Courier New", 10))
            
            text_layout.addWidget(self.text_preview)
            self.tab_widget.addTab(text_tab, "文本预览")
        
        layout.addWidget(self.tab_widget)
        
        # 底部按钮布局
        button_layout = QHBoxLayout()
        
        # 导出HTML按钮
        if html_content:
            self.export_html_btn = QPushButton("导出HTML")
            self.export_html_btn.clicked.connect(self._export_html)
            button_layout.addWidget(self.export_html_btn)
        
        # 导出文本按钮
        if text_content:
            self.export_txt_btn = QPushButton("导出文本")
            self.export_txt_btn.clicked.connect(self._export_txt)
            button_layout.addWidget(self.export_txt_btn)
        
        # 关闭按钮
        button_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _export_html(self):
        """导出HTML格式报告"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存HTML报告",
            "",
            "HTML文件 (*.html)"
        )
        
        if not file_path:
            return
        
        # 确保文件扩展名为.html
        if not file_path.lower().endswith('.html'):
            file_path += '.html'
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.html_content)
            
            QMessageBox.information(self, "保存成功", "HTML报告已成功保存")
            
            # 询问是否打开报告
            result = QMessageBox.question(
                self,
                "打开报告",
                "是否立即打开报告?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if result == QMessageBox.StandardButton.Yes:
                # 打开报告
                QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存HTML报告失败: {str(e)}")
    
    def _export_txt(self):
        """导出文本格式报告"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存文本报告",
            "",
            "文本文件 (*.txt)"
        )
        
        if not file_path:
            return
        
        # 确保文件扩展名为.txt
        if not file_path.lower().endswith('.txt'):
            file_path += '.txt'
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.text_content)
            
            QMessageBox.information(self, "保存成功", "文本报告已成功保存")
            
            # 询问是否打开报告
            result = QMessageBox.question(
                self,
                "打开报告",
                "是否立即打开报告?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if result == QMessageBox.StandardButton.Yes:
                # 打开报告
                QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存文本报告失败: {str(e)}")

class PerformanceTab(QWidget):
    """性能测试标签页类"""
    
    def __init__(self, ollama_client: OllamaClient, system_monitor: SystemMonitor):
        """
        初始化性能测试标签页
        
        参数:
            ollama_client: Ollama API客户端
            system_monitor: 系统监控器
        """
        super().__init__()
        
        # 保存引用
        self.client = ollama_client
        self.monitor = system_monitor
        self.istesting = False
        
        # 初始化性能测试器
        self.tester = PerformanceTester(self.client, self.monitor)
        
        # 初始化测试状态
        self.current_model = ""
        self.current_report = None
        self.test_worker = None
        self.is_testing = False
        self.animation_offset = 0.0
        self.is_test_paused = False  # 新增：标记测试是否暂停
        self.completed_rounds = 0    # 新增：记录已完成的轮次
        self.total_rounds = 0        # 新增：记录总轮次
        
        # 初始化进度动画定时器
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_progress_animation)
        
        # 初始化UI
        self._init_ui()
    
    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(5,5,5,5)  # 增加整体边距
        
        # 顶部标题和说明
        title_layout = QHBoxLayout()
        title = QLabel("模型性能测试")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        description = QLabel("使用不同类型的提示词测试模型性能并生成详细报告")
        description.setStyleSheet("color: #666;")
        title_layout.addWidget(title)
        title_layout.addWidget(description)
        
        # 在标题栏右侧添加提示词管理和按钮
        title_layout.addStretch(1)  # 添加弹性空间，将按钮推到右侧
        
        # 添加提示词管理标签
        prompt_label = QLabel("测试提示词管理:")
        prompt_label.setStyleSheet("font-weight: bold;")
        title_layout.addWidget(prompt_label)
        
        # 四个按钮放在右侧
        self.import_prompt_btn = QPushButton("导入")
        self.export_prompt_btn = QPushButton("导出")
        self.view_prompt_btn = QPushButton("查看")
        self.add_prompt_btn = QPushButton("添加")
        
        # 设置按钮最大宽度为60
        for btn in [self.import_prompt_btn, self.export_prompt_btn, 
                   self.view_prompt_btn, self.add_prompt_btn]:
            btn.setMaximumWidth(60)
        
        title_layout.addWidget(self.import_prompt_btn)
        title_layout.addWidget(self.export_prompt_btn)
        title_layout.addWidget(self.view_prompt_btn)
        title_layout.addWidget(self.add_prompt_btn)
        
        layout.addLayout(title_layout)
        
        # 测试控制行（直接作为顶级组件）
        control_group = QGroupBox("测试控制")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        control_layout = QHBoxLayout()

        # 创建左侧布局包含进度条和状态
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m")
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 2px;
            }
        """)
        left_layout.addWidget(self.progress_bar, 10)  # 给进度条分配更多空间

        # 在进度条右侧添加操作状态
        self.operation_status = QLabel("未开始测试")
        self.operation_status.setStyleSheet("font-weight: bold; color: #7f8c8d;")
        self.operation_status.setMinimumWidth(120)  # 设置最小宽度确保文本显示
        self.operation_status.setMaximumWidth(150)  # 设置最大宽度限制空间占用
        left_layout.addWidget(self.operation_status, 1)  # 操作状态占较少空间

        # 将左侧布局添加到主控制布局，并占据大部分空间
        control_layout.addLayout(left_layout, 10)

        # 添加弹性空间将配置项推到右侧（较少空间）
        control_layout.addStretch(1)

        # 并发数设置
        control_layout.addWidget(QLabel("并发数"))
        # 并发数提示标签 - 使用可点击的标签
        concurrency_help = ClickableLabel(" ?", self)
        control_layout.addWidget(concurrency_help)
        
        # 冒号标签
        control_layout.addWidget(QLabel(":"))
        
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 100)
        self.concurrency_spin.setValue(2)  # 默认并发数2
        self.concurrency_spin.setMaximumWidth(60)
        control_layout.addWidget(self.concurrency_spin)
        
        # 测试轮数设置
        control_layout.addWidget(QLabel("轮数:"))
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 100)
        self.rounds_spin.setValue(3)  # 默认值3轮
        self.rounds_spin.setMaximumWidth(60)
        self.rounds_spin.valueChanged.connect(self._on_rounds_changed)
        control_layout.addWidget(self.rounds_spin)

        # 按钮靠右显示
        self.start_btn = QPushButton("开始测试")
        self.start_btn.setMaximumWidth(240)  # 宽度设为原来的两倍
        self.start_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
        
        self.stop_btn = QPushButton("停止测试")
        self.stop_btn.setMaximumWidth(240)  # 宽度设为原来的两倍
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #cccccc; color: #888888; font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; padding: 4px;")  # 设置灰色禁用样式

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        # 直接添加到主布局
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)  
        
        # 测试结果概述
        results_overview_group = QGroupBox("结果概述")
        results_overview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        results_layout = QGridLayout()
        results_layout.setSpacing(10)
        
        # 第一行：模型名称、性能评级、模型载入时间
        results_layout.addWidget(QLabel("模型名称:"), 0, 0)
        self.model_name_label = QLabel("-")
        results_layout.addWidget(self.model_name_label, 0, 1)
        
        # 添加一个空白标签作为间隔
        results_layout.addWidget(QLabel(""), 0, 2)
        
        results_layout.addWidget(QLabel("性能评级:"), 0, 3)
        self.performance_rating_label = QLabel("-")
        self.performance_rating_label.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self.performance_rating_label, 0, 4)
        
        # 添加一个空白标签作为间隔
        results_layout.addWidget(QLabel(""), 0, 5)
        
        results_layout.addWidget(QLabel("模型载入时间:"), 0, 6)
        self.model_load_time_label = QLabel("-")
        results_layout.addWidget(self.model_load_time_label, 0, 7)
        
        # 第二行：模型参数量、首token延迟、平均生成速度
        results_layout.addWidget(QLabel("模型参数:"), 1, 0)
        self.parameters_input = QLineEdit()
        self.parameters_input.setPlaceholderText("例如: 7B")
        self.parameters_input.setMaximumWidth(100)
        # 设置焦点策略为ClickFocus，只有当用户点击时才获得焦点
        self.parameters_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        results_layout.addWidget(self.parameters_input, 1, 1)
        
        # 添加一个空白标签作为间隔
        results_layout.addWidget(QLabel(""), 1, 2)
        
        results_layout.addWidget(QLabel("首Token延迟:"), 1, 3)
        self.average_latency_label = QLabel("0 ms")
        results_layout.addWidget(self.average_latency_label, 1, 4)
        
        # 添加一个空白标签作为间隔
        results_layout.addWidget(QLabel(""), 1, 5)
        
        results_layout.addWidget(QLabel("平均生成速度:"), 1, 6)
        self.average_tokens_sec_label = QLabel("0 tokens/s")
        results_layout.addWidget(self.average_tokens_sec_label, 1, 7)
        
        # 第三行：编码精度、模型温度、上下文窗口可编辑文本框
        results_layout.addWidget(QLabel("编码精度:"), 2, 0)
        self.precision_input = QLineEdit()
        self.precision_input.setPlaceholderText("例如: INT8、FP16")
        self.precision_input.setText("INT4")  # 设置默认值为INT8
        self.precision_input.setMaximumWidth(120)
        # 设置焦点策略为ClickFocus
        self.precision_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        results_layout.addWidget(self.precision_input, 2, 1)
        
        # 添加一个空白标签作为间隔
        results_layout.addWidget(QLabel(""), 2, 2)
        
        results_layout.addWidget(QLabel("模型温度:"), 2, 3)
        self.temperature_input = QLineEdit()
        self.temperature_input.setPlaceholderText("例如: 0.8")
        self.temperature_input.setText("0.8")  # 设置默认值
        self.temperature_input.setMaximumWidth(120)
        # 设置焦点策略为ClickFocus
        self.temperature_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        results_layout.addWidget(self.temperature_input, 2, 4)
        
        # 添加一个空白标签作为间隔
        results_layout.addWidget(QLabel(""), 2, 5)
        
        results_layout.addWidget(QLabel("上下文窗口:"), 2, 6)
        self.context_window_input = QLineEdit()
        self.context_window_input.setPlaceholderText("例如: 128K")
        self.context_window_input.setText("128K")  # 设置默认值为128K
        self.context_window_input.setMaximumWidth(120)
        # 设置焦点策略为ClickFocus
        self.context_window_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        results_layout.addWidget(self.context_window_input, 2, 7)
        
        # 设置列的拉伸因子，使标签和值列靠近，组之间有一定间距
        results_layout.setColumnStretch(0, 1)  # 标签列
        results_layout.setColumnStretch(1, 3)  # 值列
        results_layout.setColumnStretch(2, 1)  # 间隔列
        results_layout.setColumnStretch(3, 1)  # 标签列
        results_layout.setColumnStretch(4, 3)  # 值列
        results_layout.setColumnStretch(5, 1)  # 间隔列
        results_layout.setColumnStretch(6, 1)  # 标签列
        results_layout.setColumnStretch(7, 3)  # 值列
        
        # 设置水平间距，减小标签和值之间的距离，增加组之间的距离
        results_layout.setHorizontalSpacing(5)  # 减小标签和值之间的间距
        
        results_overview_group.setLayout(results_layout)
        layout.addWidget(results_overview_group)
        
        # 测试详情区域
        details_group = QGroupBox("测试详情")
        details_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        details_layout = QVBoxLayout(details_group)
        
        # 导出按钮放在右侧
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()        
        self.export_html_btn = QPushButton("HTML报告")
        self.export_txt_btn = QPushButton("文本报告")
        self.export_html_btn.setEnabled(False)
        self.export_txt_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_html_btn)
        buttons_layout.addWidget(self.export_txt_btn)        
        details_layout.addLayout(buttons_layout)
        
        # 测试详情表格 - 添加提示词类型列
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(7)
        
        # 设置换行的表头标签
        header_labels = [
            "轮次", 
            "提示词类型",
            "提示词", 
            "首Token\n延迟(ms)", 
            "生成速度\n(tokens/s)", 
            "tokens\n总数", 
            "总时间\n(s)"
        ]
        self.details_table.setHorizontalHeaderLabels(header_labels)
        
        # 设置表头样式和高度
        header = self.details_table.horizontalHeader()
        header.setStyleSheet("QHeaderView::section { padding: 6px; background-color: #f0f0f0; }")
        header.setMinimumHeight(40)  # 增加高度以适应换行
        
        # 设置各列宽度
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)      # 轮次列固定宽度
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)      # 提示词类型列固定宽度
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)    # 提示词列拉伸填充剩余空间
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)      # 首Token延迟列固定宽度
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)      # 生成速度列固定宽度
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)      # tokens总数列固定宽度
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)      # 总时间列固定宽度
        
        # 设置固定列的宽度
        self.details_table.setColumnWidth(0, 40)   # 轮次列
        self.details_table.setColumnWidth(1, 80)  # 提示词类型列
        self.details_table.setColumnWidth(3, 70)   # 首Token延迟列
        self.details_table.setColumnWidth(4, 70)   # 生成速度列
        self.details_table.setColumnWidth(5, 60)   # tokens总数列
        self.details_table.setColumnWidth(6, 60)   # 总时间列
        
        # 确保提示词不会换行
        self.details_table.setWordWrap(False)        
        self.details_table.verticalHeader().setVisible(False)
        details_layout.addWidget(self.details_table)        
        layout.addWidget(details_group)
        
        # 绑定信号
        self.start_btn.clicked.connect(self._start_test)
        self.stop_btn.clicked.connect(self._stop_test)
        self.export_html_btn.clicked.connect(self._export_html_report)
        self.export_txt_btn.clicked.connect(self._export_txt_report)
        self.import_prompt_btn.clicked.connect(self._import_prompts)
        self.export_prompt_btn.clicked.connect(self._export_prompts)
        self.view_prompt_btn.clicked.connect(self._view_prompts)
        self.add_prompt_btn.clicked.connect(self._add_prompt)
    
    def _import_prompts(self):
        """导入提示词文件"""
        # 显示文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入提示词文件",
            "",
            "文本文件 (*.txt)"
        )
        
        if not file_path:
            return
            
        # 加载提示词文件
        try:
            self.tester._load_prompts_from_file(file_path)
            QMessageBox.information(self, "导入成功", "提示词文件导入成功")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入提示词文件失败: {str(e)}")
    
    def _export_prompts(self):
        """导出提示词到文件"""
        # 显示文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出提示词文件",
            "prompts.txt",
            "文本文件 (*.txt)"
        )
        
        if not file_path:
            return
            
        # 确保文件扩展名为.txt
        if not file_path.lower().endswith('.txt'):
            file_path += '.txt'
            
        # 保存提示词到文件
        if self.tester.save_prompts_to_file(file_path):
            QMessageBox.information(self, "导出成功", "提示词文件导出成功")
        else:
            QMessageBox.critical(self, "导出失败", "导出提示词文件失败")
    
    def _view_prompts(self):
        """查看当前所有提示词"""
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("提示词列表")
        dialog.resize(600, 400)
        
        # 主布局
        layout = QVBoxLayout(dialog)
        
        # 创建标签页控件
        tab_widget = QTabWidget()
        
        # 获取按类别分组的提示词
        prompts_by_category = self.tester.get_prompts_by_category()
        
        # 为每个类别创建一个标签页
        for category, prompts in prompts_by_category.items():
            # 创建标签页
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # 创建表格
            table = QTableWidget(len(prompts), 2)
            table.setHorizontalHeaderLabels(["名称", "内容"])
            
            # 设置列宽
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            
            # 添加提示词到表格
            for i, prompt in enumerate(prompts):
                name_item = QTableWidgetItem(prompt.name)
                content_item = QTableWidgetItem(prompt.content)
                
                table.setItem(i, 0, name_item)
                table.setItem(i, 1, content_item)
            
            # 添加表格到标签页
            tab_layout.addWidget(table)
            
            # 添加标签页到标签页控件
            tab_widget.addTab(tab, f"{category} ({len(prompts)})")
        
        # 添加标签页控件到主布局
        layout.addWidget(tab_widget)
        
        # 添加关闭按钮
        button_layout = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # 显示对话框
        dialog.exec()
    
    def _add_prompt(self):
        """添加自定义提示词"""
        # 创建自定义提示词对话框
        dialog = CustomPromptDialog(self)
        
        # 如果用户点击保存
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 获取用户输入的提示词
            prompt = dialog.get_prompt()
            
            # 添加到测试器
            self.tester.add_custom_prompt(prompt.name, prompt.content, prompt.category)
            
            # 显示成功消息
            QMessageBox.information(self, "添加成功", f"自定义提示词 '{prompt.name}' 添加成功")
    
    def _safe_access(self, attr_name, func=None, default=None):
        """
        安全地访问类属性，防止异常导致程序崩溃
        
        参数:
            attr_name: 属性名
            func: 对属性执行的函数
            default: 如果属性不存在或函数执行异常时返回的默认值
            
        返回:
            属性值或函数执行结果
        """
        try:
            if not hasattr(self, attr_name):
                return default
                
            attr = getattr(self, attr_name)
            if attr is None:
                return default
                
            if callable(func):
                return func(attr)
            else:
                return attr
        except Exception as e:
            print(f"安全访问属性出错 ({attr_name}): {e}")
            return default
    
    def _get_prompt_name_by_content(self, prompt_content: str) -> str:
        """
        根据提示词内容查找对应的提示词名称
        
        参数:
            prompt_content: 提示词内容
            
        返回:
            提示词名称，如果找不到则返回空字符串
        """
        if not hasattr(self, 'tester') or not self.tester:
            return ""
            
        # 遍历所有提示词类别和提示词
        for category, prompts in self.tester.prompts_by_category.items():
            for prompt in prompts:
                # 比较提示词内容
                if prompt.content == prompt_content:
                    return prompt.name
                    
        # 如果找不到，返回空字符串
        return ""
    
    def _on_rounds_changed(self, value):
        """当测试轮数改变时更新进度条最大值"""
        self._safe_access('progress_bar', lambda x: x.setMaximum(value))
    
    def _start_test(self):
        """开始性能测试"""
        
        self.istesting = True

        try:
            print("开始测试，istesting：", self.istesting)
            # 清空详情表格并初始化测试结果
            self._init_test_results(reset_input_values=False)
            
            # 初始化模型名称和其他所有结果显示
            self._safe_access('model_name_label', lambda x: x.setText(self.current_model))
            self._safe_access('performance_rating_label', lambda x: x.setText("-"))
            self._safe_access('model_load_time_label', lambda x: x.setText("-"))
            self._safe_access('average_latency_label', lambda x: x.setText("0 ms"))
            self._safe_access('average_tokens_sec_label', lambda x: x.setText("0 tokens/s"))
            
            # 收集测试所需的所有参数，避免多次访问UI控件
            test_params = {}
            
            # 获取测试轮数
            test_round_input = self._safe_access('rounds_spin')
            rounds = int(test_round_input.value())
            
            # 获取并发用户数
            concurrency_input = self._safe_access('concurrency_spin')
            concurrent_users = int(concurrency_input.value())
            
            # 获取模型名称 - 使用全局已选模型
            if not self.current_model:
                QMessageBox.warning(self, "参数错误", "未设置当前模型，请在主界面选择模型")
                return
            model_name = self.current_model
            test_params['model_name'] = model_name

            # 获取模型参数
            model_params_input = self._safe_access('parameters_input', lambda x: x.text())
            if model_params_input and model_params_input != "未识别" and model_params_input != "-":
                test_params['model_params'] = model_params_input
            else:
                param = self.tester.extract_model_parameter(model_name)
                if param is not None:
                    # 转换为字符串表示
                    if param >= 1:
                        model_params = f"{param}B"
                    else:
                        model_params = f"{int(param * 1000)}M"
                    test_params['model_params'] = model_params
                    # 更新输入框
                    self._safe_access('parameters_input', lambda x: x.setText(model_params))
                else:
                    test_params['model_params'] = '0B'
                    
                
            # 获取编码精度
            precision = self._safe_access('precision_input', lambda x: x.text())
            if precision:
                test_params['model_precision'] = precision
                
            # 获取温度系数
            temperature_text = self._safe_access('temperature_input', lambda x: x.text())
            if temperature_text:
                try:
                    temperature = float(temperature_text)
                    if 0 <= temperature <= 2:
                        test_params['model_temperature'] = temperature
                    else:
                        QMessageBox.warning(self, "参数错误", "温度系数必须在0-2之间")
                        return
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "温度系数必须为数字")
                    return
                    
            # 获取上下文窗口
            context_window = self._safe_access('context_window_input', lambda x: x.text())
            if context_window:
                # 支持K/M后缀
                try:
                    if context_window.upper().endswith('K') or context_window.upper().endswith('M'):
                        test_params['context_window'] = context_window
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "上下文窗口必须为有效数字，可带K/M后缀")
                    return
            
            # 计算总测试次数
            total_tests = rounds * concurrent_users
            
            # 设置进度条最大值为总测试轮次数（并发用户数 * 轮次数）
            total_rounds = self.concurrency_spin.value() * self.rounds_spin.value()
            self.progress_bar.setMaximum(total_rounds)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0/%d" % total_rounds)
            
            # 禁用相关按钮
            self._safe_access('start_btn', lambda x: {
                x.setEnabled(False),
                x.setStyleSheet("background-color: #cccccc; color: #888888; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('stop_btn', lambda x: {
                x.setText("停止测试"),
                x.setStyleSheet("color: #e74c3c; font-weight: bold; border: 1px solid #e74c3c; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('export_html_btn', lambda x: x.setEnabled(False))
            self._safe_access('export_txt_btn', lambda x: x.setEnabled(False))
            
            # 更新操作状态文字和样式 - 同步更新避免状态延迟
            self._safe_access('operation_status', lambda x: {
                x.setText("模型载入测试中..."),
                x.setStyleSheet("font-weight: bold; color: #3498db;")
            })
                        
            # 初始化测试报告
            self.current_report = PerformanceReport(
                model_name=model_name,
                system_info={},  # 由测试器填充
                test_params=test_params,
                results=[]
            )
            
            # 创建PerformanceTestWorker实例并连接信号
            self.test_worker = PerformanceTestWorker(
                client=self.client,
                model_name=model_name,
                rounds=rounds,
                test_params=test_params,
                concurrent_users=concurrent_users  # 传递并发用户数
            )
            
            # 连接信号
            self.test_worker.test_round_completed.connect(self._handle_round_completed)
            self.test_worker.test_completed.connect(self._handle_test_completed)
            self.test_worker.error_occurred.connect(self._handle_test_error)
            
            # 开始进度动画
            self.is_testing = True
            self.animation_offset = 0.0
            self.progress_timer = QTimer(self)
            self.progress_timer.timeout.connect(self._update_progress_animation)
            self.progress_timer.start(500)  # 每500毫秒更新一次
            
            # 开始执行测试
            self.test_worker.start()
            
        except Exception as e:
            print(f"启动测试时错误: {e}")
            QMessageBox.critical(self, "测试错误", f"无法启动测试: 请填写模型参数量")
            self._safe_access('start_btn', lambda x: {
                x.setEnabled(True),
                x.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('operation_status', lambda x: {
                x.setText("测试失败"),
                x.setStyleSheet("font-weight: bold; color: #e74c3c;")
            })

        self.istesting = False

    def _stop_test(self):
        """停止/暂停性能测试"""
        self.istesting = False
        try:
            # 检查测试工作线程是否存在且正在运行
            if not hasattr(self, 'test_worker') or self.test_worker is None:
                print("测试工作线程已被删除或不存在")
                return
                
            if not self.test_worker.isRunning():
                print("测试工作线程未运行")
                return
            
            # 如果是暂停状态，则继续测试
            if self.is_test_paused:
                self._continue_test()
                return
                
            # 检查其他必要的UI组件
            if not hasattr(self, 'operation_status') or self.operation_status is None:
                print("操作状态标签已被删除或不存在")
            else:
                # 更新状态
                self.operation_status.setText("已暂停测试...")
                self.operation_status.setStyleSheet("font-weight: bold; color: #f39c12;")
            
            # 将进度条设置为当前进度
            self._safe_access('progress_bar', lambda x: x.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #f39c12;
                    border-radius: 2px;
                }
            """))
            
            # 暂停测试，但不会立即终止线程，以便处理已完成的测试结果
            self.test_worker.pause_test()
            
            # 记录已完成的轮次数量
            progress_bar = self._safe_access('progress_bar')
            if progress_bar:
                self.completed_rounds = progress_bar.value()
                self.total_rounds = progress_bar.maximum()
            
            # 更新按钮状态和文本
            self._safe_access('start_btn', lambda x: {
                x.setEnabled(False),
                x.setStyleSheet("background-color: #cccccc; color: #888888; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('stop_btn', lambda x: {
                x.setText("继续测试"),
                x.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            
            # 更新状态文字
            self._safe_access('operation_status', lambda x: x.setText("测试已暂停，点击继续可恢复测试"))
            
            # 暂停测试动画，但保留进度条显示
            self.is_testing = False
            self.is_test_paused = True
            progress_timer = self._safe_access('progress_timer')
            if progress_timer and progress_timer.isActive():
                progress_timer.stop()
                
            # 启用报告导出按钮，即使测试暂停也允许导出当前结果
            self._safe_access('export_html_btn', lambda x: x.setEnabled(True))
            self._safe_access('export_txt_btn', lambda x: x.setEnabled(True))
            
            # 更新测试结果概述（基于当前已完成的测试）
            details_table = self._safe_access('details_table')
            if details_table and hasattr(self, 'current_report') and self.current_report:
                self._update_test_summary(details_table)
            
        except RuntimeError as e:
            # 忽略已删除对象的错误
            print(f"停止测试时出错: {e}")
        except Exception as e:
            # 捕获任何其他异常
            print(f"停止测试时发生未知错误: {e}")

    def _continue_test(self):
        """继续暂停的测试"""
        self.istesting = True
        try:
            # 检查测试是否处于暂停状态
            if not self.is_test_paused or not hasattr(self, 'test_worker') or self.test_worker is None:
                return
            
            # 更新按钮状态和文本
            self._safe_access('start_btn', lambda x: {
                x.setEnabled(False),
                x.setStyleSheet("background-color: #cccccc; color: #888888; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('stop_btn', lambda x: {
                x.setText("停止测试"),
                x.setStyleSheet("color: #e74c3c; font-weight: bold; border: 1px solid #e74c3c; border-radius: 4px; padding: 4px;")
            })
            
            # 更新进度条样式
            self._safe_access('progress_bar', lambda x: x.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                 stop:0 #3498db, stop:0.5 #2980b9, 
                                                 stop:1 #3498db);
                    border-radius: 2px;
                }
            """))
            
            # 更新状态文字
            self._safe_access('operation_status', lambda x: {
                x.setText("继续进行轮次测试..."),
                x.setStyleSheet("font-weight: bold; color: #27ae60;")
            })
            
            # 恢复测试状态
            self.is_testing = True
            self.is_test_paused = False
            
            # 重新启动进度动画
            self.animation_offset = 0.0
            self.progress_timer = QTimer(self)
            self.progress_timer.timeout.connect(self._update_progress_animation)
            self.progress_timer.start(100)  # 每100毫秒更新一次
            
            # 恢复测试
            self.test_worker.resume_test()
            
        except Exception as e:
            print(f"继续测试时出错: {e}")
            self._safe_access('start_btn', lambda x: {
                x.setEnabled(True),
                x.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('stop_btn', lambda x: {
                x.setEnabled(False),
                x.setStyleSheet("background-color: #cccccc; color: #888888; font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('operation_status', lambda x: {
                x.setText("恢复测试失败"),
                x.setStyleSheet("font-weight: bold; color: #e74c3c;")
            })
    
    def _init_test_results(self, reset_input_values=True):
        """初始化测试结果UI"""
        try:
            # 清空详情表格
            details_table = self._safe_access('details_table')
            if details_table:
                details_table.setRowCount(0)
            
            # 重置结果标签
            self._safe_access('performance_rating_label', lambda x: x.setText("-"))
            self._safe_access('model_load_time_label', lambda x: x.setText("-"))
            self._safe_access('average_tokens_sec_label', lambda x: x.setText("0 tokens/s"))
            self._safe_access('average_latency_label', lambda x: x.setText("0 ms"))
            
            # 如果需要，重置输入值（但保持默认设置）
            if reset_input_values:
                self._safe_access('precision_input', lambda x: x.setText("INT4"))
                self._safe_access('temperature_input', lambda x: x.setText("0.8"))
                self._safe_access('context_window_input', lambda x: x.setText("128K"))
            
            # 重置操作状态
            self._safe_access('operation_status', lambda x: x.setStyleSheet("font-weight: normal; color: black;"))
            
            # 禁用导出按钮
            self._safe_access('export_html_btn', lambda x: x.setEnabled(False))
            self._safe_access('export_txt_btn', lambda x: x.setEnabled(False))
            
            # 清除当前报告
            self.current_report = None
        
        except Exception as e:
            print(f"初始化测试结果UI时错误: {e}")
    
    @pyqtSlot(object)
    def _handle_test_completed(self, report):
        """
        处理测试完成信号
        
        参数:
            report: 性能测试报告
        """
        try:
            # 判断是否为提前停止的测试
            was_cancelled = False
            if hasattr(report, 'test_params') and report.test_params.get('cancelled', False):
                was_cancelled = True
                
            # 记录是否暂停完成
            was_paused = False
            if hasattr(report, 'test_params') and report.test_params.get('paused', False):
                was_paused = True
            
            # 更新UI状态
            self._safe_access('start_btn', lambda x: {
                x.setEnabled(True),
                x.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('stop_btn', lambda x: {
                x.setEnabled(False),
                x.setStyleSheet("background-color: #cccccc; color: #888888; font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; padding: 4px;")
            })
            self._safe_access('continue_btn', lambda x: x.setEnabled(False))
            
            # 根据测试结果更新状态文本
            if was_cancelled:
                self._safe_access('operation_status', lambda x: x.setText("测试已取消"))
                self._safe_access('operation_status', lambda x: x.setStyleSheet("font-weight: bold; color: #e74c3c;"))
            elif was_paused:
                self._safe_access('operation_status', lambda x: x.setText("测试已暂停"))
                self._safe_access('operation_status', lambda x: x.setStyleSheet("font-weight: bold; color: #f39c12;"))
            else:
                self._safe_access('operation_status', lambda x: x.setText("测试完成"))
                self._safe_access('operation_status', lambda x: x.setStyleSheet("font-weight: bold; color: #27ae60;"))
            
            # 保存报告引用
            self.current_report = report
            
            # 启用报告导出和管理按钮
            self._safe_access('export_html_btn', lambda x: x.setEnabled(True))
            self._safe_access('export_txt_btn', lambda x: x.setEnabled(True))
            
            # 强制更新测试概述
            details_table = self._safe_access('details_table')
            if details_table:
                self._update_test_summary(details_table)
            
            # 停止测试动画
            self.is_testing = False
            progress_timer = self._safe_access('progress_timer')
            if progress_timer and progress_timer.isActive():
                progress_timer.stop()
                
            # 恢复进度条默认样式
            self._safe_access('progress_bar', lambda x: x.setStyleSheet(""))
            
            # 移除对不存在方法的调用，不需要状态栏刷新
            # self._update_status_bar()
            
        except Exception as e:
            print(f"处理测试完成信号时错误: {e}")
            traceback.print_exc()
    
    def _calculate_model_load_time(self):
        """计算模型载入时间并实时更新显示"""
        try:
            # 获取表格 - 如果不存在则直接返回
            details_table = self._safe_access('details_table')
            if not details_table:
                print("表格控件不存在，无法计算模型载入时间")
                return
                
            # 获取第0轮和其他轮次的测试结果
            all_rows = details_table.rowCount()
            if all_rows < 2:  # 需要至少有第0轮和一个其他轮次
                return
                
            first_round_latency = None
            other_latencies = []
            
            # 先查找第0轮的首Token延迟
            for row in range(all_rows):
                try:
                    round_text = details_table.item(row, 0).text()
                    if round_text == "0":
                        # 找到第0轮，获取其首Token延迟
                        latency_text = details_table.item(row, 3).text()
                        # 修复：去除可能的星号标记，只保留数字部分
                        if "*" in latency_text:
                            latency_text = latency_text.split(' ')[0]
                        first_round_latency = float(latency_text)
                        break
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"查找第0轮数据时出错: {e}")
                    continue
            
            # 如果没有在表格中找到第0轮，则尝试从报告参数中获取
            if first_round_latency is None and self.current_report and 'round_zero_result' in self.current_report.test_params:
                first_round_result = self.current_report.test_params['round_zero_result']
                if first_round_result:
                    first_round_latency = first_round_result.first_token_latency
                    print(f"从报告参数中获取第0轮延迟: {first_round_latency:.2f} ms")
            
            # 收集其他轮次（大于0）的首Token延迟
            for row in range(all_rows):
                try:
                    round_text = details_table.item(row, 0).text()
                    
                    # 处理新的轮次格式 "轮次,并发编号"
                    if "," in round_text:
                        actual_round = round_text.split(",")[0]
                        if actual_round.isdigit() and int(actual_round) > 0:
                            latency_text = details_table.item(row, 3).text()
                            other_latency = float(latency_text)
                            other_latencies.append(other_latency)
                    # 处理原来的格式
                    elif round_text.isdigit() and int(round_text) > 0:
                        latency_text = details_table.item(row, 3).text()
                        other_latency = float(latency_text)
                        other_latencies.append(other_latency)
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"查找其他轮次数据时出错: {e}")
                    continue
            
            # 只要有第0轮延迟和至少一个其他轮次的延迟，就计算模型载入时间
            if first_round_latency is not None and other_latencies:
                avg_other_latency = sum(other_latencies) / len(other_latencies)
                
                # 计算模型载入时间，单位为毫秒
                model_load_time_ms = max(0, first_round_latency - avg_other_latency)
                
                # 转换为秒并更新显示（以秒为单位）
                model_load_time_s = model_load_time_ms / 1000.0
                self._safe_access('model_load_time_label', 
                                 lambda x: x.setText(f"{model_load_time_s:.2f} s"))
                
                # 如果有报告对象，同时更新报告中的数据（保存为秒值）
                if self.current_report is not None:
                    # 保存为秒
                    self.current_report.test_params['model_load_time'] = model_load_time_s
                    print(f"更新模型载入时间: {model_load_time_s:.2f} s")
                    
        except Exception as e:
            print(f"计算模型载入时间错误: {e}")

    def _update_progress_animation(self):
        """更新进度条动画效果"""
        if not self.is_testing:
            self.progress_timer.stop()
            return
            
        # 获取当前进度值和最大值，用于保持格式一致
        current = self.progress_bar.value()
        maximum = self.progress_bar.maximum()
        
        # 更新渐变效果，即使进度值不变也会产生动画效果
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                               stop:0 #3498db, stop:{self.animation_offset:.1f} #2980b9, 
                                               stop:1 #3498db);
                border-radius: 2px;
            }}
        """)
        
        # 确保进度格式保持正确
        self.progress_bar.setFormat(f"{current}/{maximum}")
        
        # 更新偏移量，产生动态效果
        self.animation_offset += 0.1
        if self.animation_offset > 1.0:
            self.animation_offset = 0.0
    
    @pyqtSlot(int, int)
    def _handle_progress_updated(self, completed, total):
        """
        处理进度更新事件
        
        参数:
            completed: 已完成的任务数
            total: 总任务数
        """
        try:
            # 安全地更新进度条值和格式
            progress_bar = self._safe_access('progress_bar')
            if progress_bar:
                # 更新进度条动画效果
                self._update_progress_animation()
                progress_bar.setValue(completed)
                # 确保进度条显示的格式始终正确
                progress_bar.setFormat(f"{completed}/{total}")
                
            # 当第0轮完成时更新状态文字（第一轮测试开始）
            if completed == 0 and total > 0:
                self._safe_access('operation_status', lambda x: {
                    x.setText("正在进行轮次测试..."),
                    x.setStyleSheet("font-weight: bold; color: #27ae60;")
                })
                
        except Exception as e:
            print(f"处理进度更新时错误: {e}")
    
    @pyqtSlot(object, str)
    def _handle_round_completed(self, result, round_num):
        """
        处理单轮测试完成信号
        
        参数:
            result: 测试结果
            round_num: 轮次编号
        """
        try:
            # 获取详情表格的引用
            details_table = self._safe_access('details_table')
            if not details_table:
                return
                
            # 如果是第0轮测试，暂时将它的首token延迟显示为模型载入时间
            if round_num == "0":
                # 更新模型载入时间标签为第0轮的首token延迟时间（转换为秒）
                model_load_time_s = result.first_token_latency / 1000.0
                self._safe_access('model_load_time_label', lambda x: x.setText(f"{model_load_time_s:.2f} s"))
                
                # 如果有报告对象，同时更新报告中的数据
                if hasattr(self, 'current_report') and self.current_report is not None:
                    self.current_report.test_params['model_load_time'] = model_load_time_s
            
            # 存储第一轮测试结果，用于后续计算
            if round_num == "1":
                self.first_round_result = result
                # 当开始正式轮次测试时更新状态文字
                self._safe_access('operation_status', lambda x: {
                    x.setText("正在进行轮次测试..."),
                    x.setStyleSheet("font-weight: bold; color: #27ae60;")
                })
                
            # 转换总时间（毫秒->秒）
            total_time_seconds = result.total_time / 1000
            
            # 插入新行
            row_position = details_table.rowCount()
            details_table.insertRow(row_position)
            
            # 设置单元格内容
            # 轮次编号处理
            if round_num == "0":
                # 第0轮保持不变
                display_round = "0"
            else:
                # 获取当前的并发用户数
                concurrent_users = self._safe_access('concurrency_spin', lambda x: x.value(), 1)
                
                # 根据并发用户数重新格式化轮次编号
                if concurrent_users > 1:
                    # 多并发情况处理
                    # 检查是否已经是正确的格式 "(round,index)"
                    if isinstance(round_num, str) and round_num.startswith("(") and round_num.endswith(")"):
                        # 已经是正确的括号格式，保持原样
                        display_round = round_num
                    elif isinstance(round_num, str) and "," in round_num:
                        # 是逗号分隔但没有括号，添加括号
                        parts = round_num.split(",")
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            display_round = f"({parts[0]},{parts[1]})"
                        else:
                            # 格式不符合预期，使用安全值
                            round_index = details_table.rowCount() % concurrent_users + 1
                            round_num = (details_table.rowCount() // concurrent_users) + 1
                            display_round = f"({round_num},{round_index})"
                    else:
                        # 其他格式，根据行号重新计算轮次和索引
                        try:
                            if isinstance(round_num, int) or (isinstance(round_num, str) and round_num.isdigit()):
                                # 是纯数字，转换为(轮次,索引)格式
                                current_row = int(round_num) if isinstance(round_num, int) else int(round_num)
                                round_part = (current_row - 1) // concurrent_users + 1
                                index_part = (current_row - 1) % concurrent_users + 1
                                display_round = f"({round_part},{index_part})"
                            else:
                                # 异常情况，使用行号重新计算
                                round_index = details_table.rowCount() % concurrent_users + 1
                                round_num = (details_table.rowCount() // concurrent_users) + 1
                                display_round = f"({round_num},{round_index})"
                        except (ValueError, TypeError):
                            # 计算失败，使用安全值
                            round_index = details_table.rowCount() % concurrent_users + 1
                            round_num = (details_table.rowCount() // concurrent_users) + 1
                            display_round = f"({round_num},{round_index})"
                else:
                    # 单并发情况处理，直接使用数字
                    try:
                        if isinstance(round_num, int):
                            display_round = str(round_num)
                        elif isinstance(round_num, str):
                            # 尝试移除所有非数字内容
                            if round_num.isdigit():
                                display_round = round_num
                            elif "," in round_num and "(" in round_num and ")" in round_num:
                                # 是(轮次,索引)格式，只取轮次部分
                                parts = round_num.replace("(", "").replace(")", "").split(",")
                                if parts[0].isdigit():
                                    display_round = parts[0]
                                else:
                                    # 使用行号作为安全值
                                    display_round = str(details_table.rowCount())
                            else:
                                # 使用行号作为安全值
                                display_round = str(details_table.rowCount())
                        else:
                            # 使用行号作为安全值
                            display_round = str(details_table.rowCount())
                    except (ValueError, TypeError, IndexError):
                        # 计算失败，使用行号作为安全值
                        display_round = str(details_table.rowCount())
                    
            details_table.setItem(row_position, 0, QTableWidgetItem(display_round))
            
            # 提示词类型（实际显示提示词名称）
            prompt_name = self._get_prompt_name_by_content(result.prompt_content)
            type_cell = QTableWidgetItem(prompt_name if prompt_name else result.prompt_type)
            # 检查是否有错误，如果有，在类型单元格中添加错误标记
            if result.error:
                type_cell.setText(f"{type_cell.text()} [错误]")
                type_cell.setForeground(QColor("red"))
                type_cell.setToolTip(result.error)
            details_table.setItem(row_position, 1, type_cell)
            
            # 提示词 - 预处理处理内容，去除换行符，显示更多内容
            processed_prompt = result.prompt.replace('\n', ' ').strip()
            details_table.setItem(row_position, 2, QTableWidgetItem(processed_prompt))
            
            # 首Token延迟(ms)
            latency_cell = QTableWidgetItem(f"{result.first_token_latency:.2f}")
            latency_cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # 特殊标记第0轮的延迟
            if round_num == "0":
                latency_cell.setForeground(QColor("#3498db"))  # 蓝色标记
                
            if result.error:
                latency_cell = QTableWidgetItem("0.00")
                latency_cell.setForeground(QColor("red"))
                latency_cell.setToolTip(result.error)
            details_table.setItem(row_position, 3, latency_cell)
            
            # 生成速度(tokens/s)
            speed_cell = QTableWidgetItem(f"{result.tokens_per_second:.2f}")
            speed_cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if result.error:
                speed_cell = QTableWidgetItem("0.00")
                speed_cell.setForeground(QColor("red"))
                speed_cell.setToolTip(result.error)
            details_table.setItem(row_position, 4, speed_cell)
            
            # Token总数
            tokens_cell = QTableWidgetItem(str(result.total_tokens))
            tokens_cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if result.error:
                tokens_cell = QTableWidgetItem("0")
                tokens_cell.setForeground(QColor("red"))
                tokens_cell.setToolTip(result.error)
            details_table.setItem(row_position, 5, tokens_cell)
            
            # 总时间(秒)
            time_cell = QTableWidgetItem(f"{total_time_seconds:.2f}")
            time_cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if result.error:
                time_cell = QTableWidgetItem("0.00")
                time_cell.setForeground(QColor("red"))
                time_cell.setToolTip(result.error)
            details_table.setItem(row_position, 6, time_cell)
            
            # 轮次列也居中显示
            round_cell = details_table.item(row_position, 0)
            round_cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # 提示词类型列也居中显示
            type_cell = details_table.item(row_position, 1)
            type_cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # 自动调整行高
            details_table.resizeRowsToContents()
            
            # 只在第2轮测试结束时首次计算模型加载时间
            if round_num == "2" and details_table.rowCount() >= 2:
                self._calculate_model_load_time()
            
            # 更新进度条 - 只对非第0轮的轮次更新进度
            if round_num != "0":  
                # 获取进度条当前和最大值
                progress_bar = self._safe_access('progress_bar')
                if progress_bar:
                    # 获取总轮次数
                    rounds = self._safe_access('rounds_spin', lambda x: x.value(), 5)
                    # 获取当前并发用户数
                    concurrent_users = self._safe_access('concurrency_spin', lambda x: x.value(), 1)
                    
                    # 计算当前完成的轮次
                    completed_rounds = 0
                    
                    # 遍历详情表格中所有行
                    for r in range(details_table.rowCount()):
                        try:
                            round_text = details_table.item(r, 0).text()
                            
                            # 跳过第0轮
                            if round_text == "0":
                                continue
                                
                            # 处理括号格式: (轮次,索引)
                            if round_text.startswith("(") and round_text.endswith(")"):
                                # 提取括号内的内容
                                inner_text = round_text[1:-1]  # 去掉括号
                                parts = inner_text.split(",")
                                
                                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                    round_num = int(parts[0])
                                    index_num = int(parts[1])
                                    
                                    # 只在每轮的最后一个并发请求完成时更新进度
                                    if index_num == concurrent_users:
                                        completed_rounds = max(completed_rounds, round_num)
                            # 处理单并发的数字格式
                            elif round_text.isdigit():
                                completed_rounds = max(completed_rounds, int(round_text))
                        except (AttributeError, IndexError, ValueError):
                            continue
                    
                    # 更新进度条，确保不超过最大值
                    completed_rounds = min(completed_rounds, rounds)
                    progress_bar.setValue(completed_rounds)
                    progress_bar.setFormat(f"{completed_rounds}/{rounds}")
                
            # 轮次完成后更新进度条
            if round_num != "0":  # 跳过第0轮
                # 获取总轮次数
                total_rounds = self._safe_access('rounds_spin', lambda x: x.value(), 5)
                concurrent_users = self._safe_access('concurrency_spin', lambda x: x.value(), 1)
                max_rounds = total_rounds * concurrent_users
                
                # 计算当前已完成的测试数
                # 检查详情表格中有多少非0轮次的测试
                completed_tests = 0
                for row in range(details_table.rowCount()):
                    try:
                        round_text = details_table.item(row, 0).text()
                        if round_text != "0":
                            completed_tests += 1
                    except (AttributeError, IndexError):
                        continue
                
                # 更新进度条
                progress_bar = self._safe_access('progress_bar')
                if progress_bar:
                    current_value = min(completed_tests, max_rounds)
                    progress_bar.setValue(current_value)
                    progress_bar.setFormat(f"{current_value}/{max_rounds}")
                    
                    # 更新进度条动画效果
                    self._update_progress_animation()
                    
        except Exception as e:
            print(f"处理轮次完成信号时错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_test_summary(self, details_table):
        """
        更新测试结果概述
        
        参数:
            details_table: 测试详情表格
        """
        try:
            all_rows = details_table.rowCount()
            total_latency = 0
            total_speed = 0
            count = 0
            first_round_latency = None  # 用于存储第0轮的首token延迟
            
            # 先找到第0轮的首token延迟
            for row in range(all_rows):
                try:
                    round_text = details_table.item(row, 0).text()
                    if round_text == "0":
                        # 找到第0轮，获取其首Token延迟
                        latency_text = details_table.item(row, 3).text()
                        # 去除可能的星号标记，只保留数字部分
                        if "*" in latency_text:
                            latency_text = latency_text.split(' ')[0]
                        first_round_latency = float(latency_text)
                        break
                except (ValueError, AttributeError, IndexError):
                    continue
            
            # 计算正式轮次的平均值
            for row in range(all_rows):
                try:
                    # 获取轮次，跳过第0轮
                    round_text = details_table.item(row, 0).text()
                    
                    # 跳过第0轮
                    if round_text == "0":
                        continue
                    
                    # 获取当前行的延迟和速度数据
                    latency_item = details_table.item(row, 3)
                    speed_item = details_table.item(row, 4)
                    
                    # 确保数据有效
                    if not latency_item or not speed_item:
                        continue
                    
                    try:
                        latency = float(latency_item.text())
                        speed = float(speed_item.text())
                    except ValueError:
                        continue
                    
                    # 处理括号格式的轮次编号: (轮次,索引)
                    if round_text.startswith("(") and round_text.endswith(")"):
                        inner_text = round_text[1:-1]  # 去掉括号
                        parts = inner_text.split(",")
                        
                        # 有效的轮次数据
                        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) > 0:
                            total_latency += latency
                            total_speed += speed
                            count += 1
                    # 处理纯数字格式 (单并发情况)
                    elif round_text.isdigit() and int(round_text) > 0:
                        total_latency += latency
                        total_speed += speed
                        count += 1
                except (ValueError, AttributeError, IndexError):
                    # 跳过无法解析的值
                    continue
            
            if count > 0:
                avg_latency = total_latency / count
                avg_speed = total_speed / count
                
                # 更新结果概述 - 第二行：首token延迟、平均生成速度
                self._safe_access('model_name_label', lambda x: x.setText(self.current_model))
                self._safe_access('average_latency_label', lambda x: x.setText(f"{avg_latency:.2f} ms"))
                self._safe_access('average_tokens_sec_label', lambda x: x.setText(f"{avg_speed:.2f} tokens/s"))
                
                # 计算真实的模型加载时间（如果有第0轮数据）
                if first_round_latency is not None:
                    # 模型载入时间 = 第0轮延迟 - 平均延迟
                    model_load_time_ms = max(0, first_round_latency - avg_latency)
                    # 转换为秒
                    model_load_time_s = model_load_time_ms / 1000.0
                    self._safe_access('model_load_time_label', lambda x: x.setText(f"{model_load_time_s:.2f} s"))
                    
                    # 如果有报告对象，同时更新报告中的数据
                    if hasattr(self, 'current_report') and self.current_report is not None:
                        self.current_report.test_params['model_load_time'] = model_load_time_s
                
                # 更新结果概述 - 第三行：从输入框获取编码精度、温度、上下文窗口
                precision = self._safe_access('precision_input', lambda x: x.text().strip(), "")
                temperature = self._safe_access('temperature_input', lambda x: x.text().strip(), "0.8")
                context_window = self._safe_access('context_window_input', lambda x: x.text().strip(), "4096")
                
                self._safe_access('precision_input', lambda x: x.setText(precision if precision else ""))
                self._safe_access('temperature_input', lambda x: x.setText(f"{temperature}"))
                self._safe_access('context_window_input', lambda x: x.setText(f"{context_window}"))
                
                # 根据当前性能设置评级和性能评级标签
                if avg_speed > 60:
                    rating = "极佳"
                    self._safe_access('performance_rating_label', 
                                    lambda x: x.setStyleSheet("font-weight: bold; color: #27ae60;"))
                elif avg_speed > 30:
                    rating = "优秀"
                    self._safe_access('performance_rating_label', 
                                    lambda x: x.setStyleSheet("font-weight: bold; color: #2980b9;"))
                elif avg_speed > 10:
                    rating = "良好"
                    self._safe_access('performance_rating_label', 
                                    lambda x: x.setStyleSheet("font-weight: bold; color: #f39c12;"))
                else:
                    rating = "较差"
                    self._safe_access('performance_rating_label', 
                                    lambda x: x.setStyleSheet("font-weight: bold; color: #c0392b;"))
                
                # 更新操作状态和性能评级标签
                self._safe_access('operation_status', lambda x: x.setText(rating))
                self._safe_access('performance_rating_label', lambda x: x.setText(rating))
        except Exception as e:
            print(f"更新测试结果概述时错误: {e}")
    
    @pyqtSlot(str)
    def _handle_test_error(self, error_message):
        """
        处理测试错误事件
        
        参数:
            error_message: 错误消息
        """
        # 恢复UI状态
        self._safe_access('start_btn', lambda x: x.setEnabled(True))
        self._safe_access('stop_btn', lambda x: x.setEnabled(False))
        
        # 停止测试状态
        self.is_testing = False
        progress_timer = self._safe_access('progress_timer')
        if progress_timer and progress_timer.isActive():
            progress_timer.stop()
        
        # 恢复进度条默认样式
        self._safe_access('progress_bar', lambda x: x.setStyleSheet(""))
        
        # 更新操作状态
        self._safe_access('operation_status', lambda x: x.setText("测试错误"))
        self._safe_access('operation_status', lambda x: x.setStyleSheet("font-weight: bold; color: #e74c3c;"))
        
        # 显示错误消息
        QMessageBox.critical(self, "测试错误", error_message)
        
    def _export_html_report(self):
        """预览HTML格式的测试报告"""
        self.update_parameters()

        if not hasattr(self, 'current_report') or self.current_report is None:
            QMessageBox.warning(self, "无可用报告", "当前没有可用的测试报告，请先完成测试")
            return
        
        if not hasattr(self, 'tester') or self.tester is None:
            QMessageBox.error(self, "系统错误", "测试器对象不可用")
            return
        
        try:
            # 确保测试报告包含必要的系统信息
            if not self.current_report.system_info:
                self.current_report.system_info = self.monitor.get_system_info().__dict__ if self.monitor else {}
            
            # 生成HTML报告
            html_report = self.tester.generate_html_report(self.current_report)
            
            # 显示预览对话框
            dialog = ReportPreviewDialog(
                self,
                html_content=html_report,
                text_content=None
            )
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "报告生成错误", f"生成HTML报告时出错: {str(e)}")
    
    def _export_txt_report(self):
        """预览文本格式的测试报告"""
        self.update_parameters()

        if not hasattr(self, 'current_report') or self.current_report is None:
            QMessageBox.warning(self, "无可用报告", "当前没有可用的测试报告，请先完成测试")
            return
        
        if not hasattr(self, 'tester') or self.tester is None:
            QMessageBox.error(self, "系统错误", "测试器对象不可用")
            return
            
        try:
            # 确保测试报告包含必要的系统信息
            if not self.current_report.system_info:
                self.current_report.system_info = self.monitor.get_system_info().__dict__ if self.monitor else {}
            
            # 生成文本报告
            text_report = self.tester.generate_text_report(self.current_report)
            
            # 显示预览对话框
            dialog = ReportPreviewDialog(
                self,
                html_content=None,
                text_content=text_report
            )
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "报告生成错误", f"生成文本报告时出错: {str(e)}")
    
    def update_parameters(self):
        if hasattr(self, 'parameters_input') and self.parameters_input is not None:
            model_params = self.parameters_input.text()
        if hasattr(self, 'precision_input') and self.precision_input is not None:
            precision_input = self.precision_input.text()
        if hasattr(self, 'temperature_input') and self.temperature_input is not None:
            temperature_input = self.temperature_input.text()
        if hasattr(self, 'context_window_input') and self.context_window_input is not None:
            context_window_input = self.context_window_input.text()
        print("当前报告参数:", self.current_report.test_params)
        print("更新参数:", model_params, precision_input, temperature_input, context_window_input)
        # 更新当前报告的测试参数
        test_params = self.current_report.test_params 
        test_params['model_params'] = model_params
        test_params['model_precision'] = precision_input
        test_params['model_temperature'] = float(temperature_input)
        test_params['context_window'] = context_window_input
        # 更新当前报告的测试参数
        self.current_report.test_params = test_params
        print("更新后报告参数:", self.current_report.test_params)

    def set_model(self, model_name: str):
        """
        设置当前使用的模型
        
        参数:
            model_name: 模型名称
        """
        # 直接设置当前模型名称，不再需要更新下拉框
        self.current_model = model_name
    
        # 更新模型名称显示 - 添加安全检查
        try:
            if hasattr(self, 'model_name_label') and self.model_name_label is not None:
                self.model_name_label.setText(model_name)
            
            # 尝试从模型名称中识别参数量并自动填充
            if hasattr(self, 'parameters_input') and self.parameters_input is not None:
                model_params = self.tester.extract_model_parameter(model_name)
                self.parameters_input.setText(model_params if model_params else "未识别")
                
                # 将焦点设置回窗口本身，防止参数输入框获得焦点
                self.setFocus()
        except RuntimeError:
            # 忽略已删除对象的错误
            pass
    
    def on_tab_selected(self):
        """标签页被选中时调用"""
        # 当标签页被选中时，尝试自动填充模型参数（如果当前未填充）
        try:            
            if (hasattr(self, 'parameters_input') and self.parameters_input is not None and 
                    (not self.parameters_input.text() or self.parameters_input.text() == "-" or self.parameters_input.text() == "未识别")):
                if self.current_model:
                    # 使用tester中的提取方法（静态方法）
                    model_params = self.tester.extract_model_parameter(self.current_model)                    
                    self.parameters_input.setText(model_params if model_params else "未识别")
                    
                    # 将焦点设置回窗口本身，防止参数输入框获得焦点
                    self.setFocus()
        except Exception as e:
            print(f"自动填充模型参数时出错: {str(e)}")
            # 忽略错误继续执行
    
    def on_server_changed(self):
        """当服务器改变时重置UI状态"""
        try:
            # 重置当前报告
            self.current_report = None
            
            # 更新UI元素
            self._safe_access('model_name_label', lambda x: x.setText("-"))
            self._safe_access('parameters_input', lambda x: x.setText(""))
            self._safe_access('average_tokens_sec_label', lambda x: x.setText("0 tokens/s"))
            self._safe_access('average_latency_label', lambda x: x.setText("0 ms"))
            self._safe_access('performance_rating_label', lambda x: x.setText("-"))
            self._safe_access('model_load_time_label', lambda x: x.setText("-"))
            
            # 设置默认值
            self._safe_access('precision_input', lambda x: x.setText("INT4"))
            self._safe_access('temperature_input', lambda x: x.setText("0.8"))
            self._safe_access('context_window_input', lambda x: x.setText("128K"))
            
            # 清空详情表格
            details_table = self._safe_access('details_table')
            if details_table:
                details_table.setRowCount(0)
                
            # 禁用导出按钮
            self._safe_access('export_html_btn', lambda x: x.setEnabled(False))
            self._safe_access('export_txt_btn', lambda x: x.setEnabled(False))
            
            # 重置操作状态标签
            self._safe_access('operation_status', lambda x: x.setText("就绪"))
            self._safe_access('operation_status', lambda x: x.setStyleSheet("font-weight: normal; color: black;"))
            
        except Exception as e:
            # 可能是在对象已销毁的情况下尝试访问
            print(f"服务器更改重置UI时错误: {e}")
    
    def on_close(self):
        """应用程序关闭时调用"""
        try:
            # 停止正在进行的测试
            if hasattr(self, 'test_worker') and self.test_worker is not None and self.test_worker.isRunning():
                self.test_worker.terminate()
                self.test_worker = None 
        
            # 停止进度动画定时器
            if hasattr(self, 'progress_timer') and self.progress_timer is not None and self.progress_timer.isActive():
                self.progress_timer.stop()
        except RuntimeError:
            # 忽略已删除对象的错误
            pass 

class PerformanceTestWorker(QThread):
    """性能测试工作线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, int)
    test_completed = pyqtSignal(object)
    test_round_completed = pyqtSignal(object, str)  # 每轮测试完成的信号
    error_occurred = pyqtSignal(str)
    
    def __init__(
        self, 
        client: OllamaClient,
        model_name: str,
        rounds: int = 3,  # 默认轮数
        test_params: Dict[str, Any] = None,
        concurrent_users: int = 2,  # 默认并发用户数
        selected_prompts: Optional[List[TestPrompt]] = None
    ):
        """
        初始化性能测试工作线程
        
        参数:
            client: Ollama API客户端
            model_name: 模型名称
            rounds: 测试轮次
            test_params: 测试参数
            concurrent_users: 并发用户数
            selected_prompts: 已选择的提示词列表
        """
        super().__init__()
        self.client = client
        self.model = model_name
        self.num_tests = rounds
        self.test_params = test_params
        self.concurrent_users = concurrent_users
        self.selected_prompts = selected_prompts
        self.actual_tests = rounds + 1  # 实际执行的测试轮次比显示的多一轮，用于计算模型载入时间
        self.first_round_result = None  # 用于保存第一轮的测试结果
        self.is_cancelled = False  # 添加取消标志
        self.is_paused = False     # 添加暂停标志
        self.current_round = 0     # 当前测试轮次

        if self.client.check_parallel_support():
            self.client.configure_parallel(3,1)
        
        # 记录测试参数
        self.test_params['rounds'] = rounds
        self.test_params['concurrent_users'] = concurrent_users
        
        # 创建测试器
        from ...utils.system_monitor import SystemMonitor
        self.system_monitor = SystemMonitor()
        self.tester = PerformanceTester(self.client, self.system_monitor)
        
        # 保存原始并发配置，以便测试结束后恢复
        self.original_parallel_config = None
        self.parallel_modified = False
    
    def cancel_test(self):
        """取消测试"""
        self.is_cancelled = True
        print("测试取消请求已提交 - 等待当前任务完成后停止测试")
    
    def pause_test(self):
        """暂停测试"""
        self.is_paused = True
    
    def resume_test(self):
        """继续测试"""
        self.is_paused = False

    def run(self):
        """运行测试线程"""
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("-"*50)
        print(f"开始测试 {self.model} 的性能 {date_time}")
        print("-"*50)

        def config_restart_ollama_service(num_users,max_models=1):
            if num_users is None:
                success1, message = (True,"")
            else:
                success1, message = self.client.configure_parallel(num_users,max_models,False)
            success2 = False
            max_attempts = 3
            attempt_count = 0
            
            print("\n正在重启Ollama服务...")         
            # 首先确保服务停止
            if self.client.service_controller.is_running():
                stop_success = self.client.service_controller.stop_service()
                if not stop_success:
                    print("警告: Ollama服务停止失败，将尝试强制停止...")
                    time.sleep(1)  # 等待一段时间再检查
            
            # 确认服务已经停止
            while self.client.service_controller.is_running() and attempt_count < max_attempts:
                print(f"等待Ollama服务停止... (尝试 {attempt_count+1}/{max_attempts})")
                time.sleep(1)
                attempt_count += 1
            
            # 重置尝试计数
            attempt_count = 0
            
            # 尝试启动服务
            while not self.client.service_controller.is_running() and attempt_count < max_attempts:
                print(f"尝试启动Ollama服务... (尝试 {attempt_count+1}/{max_attempts})")
                start_success = self.client.service_controller.start_service()
                if start_success:
                    message="Ollama服务已成功启动"
                    success2 = True
                    break
                attempt_count += 1
                time.sleep(2)  # 等待几秒后再次尝试
            return success1 and success2,message
        
        # 添加异常处理和资源管理
        executor = None
        try:
            # 总轮次包括：1个第0轮 + (轮次数 * 并发用户数)的正式测试
            total_tests = self.num_tests * self.concurrent_users
            total_rounds = 1 + total_tests  # 加上第0轮
            
            # 更新进度条总数 - 但不显示第0轮
            self.progress_updated.emit(0, total_tests)
            
            # 检查并发用户设置
            parallel_warning = None
            clients = [self.tester.client]
            
            # 获取系统信息
            system_info = self.tester.system_monitor.get_system_info()

            test_params = self.test_params.copy()
            test_params['test_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 初始化性能报告
            report = PerformanceReport(
                model_name=self.model,
                system_info=system_info,
                test_params=test_params,
                results=[]
            )
            # print(f"当前系统信息: {system_info}")

            self.original_parallel_config = self.client.get_current_config()
            # print(f"\n记录原始并发配置: {self.original_parallel_config}")

            #重启ollama服务，以保证第0轮测试的准确性
            print("\n为测试模型载入速度，需要重启Ollama服务")
            success_restart, message = config_restart_ollama_service(None)            
            if success_restart:
                print(f"已经完成重启ollama服务\n")

            # 第0轮使用特殊提示词 - 模型载入测试
            try:
                print(f"\n执行模型载入测试")
                special_prompt = TestPrompt("载入测试", "模型载入测试", f"你好 {self.model}")
                result_zero = self.run_single_test(
                    model=self.model,
                    prompt=special_prompt.content,
                    prompt_type=special_prompt.category,
                    timeout=60  # 第0轮最多等待60秒
                )
                report.results.append(result_zero)
                self.test_round_completed.emit(result_zero, "0")
            except Exception as e:
                error_result = TestResult(
                    prompt="载入测试失败",
                    prompt_content=f"你好 {self.model}",
                    prompt_type="载入测试",
                    response_text=f"错误: {str(e)}",
                    first_token_latency=0,
                    tokens_per_second=0,
                    total_tokens=0,
                    total_time=0,
                    error=str(e)
                )
                report.results.append(error_result)
                self.test_round_completed.emit(error_result, "0")
            
            # 如果并发用户数大于1，则需要创建多个独立客户端实例
            if self.concurrent_users > 1:
                # 获取Ollama版本,检查版本是否支持多实例并发,获取当前并发配置
                version, error = self.client.get_version()
                supports_parallel = self.client.check_parallel_support(version)
                if error:
                    parallel_warning = f"无法检测Ollama版本: {error}，并发测试可能无法正常工作"
                if not supports_parallel:
                    parallel_warning = f"当前Ollama版本 {version} 不支持多用户并发，请升级到0.1.33及以上版本。测试将以单用户模式进行，可能无法体现真实的并发性能。"
                    self.concurrent_users = 1     
                
                # 检查并调整并发配置
                if self.client.get_current_config()['num_parallel'] < self.concurrent_users:
                    print(f"当前并发配置: {self.client.get_current_config()},当前并发用户数: {self.concurrent_users}")
                    print("\n为使并发设置生效，测试并发性能，需要重启Ollama服务")
                    success_restart, message = config_restart_ollama_service(self.concurrent_users)            
                    if success_restart:
                        self.parallel_modified = True
                        print(f"已临时开启多用户并发: {message}")
                    else:
                        parallel_warning = f"并发设置失败: {message}，使用单用户模式"
                        self.concurrent_users = 1
                        clients = [self.tester.client]
                    success , _ = self.tester.test_model(self.model)
                    if not success:
                        parallel_warning = f"并发测试失败: {message}，使用单用户模式"
                    
                # 如果有并发相关的警告，添加到测试参数中
                if parallel_warning:
                    self.test_params['parallel_warning'] = parallel_warning
                    print(parallel_warning)
                # 创建多个独立客户端实例
                clients = []
                for i in range(self.concurrent_users):
                    new_client = OllamaClient(base_url=self.tester.client.base_url)
                    clients.append(new_client)

            # 准备正式测试轮次
            completed = 0  # 已完成的测试数（不包括第0轮）
            prompts = self.tester.select_balanced_prompts(total_tests)  # 平衡选择提示词
            
            # 设置任务超时时间（秒），防止单个任务卡住整个测试
            task_timeout = 180  # 3分钟超时
            
            # 执行正式测试轮次
            for round_num in range(self.num_tests):
                print(f"\n执行第{round_num+1}轮测试")
                # 检查取消标志
                if self.is_cancelled:
                    break
                
                # 处理暂停状态
                while self.is_paused and not self.is_cancelled:
                    report.test_params['paused'] = True
                    report.test_params['completion_status'] = "暂停完成"
                    # print("测试被暂停 - 不再继续执行")
                    time.sleep(2)
                    continue
                
                # 获取本轮所有提示词
                start_idx = round_num * self.concurrent_users
                end_idx = start_idx + self.concurrent_users
                current_prompts = prompts[start_idx:end_idx]
                
                # 明确定义当前轮次编号（从1开始计数）
                current_round = round_num + 1
                
                # 使用线程池执行并发测试 - 限制最大工作线程数
                # 为每轮测试创建新的线程池，避免线程资源耗尽
                with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
                    futures = []
                    # 创建一个映射存储每个future对应的轮次和并发编号
                    future_to_round_info = {}
                    
                    # 第一步: 同时提交所有并发请求，确保它们尽可能同时开始
                    # 明确跟踪并发编号（从1开始计数）
                    for concurrent_index, prompt in enumerate(current_prompts, 1):
                        # 分配客户端实例
                        client = clients[(concurrent_index-1) % len(clients)]                        
                        # 计算当前轮次的显示值，使用明确定义的轮次和并发编号
                        if self.concurrent_users > 1:
                            # 多并发用户: 使用括号格式 (轮次,并发编号)
                            # print(f"当前轮次: {current_round}, 并发编号: {concurrent_index}")
                            current_display_round = f"({current_round},{concurrent_index})"
                        else:
                            # 单用户: 直接使用轮次数字
                            current_display_round = current_round
                        
                        # 定义要执行的函数，增强错误处理
                        def test_task(p, pt, c, dr):
                            try:
                                # print(f"启动测试任务 - 轮次: {dr}, 客户端: {c.base_url if hasattr(c, 'base_url') else '未知'}")
                                result = self.run_single_test(
                                    model=self.model,
                                    prompt=p,
                                    prompt_type=pt,
                                    client=c,
                                    timeout=task_timeout  # 添加超时机制
                                )
                                return (result, dr)
                            except Exception as e:
                                print(f"测试任务执行出错 - 轮次: {dr}, 错误: {str(e)}")
                                raise  # 重新抛出异常，让外层catch处理
                        
                        # 提交任务到线程池 - 此时任务会立即被调度执行
                        future = executor.submit(
                            test_task, 
                            prompt.content,
                            prompt.category,
                            client,
                            current_display_round
                        )
                        
                        # 保存future与轮次信息的映射
                        future_to_round_info[future] = {
                            'round': current_round, 
                            'index': concurrent_index, 
                            'prompt': prompt
                        }
                        futures.append(future)
                    
                    # 第二步: 等待所有任务完成并处理结果
                    # 注意: 这里不使用as_completed，因为我们希望每个请求的首token延迟是
                    # 独立计算的，而不是基于它们完成的顺序
                    for future in futures:
                        if self.is_cancelled:
                            # 如果测试被取消，尝试取消剩余任务
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
                            
                        # 获取该future对应的轮次信息
                        round_info = future_to_round_info.get(future, {})
                        current_round = round_info.get('round', 1)
                        concurrent_index = round_info.get('index', 1)
                        current_prompt = round_info.get('prompt')
                            
                        try:
                            # 使用timeout获取结果，防止任务卡住
                            result, display_round = future.result(timeout=task_timeout)
                            report.results.append(result)
                            self.test_round_completed.emit(result, str(display_round))
                            completed += 1
                        except TimeoutError:
                            # 使用存储的轮次信息创建错误结果
                            if current_prompt:
                                prompt_text = current_prompt.content
                                prompt_type = current_prompt.category
                                prompt_display = current_prompt.name
                            else:
                                # 如果没有找到轮次信息，使用默认值
                                prompt_text = "未知提示词"
                                prompt_type = "未知"
                                prompt_display = "轮次错误"
                                
                            # 使用存储的轮次和并发编号创建显示轮次
                            if self.concurrent_users > 1:
                                error_round_num = f"({current_round},{concurrent_index})"
                            else:
                                error_round_num = str(current_round)
                                
                            error_message = f"任务执行超时（超过{task_timeout}秒）"
                            error_result = TestResult(
                                prompt=prompt_display,
                                prompt_content=prompt_text,
                                prompt_type=prompt_type,
                                response_text=error_message,
                                first_token_latency=0,
                                tokens_per_second=0,
                                total_tokens=0,
                                total_time=0,
                                error=error_message
                            )
                            report.results.append(error_result)
                            self.test_round_completed.emit(error_result, error_round_num)
                            completed += 1
                        except Exception as e:
                            # 使用存储的轮次信息创建错误结果
                            if current_prompt:
                                prompt_text = current_prompt.content
                                prompt_type = current_prompt.category
                                prompt_display = current_prompt.name
                            else:
                                # 如果没有找到轮次信息，使用默认值
                                prompt_text = "未知提示词"
                                prompt_type = "未知"
                                prompt_display = "轮次错误"
                            
                            # 使用存储的轮次和并发编号创建显示轮次
                            if self.concurrent_users > 1:
                                error_round_num = f"({current_round},{concurrent_index})"
                            else:
                                error_round_num = str(current_round)
                                
                            error_message = f"执行测试时出错: {str(e)}"
                            error_result = TestResult(
                                prompt=prompt_display,
                                prompt_content=prompt_text,
                                prompt_type=prompt_type,
                                response_text=error_message,
                                first_token_latency=0,
                                tokens_per_second=0,
                                total_tokens=0,
                                total_time=0,
                                error=str(e)
                            )
                            report.results.append(error_result)
                            self.test_round_completed.emit(error_result, error_round_num)
                            completed += 1

            # 恢复原始并发配置（如果有修改）
            # if self.parallel_modified and self.original_parallel_config:
            #     try:
            #         success, msg = self.client.configure_parallel(self.original_parallel_config["num_parallel"],self.original_parallel_config["max_models"],False)
            #         # print(f"\n恢复原始并发配置: {msg}")
            #     except Exception as e:
            #         print(f"\n恢复配置失败: {e}")
            
            # 生成最终报告评级
            if len(report.results) > 1:
                original_results = report.results.copy()
                # 计算模型载入时间（第0轮与后续轮次的差异）
                other_latencies = [r.first_token_latency for r in report.results[1:] if r.first_token_latency > 0]
                if other_latencies:
                    avg_latency = sum(other_latencies) / len(other_latencies)
                    
                    # 计算模型载入时间（毫秒）
                    model_load_time_ms = max(0, report.results[0].first_token_latency - avg_latency)
                    # 转换为秒
                    model_load_time_s = model_load_time_ms / 1000.0
                    report.test_params['model_load_time'] = model_load_time_s
                    report.test_params['avg_latency'] = avg_latency
                
                # 计算平均性能指标（排除第0轮）
                valid_results = [r for r in report.results[1:] if not r.error and r.tokens_per_second > 0]
                if valid_results:
                    report.avg_first_token_latency = sum(r.first_token_latency for r in valid_results) / len(valid_results)
                    report.avg_tokens_per_second = sum(r.tokens_per_second for r in valid_results) / len(valid_results)
                    # 计算性能评级
                    if report.avg_tokens_per_second > 60:
                        report.performance_rating = "极佳"
                    elif report.avg_tokens_per_second > 30:
                        report.performance_rating = "优秀"
                    elif report.avg_tokens_per_second > 10:
                        report.performance_rating = "良好"
                    else:
                        report.performance_rating = "较差"
            
            # 恢复完整的结果列表，保留第0轮结果在UI显示中
            # report.results = original_results
            
            # 更新进度条到最终状态
            self.progress_updated.emit(total_tests, total_tests)
            
            # 提交最终测试报告
            self.test_completed.emit(report)
            print(f"\n测试完成，已经生成报告")       

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("-"*50)
        print(f"模型 {self.model} 性能测试完成 {date_time}")
        print("-"*50)
            
    def _progress_callback(self, completed, total):
        """进度回调函数"""
        if not self.is_cancelled:
            self.progress_updated.emit(completed, total)
        
    def _round_callback(self, result, round_num):
        """轮次完成回调函数"""
        if not self.is_cancelled:
            # 发送轮次完成信号
            self.test_round_completed.emit(result, round_num)
            
            # 跳过第0轮的进度更新
            if round_num == "0":
                return
                
            # 基于轮次编号计算进度
            try:
                # 处理字符串格式的轮次编号
                # 处理 "(1,2)" 或 "1,2" 格式
                if "," in round_num:
                    # 提取轮次部分
                    cleaned_num = round_num.replace("(", "").replace(")", "")
                    parts = cleaned_num.split(",")
                    
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        actual_round = int(parts[0])
                        index = int(parts[1])
                        
                        # 只在每轮的最后一个并发请求完成时更新进度
                        if index == self.concurrent_users:
                            self._progress_callback(actual_round, self.num_tests)
                else:
                    # 非并发格式，尝试直接转为数字
                    try:
                        round_value = int(round_num)
                        self._progress_callback(round_value, self.num_tests)
                    except ValueError:
                        pass
            except Exception as e:
                print(f"处理轮次回调进度更新时出错: {e}")
    
    def run_single_test(self, model: str, prompt: str, prompt_type: str = "未分类", system: str = "", client = None, timeout=None) -> TestResult:
        """
        执行单次测试
        
        参数:
            model: 模型名称
            prompt: 提示词内容
            prompt_type: 提示词类型/分类
            system: 系统提示词
            client: 客户端实例
            timeout: 超时时间（秒）
            
        返回:
            测试结果对象
        """
        # 如果未指定客户端，使用当前的客户端
        if client is None:
            client = self.tester.client
        
        try:            
            # 设置默认超时
            if timeout is None:
                timeout = 600  # 默认10分钟超时
                
            # 执行测试
            return self.tester.test_case(
                model=model,
                prompt=prompt,
                prompt_type=prompt_type,
                system=system,
                client=client,
                timeout=timeout  # 添加超时参数
            )
            
        except Exception as e:
            # 创建错误结果
            error_result = TestResult(
                prompt=prompt[:50] + "...",
                prompt_content=prompt,
                prompt_type=prompt_type,
                response_text=f"错误: {str(e)}",
                first_token_latency=0,
                tokens_per_second=0,
                total_tokens=0,
                total_time=0,
                error=str(e)
            )
            return error_result






