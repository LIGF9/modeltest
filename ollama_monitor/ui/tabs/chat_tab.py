#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聊天测试标签页模块
提供与模型交互的聊天界面
"""

import time
import threading
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import random
from functools import partial

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QScrollArea, QFrame, QLabel, 
    QSizePolicy, QDialog, QLineEdit, QSplitter,
    QComboBox, QMessageBox, QMenu, QCheckBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread, QRect, QTimer
from PyQt6.QtGui import QColor, QPalette, QTextCursor, QKeyEvent, QAction, QFont

from ...utils.ollama_client import OllamaClient


@dataclass
class ChatMessage:
    """聊天消息数据类"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: str = ""
    
    def __post_init__(self):
        """初始化后设置时间戳"""
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


class MessageBubble(QFrame):
    """气泡式聊天消息组件"""
    
    def __init__(self, message: ChatMessage, parent=None):
        """
        初始化消息气泡
        
        参数:
            message: 聊天消息对象
            parent: 父级部件
        """
        super().__init__(parent)
        
        # 设置框架样式
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setAutoFillBackground(True)
        
        # 根据消息角色设置气泡样式
        self._set_style(message.role)
        
        # 创建布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(3)
        
        # 添加角色标签（仅系统消息显示）
        if message.role == 'system':
            message.content = "系统："+message.content
        
        # 添加内容
        content = QLabel(message.content)
        content.setWordWrap(True)
        content.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # 移除内容框边框
        content.setStyleSheet("border: none;")
        main_layout.addWidget(content)
        
        # 添加时间戳
        timestamp_label = QLabel(message.timestamp)
        timestamp_label.setStyleSheet("color: #888; font-size: 9px; border: none;")
        timestamp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(timestamp_label)
        
        # 设置大小策略
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    
    def _set_style(self, role: str):
        """
        根据消息角色设置气泡样式
        
        参数:
            role: 消息角色
        """
        palette = self.palette()
        
        if role == 'user':
            # 用户消息 - 绿色背景
            palette.setColor(QPalette.ColorRole.Base, QColor(226, 255, 226))
            self.setStyleSheet("""
                QFrame {
                    border-radius: 10px;
                    border: 1px solid #d1e9d1;
                    background-color: #e2ffe2;
                }
            """)
            # 设置右对齐
            self.setContentsMargins(100, 5, 20, 5)
        
        elif role == 'assistant':
            # 模型回复 - 淡蓝色背景
            palette.setColor(QPalette.ColorRole.Base, QColor(225, 240, 255))
            self.setStyleSheet("""
                QFrame {
                    border-radius: 10px;
                    border: 1px solid #d0e1f5;
                    background-color: #e1f0ff;
                }
            """)
            # 设置左对齐
            self.setContentsMargins(20, 5, 100, 5)
        
        else:  # 系统消息
            # 系统消息 - 灰色背景
            palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))
            self.setStyleSheet("""
                QFrame {
                    border-radius: 10px;
                    border: 1px solid #ddd;
                    background-color: #f0f0f0;
                }
            """)
            # 设置居中
            self.setContentsMargins(60, 5, 60, 5)
        
        self.setPalette(palette)


class SystemPromptDialog(QDialog):
    """系统提示词设置对话框"""
    
    def __init__(self, current_prompt: str = "", parent=None):
        """
        初始化对话框
        
        参数:
            current_prompt: 当前系统提示词
            parent: 父级部件
        """
        super().__init__(parent)
        
        self.setWindowTitle("设置系统提示词")
        self.resize(500, 300)
        
        # 主布局
        layout = QVBoxLayout(self)
        
        # 提示文本
        info_label = QLabel("系统提示词可以设置模型的角色和行为指导。")
        layout.addWidget(info_label)
        
        # 预设提示词选择
        preset_layout = QHBoxLayout()
        preset_label = QLabel("预设:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("自定义")
        self.preset_combo.addItem("助手", "你是一个乐于助人的AI助手。")
        self.preset_combo.addItem("专家", "你是一位拥有广泛知识的AI专家，能够提供深入而专业的解答。")
        self.preset_combo.addItem("数学导师", "你是一位数学导师，擅长用清晰的方式解释数学概念和解答问题。")
        self.preset_combo.addItem("代码专家", "你是一位编程专家，专注于提供高质量、可维护的代码示例和解决方案。")
        self.preset_combo.addItem("创意伙伴", "你是一位创意伙伴，善于激发灵感并帮助发展创新想法。")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)
        
        # 文本编辑框
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("在此输入系统提示词...")
        self.text_edit.setText(current_prompt)
        layout.addWidget(self.text_edit)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 清空按钮
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self.text_edit.clear)
        
        # 取消和确定按钮
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.accept)
        save_button.setDefault(True)
        
        button_layout.addWidget(clear_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
    
    def _on_preset_changed(self, index: int):
        """
        预设提示词选择变更处理
        
        参数:
            index: 选项索引
        """
        if index > 0:  # 不是"自定义"选项
            preset_text = self.preset_combo.currentData()
            self.text_edit.setText(preset_text)
    
    def get_prompt(self) -> str:
        """
        获取编辑后的提示词
        
        返回:
            提示词文本
        """
        return self.text_edit.toPlainText().strip()


class ChatWorker(QThread):
    """聊天处理线程"""
    
    # 信号定义
    response_received = pyqtSignal(str)
    thinking_status = pyqtSignal(bool)
    chat_completed = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, client: OllamaClient, model: str, messages: List[Dict[str, str]]):
        """
        初始化聊天线程
        
        参数:
            client: Ollama API客户端
            model: 模型名称
            messages: 聊天消息列表
        """
        super().__init__()
        
        self.client = client
        self.model = model
        self.messages = messages
        self.stop_requested = False
    
    def run(self):
        """线程执行函数"""
        try:
            # 发送思考状态信号
            self.thinking_status.emit(True)
            
            # 拼接响应文本
            response_text = ""
            
            # 发送聊天请求
            for chunk in self.client.chat_completion(self.model, self.messages):
                # 检查是否请求停止
                if self.stop_requested:
                    break
                
                # 处理错误
                if "error" in chunk:
                    self.error_occurred.emit(f"聊天请求失败: {chunk['error']}")
                    break
                
                # 处理响应内容
                if "message" in chunk:
                    message = chunk.get("message", {})
                    content = message.get("content", "")
                    
                    if content:
                        response_text += content
                        self.response_received.emit(response_text)
            
            # 聊天完成（无论是正常完成还是中断）
            self.chat_completed.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"聊天处理异常: {str(e)}")
        
        finally:
            # 恢复状态
            self.thinking_status.emit(False)
    
    def stop(self):
        """请求停止处理"""
        self.stop_requested = True


class ChatTab(QWidget):
    """聊天测试标签页类"""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        初始化聊天测试标签页
        
        参数:
            ollama_client: Ollama API客户端
        """
        super().__init__()
        
        # 保存客户端引用
        self.client = ollama_client
        
        # 初始化聊天状态
        self.current_model = ""
        self.messages = []
        self.system_prompt = "你是一个乐于助人的AI助手。"
        self.chat_worker = None
        
        # 初始化UI
        self._init_ui()
        
        # 添加初始化系统消息
        self._add_system_message("欢迎使用聊天测试。请在下方输入框中输入您的问题，按回车键发送。")
    
    def _init_ui(self):
        """初始化UI元素"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        
        # 创建上部聊天区域
        chat_area = QScrollArea()
        chat_area.setWidgetResizable(True)
        chat_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # 聊天内容容器
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_layout.setContentsMargins(5,5,5,5)
        
        chat_area.setWidget(self.chat_container)
        
        # 创建底部输入区域
        input_area = QWidget()
        input_layout = QVBoxLayout(input_area)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        # 系统提示词按钮
        self.prompt_button = QPushButton("系统提示词")
        self.prompt_button.clicked.connect(self._open_system_prompt_dialog)
        toolbar_layout.addWidget(self.prompt_button)
        
        # 停止生成按钮
        self.stop_button = QPushButton("停止生成")
        self.stop_button.clicked.connect(self._stop_generation)
        self.stop_button.setEnabled(False)
        toolbar_layout.addWidget(self.stop_button)
        
        # 状态标签
        self.status_label = QLabel("模型准备就绪")
        self.status_label.setStyleSheet("color: #27ae60;") # 绿色表示就绪
        toolbar_layout.addWidget(self.status_label)
        
        # 添加空白区域
        toolbar_layout.addStretch()
        
        # 清空聊天按钮
        self.clear_button = QPushButton("清空聊天")
        self.clear_button.clicked.connect(self._clear_chat)
        toolbar_layout.addWidget(self.clear_button)
        
        input_layout.addLayout(toolbar_layout)
        
        # 输入框
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("在此输入您的问题...\n按Enter发送，Ctrl+Enter换行")
        self.input_box.setMinimumHeight(25)  # 调整为适合三行文本的高度
        self.input_box.setMaximumHeight(100)
        self.input_box.keyPressEvent = self._input_key_press
        input_layout.addWidget(self.input_box)
        
        # 引导问题和发送按钮区域（合并到同一行）
        bottom_area_layout = QHBoxLayout()
        bottom_area_layout.setContentsMargins(5, 5, 5, 5)
        
        # 提问引导标签
        guide_label = QLabel("您可能想问：")
        guide_label.setStyleSheet("color: #666666;")
        bottom_area_layout.addWidget(guide_label)
        
        # 创建四个动态引导按钮
        self.guide_buttons = []
        for i in range(4):  # 4个按钮
            btn = QPushButton("")
            btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 4px; padding: 3px 8px; color: #555555;")
            btn.setVisible(False)  # 初始状态下不可见
            bottom_area_layout.addWidget(btn)
            self.guide_buttons.append(btn)
        
        # 添加弹性空间，使发送按钮靠右
        bottom_area_layout.addStretch()
        
        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self._send_message)
        self.send_button.setDefault(True)
        self.send_button.setMinimumWidth(80)  # 设置最小宽度
        bottom_area_layout.addWidget(self.send_button)
        
        input_layout.addLayout(bottom_area_layout)
        
        # 创建分隔器，允许调整聊天区域和输入区域的比例
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(chat_area)
        splitter.addWidget(input_area)
        
        # 设置初始大小比例
        splitter.setSizes([700, 200])
        
        # 添加到主布局
        main_layout.addWidget(splitter)
        
        # 设置布局
        self.setLayout(main_layout)
        
        # 初始化自动滚动计时器
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self._auto_scroll)
        self.scroll_timer.setSingleShot(True)
        
        # 初始化引导按钮内容
        self._update_initial_guide_buttons()
    
    def _input_key_press(self, event: QKeyEvent):
        """
        输入框按键事件处理
        
        参数:
            event: 按键事件
        """
        # 检查是否按下Enter键（不是Ctrl+Enter）
        if (event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._send_message()
            return
        
        # 如果是Ctrl+Enter，插入换行符
        if (event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            QTextEdit.keyPressEvent(self.input_box, event)
            return
        
        # 其他按键正常处理
        QTextEdit.keyPressEvent(self.input_box, event)
    
    def _add_message(self, message: ChatMessage):
        """
        添加消息到聊天区域
        
        参数:
            message: 聊天消息对象
        """
        # 创建消息气泡
        bubble = MessageBubble(message, self)
        
        # 添加到聊天布局
        self.chat_layout.addWidget(bubble)
        
        # 延迟滚动到底部
        self.scroll_timer.start(100)
    
    def _add_system_message(self, content: str):
        """
        添加系统消息
        
        参数:
            content: 消息内容
        """
        message = ChatMessage(
            role="system",
            content=content
        )
        self._add_message(message)
        
        # 更新引导按钮，确保即使只有系统消息也能显示引导按钮
        self._update_initial_guide_buttons()
    
    def _add_user_message(self, content: str):
        """
        添加用户消息
        
        参数:
            content: 消息内容
        """
        message = ChatMessage(
            role="user",
            content=content
        )
        self._add_message(message)
        
        # 添加到消息历史
        self.messages.append({
            "role": "user",
            "content": content
        })
    
    def _add_assistant_message(self, content: str):
        """
        添加模型回复消息
        
        参数:
            content: 消息内容
        """
        message = ChatMessage(
            role="assistant",
            content=content
        )
        self._add_message(message)
        
        # 添加到消息历史
        self.messages.append({
            "role": "assistant",
            "content": content
        })
    
    def _add_thinking_message(self):
        """添加模型思考中的提示消息"""
        # 添加一个临时的系统消息，表示模型正在思考
        message = ChatMessage(
            role="system",
            content="模型思考中..."
        )
        self._add_message(message)
    
    def _send_message(self):
        """发送用户消息并获取模型回复"""
        # 获取用户输入
        user_input = self.input_box.toPlainText().strip()
        
        # 验证输入
        if not user_input:
            return
        
        # 验证模型是否已设置
        if not self.current_model:
            self._add_system_message("错误：未选择模型。请在顶部工具栏中选择一个模型。")
            return
        
        # 清空输入框
        self.input_box.clear()
        
        # 添加用户消息
        self._add_user_message(user_input)
        
        # 添加"模型生成中..."提示
        self._add_thinking_message()
        
        # 准备进行请求
        all_messages = []
        
        # 添加系统提示词（如果有）
        if self.system_prompt:
            all_messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # 添加聊天历史
        all_messages.extend(self.messages)
        
        # 禁用输入区域
        self.input_box.setEnabled(False)
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("思考中...")
        
        # 移除卸载当前模型的操作，直接向目标模型发送消息
        # 创建并启动聊天线程
        self.chat_worker = ChatWorker(self.client, self.current_model, all_messages)
        self.chat_worker.response_received.connect(self._handle_response)
        self.chat_worker.thinking_status.connect(self._handle_thinking_status)
        self.chat_worker.chat_completed.connect(self._handle_chat_completed)
        self.chat_worker.error_occurred.connect(self._handle_error)
        self.chat_worker.start()
    
    @pyqtSlot(str)
    def _handle_response(self, response_text: str):
        """
        处理模型响应
        
        参数:
            response_text: 响应文本
        """
        # 移除"思考中"的系统消息（如果存在）
        if self.chat_layout.count() > 0:
            last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if last_item and last_item.widget():
                # 检查最后一个消息是否为系统消息且内容为"模型思考中..."
                # 由于无法直接访问MessageBubble内部的内容，我们只能通过消息历史间接判断
                if (len(self.messages) > 0 and self.messages[-1]["role"] == "user") or \
                   (self.chat_layout.count() > 1 and self.chat_layout.itemAt(self.chat_layout.count() - 2).widget()):
                    # 如果最后一个消息是用户消息，或者前一个消息存在，说明当前最后一个可能是"思考中"提示
                    # 安全地移除它
                    last_item.widget().deleteLater()
        
        # 添加新的助手消息到消息列表
        if self.messages and self.messages[-1]["role"] == "assistant":
            # 如果最后一条是助手消息，更新它
            self.messages[-1]["content"] = response_text
        else:
            # 否则添加新消息
            self.messages.append({
                "role": "assistant",
                "content": response_text
            })
        
        # 添加助手消息到UI
        message = ChatMessage(
            role="assistant",
            content=response_text
        )
        self._add_message(message)
    
    @pyqtSlot(bool)
    def _handle_thinking_status(self, is_thinking: bool):
        """
        处理思考状态变化
        
        参数:
            is_thinking: 是否正在思考
        """
        if is_thinking:
            self.status_label.setText("生成中...")
            self.status_label.setStyleSheet("color: #0066CC;") # 蓝色表示生成中
        else:
            self.status_label.setText("模型准备就绪")
            self.status_label.setStyleSheet("color: #27ae60;") # 绿色表示就绪
    
    @pyqtSlot()
    def _handle_chat_completed(self):
        """处理聊天完成事件"""
        # 恢复输入区域
        self.input_box.setEnabled(True)
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("模型准备就绪")
        self.status_label.setStyleSheet("color: #27ae60;") # 绿色表示就绪
        
        # 更新引导按钮
        self._update_guide_buttons()
        
        # 清除worker引用
        self.chat_worker = None
    
    @pyqtSlot(str)
    def _handle_error(self, error_message: str):
        """
        处理错误
        
        参数:
            error_message: 错误消息
        """
        # 移除"思考中"的系统消息（如果存在）
        if self.chat_layout.count() > 0:
            last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if last_item and last_item.widget():
                # 使用与_handle_response相同的逻辑确定是否为思考中消息
                if (len(self.messages) > 0 and self.messages[-1]["role"] == "user") or \
                   (self.chat_layout.count() > 1 and self.chat_layout.itemAt(self.chat_layout.count() - 2).widget()):
                    last_item.widget().deleteLater()
        
        # 显示错误消息
        self._add_system_message(f"错误：{error_message}")
        
        # 恢复输入区域
        self.input_box.setEnabled(True)
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("错误")
        self.status_label.setStyleSheet("color: #CC0000; font-weight: bold;") # 红色表示错误
    
    def _stop_generation(self):
        """停止生成响应"""
        if self.chat_worker:
            self.chat_worker.stop()
            self.status_label.setText("正在停止...")
            self.status_label.setStyleSheet("color: #FF6600; font-weight: bold;") # 橙色表示正在停止
    
    def _clear_chat(self):
        """清空聊天记录"""
        # 确认对话框
        result = QMessageBox.question(
            self,
            "清空聊天",
            "确定要清空所有聊天记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            # 清空消息历史
            self.messages = []
            
            # 清空UI
            while self.chat_layout.count():
                item = self.chat_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            
            # 添加新的系统消息
            self._add_system_message("聊天记录已清空。")
    
    def _open_system_prompt_dialog(self):
        """打开系统提示词设置对话框"""
        dialog = SystemPromptDialog(self.system_prompt, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.system_prompt = dialog.get_prompt()
            
            if self.system_prompt:
                self._add_system_message(f"系统提示词已设置为：\n{self.system_prompt}")
            else:
                self._add_system_message("系统提示词已清空。")
    
    def _auto_scroll(self):
        """自动滚动到聊天区域底部"""
        # 获取聊天区域的滚动区域
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            # 滚动到底部
            vbar = scroll_area.verticalScrollBar()
            vbar.setValue(vbar.maximum())
    
    def set_model(self, model_name: str):
        """
        设置当前使用的模型
        
        参数:
            model_name: 模型名称
        """
        if self.current_model != model_name:
            self.current_model = model_name
            self._add_system_message(f"当前已切换到模型：{model_name}")
    
    def on_tab_selected(self):
        """标签页被选中时调用"""
        # 聊天标签页不需要额外处理
        pass
    
    def on_server_changed(self):
        """服务器连接更改时调用"""
        # 添加系统消息
        self._add_system_message("服务器连接已更改。")
    
    def on_close(self):
        """应用程序关闭时调用"""
        # 停止正在进行的请求
        if self.chat_worker:
            self.chat_worker.stop()
    
    def _fill_guide_text(self, text: str):
        """
        填充引导文本到输入框
        
        参数:
            text: 引导文本
        """
        self.input_box.setText(text)
        self.input_box.setFocus()
        # 将光标移到文本末尾
        cursor = self.input_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.input_box.setTextCursor(cursor)
    
    def _update_guide_buttons(self):
        """根据聊天内容更新引导按钮"""
        # 如果没有聊天历史，使用初始引导问题
        if not self.messages:
            self._update_initial_guide_buttons()
            return
        
        # 获取最后一条助手消息的内容（如果存在）
        last_assistant_message = ""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                last_assistant_message = msg["content"]
                break
        
        # 如果没有找到助手消息，使用初始引导问题
        if not last_assistant_message:
            self._update_initial_guide_buttons()
            return
            
        # 根据最后一条助手消息内容生成可能的后续问题
        suggested_questions = self._generate_follow_up_questions(last_assistant_message)
        
        # 更新按钮文本和可见性
        for i, btn in enumerate(self.guide_buttons):
            # 断开之前的所有连接
            try:
                btn.clicked.disconnect()
            except TypeError:
                # 如果没有连接的信号，会抛出TypeError
                pass
                
            if i < len(suggested_questions):
                question = suggested_questions[i]
                btn.setText(question)
                btn.setVisible(True)
                # 创建点击事件处理函数
                self._create_button_click_handler(btn, question)
            else:
                btn.setVisible(False)
    
    def _generate_follow_up_questions(self, assistant_message: str) -> list:
        """
        根据助手的回复生成后续可能的问题
        
        参数:
            assistant_message: 助手的回复内容
            
        返回:
            建议的问题列表，包含2个相关问题和2个拓展问题
        """
        # 话题相关问题集
        topic_related_questions = []
        
        # 检测是否提到了概念或术语
        if "概念" in assistant_message or "定义" in assistant_message or "术语" in assistant_message:
            topic_related_questions.append("能详细解释一下这个概念吗？")
        
        # 检测是否提到了方法或步骤
        if "方法" in assistant_message or "步骤" in assistant_message or "过程" in assistant_message:
            topic_related_questions.append("这个方法有什么优缺点？")
        
        # 检测是否提到了代码
        if "代码" in assistant_message or "程序" in assistant_message or "函数" in assistant_message:
            topic_related_questions.append("能提供完整的代码示例吗？")
        
        # 检测是否提到了多个选项
        if "可以" in assistant_message or "或者" in assistant_message or "另一种" in assistant_message:
            topic_related_questions.append("哪种方法最适合我的情况？")
        
        # 检测是否提到了问题或错误
        if "问题" in assistant_message or "错误" in assistant_message or "异常" in assistant_message:
            topic_related_questions.append("如何解决这个问题？")
        
        # 检测是否提到了最佳实践
        if "最佳实践" in assistant_message or "建议" in assistant_message or "推荐" in assistant_message:
            topic_related_questions.append("为什么这是最佳实践？")
        
        # 默认话题相关问题（如果分析不出合适问题，使用这些）
        default_related_questions = [
            "能详细解释一下吗？",
            "还有其他方法吗？", 
            "请给出一个实际例子",
            "这个有什么应用场景？",
            "这个技术的优势是什么？",
            "有没有需要注意的问题？"
        ]
        
        # 话题拓展问题集
        expansion_questions = [
            "这与其他领域有什么联系？",
            "这个技术的发展历史是怎样的？",
            "未来这个领域可能有什么新趋势？",
            "初学者应该如何入门这个领域？",
            "有没有推荐的学习资源？",
            "这个知识在实际工作中如何应用？",
            "这个领域有哪些常见的误区？",
            "相比其他方案，这个方法有什么独特之处？",
            "这个技术背后的原理是什么？",
            "这个方法如何与其他技术结合使用？"
        ]
        
        # 确保有足够的话题相关问题
        if len(topic_related_questions) < 2:
            # 从默认问题中补充
            remaining = set(default_related_questions) - set(topic_related_questions)
            topic_related_questions.extend(random.sample(list(remaining), min(2 - len(topic_related_questions), len(remaining))))
        
        # 选择最终的2个话题相关问题
        related_questions = random.sample(topic_related_questions, min(2, len(topic_related_questions)))
        
        # 选择2个话题拓展问题
        expansion_selected = random.sample(expansion_questions, 2)
        
        # 组合所有问题（先相关问题，后拓展问题）
        all_questions = related_questions + expansion_selected
        
        return all_questions
    
    def _update_initial_guide_buttons(self):
        """更新初始引导按钮，显示开放性问题"""
        # 开放性引导问题
        initial_questions = [
            "我想了解人工智能的基础知识",
            "请介绍一下大语言模型的工作原理",
            "你能帮我解决什么问题？",
            "如何使用Python进行数据分析？"
        ]
        
        # 更新按钮文本和可见性
        for i, btn in enumerate(self.guide_buttons):
            # 断开之前的所有连接
            try:
                btn.clicked.disconnect()
            except TypeError:
                # 如果没有连接的信号，会抛出TypeError
                pass
                
            question = initial_questions[i]
            btn.setText(question)
            btn.setVisible(True)
            # 创建点击事件处理函数
            self._create_button_click_handler(btn, question)
    
    def _create_button_click_handler(self, button, question_text):
        """
        为按钮创建点击事件处理函数
        
        参数:
            button: 按钮控件
            question_text: 问题文本
        """
        # 使用functools.partial解决闭包问题
        handler = partial(self._fill_guide_text, question_text)
        button.clicked.connect(handler) 