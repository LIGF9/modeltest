#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主窗口模块
定义应用程序的主窗口和界面布局
"""

from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QComboBox, QToolBar, 
    QLabel, QPushButton, QWidget, QVBoxLayout, QStatusBar, QMessageBox,
    QSizePolicy, QDialog
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon, QAction

from ..utils.ollama_client import OllamaClient
from ..utils.system_monitor import SystemMonitor
from ..utils.concurrency_settings import OllamaConcurrencySettings
from .tabs.system_monitor_tab import SystemMonitorTab
from .tabs.chat_tab import ChatTab
from .tabs.performance_tab import PerformanceTab
from .dialogs.concurrency_dialog import ConcurrencySettingsDialog


class MainWindow(QMainWindow):
    """应用程序主窗口类"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("Ollama 模型监控测试工具")
        self.resize(1000, 700)
        
        # 设置窗口图标
        self.setWindowIcon(QIcon("ollamaIcon.ico")) 
        
        # 初始化客户端和监控器
        self.ollama_client = OllamaClient()
        self.system_monitor = SystemMonitor()
        self.concurrency_settings = OllamaConcurrencySettings(self.ollama_client)

        #确保ollama服务正常运行
        if not self.ollama_client.service_controller.is_running():
            self.ollama_client.service_controller.start_service()
            print("\nollama已启动服务")        
        else:
            print("\nollama运行正常")
        config = self.ollama_client.get_current_config()
        # print("config",config)
        # 检查并发设置
        if config["parallel_enabled"]:
            print(f"ollama并发配置: {config['num_parallel']}并发/{config['max_models']}模型","\n")
        else:
            print("ollama并发配置: 未启用","\n")
        
        # 初始化UI元素
        self._init_ui()
        
        # 加载模型列表
        self._load_models()
        
        # 设置定时器更新状态栏
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.start(3000)  # 每3秒更新一次
        self._update_status_bar()
    
    def _init_ui(self):
        """初始化UI元素"""
        # 创建主布局和中心窗口部件
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # 创建工具栏
        self._create_toolbar()
        
        # 创建标签页控件
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setDocumentMode(True)
        
        # 创建各个标签页
        self.system_tab = SystemMonitorTab(self.ollama_client, self.system_monitor)
        self.chat_tab = ChatTab(self.ollama_client)
        self.performance_tab = PerformanceTab(self.ollama_client, self.system_monitor)
        
        # 添加标签页
        self.tab_widget.addTab(self.system_tab, "系统监控")
        self.tab_widget.addTab(self.chat_tab, "聊天测试")
        self.tab_widget.addTab(self.performance_tab, "性能测试")
        
        # 添加标签页切换事件处理
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # 添加到主布局
        main_layout.addWidget(self.tab_widget)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状态栏元素 - 从左到右的顺序定义
        self.ollama_version_label = QLabel("Ollama版本: --")
        self.connection_status = QLabel()
        self.server_latency = QLabel()
        self.concurrency_status = QLabel()
        self.cpu_usage = QLabel()
        self.memory_usage = QLabel()
        self.network_traffic = QLabel("网速: 上传 0 KB/s | 下载 0 KB/s")
        
        # 添加到状态栏 - 确保Ollama版本在最左侧
        self.status_bar.addWidget(self.ollama_version_label)  # 最左侧第一个
        self.status_bar.addWidget(self.connection_status)     # 连接状态放在版本右侧
        self.status_bar.addWidget(self.server_latency)        # 延迟放在连接状态右侧
        self.status_bar.addWidget(self.concurrency_status)    # 并发状态放在第四位置
        # 右侧元素
        self.status_bar.addPermanentWidget(self.cpu_usage)
        self.status_bar.addPermanentWidget(self.memory_usage)
        self.status_bar.addPermanentWidget(self.network_traffic)
        
        # 设置中心部件
        self.setCentralWidget(central_widget)
    
    def _create_toolbar(self):
        """创建工具栏"""
        self.toolbar = QToolBar("主工具栏")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        
        # 模型选择下拉框
        model_label = QLabel("模型:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self._load_models)
        
        # 服务器连接配置
        server_label = QLabel("服务器:")
        self.server_input = QComboBox()
        self.server_input.setEditable(True)
        self.server_input.setMinimumWidth(200)
        self.server_input.addItem("http://localhost:11434")
        self.server_input.addItem("http://127.0.0.1:11434")
        self.server_input.currentTextChanged.connect(self._on_server_changed)
        
        # 连接按钮
        self.connect_button = QPushButton("连接")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        
        # 添加到工具栏
        self.toolbar.addWidget(model_label)
        self.toolbar.addWidget(self.model_combo)
        self.toolbar.addWidget(self.refresh_button)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(server_label)
        self.toolbar.addWidget(self.server_input)
        self.toolbar.addWidget(self.connect_button)
        
        # 添加伸缩空间
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # 添加并发设置按钮
        self.concurrency_action = QAction("设置", self)
        self.concurrency_action.triggered.connect(self._show_concurrency_settings)
        self.toolbar.addAction(self.concurrency_action)
        
        # 添加关于按钮
        self.about_action = QAction("关于", self)
        self.about_action.triggered.connect(self._show_about)
        self.toolbar.addAction(self.about_action)
    
    def _load_models(self):
        """加载Ollama模型列表"""
        
        if self.performance_tab.istesting:
            QMessageBox.warning(self, "测试中", "正在进行性能测试，请手动停止测试，或等待测试结束后再试。")
            return

        try:
            # 保存当前选择的模型
            current_model = self.model_combo.currentText()
            
            # 清空列表并添加加载提示
            self.model_combo.clear()
            self.model_combo.addItem("加载中...")
            self.refresh_button.setEnabled(False)
            
            # 获取模型列表
            models = self.ollama_client.get_models()
            
            # 更新下拉框
            self.model_combo.clear()
            if not models:
                self.model_combo.addItem("未找到模型")
            else:
                for model in models:
                    name = model.get("name", "未知模型")
                    self.model_combo.addItem(name)
                
                # 恢复之前选择的模型
                if current_model and current_model != "加载中..." and current_model != "未找到模型":
                    index = self.model_combo.findText(current_model)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                
                # 如果没有之前的选择或未找到，选择第一个模型
                if self.model_combo.currentText() == "":
                    self.model_combo.setCurrentIndex(0)
                
                # 通知所有标签页模型已加载
                self._update_active_model()
            
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItem(f"加载失败: {str(e)}")
        
        finally:
            self.refresh_button.setEnabled(True)
    
    def _on_model_changed(self, model_name):
        """
        当选择的模型改变时的处理
        
        参数:
            model_name: 选择的模型名称
        """

        if self.performance_tab.istesting:
            QMessageBox.warning(self, "测试中", "正在进行性能测试，请手动停止测试，或等待测试结束后再试。")
            return

        if model_name and model_name not in ["加载中...", "未找到模型", "加载失败"]:
            self._update_active_model()
    
    def _update_active_model(self):
        """更新当前选择的模型到所有标签页"""
        model_name = self.model_combo.currentText()
        try:
            # 更新聊天标签页的模型
            if hasattr(self, 'chat_tab') and self.chat_tab is not None:
                self.chat_tab.set_model(model_name)
            
            # 更新性能测试标签页的模型
            if hasattr(self, 'performance_tab') and self.performance_tab is not None:
                self.performance_tab.set_model(model_name)
                
        except RuntimeError as e:
            # 忽略已删除对象的错误
            print(f"更新模型时出错: {e}")
    
    def _on_server_changed(self, server_url):
        """
        当服务器URL改变时的处理
        
        参数:
            server_url: 新的服务器URL
        """
        # 这里只记录改变，不立即连接
        pass
    
    def _on_connect_clicked(self):
        if self.performance_tab.istesting:
            QMessageBox.warning(self, "测试中", "正在进行性能测试，请手动停止测试，或等待测试结束后再试。")
            return

        """连接按钮点击事件处理"""
        server_url = self.server_input.currentText()
        
        # 验证URL格式
        if not (server_url.startswith("http://") or server_url.startswith("https://")):
            QMessageBox.warning(self, "URL格式错误", "服务器URL必须以http://或https://开头")
            return
        
        # 设置新的服务器URL
        self.ollama_client.base_url = server_url.rstrip('/')
        
        # 保存到下拉框历史
        if self.server_input.findText(server_url) == -1:
            self.server_input.addItem(server_url)
        
        # 重新加载模型列表
        self._load_models()
        
        # 更新状态栏
        self._update_status_bar()
        
        try:
            # 通知所有标签页服务器已更改
            if hasattr(self, 'system_tab') and self.system_tab is not None:
                self.system_tab.on_server_changed()
            
            if hasattr(self, 'chat_tab') and self.chat_tab is not None:
                self.chat_tab.on_server_changed()
            
            if hasattr(self, 'performance_tab') and self.performance_tab is not None:
                self.performance_tab.on_server_changed()
        except RuntimeError as e:
            # 忽略已删除对象的错误
            print(f"通知标签页服务器变更时出错: {e}")
    
    def _show_concurrency_settings(self):
        """显示并发设置对话框"""
        dialog = ConcurrencySettingsDialog(self, self.concurrency_settings, self.ollama_client)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            # 获取用户设置的值
            max_users = self.concurrency_settings.max_concurrent_users
            max_models = self.concurrency_settings.max_in_memory_models
            
            # 调用OllamaClient的configure_parallel方法实际应用设置
            try:
                success, message = self.ollama_client.configure_parallel(max_users, max_models)
                if success:
                    print(f"成功应用并发设置: {message}")
                    # 在状态栏显示成功消息
                    QMessageBox.warning(self, "设置成功", f"已应用并发设置: {max_users}并发/{max_models}模型。请重启计算机生效。")
                    self.status_bar.showMessage(f"已应用并发设置: {max_users}并发/{max_models}模型。请重启计算机生效。", 10000)
                else:
                    print(f"应用并发设置失败: {message}")
                    # 显示错误消息框
                    QMessageBox.warning(self, "设置失败", f"应用并发设置失败: {message}")
                    self.status_bar.showMessage(f"应用并发设置失败: {message}", 10000)
            except Exception as e:
                print(f"应用并发设置时出错: {e}")
                QMessageBox.warning(self, "设置错误", f"应用并发设置时出错: {str(e)}")
                self.status_bar.showMessage(f"应用并发设置失败: {str(e)}", 10000)
            
            # 更新状态栏显示
            self._update_status_bar()
    
    def _on_tab_changed(self, index):
        """
        标签页切换事件处理
        
        参数:
            index: 新的标签页索引
        """
        try:
            # 通知各标签页
            if index == 0:  # 系统监控标签页
                if hasattr(self, 'system_tab') and self.system_tab is not None:
                    self.system_tab.on_tab_selected()
            elif index == 1:  # 聊天标签页
                if hasattr(self, 'chat_tab') and self.chat_tab is not None:
                    self.chat_tab.on_tab_selected()
            elif index == 2:  # 性能测试标签页
                if hasattr(self, 'performance_tab') and self.performance_tab is not None:
                    self.performance_tab.on_tab_selected()
        except RuntimeError as e:
            # 忽略已删除对象的错误
            print(f"标签页切换时出错: {e}")
    
    def _update_status_bar(self):
        """更新状态栏信息"""
        # 更新Ollama版本
        version = self.ollama_client.get_version()
        self.ollama_version_label.setText(f"Ollama版本: {version}")
        
        # 更新连接状态
        connected, latency = self.ollama_client.ping()
        if connected:
            self.connection_status.setText(f"状态: 已连接")
            self.connection_status.setStyleSheet("color:black;")  # 黑色
            self.server_latency.setText(f"延迟: {latency:.1f} ms")
        else:
            self.connection_status.setText("状态: 未连接")
            self.connection_status.setStyleSheet("color: red;")  # 红色
            self.server_latency.setText("延迟: -- ms")
        
        # 更新并发设置状态
        self.concurrency_status.setText(self.concurrency_settings.get_status_text())
        
        # 更新系统指标
        metrics = self.system_monitor.get_metrics()
        self.cpu_usage.setText(f"CPU: {metrics.cpu_percent:.1f}%")
        self.memory_usage.setText(f"内存: {metrics.memory_percent:.1f}%")
        
        # 更新网络流量信息
        self.network_traffic.setText(f"网速: 上传 {metrics.network_sent:.1f} KB/s | 下载 {metrics.network_recv:.1f} KB/s")
    
    def _show_about(self):
        """显示关于对话框"""
        about_text = """
<h3>Ollama 模型监控测试工具</h3>
<p>版本: 1.0.0</p>
<p>这是一个用于测试和监控本地Ollama大语言模型性能的桌面应用程序。</p>
<p>主要功能:</p>
<ul>
<li>系统资源监控</li>
<li>聊天测试</li>
<li>性能评估</li>
</ul>
<p>© 2023 Ollama Monitor</p>
"""
        QMessageBox.about(self, "关于", about_text)
    
    def closeEvent(self, event):
        """
        窗口关闭事件处理
        
        参数:
            event: 关闭事件
        """
        try:
            # 停止定时器
            if hasattr(self, 'status_timer') and self.status_timer is not None:
                self.status_timer.stop()
            
            # 通知所有标签页应用程序即将关闭
            if hasattr(self, 'system_tab') and self.system_tab is not None:
                self.system_tab.on_close()
            
            if hasattr(self, 'chat_tab') and self.chat_tab is not None:
                self.chat_tab.on_close()
            
            if hasattr(self, 'performance_tab') and self.performance_tab is not None:
                self.performance_tab.on_close()

        except AttributeError as e:
            # 忽略属性错误
            print(f"关闭应用程序时出错: {e}")
        except RuntimeError as e:
            # 忽略已删除对象的错误
            print(f"关闭应用程序时出错: {e}")

        
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("-"*100)
        print(f"结束运行Ollama模型监控测试工具 {date_time}")
        print("-"*100)
        
        # 接受关闭事件
        event.accept() 