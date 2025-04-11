#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama模型监控测试工具 - 主入口文件
"""

import sys
import os
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QIcon
from qtpy.QtCore import QSharedMemory
from ollama_monitor.ui.main_window import MainWindow
import multiprocess
import ctypes
# 隐藏控制台窗口
if hasattr(ctypes, 'windll'):
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

def is_already_running():
    """检查是否已有实例在运行"""
    shared_mem = QSharedMemory("YourUniqueAppKey")  # 替换为你的唯一标识
    if shared_mem.attach():
        return True
    if not shared_mem.create(1):
        return True
    return False

def main():
    """主函数，应用程序入口点"""
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-"*100)
    print(f"开始运行Ollama模型监控测试工具 {date_time}")
    print("-"*100)
    # 确保资源目录存在
    os.makedirs(os.path.join(os.path.dirname(__file__), 'ollama_monitor', 'resources'), exist_ok=True)
    
    if is_already_running():
        QMessageBox.warning(None, "警告", "程序已在运行！")
        sys.exit(1)

    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 可选：同时设置应用程序图标（影响任务栏图标）
    app.setWindowIcon(QIcon("ollamaIcon.ico"))
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocess.freeze_support()  # 支持多进程
    main()