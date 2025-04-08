#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama模型监控测试工具 - 主入口文件
"""

import sys
import os
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from ollama_monitor.ui.main_window import MainWindow


def main():
    """主函数，应用程序入口点"""
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-"*100)
    print(f"开始运行Ollama模型监控测试工具 {date_time}")
    print("-"*100)
    # 确保资源目录存在
    os.makedirs(os.path.join(os.path.dirname(__file__), 'ollama_monitor', 'resources'), exist_ok=True)
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()