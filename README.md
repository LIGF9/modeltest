# Ollama 模型监控测试工具

这是一个用于测试和监控本地Ollama大语言模型性能的桌面应用程序。该应用提供系统资源监控、聊天测试、性能评估三个主要功能，通过PyQt6构建图形界面，与本地或远程Ollama服务进行交互。

## 功能特点

### 1. 系统监控
- 静态信息：系统版本，CPU、内存、GPU信息，Python版本，Ollama版本
- 实时监控：CPU、内存使用率，GPU和显存使用率(支持NVIDIA)
- Ollama服务器连接状态、延迟，网络上下行速度监控

### 2. 聊天测试
- 支持流式响应显示
- 支持定制系统提示词
- 支持清空聊天记录

### 3. 性能测试
- 可配置测试参数(测试次数，并发用户数等)
- 预设和自定义提示词
- 测试指标：首token延迟、生成速度(tokens/s)等
- 性能评级和测试报告生成

## 安装说明

1. 确保已安装Python 3.8+
2. 安装依赖：
```
pip install -r requirements.txt
```
3. 确保Ollama服务已启动并可访问

## 使用方法

运行入口文件：
```
python main.py
```
运行入口文件：(将log重定向到log.txt)
```
python main.py >> log.txt
```

## 系统要求
- 操作系统：Windows 10/11 或 Linux
- Python版本：3.8+
- Ollama：已安装并运行 