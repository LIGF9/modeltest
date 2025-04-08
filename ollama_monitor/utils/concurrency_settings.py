#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama并发设置管理模块
管理Ollama的并发用户数和内存中模型数设置
"""

class OllamaConcurrencySettings:
    """Ollama并发设置管理类"""
    
    def __init__(self, client=None):
        """
        初始化并发设置管理
        如果提供了OllamaClient实例，则从中获取实际配置值，
        否则使用默认设置
        
        参数:
            client: OllamaClient实例，可选
        """
        # 默认设置
        self._max_concurrent_users = 3
        self._max_in_memory_models = 1
        
        # 如果提供了客户端，获取实际配置
        if client:
            try:
                config = client.get_current_config()
                if config["parallel_enabled"]:
                    self._max_concurrent_users = config["num_parallel"] or 3
                    self._max_in_memory_models = config["max_models"] or 1
                    # print(f"从Ollama服务获取的并发设置: {self._max_concurrent_users}并发/{self._max_in_memory_models}模型")
            except Exception as e:
                print(f"获取Ollama并发设置失败: {str(e)}")
    
    @property
    def max_concurrent_users(self) -> int:
        """获取最大并发用户数"""
        return self._max_concurrent_users
    
    @max_concurrent_users.setter
    def max_concurrent_users(self, value: int):
        """设置最大并发用户数"""
        if value < 1:
            value = 1  # 至少允许1个并发用户
        self._max_concurrent_users = value
    
    @property
    def max_in_memory_models(self) -> int:
        """获取最大内存中模型数"""
        return self._max_in_memory_models
    
    @max_in_memory_models.setter
    def max_in_memory_models(self, value: int):
        """设置最大内存中模型数"""
        if value < 1:
            value = 1  # 至少允许1个内存中模型
        self._max_in_memory_models = value
    
    def get_status_text(self) -> str:
        """
        获取用于状态栏显示的状态文本
        
        返回:
            格式化的状态文本
        """
        return f"模型数: {self._max_in_memory_models}   并发数: {self._max_concurrent_users}" 