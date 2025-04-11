#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama API 客户端模块
处理与Ollama服务的API交互
"""

import json
import time
import re
import os
import platform
import requests
import threading
import subprocess
from typing import Dict, List, Optional, Generator, Any, Tuple


class OllamaClient:
    """Ollama API 客户端类"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        初始化Ollama客户端
        
        参数:
            base_url: Ollama服务的基础URL，默认为本地11434端口
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        self.service_controller = OllamaServiceController()
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        获取可用模型列表
        
        返回:
            模型信息列表
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            print(f"获取模型列表失败: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取指定模型的详细信息
        
        参数:
            model_name: 模型名称
            
        返回:
            模型详细信息字典
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"获取模型信息失败: {e}")
            return {}
    
    def generate_completion(
        self, 
        model: str, 
        prompt: str, 
        system: str = "",
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        生成文本完成（支持流式输出）
        
        参数:
            model: 模型名称
            prompt: 提示文本
            system: 系统提示词
            stream: 是否使用流式输出
            options: 其他选项参数
            cancel_event: 取消事件标志，用于中断流式输出
            
        返回:
            生成的文本响应生成器
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            data["system"] = system
            
        if options:
            data["options"] = options
            
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                for line in response.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        response.close()
                        break
                        
                    if line:
                        yield json.loads(line)
            else:
                yield response.json()
                
        except requests.RequestException as e:
            print(f"生成文本完成失败: {e}")
            yield {"error": str(e)}
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        聊天完成接口（支持流式输出）
        
        参数:
            model: 模型名称
            messages: 聊天消息列表
            stream: 是否使用流式输出
            options: 其他选项参数
            
        返回:
            聊天响应生成器
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if options:
            data["options"] = options
            
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=data,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                yield response.json()
                
        except requests.RequestException as e:
            print(f"聊天请求失败: {e}")
            yield {"error": str(e)}
    
    def ping(self) -> Tuple[bool, float]:
        """
        测试与Ollama服务的连接状态和延迟
        
        返回:
            (连接状态, 延迟时间ms)
        """
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            return True, latency
        except requests.RequestException:
            return False, 0
    
    def get_version(self) -> str:
        """
        获取Ollama版本信息
        
        返回:
            版本字符串
        """
        try:
            response = self.session.get(f"{self.base_url}/api/version")
            response.raise_for_status()
            return response.json().get("version", "未知")
        except requests.RequestException as e:
            print(f"获取Ollama版本失败: {e}")
            return "未知"
    
    def unload_model(self, model_name: str) -> bool:
        """
        从Ollama服务器内存中卸载指定模型
        
        参数:
            model_name: 要卸载的模型名称
            
        返回:
            bool: 卸载是否成功
        """
        try:
            # 在新版Ollama中，可以通过API直接控制模型的加载/卸载
            # 发送特殊空请求来卸载模型
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                    "options": {
                        "num_keep": 0  # 强制卸载模型
                    }
                }
            )
            
            response.raise_for_status()
            # print("卸载响应：",response.json())
            return True
        except requests.RequestException as e:
            print(f"卸载模型 {model_name} 失败: {e}")
            return False 
        
    def check_parallel_support(self, version: Optional[str] = None) -> bool:
        """检查版本是否支持多实例并发"""
        if version is None:
            version = self.get_version()
            if version == "未知":
                return False
        try:
            return tuple(map(int, version.split('.'))) >= (0, 1, 33)
        except:
            return False

    def get_current_config(self) -> Dict:
        """获取当前并发配置"""
        config = {
            "parallel_enabled": False,
            "num_parallel": None,
            "max_models": None,
            "env_source": "none"
        }

        # 检查环境变量
        if os.getenv("OLLAMA_NUM_PARALLEL"):
            config.update({
                "parallel_enabled": True,
                "num_parallel": int(os.getenv("OLLAMA_NUM_PARALLEL")),
                "max_models": int(os.getenv("OLLAMA_MAX_LOADED_MODELS", 1)),
                "env_source": "env"
            })
            return config

        # 检查systemd配置（仅Linux）
        if platform.system() != 'Windows':
            try:
                output = subprocess.run(["systemctl", "show", "ollama", "--property=Environment"],
                                    capture_output=True, text=True,creationflags=subprocess.CREATE_NO_WINDOW).stdout
                if "OLLAMA_NUM_PARALLEL" in output:
                    num_parallel = re.search(r"OLLAMA_NUM_PARALLEL=(\d+)", output).group(1)
                    max_models = re.search(r"OLLAMA_MAX_LOADED_MODELS=(\d+)", output)
                    config.update({
                        "parallel_enabled": True,
                        "num_parallel": int(num_parallel),
                        "max_models": int(max_models.group(1)) if max_models else 1,
                        "env_source": "systemd"
                    })
            except:
                pass

        return config

    def configure_parallel(self,
        num_parallel: int,
        max_models: Optional[int] = 1,
        persist: bool = True
    ) -> Tuple[bool, str]:
        """配置并发参数"""
        max_models = max_models or num_parallel
        
        try:
            # 设置环境变量(临时生效)
            os.environ["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(max_models)
            
            if persist:
                def is_root() -> bool:
                    """检查是否具有管理员权限（跨平台实现）"""
                    if platform.system() == 'Windows':
                        try:
                            import ctypes
                            return ctypes.windll.shell32.IsUserAnAdmin() != 0
                        except:
                            return False
                    else:
                        try:
                            return os.getuid() == 0
                        except AttributeError:
                            return False
                            
                if platform.system() == 'Windows':
                    if not is_root():
                        return (False, "需要管理员权限修改系统配置")
                    
                    try:
                        from winreg import (
                            HKEY_LOCAL_MACHINE, 
                            KEY_SET_VALUE, 
                            REG_EXPAND_SZ,
                            OpenKey, 
                            SetValueEx
                        )
                        
                        with OpenKey(
                            HKEY_LOCAL_MACHINE, 
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                            0, KEY_SET_VALUE
                        ) as key:
                            SetValueEx(key, "OLLAMA_NUM_PARALLEL", 0, REG_EXPAND_SZ, str(num_parallel))
                            SetValueEx(key, "OLLAMA_MAX_LOADED_MODELS", 0, REG_EXPAND_SZ, str(max_models))
                        
                        # 通知系统环境变量变更
                        import ctypes
                        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 0, 1000, None)
                        return (True, "Windows 系统环境变量已设置，需重启生效")
                    except ImportError:
                        return (False, "Windows 注册表操作需要 pywin32 库")
                    except Exception as e:
                        return (False, f"注册表修改失败: {str(e)}")
                else:
                    # Linux/macOS
                    service_file = "/etc/systemd/system/ollama.service"
                    if not os.path.exists(service_file):
                        return (False, "未找到systemd服务文件")
                    
                    try:
                        with open(service_file, "r+") as f:
                            content = f.read()
                            if "Environment=" in content:
                                content = re.sub(
                                    r"Environment=(.*)",
                                    f"Environment=\"OLLAMA_NUM_PARALLEL={num_parallel} OLLAMA_MAX_LOADED_MODELS={max_models}\"",
                                    content
                                )
                            else:
                                content = content.replace(
                                    "[Service]",
                                    f"[Service]\nEnvironment=\"OLLAMA_NUM_PARALLEL={num_parallel} OLLAMA_MAX_LOADED_MODELS={max_models}\""
                                )
                            f.seek(0)
                            f.write(content)
                            f.truncate()
                        
                        subprocess.run(["systemctl", "daemon-reload"], check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                        subprocess.run(["systemctl", "restart", "ollama"], check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                        return (True, "systemd 配置已持久化")
                    except subprocess.CalledProcessError as e:
                        return (False, f"服务重启失败: {str(e)}")
                    except Exception as e:
                        return (False, f"文件操作失败: {str(e)}")
            
            return (True, f"并发配置已{'临时' if not persist else '持久化'}设置: {num_parallel}并行/{max_models}模型")
        
        except Exception as e:
            return (False, f"配置失败: {str(e)}")

class OllamaServiceController:
    """
    Ollama 服务控制器（纯命令行实现）
    功能：启动/停止/重启 Ollama 服务，无需知道可执行文件路径
    """

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        self.is_mac = platform.system() == "Darwin"

    def _run_command(self, command, check_output=False):
        """执行命令行命令"""
        try:
            if self.is_windows:
                # Windows 系统下的命令执行优化
                if "start /B" in command and not check_output:
                    # 使用subprocess.CREATE_NO_WINDOW处理后台进程
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        text=True
                    )
                    return True
                else:
                    command = f"cmd /c {command}"
            
            if check_output:
                return subprocess.check_output(
                    command, 
                    shell=True, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                ).strip()
            else:
                subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    start_new_session=not self.is_windows,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                return True
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {str(e)}")
            if hasattr(e, 'stderr'):
                print(f"错误输出: {e.stderr}")
            return None
        except Exception as e:
            print(f"执行命令时发生错误: {str(e)}")
            return None

    def _find_ollama_pids(self):
        """查找所有 Ollama 进程的 PID"""
        if self.is_windows:
            # Windows 使用 tasklist 命令
            output = self._run_command("tasklist /FI \"IMAGENAME eq ollama.exe\" /NH", check_output=True)
            if not output:
                return []
            
            pids = []
            for line in output.splitlines():
                if "ollama" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            pids.append(pid)
                        except ValueError:
                            # 跳过无法转换为整数的PID
                            print(f"警告: 无法解析PID: {parts[1]}")
            
            return pids
        else:
            # Unix-like 系统使用 ps 命令
            output = self._run_command("ps aux | grep -i [o]llama", check_output=True)
            if not output:
                return []
            pids = [int(line.split()[1]) for line in output.splitlines()]
            return pids

    def is_running(self):
        """检查 Ollama 服务是否正在运行"""
        return len(self._find_ollama_pids()) > 0

    def stop_service(self):
        """停止 Ollama 服务"""
        pids = self._find_ollama_pids()
        if not pids:
            print("Ollama 服务未运行")
            return True

        try:
            if self.is_windows:
                # Windows 使用 taskkill 命令
                for pid in pids:
                    self._run_command(f"taskkill /F /PID {pid}")
            else:
                # Unix-like 系统使用 kill 命令
                self._run_command(f"kill -9 {' '.join(map(str, pids))}")
            
            # 等待进程结束
            time.sleep(2)
            
            # 验证是否已停止
            remaining_pids = self._find_ollama_pids()
            if remaining_pids:
                print(f"未能停止的 Ollama 进程: {remaining_pids}")
                return False
            
            # print("Ollama 服务已成功停止")
            return True
        except Exception as e:
            print(f"停止服务时出错: {str(e)}")
            return False

    def restart_service(self):
        """重启 Ollama 服务"""
        if not self.stop_service():
            return False
        return self.start_service()

    def start_service(self):
        """启动 Ollama 服务"""
        if self.is_running():
            print("Ollama 服务已在运行")
            return True

        try:
            # 直接使用 ollama 命令（假设已在 PATH 中）
            if self.is_windows:
                # Windows 具体指定如何启动 Ollama
                try:
                    # 首先尝试使用 start 命令
                    self._run_command("start /B ollama serve", check_output=False)
                    
                    # 等待服务启动
                    max_wait = 10  # 最多等待10秒
                    wait_count = 0
                    while wait_count < max_wait:
                        time.sleep(1)
                        wait_count += 1
                        if self.is_running():
                            # print("Ollama 服务已成功启动")
                            return True
                    
                    # 如果启动失败，尝试直接运行
                    if not self.is_running():
                        print("使用 start 命令启动失败，尝试直接运行...")
                        self._run_command("ollama serve", check_output=False)
                        time.sleep(5)  # 等待服务启动
                except Exception as e:
                    print(f"Windows 启动 Ollama 出错: {str(e)}")
                    # 尝试查找 Ollama 可执行文件
                    try:
                        # 搜索可能的路径
                        possible_paths = [
                            os.path.expanduser("~\\AppData\\Local\\ollama\\ollama.exe"),
                            "C:\\Program Files\\ollama\\ollama.exe"
                        ]
                        for path in possible_paths:
                            if os.path.exists(path):
                                print(f"找到 Ollama 可执行文件: {path}")
                                self._run_command(f"start /B \"{path}\" serve", check_output=False)
                                time.sleep(5)  # 等待服务启动
                                break
                    except Exception as e2:
                        print(f"尝试查找 Ollama 可执行文件失败: {str(e2)}")
            else:
                # Unix-like 系统使用 nohup 后台运行
                self._run_command("nohup ollama serve > /dev/null 2>&1 &")
                time.sleep(3)
            
            # 确认服务是否已启动
            max_checks = 5
            for i in range(max_checks):
                if self.is_running():
                    # print("Ollama 服务已成功启动")
                    return True
                # print(f"等待 Ollama 服务启动... ({i+1}/{max_checks})")
                time.sleep(2)
                
            print("Ollama 服务启动失败，请确保 ollama 命令在 PATH 中或在正确位置")
            return False
        except Exception as e:
            print(f"启动服务时出错: {str(e)}")
            return False

if __name__ == "__main__":
    manager = OllamaServiceController()
    client = OllamaClient()
    print("当前服务状态:", "运行中" if manager.is_running() else "未运行")
    print("当前并发配置:", client.get_current_config())
    success, message = client.configure_parallel(3)
    print("配置结果:", success, message)
    print("当前并发配置:", client.get_current_config())