#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统监控模块
用于获取系统资源信息和实时监控数据
"""

import sys
import platform
import psutil
import cpuinfo
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import logging
from ollama_monitor.utils.ollama_client import OllamaClient

# 定义全局变量控制GPU功能
GPU_AVAILABLE = False
GPUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# 导入自定义的GPU信息收集器
from .gpuInfo import GPUInfoCollector


@dataclass
class SystemInfo:
    """系统静态信息数据类"""
    os_name: str
    os_version: str
    cpu_brand: str
    cpu_cores: int
    memory_total: float  # GB
    gpu_info: List[Dict[str, Any]]
    python_version: str
    has_gpu: bool
    ollama_version: str


@dataclass
class SystemMetrics:
    """系统实时监控指标数据类"""
    cpu_percent: float
    memory_percent: float
    network_sent: float  # KB/s
    network_recv: float  # KB/s
    gpu_metrics: List[Dict[str, Any]]


class SystemMonitor:
    """系统资源监控类"""
    
    def __init__(self):
        """初始化系统监控器"""
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()
        self.prev_net_io_time = time.time()
        # 初始化GPU信息收集器
        self.gpu_collector = GPUInfoCollector()
        # 初始化系统静态信息
        self._system_info = self._get_system_info()
        # 初始化Ollama客户端
        self.ollama_client = OllamaClient()
        # 初始化GPU可用性标志
        self.nvidia_gpu_available = False
    
    def _get_system_info(self) -> SystemInfo:
        """
        获取系统静态信息
        
        返回:
            系统静态信息对象
        """
        # 操作系统信息
        os_name = platform.system()
        if os_name == "Windows":
            os_version = f"Windows {platform.release()} {platform.version()}"
        elif os_name == "Linux":
            try:
                import distro
                os_version = f"{distro.name()} {distro.version()}"
            except ImportError:
                os_version = platform.version()
        else:
            os_version = platform.version()
            
        # CPU信息
        try:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get('brand_raw', '未知')
        except Exception:
            cpu_brand = "未知"
        
        cpu_cores = psutil.cpu_count(logical=True)
        
        # 内存信息
        memory_total = psutil.virtual_memory().total / (1024 ** 3)  # 转换为GB
        
        # GPU信息
        gpu_info = []
        has_gpu = False
        
        # 检查NVIDIA GPU是否可用
        nvidia_gpus = []
        if GPUTIL_AVAILABLE:
            try:
                # 安全检查NVIDIA驱动是否可用
                gpus = []
                try:
                    gpus = GPUtil.getGPUs()
                except Exception as e:
                    print(f"获取NVIDIA GPU信息失败: {e}")
                    # 忽略错误，继续执行
                
                has_nvidia_gpu = len(gpus) > 0
                self.nvidia_gpu_available = has_nvidia_gpu
                
                # 如果有NVIDIA GPU，添加到列表
                if has_nvidia_gpu:
                    for i, gpu in enumerate(gpus):
                        try:
                            memory_total = round(gpu.memoryTotal / 1024, 2)  # 转换为GB
                        except:
                            memory_total = 0
                            
                        nvidia_gpu = {
                            "index": i,
                            "name": gpu.name,
                            "memory_total": memory_total,
                            "driver": "NVIDIA",
                            "vendor": "NVIDIA"
                        }
                        nvidia_gpus.append(nvidia_gpu)
                        gpu_info.append(nvidia_gpu)
                    
                    has_gpu = True
            except Exception as e:
                print(f"处理NVIDIA GPU信息时出错: {e}")
                self.nvidia_gpu_available = False
        
        # 如果没有找到NVIDIA GPU，使用GPUInfoCollector收集其他GPU信息
        if not nvidia_gpus:
            try:
                non_nvidia_gpu_info = self.gpu_collector.collect()
                if non_nvidia_gpu_info:
                    for i, gpu in enumerate(non_nvidia_gpu_info):
                        non_nvidia_gpu = {
                            "index": i,
                            "name": gpu["name"],
                            "memory_total": gpu["total"] / 1024,  # 转换为GB（从MB转换）
                            "dedicated": gpu.get("dedicated", 0) / 1024,  # 转换为GB
                            "shared": gpu.get("shared", 0) / 1024,  # 转换为GB
                            "driver": "非NVIDIA",
                            "vendor": "非NVIDIA",
                            "source": gpu.get("source", "未知")
                        }
                        gpu_info.append(non_nvidia_gpu)
                    
                    has_gpu = len(non_nvidia_gpu_info) > 0
            except Exception as e:
                print(f"处理非NVIDIA GPU信息时出错: {e}")
        
        # Python版本
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        return SystemInfo(
            os_name=os_name,
            os_version=os_version,
            cpu_brand=cpu_brand,
            cpu_cores=cpu_cores,
            memory_total=memory_total,
            gpu_info=gpu_info,
            python_version=python_version,
            has_gpu=has_gpu,
            ollama_version="未知"
        )
    
    def get_system_info(self) -> SystemInfo:
        """
        获取系统静态信息
        
        返回:
            系统静态信息对象
        """
        # 返回原本初始化时获取的系统信息，但添加ollama版本
        self._system_info.ollama_version = self._get_ollama_version()
        return self._system_info
    
    def _get_cpu_brand(self) -> str:
        """
        获取CPU品牌信息
        
        返回:
            CPU品牌名称字符串
        """
        try:
            cpu_info = cpuinfo.get_cpu_info()
            return cpu_info.get('brand_raw', '未知')
        except Exception:
            return "未知"
    
    def _get_ollama_version(self) -> str:
        """
        获取Ollama版本
        
        返回:
            Ollama版本字符串
        """
        try:
            version = self.ollama_client.get_version()
            return version
        except Exception:
            return "未知"
    
    def _get_gpu_info(self) -> list:
        """
        获取GPU信息，尝试多种方法获取GPU型号和显存大小
        
        返回:
            包含GPU信息的列表，每个GPU一个字典
        """
        gpu_info = []
        
        # 1. 尝试使用pynvml（NVIDIA专用）
        try:
            import importlib.util
            pynvml_spec = importlib.util.find_spec("pynvml")
            
            if pynvml_spec is not None:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        for i in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            name = pynvml.nvmlDeviceGetName(handle)
                            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            # 获取更详细的GPU信息
                            try:
                                uuid = pynvml.nvmlDeviceGetUUID(handle)
                                driver_version = pynvml.nvmlSystemGetDriverVersion()
                                vbios_version = pynvml.nvmlDeviceGetVbiosVersion(handle)
                            except Exception:
                                uuid = "未知"
                                driver_version = "未知"
                                vbios_version = "未知"
                            
                            gpu_info.append({
                                "index": i,
                                "name": name.decode('utf-8') if isinstance(name, bytes) else name,
                                "uuid": uuid.decode('utf-8') if isinstance(uuid, bytes) else uuid,
                                "memory_total": round(memory.total / (1024 ** 3), 2),  # GB
                                "memory_used": round(memory.used / (1024 ** 3), 2),    # GB
                                "memory_free": round(memory.free / (1024 ** 3), 2),    # GB
                                "driver_version": driver_version,
                                "vbios_version": vbios_version,
                                "vendor": "NVIDIA"
                            })
                    
                    pynvml.nvmlShutdown()
                    if gpu_info:  # 如果已经获取到了GPU信息，就不需要再使用其他方法了
                        return gpu_info
                except Exception as e:
                    logging.warning(f"使用pynvml获取GPU信息失败: {e}")
        except Exception as e:
            logging.warning(f"检查pynvml模块时发生错误: {e}")
        
        # 2. 尝试使用GPUtil
        try:
            import importlib.util
            gputil_spec = importlib.util.find_spec("GPUtil")
            
            if gputil_spec is not None:
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    
                    if not gpu_info:  # 如果pynvml没有获取到信息，再使用GPUtil获取
                        for i, gpu in enumerate(gpus):
                            gpu_info.append({
                                "index": i,
                                "name": gpu.name,
                                "memory_total": round(gpu.memoryTotal / 1024, 2),  # GB
                                "memory_used": round(gpu.memoryUsed / 1024, 2),    # GB
                                "memory_free": round((gpu.memoryTotal - gpu.memoryUsed) / 1024, 2),  # GB
                                "driver_version": "未知",
                                "vbios_version": "未知",
                                "vendor": "NVIDIA"
                            })
                        
                        if gpu_info:  # 如果GPUtil获取到了信息，就不需要再使用其他方法了
                            return gpu_info
                except Exception as e:
                    logging.warning(f"使用GPUtil获取GPU信息失败: {e}")
        except Exception as e:
            logging.warning(f"检查GPUtil模块时发生错误: {e}")
        
        # 3. Windows系统特有的GPU检测方法
        if platform.system() == "Windows" and not gpu_info:
            # 3.1 使用wmic命令获取所有视频控制器信息
            try:
                import subprocess
                import re
                
                # 使用wmic命令获取GPU信息
                gpu_cmd = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if gpu_cmd.returncode == 0:
                    # 解析输出
                    lines = gpu_cmd.stdout.strip().split('\n')[1:]  # 跳过表头
                    for i, line in enumerate(lines):
                        if line.strip():  # 跳过空行
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                # 尝试提取显存大小
                                adapter_ram = 0
                                driver_version = "未知"
                                try:
                                    ram_match = re.search(r'(\d+)', parts[0])
                                    if ram_match:
                                        adapter_ram = int(ram_match.group(1)) / (1024 ** 3)  # 转换为GB
                                    
                                    # 尝试提取驱动版本
                                    for part in parts:
                                        if re.match(r'\d+\.\d+', part):
                                            driver_version = part
                                            break
                                except:
                                    pass
                                
                                # 最后一个部分是名称
                                name = ' '.join(parts[1:])
                                
                                # 判断GPU供应商
                                vendor = "未知"
                                if "NVIDIA" in name.upper():
                                    vendor = "NVIDIA"
                                elif "AMD" in name.upper() or "RADEON" in name.upper() or "ATI" in name.upper():
                                    vendor = "AMD"
                                elif "INTEL" in name.upper():
                                    vendor = "Intel"
                                
                                gpu_info.append({
                                    "index": i,
                                    "name": name,
                                    "memory_total": round(adapter_ram, 2),
                                    "memory_used": 0,
                                    "memory_free": 0,
                                    "driver_version": driver_version,
                                    "vbios_version": "未知",
                                    "vendor": vendor
                                })
            except Exception as e:
                logging.warning(f"使用wmic命令获取GPU信息失败: {e}")
            
            # 3.2 如果之前的方法未识别到显存大小，特别是对AMD显卡，尝试使用PowerShell获取更多信息
            if not gpu_info or any(gpu["memory_total"] == 0 for gpu in gpu_info):
                try:
                    import subprocess
                    import re
                    
                    # 使用PowerShell获取更详细的显卡信息
                    ps_cmd = "Get-WmiObject Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | Format-List"
                    ps_result = subprocess.run(
                        ["powershell", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if ps_result.returncode == 0:
                        # 解析PowerShell输出
                        output = ps_result.stdout
                        gpu_blocks = re.split(r'\r?\n\r?\n', output)
                        
                        for block in gpu_blocks:
                            if not block.strip():
                                continue
                                
                            name_match = re.search(r'Name\s*:\s*(.*)', block)
                            ram_match = re.search(r'AdapterRAM\s*:\s*(\d+)', block)
                            driver_match = re.search(r'DriverVersion\s*:\s*([\d\.]+)', block)
                            
                            if name_match:
                                name = name_match.group(1).strip()
                                
                                # 判断GPU供应商
                                vendor = "未知"
                                if "NVIDIA" in name.upper():
                                    vendor = "NVIDIA"
                                elif "AMD" in name.upper() or "RADEON" in name.upper() or "ATI" in name.upper():
                                    vendor = "AMD"
                                elif "INTEL" in name.upper():
                                    vendor = "Intel"
                                
                                # 提取显存大小
                                memory_total = 0
                                if ram_match:
                                    ram_bytes = int(ram_match.group(1))
                                    memory_total = round(ram_bytes / (1024**3), 2)  # 转换为GB
                                
                                # 提取驱动版本
                                driver_version = "未知"
                                if driver_match:
                                    driver_version = driver_match.group(1)
                                
                                # 查找现有的相同名称的GPU条目
                                found = False
                                for gpu in gpu_info:
                                    if gpu["name"] == name:
                                        # 更新现有条目的显存信息
                                        if gpu["memory_total"] == 0 and memory_total > 0:
                                            gpu["memory_total"] = memory_total
                                        if gpu["driver_version"] == "未知" and driver_version != "未知":
                                            gpu["driver_version"] = driver_version
                                        found = True
                                        break
                                
                                # 如果没找到匹配的条目，添加新条目
                                if not found:
                                    gpu_info.append({
                                        "index": len(gpu_info),
                                        "name": name,
                                        "memory_total": memory_total,
                                        "memory_used": 0,
                                        "memory_free": 0,
                                        "driver_version": driver_version,
                                        "vbios_version": "未知",
                                        "vendor": vendor
                                    })
                except Exception as e:
                    logging.warning(f"使用PowerShell获取GPU信息失败: {e}")
            
            # 3.3 专门针对AMD显卡的注册表查询方法
            if not gpu_info or any(gpu["vendor"] == "AMD" and gpu["memory_total"] == 0 for gpu in gpu_info):
                try:
                    import subprocess
                    import re
                    
                    # 使用reg query命令查询AMD显卡的注册表信息
                    reg_cmd = "reg query \"HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\" /s /v \"HardwareInformation.qwMemorySize\""
                    reg_result = subprocess.run(
                        ["cmd", "/c", reg_cmd],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if reg_result.returncode == 0:
                        # 解析注册表输出
                        lines = reg_result.stdout.split('\n')
                        current_key = ""
                        
                        for line in lines:
                            # 找到注册表键
                            if "HKEY_LOCAL_MACHINE" in line:
                                current_key = line.strip()
                            # 找到内存大小信息
                            elif "HardwareInformation.qwMemorySize" in line:
                                # 提取十六进制值并转换为GB
                                hex_match = re.search(r'REG_QWORD\s+(0x[0-9a-fA-F]+)', line)
                                if hex_match:
                                    mem_hex = hex_match.group(1)
                                    try:
                                        mem_bytes = int(mem_hex, 16)
                                        mem_gb = round(mem_bytes / (1024**3), 2)
                                        
                                        # 需要查询对应项的显卡名称
                                        device_desc_cmd = f"reg query \"{current_key}\" /v \"DriverDesc\""
                                        desc_result = subprocess.run(
                                            ["cmd", "/c", device_desc_cmd],
                                            capture_output=True,
                                            text=True,
                                            check=False
                                        )
                                        
                                        if desc_result.returncode == 0:
                                            desc_match = re.search(r'DriverDesc\s+REG_SZ\s+(.*)', desc_result.stdout)
                                            if desc_match:
                                                gpu_name = desc_match.group(1).strip()
                                                
                                                # 判断是否为AMD显卡
                                                if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper() or "ATI" in gpu_name.upper():
                                                    # 查找现有的相同名称的GPU条目
                                                    found = False
                                                    for gpu in gpu_info:
                                                        if gpu["name"] == gpu_name:
                                                            # 更新现有条目的显存信息
                                                            if gpu["memory_total"] == 0:
                                                                gpu["memory_total"] = mem_gb
                                                            found = True
                                                            break
                                                    
                                                    # 如果没找到匹配的条目，添加新条目
                                                    if not found:
                                                        gpu_info.append({
                                                            "index": len(gpu_info),
                                                            "name": gpu_name,
                                                            "memory_total": mem_gb,
                                                            "memory_used": 0,
                                                            "memory_free": 0,
                                                            "driver_version": "未知",
                                                            "vbios_version": "未知",
                                                            "vendor": "AMD"
                                                        })
                                    except ValueError:
                                        pass
                except Exception as e:
                    logging.warning(f"使用注册表查询AMD显卡信息失败: {e}")
            
            # 如果已经获取到GPU信息，就不需要继续尝试其他方法了
            if gpu_info:
                return gpu_info
        
        # 4. 尝试系统命令获取GPU信息 - Linux
        if platform.system() == "Linux" and not gpu_info:
            try:
                import subprocess
                import re
                
                # 首先尝试使用lspci命令
                gpu_cmd = subprocess.run(
                    ["lspci", "-v"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if gpu_cmd.returncode == 0:
                    # 查找VGA兼容控制器和3D控制器
                    vga_controllers = re.findall(r'(VGA compatible controller|3D controller):(.*?)(?=^\S|\Z)', 
                                               gpu_cmd.stdout, re.MULTILINE | re.DOTALL)
                    
                    for i, (_, controller_info) in enumerate(vga_controllers):
                        # 提取控制器名称
                        name_match = re.search(r'([A-Za-z0-9 ]+)', controller_info.split('\n')[0])
                        if name_match:
                            name = name_match.group(1).strip()
                            
                            # 判断GPU供应商
                            vendor = "未知"
                            if "NVIDIA" in name.upper():
                                vendor = "NVIDIA"
                            elif "AMD" in name.upper() or "RADEON" in name.upper() or "ATI" in name.upper():
                                vendor = "AMD"
                            elif "INTEL" in name.upper():
                                vendor = "Intel"
                            
                            gpu_info.append({
                                "index": i,
                                "name": name,
                                "memory_total": 0,  # 这里获取不到显存大小
                                "memory_used": 0,
                                "memory_free": 0,
                                "driver_version": "未知",
                                "vbios_version": "未知",
                                "vendor": vendor
                            })
                
                # 尝试使用nvidia-smi获取NVIDIA GPU信息
                if not gpu_info or any(gpu["memory_total"] == 0 for gpu in gpu_info):
                    nvidia_cmd = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"], 
                        capture_output=True, 
                        text=True, 
                        check=False
                    )
                    
                    if nvidia_cmd.returncode == 0:
                        # 检查是否有错误信息
                        if "has failed because you do not have suffient permissions" in nvidia_cmd.stdout:
                            logging.warning("NVIDIA-SMI需要管理员权限才能运行")
                        elif "NVIDIA-SMI has failed" in nvidia_cmd.stdout:
                            # NVIDIA驱动未安装或存在其他问题
                            logging.warning(f"NVIDIA-SMI执行失败: {nvidia_cmd.stdout.strip()}")
                            print(f"获取GPU信息失败: {nvidia_cmd.stdout.strip()}")
                        else:
                            lines = nvidia_cmd.stdout.strip().split('\n')[1:]  # 跳过表头
                            for line in lines:
                                if line.strip():
                                    # 检查行是否包含错误信息
                                    if "NVIDIA-SMI has failed" in line:
                                        logging.warning(f"NVIDIA-SMI执行失败: {line}")
                                        print(f"获取GPU信息失败: {line}")
                                        continue
                                        
                                    parts = line.split(', ')
                                    if len(parts) >= 3:
                                        try:
                                            # 确保第一部分是索引号
                                            if not parts[0].isdigit():
                                                logging.warning(f"无效的GPU索引: {parts[0]}")
                                                continue
                                                
                                            index = int(parts[0])
                                            name = parts[1]
                                            memory_str = parts[2]
                                            driver_version = parts[3] if len(parts) > 3 else "未知"
                                            
                                            # 提取显存大小
                                            memory_match = re.search(r'(\d+)', memory_str)
                                            memory_total = 0
                                            if memory_match:
                                                memory_value = int(memory_match.group(1))
                                                if "MiB" in memory_str:
                                                    memory_total = memory_value / 1024  # 转换为GB
                                                elif "GiB" in memory_str:
                                                    memory_total = memory_value
                                            
                                            # 更新或添加GPU信息
                                            found = False
                                            for gpu in gpu_info:
                                                if gpu["index"] == index:
                                                    gpu["memory_total"] = round(memory_total, 2)
                                                    gpu["driver_version"] = driver_version
                                                    found = True
                                                    break
                                            
                                            if not found:
                                                gpu_info.append({
                                                    "index": index,
                                                    "name": name,
                                                    "memory_total": round(memory_total, 2),
                                                    "memory_used": 0,
                                                    "memory_free": 0,
                                                    "driver_version": driver_version,
                                                    "vbios_version": "未知",
                                                    "vendor": "NVIDIA"
                                                })
                                        except ValueError as e:
                                            logging.warning(f"解析NVIDIA-SMI输出时出错: {e}")
                                            continue
                
                # 尝试使用rocm-smi获取AMD GPU信息
                if not gpu_info or any(gpu["vendor"] == "AMD" and gpu["memory_total"] == 0 for gpu in gpu_info):
                    rocm_cmd = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram"], 
                        capture_output=True, 
                        text=True, 
                        check=False
                    )
                    
                    if rocm_cmd.returncode == 0:
                        # 解析ROCm-SMI输出
                        for gpu in gpu_info:
                            if gpu["vendor"] == "AMD":
                                # 尝试从输出中找到对应索引的AMD GPU
                                memory_match = re.search(rf'GPU\[{gpu["index"]}\].*?VRAM:\s+(\d+)\s+(\w+)', 
                                                       rocm_cmd.stdout, re.DOTALL)
                                if memory_match:
                                    memory_value = int(memory_match.group(1))
                                    memory_unit = memory_match.group(2)
                                    
                                    if memory_unit.upper() == "MB":
                                        gpu["memory_total"] = round(memory_value / 1024, 2)
                                    elif memory_unit.upper() == "GB":
                                        gpu["memory_total"] = memory_value
                        
                        # 检查是否有尚未添加的AMD GPU
                        gpu_indices = [gpu["index"] for gpu in gpu_info if gpu["vendor"] == "AMD"]
                        for i in range(10):  # 假设最多10个GPU
                            if i not in gpu_indices:
                                # 尝试在输出中查找该索引的GPU
                                memory_match = re.search(rf'GPU\[{i}\].*?VRAM:\s+(\d+)\s+(\w+)', 
                                                       rocm_cmd.stdout, re.DOTALL)
                                if memory_match:
                                    memory_value = int(memory_match.group(1))
                                    memory_unit = memory_match.group(2)
                                    
                                    memory_total = 0
                                    if memory_unit.upper() == "MB":
                                        memory_total = round(memory_value / 1024, 2)
                                    elif memory_unit.upper() == "GB":
                                        memory_total = memory_value
                                    
                                    # 添加新的AMD GPU
                                    gpu_info.append({
                                        "index": i,
                                        "name": f"AMD GPU #{i}",
                                        "memory_total": memory_total,
                                        "memory_used": 0,
                                        "memory_free": 0,
                                        "driver_version": "未知",
                                        "vbios_version": "未知",
                                        "vendor": "AMD"
                                    })
            except Exception as e:
                logging.warning(f"使用Linux命令获取GPU信息失败: {e}")
        
        # 5. 尝试系统命令获取GPU信息 - macOS
        if platform.system() == "Darwin" and not gpu_info:
            try:
                import subprocess
                import re
                
                # 使用system_profiler获取GPU信息
                gpu_cmd = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if gpu_cmd.returncode == 0:
                    # 提取GPU型号
                    gpu_blocks = re.split(r'\n\n', gpu_cmd.stdout)
                    for i, block in enumerate(gpu_blocks):
                        if "Chipset Model:" in block:
                            name_match = re.search(r'Chipset Model: (.*)', block)
                            vram_match = re.search(r'VRAM \(.*\): (\d+) (\w+)', block)
                            
                            if name_match:
                                name = name_match.group(1).strip()
                                
                                # 提取显存大小
                                memory_total = 0
                                if vram_match:
                                    memory_value = int(vram_match.group(1))
                                    memory_unit = vram_match.group(2)
                                    
                                    if memory_unit.upper() == "MB":
                                        memory_total = memory_value / 1024
                                    elif memory_unit.upper() == "GB":
                                        memory_total = memory_value
                                
                                # 判断GPU供应商
                                vendor = "未知"
                                if "NVIDIA" in name.upper():
                                    vendor = "NVIDIA"
                                elif "AMD" in name.upper() or "RADEON" in name.upper() or "ATI" in name.upper():
                                    vendor = "AMD"
                                elif "INTEL" in name.upper():
                                    vendor = "Intel"
                                elif "APPLE" in name.upper():
                                    vendor = "Apple"
                                
                                gpu_info.append({
                                    "index": i,
                                    "name": name,
                                    "memory_total": round(memory_total, 2),
                                    "memory_used": 0,
                                    "memory_free": 0,
                                    "driver_version": "未知",
                                    "vbios_version": "未知",
                                    "vendor": vendor
                                })
            except Exception as e:
                logging.warning(f"使用macOS命令获取GPU信息失败: {e}")
        
        # 如果所有方法都无法获取GPU信息，返回空列表
        return gpu_info
    
    def get_metrics(self) -> SystemMetrics:
        """
        获取系统当前监控指标
        
        返回:
            系统监控指标对象
        """
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 获取网络IO
        net_io = psutil.net_io_counters()
        
        # 如果是第一次调用，前一次值为当前值
        if self.prev_net_io is None:
            self.prev_net_io = net_io
            self.prev_net_io_time = time.time()
            network_sent = 0
            network_recv = 0
        else:
            # 计算时间差
            now = time.time()
            time_diff = now - self.prev_net_io_time
            
            # 计算网络IO速率 (KB/s)
            if time_diff > 0:
                network_sent = (net_io.bytes_sent - self.prev_net_io.bytes_sent) / 1024 / time_diff
                network_recv = (net_io.bytes_recv - self.prev_net_io.bytes_recv) / 1024 / time_diff
            else:
                network_sent = 0
                network_recv = 0
            
            # 更新前一次值
            self.prev_net_io = net_io
            self.prev_net_io_time = now
        
        # 获取GPU指标
        gpu_metrics = []
        
        # 仅当NVIDIA GPU可用时才尝试获取其指标
        if self.nvidia_gpu_available:
            # 1. 使用pynvml获取NVIDIA GPU指标
            if hasattr(self, '_pynvml_available') and self._pynvml_available:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                        
                        gpu_metrics.append({
                            "index": i,
                            "gpu_utilization": utilization.gpu,
                            "memory_utilization": utilization.memory,
                            "memory_used": round(memory.used / (1024**3), 2),  # GB
                            "memory_total": round(memory.total / (1024**3), 2),  # GB
                            "temperature": temperature,
                            "power_usage": round(power_usage, 2),  # 瓦特
                            "vendor": "NVIDIA"
                        })
                    
                    pynvml.nvmlShutdown()
                except Exception as e:
                    print(f"获取NVIDIA GPU指标失败: {e}")
                    
                    # 如果因为权限问题失败，尝试使用命令行工具（仅适用于NVIDIA GPU）
                    import subprocess
                    import re
                    
                    try:
                        # 使用nvidia-smi命令获取GPU利用率
                        nvidia_cmd = subprocess.run(
                            ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv"], 
                            capture_output=True, 
                            text=True, 
                            check=False
                        )
                        
                        if nvidia_cmd.returncode == 0:
                            # 检查是否有错误信息
                            if "has failed because you do not have suffient permissions" in nvidia_cmd.stdout:
                                logging.warning("NVIDIA-SMI需要管理员权限才能运行")
                            elif "NVIDIA-SMI has failed" in nvidia_cmd.stdout:
                                # NVIDIA驱动未安装或存在其他问题
                                logging.warning(f"NVIDIA-SMI执行失败: {nvidia_cmd.stdout.strip()}")
                                print(f"获取GPU性能指标失败: {nvidia_cmd.stdout.strip()}")
                            else:
                                lines = nvidia_cmd.stdout.strip().split('\n')[1:]  # 跳过表头
                                for line in lines:
                                    if line.strip():
                                        # 检查行是否包含错误信息
                                        if "NVIDIA-SMI has failed" in line:
                                            logging.warning(f"NVIDIA-SMI执行失败: {line}")
                                            print(f"获取GPU性能指标失败: {line}")
                                            continue
                                            
                                        try:
                                            parts = line.split(', ')
                                            if len(parts) >= 6:
                                                # 确保第一部分是索引号
                                                if not parts[0].isdigit():
                                                    logging.warning(f"无效的GPU索引: {parts[0]}")
                                                    continue
                                                    
                                                index = int(parts[0])
                                                gpu_util = int(re.search(r'(\d+)', parts[1]).group(1))
                                                mem_util = int(re.search(r'(\d+)', parts[2]).group(1))
                                                mem_used = float(re.search(r'(\d+)', parts[3]).group(1))
                                                mem_total = float(re.search(r'(\d+)', parts[4]).group(1))
                                                temperature = int(re.search(r'(\d+)', parts[5]).group(1))
                                                power_usage = float(re.search(r'(\d+\.\d+)', parts[6]).group(1)) if len(parts) > 6 else 0
                                                
                                                # 转换单位
                                                if "MiB" in parts[3]:
                                                    mem_used /= 1024  # 转换为GB
                                                if "MiB" in parts[4]:
                                                    mem_total /= 1024  # 转换为GB
                                                
                                                gpu_metrics.append({
                                                    "index": index,
                                                    "gpu_utilization": gpu_util,
                                                    "memory_utilization": mem_util,
                                                    "memory_used": round(mem_used, 2),
                                                    "memory_total": round(mem_total, 2),
                                                    "temperature": temperature,
                                                    "power_usage": round(power_usage, 2),
                                                    "vendor": "NVIDIA"
                                                })
                                        except (ValueError, AttributeError, IndexError) as e:
                                            logging.warning(f"解析NVIDIA-SMI输出时出错: {e}")
                                            continue
                    except Exception as e:
                        print(f"使用nvidia-smi命令获取GPU指标失败: {e}")
        
        # 对于非NVIDIA显卡，添加占位指标（不支持动态监控）
        system_info = self._system_info
        for gpu in system_info.gpu_info:
            if gpu.get("driver", "").upper() != "NVIDIA" and gpu.get("vendor", "").upper() != "NVIDIA":
                # 检查是否已经在metrics列表中
                if not any(m.get("index") == gpu.get("index") for m in gpu_metrics):
                    gpu_metrics.append({
                        "index": gpu.get("index", 0),
                        "name": gpu.get("name", "未知"),
                        "vendor": "非NVIDIA",
                        "no_dynamic_monitoring": True,  # 标记不支持动态监控
                        "memory_total": gpu.get("memory_total", 0)
                    })
        
        # 返回系统指标
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            network_sent=network_sent,
            network_recv=network_recv,
            gpu_metrics=gpu_metrics
        ) 