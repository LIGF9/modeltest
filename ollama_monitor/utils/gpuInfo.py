import re
import subprocess
from typing import List, Dict, Optional
import tempfile
import os
import csv

class GPUInfoCollector:
    def __init__(self):
        self.results: List[Dict] = []

    def collect(self):
        """主收集方法"""
        try:
            # DXDIAG信息
            self.get_with_dxdiag()
            counter = 0
            # 处理dxdiag结果
            for device in self.results:
                # 检查显存合理性
                if device['dedicated'] <= 0 and device['shared'] <= 0:
                    continue
                else:
                    counter += 1
            if counter == 0:
                self._fallback_collect()
                            
        except Exception as e:
            print(f"[收集错误] {str(e)}")      
            # # 最终后备方案
            self._fallback_collect()
        
        finally:
            # # 新增过滤步骤
            self._filter_gpu_devices()
            return self.results

    def _filter_gpu_devices(self):
        """过滤非显卡设备"""
        filtered = []
        gpu_keywords = [
            r'radeon', r'geforce', r'intel', r'arc', 
            r'rtx', r'gtx', r'quadro', r'graphics', r'gpu',
            r'hd graphics', r'iris xe', r'firepro'
        ]
        
        for device in self.results:
            name = device['name'].lower()
            names = [device["name"] for device in filtered]
            
            # 排除常见虚拟设备
            if re.search(r'oray|virtual|software|microsoft basic|parsec|idd|device|driver|display', name):
                continue
                
            # 检查显存合理性
            if device['dedicated'] <= 0 and device['shared'] <= 0:
                continue
                
            # 检查是否包含显卡关键词
            if any(re.search(kw, name) for kw in gpu_keywords) and name not in names:
                filtered.append(device)
                
        self.results = filtered

    def print_report(self):
        """带使用情况的报告"""
        def custom_round(x):
            integer_part = int(x)
            decimal_part = x - integer_part
            return integer_part + (1 if decimal_part > 0.5 else 0)

        print("\n=== GPU 信息报告 ===")
        # print("self.results:",self.results)
        for idx, gpu in enumerate(self.results, 1):
            print(f"GPU #{idx}")
            print(f"  名称: {gpu['name']}")
            print(f"  总可用显存: {custom_round(gpu['total']/1024)}GB")
            print(f"  显卡专用显存: {custom_round(gpu['dedicated']/1024)}GB")
            print(f"  系统共享显存: {custom_round(gpu['shared']/1024)}GB")
            print(f"  数据来源: {gpu['source']}")
        print("===================\n")

    def get_system_memory(self) -> int:
        """获取系统总物理内存 (MB)"""
        try:
            # 执行命令获取内存条容量列表
            output = subprocess.check_output(
                "wmic MemoryChip get Capacity",
                shell=True,
                stderr=subprocess.STDOUT
            ).decode('utf-8', errors='ignore')
            
            # 提取所有内存容量并求和
            memory_sizes = [
                int(line.strip()) 
                for line in output.splitlines() 
                if line.strip().isdigit()
            ]
            total_bytes = sum(memory_sizes)
            return total_bytes  # 转换为GB
        except Exception as e:
            return f"Error: {e}"

    def _fallback_collect(self):

        def parse_int(value_str, default=0):
            """将字符串转换为整数，失败时返回默认值。"""
            try:
                return int(value_str)
            except ValueError:
                return default
    
        # 执行PowerShell命令
        cmd = [
            'powershell',
            '-Command',
            'Get-CimInstance Win32_VideoController | '
            'Select-Object Name, AdapterRAM, SharedSystemMemory, VideoMemoryType | '
            'Format-List'
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'  # 忽略编码错误
        )

        if result.returncode != 0:
            return []
        
        output = result.stdout.strip()
        with open('gpu_raw_output.txt', 'w', encoding='utf-8') as f:
            f.write(output)
        blocks = []
        current_block = None
        
        # 按行分割并处理块
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('Name'):
                if current_block is not None:
                    blocks.append(current_block)
                current_block = [line]
            else:
                if current_block is not None:
                    current_block.append(line)
        if current_block:
            blocks.append(current_block)
        for block in blocks:
            props = {}
            for line in block:
                if ':' not in line:
                    continue  # 忽略无效行
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                props[key] = value
            
            # 提取字段并处理转换
            name = props.get('Name', 'Unknown')
            adapter_ram = parse_int(props.get('AdapterRAM', '0'))
            shared_system = parse_int(props.get('SharedSystemMemory', '0'))
            shared_system = shared_system if shared_system else self.get_system_memory()// 2 
            total = adapter_ram + shared_system
            
            gpu_info = {
                'name': name,
                'total': total //1024**2,
                'dedicated':  adapter_ram//1024**2,
                'shared': shared_system //1024**2,
                'source': 'PowerShell后备方案'
            }
            self.results.append(gpu_info)

    def get_with_dxdiag(self):
        """通过dxdiag获取显卡信息并解析"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # 执行dxdiag并保存到临时文件
            result = subprocess.run(
                ['dxdiag', '/t', temp_filename],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                text=True
            )
            # 重置结果集
            self.results = []
            # 状态机变量
            in_display_section = False
            current_device = {}
            separator_count = 0
            # 读取临时文件内容
            with open(temp_filename, 'r') as f:
                for line in f:
                    stripped = line.strip()                    
                    # 检测Display Devices章节开始
                    if not in_display_section:
                        if stripped == 'Display Devices':
                            separator_count += 1
                            continue
                        elif separator_count == 1 and '-------' in stripped:
                            separator_count += 1
                            continue
                        elif separator_count >= 2:
                            in_display_section = True
                            separator_count = 0
                    
                    # 检测章节结束（遇到两个连续分隔线）
                    if '-------' in stripped:
                        separator_count += 1
                        if separator_count >= 2 and current_device:
                            self._finalize_device(current_device)
                            current_device = {}
                            break  # 结束章节处理
                        continue
                    else:
                        separator_count = 0

                    # 处理设备信息
                    if ':' in line:  # 仅处理包含冒号的键值对
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 新设备开始
                        if key == 'Card name':
                            if current_device:
                                self._finalize_device(current_device)
                            current_device = {'Card name': value}
                            # print(key,value)
                        # 显存相关字段
                        elif key in ('Display Memory', 'Dedicated Memory', 'Shared Memory'):
                            current_device[key] = value
                            # print(key,value)
                # 处理最后一个设备
                if current_device:
                    self._finalize_device(current_device)

        except subprocess.CalledProcessError as e:
            print(f"命令执行失败，错误码：{e.returncode}")
            print("标准错误：", e.stderr)
        finally:
            # 清理临时文件
            os.remove(temp_filename)

    def _finalize_device(self, device):
        """验证并添加设备到结果集"""
        try:
            # 提取关键字段
            # print(device)
            name = device.get('Card name', '未知显卡').strip()
            total = self._parse_memory(device.get('Display Memory', 0))
            dedicated = self._parse_memory(device.get('Dedicated Memory', 0))
            shared = self._parse_memory(device.get('Shared Memory', 0))
            
            # 验证显存总量
            if total <= 0 and (dedicated >= 0 or shared >= 0):
                total = dedicated + shared
            
            # 如果专用显存为0但有总显存，则假设全部为专用显存（针对某些显示不完整的情况）
            if dedicated <= 0 and total >= 0:
                dedicated = total
                shared = 0
            
            # 组装设备信息
            gpu_info = {
                'name': name,
                'total': total,
                'dedicated': dedicated,
                'shared': shared,
                'source': 'DxDiag'
            }
            
            self.results.append(gpu_info)
        except Exception as e:
            print(f"处理设备信息时出错: {e}")
            
    def _parse_memory(self, memory_str):
        """解析显存字符串为MB数值"""
        if not memory_str:
            return 0
            
        try:
            # 匹配数字和单位
            match = re.search(r'(\d+)\s*(\w+)', memory_str)
            if not match:
                return 0
                
            value = int(match.group(1))
            unit = match.group(2).lower()
            
            # 转换为MB
            if 'gb' in unit:
                return value * 1024
            elif 'mb' in unit:
                return value
            elif 'kb' in unit:
                return value / 1024
            elif 'tb' in unit:
                return value * 1024 * 1024
            else:
                # 默认按MB算
                return value
        except:
            return 0
        
if __name__ == "__main__":
    collector = GPUInfoCollector()
    collector.collect()
    collector.print_report()