#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能测试模块
用于测试Ollama模型的性能指标
"""

import time
import json
import random
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .system_monitor import SystemMonitor
from ollama_monitor.utils.ollama_client import OllamaClient
import re
import platform
from typing import Dict, Optional


@dataclass
class TestPrompt:
    """测试提示词数据类"""    
    category: str
    name: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "name": self.name,
            "category": self.category,
            "content": self.content
        }

@dataclass
class TestResult:
    """单次测试结果数据类"""
    prompt: str
    prompt_content: str  # 完整提示词内容
    prompt_type: str = "未分类"  # 提示词类型/类别
    response_text: str = ""  # 模型回复内容
    first_token_latency: float = 0.0  # 毫秒
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0  # 毫秒
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    error: Optional[str] = None

@dataclass
class PerformanceReport:
    """性能测试报告数据类"""
    model_name: str
    system_info: Dict[str, Any]
    test_params: Dict[str, Any]
    results: List[TestResult]
    avg_first_token_latency: float = 0
    avg_tokens_per_second: float = 0
    performance_rating: str = "未评级"
    
    def __post_init__(self):
        """初始化后计算平均值和性能评级"""
        if not self.results or any(r.error for r in self.results):
            return
        
        # 计算平均值
        self.avg_first_token_latency = sum(r.first_token_latency for r in self.results) / len(self.results)
        self.avg_tokens_per_second = sum(r.tokens_per_second for r in self.results) / len(self.results)
        
        # 性能评级 - 仅基于生成速度
        if self.avg_tokens_per_second >= 60:
            self.performance_rating = "极佳"
        elif self.avg_tokens_per_second >= 30:
            self.performance_rating = "优秀"
        elif self.avg_tokens_per_second >= 10:
            self.performance_rating = "良好"
        else:
            self.performance_rating = "较差"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "system_info": self.system_info,
            "test_params": self.test_params,
            "results": [asdict(r) for r in self.results],
            "avg_first_token_latency": self.avg_first_token_latency,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "performance_rating": self.performance_rating
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

class PerformanceTester:
    """Ollama模型性能测试器"""
    
    # 预设提示词列表
    DEFAULT_PROMPTS = [
    # 文本处理类 - 10条
    TestPrompt("文本处理", "文本总结", "请总结以下文本的主要内容：近年来，人工智能技术快速发展，从机器学习到深度学习，再到大型语言模型，技术进步日新月异。特别是在2023年，生成式AI成为主流，各种大模型如ChatGPT、Claude等展现出惊人的能力，能够进行文本生成、代码编写、内容创作等多种任务。然而，这些技术的广泛应用也带来了隐私保护、版权归属、内容真实性等一系列问题，需要社会各界共同关注和解决。"),
    TestPrompt("文本处理", "关键词提取", "从以下文本中提取5个最重要的关键词：数字化转型已成为全球企业的必然选择，它不仅涉及技术升级，还包括组织结构、业务流程和企业文化的全面变革。企业通过引入云计算、大数据、人工智能等技术，实现内部流程优化和对外服务提升。数字化转型成功的关键在于清晰的战略规划、灵活的组织结构、持续的创新能力以及强大的数据管理能力。"),
    TestPrompt("文本处理", "文本分类", "请将以下文本分类到最合适的类别（科技、健康、教育、商业、娱乐、体育）：随着量子计算技术的突破，科学家们已经能够在实验室中实现50个量子比特的纠缠，这一突破将为密码学带来革命性的变化，同时也可能挑战当前互联网安全的基础架构。"),
    TestPrompt("文本处理", "文本改写", "请用更正式的语言重写以下文本：那个会议真是太无聊了，大家都快睡着了，而且那个演讲者说的东西没人懂，感觉完全浪费时间。"),
    TestPrompt("文本处理", "情感分析", "分析以下评论的情感倾向（积极、消极或中性）并解释理由：这家餐厅的服务态度实在太差了，等了一个小时才上菜，而且菜品质量一般，价格却很高，绝对不会再来第二次。"),
    TestPrompt("文本处理", "文本对比", "比较以下两段文本的相似点和不同点：文本1：人工智能正在改变我们的生活方式，从智能手机助手到自动驾驶汽车，AI技术无处不在。文本2：AI技术正在渗透到日常生活的各个方面，包括智能家居、语音助手和自动驾驶技术，彻底改变了人们的生活方式。"),
    TestPrompt("文本处理", "摘要生成", "为以下学术文章生成一个200字的摘要：本研究探讨了气候变化对全球粮食安全的潜在影响。通过分析过去50年的气象数据和农作物产量记录，我们发现全球平均气温每升高1摄氏度，主要谷物产量将下降约5.6%。研究还表明，发展中国家将承受气候变化带来的最严重后果，因为这些地区的农业生产更依赖稳定的气候条件，同时适应能力较弱。我们建议各国政府增加对抗旱品种研发的投资，改进灌溉系统，并建立农作物多样化计划，以增强粮食系统对气候变化的适应能力。"),
    TestPrompt("文本处理", "文本纠错", "请找出并纠正以下文本中的语法和拼写错误：我昨天去了图书馆，借了三本是关于历史的书。这些书很有趣，但是有一点难懂，因为里面有好多我不认识的词语。我打算每天都看一点，这样可以提高我的阅读能力吧。"),
    TestPrompt("文本处理", "文本扩写", "请将以下简短描述扩写成一个200字的段落：夕阳西下，湖面泛起金光。"),
    TestPrompt("文本处理", "文本简化", "请将以下复杂的技术文本简化为普通人也能理解的内容：量子纠缠是量子力学中的一种现象，当两个或多个粒子以某种方式相互作用或共享空间接近度时，即使粒子相隔很远，其中一个粒子的量子态也不能独立于其他粒子的状态来描述。"),

    # 知识问答类 - 10条
    TestPrompt("知识问答", "量子计算机", "请解释量子计算机的工作原理和它与传统计算机的主要区别。"),
    TestPrompt("知识问答", "历史事件", "请详细介绍第一次工业革命的起因、过程和影响。"),
    TestPrompt("知识问答", "科学原理", "请解释光合作用的过程以及它对地球生态系统的重要性。"),
    TestPrompt("知识问答", "文化差异", "请比较中国文化和西方文化在家庭观念上的主要差异。"),
    TestPrompt("知识问答", "技术发展", "请介绍人工智能的发展历程，包括主要的里程碑事件和突破。"),
    TestPrompt("知识问答", "医学知识", "请解释人体免疫系统的工作原理以及疫苗如何帮助预防疾病。"),
    TestPrompt("知识问答", "天文学", "请描述黑洞的形成过程、特性以及它们在宇宙中的作用。"),
    TestPrompt("知识问答", "地理知识", "请解释季风气候的形成原因及其对亚洲地区农业生产的影响。"),
    TestPrompt("知识问答", "经济概念", "请解释通货膨胀的原因、类型以及政府可以采取哪些措施来控制通货膨胀。"),
    TestPrompt("知识问答", "哲学思想", "请解释并比较康德和黑格尔哲学的核心观点及其对现代思想的影响。"),

    # 编程类 - 10条
    TestPrompt("编程", "代码生成", "请用Python写一个函数，实现快速排序算法，并解释其时间复杂度。"),
    TestPrompt("编程", "代码优化", "请优化以下Python代码的效率，并解释你做的改进：def find_duplicates(arr):\n    duplicates = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in duplicates:\n                duplicates.append(arr[i])\n    return duplicates"),
    TestPrompt("编程", "算法设计", "请设计一个算法来解决以下问题：给定一个整数数组，找出其中和为特定值的两个数。例如，输入[2, 7, 11, 15]和目标值9，应该返回[0, 1]，因为nums[0] + nums[1] = 9。"),
    TestPrompt("编程", "数据结构", "请解释二叉树和二叉搜索树的区别，并用Java实现一个二叉搜索树的插入操作。"),
    TestPrompt("编程", "接口设计", "设计一个RESTful API来管理一个在线书店的图书库存。包括增加新书、更新书籍信息、查询图书和删除图书的接口。"),
    TestPrompt("编程", "调试问题", "以下JavaScript代码有一个bug，它应该计算数组中所有数字的和，但结果不正确。请找出问题并修复：function sum(arr) {\n  let result = 0;\n  for (let i = 0; i <= arr.length; i++) {\n    result += arr[i];\n  }\n  return result;\n}"),
    TestPrompt("编程", "前端开发", "请用HTML和CSS创建一个响应式导航栏，在大屏幕上水平显示菜单项，在小屏幕上显示一个汉堡菜单按钮。"),
    TestPrompt("编程", "后端开发", "用Node.js和Express框架编写代码，创建一个简单的API端点，从数据库中获取用户列表并返回JSON格式的响应。"),
    TestPrompt("编程", "数据库查询", "为一个电子商务网站编写SQL查询，找出过去30天内购买总金额超过1000元的客户，并按购买金额降序排列。"),
    TestPrompt("编程", "系统设计", "设计一个高并发的短链接服务，要求能够处理每秒数千次的链接生成和重定向请求，并考虑可扩展性和容错性。"),

    # 创意类 - 10条
    TestPrompt("创意", "创意写作", "请以'未来城市'为主题，写一篇短篇科幻小说的开头，约300字。"),
    TestPrompt("创意", "诗歌创作", "请创作一首关于春天的现代诗，表达对新生命和希望的赞美。"),
    TestPrompt("创意", "故事构思", "请构思一个发生在古代中国的悬疑故事的情节梗概，包括主要角色、核心谜题和故事发展方向。"),
    TestPrompt("创意", "角色设计", "请设计一个奇幻小说的主角，包括其外貌、性格、能力、背景故事和成长轨迹。"),
    TestPrompt("创意", "世界观构建", "请构建一个后启示录世界的设定，描述灾难的性质、幸存者社会的组织形式、面临的挑战以及新兴的文化特点。"),
    TestPrompt("创意", "对话创作", "请创作一段有冲突的对话，场景是两个性格迥异的朋友在讨论是否应该放弃安稳的工作去创业。"),
    TestPrompt("创意", "场景描写", "请描写一个废弃多年的游乐园在暴雨夜晚的场景，营造出神秘而略带恐怖的氛围。"),
    TestPrompt("创意", "创意标题", "为一部讲述人工智能觉醒并与人类和平共处的科幻电影想出10个吸引人的标题。"),
    TestPrompt("创意", "创意广告", "请为一款能够帮助人们改善睡眠质量的智能床垫写一段富有创意的广告文案。"),
    TestPrompt("创意", "替代历史", "请构想一个历史转折点的替代发展：如果中国在15世纪继续郑和下西洋的航海探索，世界历史可能会如何发展？"),

    # 推理类 - 10条
    TestPrompt("推理", "逻辑推理", "小明比小华大2岁，小华比小红小3岁，小红今年15岁。请问小明今年多少岁？请给出推理过程。"),
    TestPrompt("推理", "数学推理", "一个水箱，上面1/4的容积有水，往里面倒入18升水后，水箱的3/5容积有水，求水箱的容积。"),
    TestPrompt("推理", "概率问题", "一个袋子里有3个红球和5个蓝球。如果随机取出2个球，求取出的两个球都是红球的概率。请给出计算过程。"),
    TestPrompt("推理", "侦探谜题", "在一个密室中发现一具尸体，房间里有一把手枪、一根绳子和一杯水。根据以下线索推理可能的死因：死者没有明显外伤；房间门窗从内部锁住；手枪没有被使用过；水杯中有残留的白色粉末。"),
    TestPrompt("推理", "归纳推理", "观察以下数列，找出规律并计算下一个数：2, 6, 12, 20, 30, ？"),
    TestPrompt("推理", "条件推理", "如果所有的A都是B，所有的B都是C，那么我们能否确定所有的A都是C？请解释你的推理。"),
    TestPrompt("推理", "组合问题", "一个班级有10名学生，需要选出3人组成一个小组。请计算有多少种不同的组合方式，并给出推导过程。"),
    TestPrompt("推理", "逻辑难题", """村里有两种人：诚实者总是说真话，说谎者总是说假话。你遇到三个人A、B和C。A说："B是诚实者"，B说："C是说谎者"，C说："A和B类型不同"。请确定每个人的类型。"""),
    TestPrompt("推理", "决策分析", "假设你是一家公司的经理，需要在两个项目中选择一个投资。项目A有60%的概率获得100万利润，40%的概率亏损30万。项目B有80%的概率获得50万利润，20%的概率亏损20万。从期望值角度分析，应该选择哪个项目？"),
    TestPrompt("推理", "时间推理", "小李和小王约定在图书馆见面。小李提前15分钟到达，等了25分钟后，比小王早到50分钟的小张也到了。请问小王迟到了多少分钟？"),

    # 复杂任务类 - 10条
    TestPrompt("复杂任务", "多步骤任务", "1. 生成一个包含10个随机整数（1-100）的列表\n2. 对这个列表进行升序排序\n3. 计算这些数字的平均值、中位数和标准差\n4. 绘制一个简单的直方图描述这些数据的分布"),
    TestPrompt("复杂任务", "数据分析", "分析以下销售数据，找出趋势和异常，并提供改进建议：\n2022年1月：$12,500\n2022年2月：$13,200\n2022年3月：$15,600\n2022年4月：$14,800\n2022年5月：$21,500\n2022年6月：$10,300\n2022年7月：$16,400\n2022年8月：$16,900\n2022年9月：$18,200\n2022年10月：$17,500\n2022年11月：$19,800\n2022年12月：$25,600"),
    TestPrompt("复杂任务", "教学计划", "为一个高中物理课设计一个关于牛顿运动定律的完整教学计划，包括教学目标、所需材料、课前准备、教学步骤、实验演示、小组活动、评估方法和家庭作业。"),
    TestPrompt("复杂任务", "项目规划", "为一个小型创业公司设计一个社交媒体营销策略，包括目标受众分析、内容计划、发布时间表、预算分配、绩效指标和风险管理。"),
    TestPrompt("复杂任务", "比较分析", "比较三种主流编程语言（Python、JavaScript和Java）在以下方面的优缺点：语法简洁性、学习曲线、性能效率、社区支持、应用场景和未来发展前景。"),
    TestPrompt("复杂任务", "决策矩阵", "创建一个决策矩阵，帮助一个年轻专业人士在三个工作机会之间做选择：一家知名大公司的普通职位、一家中等规模公司的管理职位和一家有风险但有股权的创业公司。考虑因素包括：薪资水平、职业发展、工作稳定性、工作生活平衡和个人成长。"),
    TestPrompt("复杂任务", "用户旅程", "为一款健康饮食应用设计详细的用户旅程，从初次下载到成为忠实用户，包括用户动机、行为、痛点和关键互动节点，以及应用如何在每个阶段满足用户需求。"),
    TestPrompt("复杂任务", "风险评估", "为一个计划在东南亚开设新工厂的制造企业进行全面的风险评估，包括政治风险、经济风险、供应链风险、法律合规风险、环境风险和社会风险，并提出相应的风险管理策略。"),
    TestPrompt("复杂任务", "伦理分析", "分析人工智能在医疗诊断中应用的伦理问题，包括隐私保护、算法偏见、医生责任边界、患者知情权和技术依赖性，并提出相应的伦理准则和监管建议。"),
    TestPrompt("复杂任务", "综合方案", "为一个中型企业设计一个环保转型方案，包括能源使用、废弃物管理、供应链优化、员工参与和品牌传播等方面的具体措施，同时考虑成本效益和实施时间表。")
]
    
    def __init__(self, ollama_client: OllamaClient, system_monitor: SystemMonitor):
        """
        初始化性能测试器
        
        参数:
            ollama_client: Ollama API客户端
            system_monitor: 系统监控器
        """
        self.client = ollama_client
        self.system_monitor = system_monitor
        
        # 按类别组织提示词
        self.prompts_by_category = {}
        self.custom_prompts = []  # 存储用户自定义提示词
        
        # 初始化默认提示词
        self._init_default_prompts()
        
        # 尝试从文件加载提示词
        self._load_prompts_from_file()
    
    def _init_default_prompts(self):
        """初始化默认提示词"""
        # 按类别对默认提示词进行分组
        for prompt in self.DEFAULT_PROMPTS:
            if prompt.category not in self.prompts_by_category:
                self.prompts_by_category[prompt.category] = []
            self.prompts_by_category[prompt.category].append(prompt)
    
    def _load_prompts_from_file(self, filepath: str = "prompts.txt"):
        """
        从文件加载提示词
        
        参数:
            filepath: 提示词文件路径
        """
        try:
            if not os.path.exists(filepath):
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                current_category = "未分类"
                
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 检查是否是类别行
                    if line.startswith('[') and line.endswith(']'):
                        current_category = line[1:-1]
                        if current_category not in self.prompts_by_category:
                            self.prompts_by_category[current_category] = []
                    else:
                        # 假设格式为：提示词名称: 提示词内容
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            name = parts[0].strip()
                            content = parts[1].strip()
                            if name and content:
                                prompt = TestPrompt(name, content, current_category)
                                self.prompts_by_category[current_category].append(prompt)
        except Exception as e:
            print(f"加载提示词文件失败: {e}")

    def save_prompts_to_file(self, filepath: str = "prompts.txt"):
        """
        将提示词保存到文件
        
        参数:
            filepath: 提示词文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# Ollama性能测试提示词文件\n")
                f.write("# 格式: [类别]\n")
                f.write("# 提示词名称: 提示词内容\n\n")
                
                # 先写入自定义提示词
                if self.custom_prompts:
                    f.write("[自定义]\n")
                    for prompt in self.custom_prompts:
                        f.write(f"{prompt.name}: {prompt.content}\n")
                    f.write("\n")
                
                # 再写入其他类别的提示词
                for category, prompts in self.prompts_by_category.items():
                    if category != "自定义":  # 自定义类别已经单独处理
                        f.write(f"[{category}]\n")
                        for prompt in prompts:
                            f.write(f"{prompt.name}: {prompt.content}\n")
                        f.write("\n")
            
            return True
        except Exception as e:
            print(f"保存提示词文件失败: {e}")
            return False
    
    def add_custom_prompt(self, name: str, content: str, category: str = "自定义"):
        """
        添加自定义测试提示词
        
        参数:
            name: 提示词名称
            content: 提示词内容
            category: 提示词类别
        """
        # 创建提示词
        prompt = TestPrompt(name, content, category)
        
        # 添加到自定义提示词列表
        self.custom_prompts.append(prompt)
        
        # 同时添加到类别分组中
        if category not in self.prompts_by_category:
            self.prompts_by_category[category] = []
        self.prompts_by_category[category].append(prompt)
    
    def get_prompts(self) -> List[TestPrompt]:
        """
        获取所有可用的测试提示词
        
        返回:
            测试提示词列表
        """
        all_prompts = []
        for prompts in self.prompts_by_category.values():
            all_prompts.extend(prompts)
        return all_prompts
    
    def get_prompts_by_category(self) -> Dict[str, List[TestPrompt]]:
        """
        按类别获取所有提示词
        
        返回:
            按类别分组的提示词字典
        """
        return self.prompts_by_category
    
    def select_balanced_prompts(self, num_prompts: int) -> List[TestPrompt]:
        """
        均衡选择各类别的提示词，确保每个类别都有平等的机会被选择
        
        参数:
            num_prompts: 需要选择的提示词数量
            
        返回:
            选择的提示词列表
        """
        # 如果有自定义提示词，优先使用
        selected_prompts = []
        available_categories = list(self.prompts_by_category.keys())
        
        # 排除空类别
        available_categories = [cat for cat in available_categories 
                               if self.prompts_by_category[cat]]
        
        if not available_categories:
            return []  # 没有可用的提示词
        
        # 如果有自定义提示词，优先添加一部分（最多占总数的20%）
        if "自定义" in self.prompts_by_category and self.prompts_by_category["自定义"]:
            custom_prompts = self.prompts_by_category["自定义"]
            # 确定要添加的自定义提示词数量（最多不超过20%的总数，且不超过自定义提示词总数）
            custom_count = min(len(custom_prompts), max(1, num_prompts // 5))
            # 随机选择一部分自定义提示词
            custom_selected = random.sample(custom_prompts, custom_count)
            selected_prompts.extend(custom_selected)
            # 从类别列表中移除自定义类别
            if "自定义" in available_categories:
                available_categories.remove("自定义")
        
        # 确定剩余需要的提示词数量
        remaining = num_prompts - len(selected_prompts)
        
        # 如果剩余数量大于0且有可用类别
        if remaining > 0 and available_categories:
            # 第一轮：按类别平均分配
            # 计算每个类别至少分配多少个提示词
            initial_per_category = remaining // len(available_categories)
            
            # 均匀地从每个类别中选择提示词
            for category in available_categories:
                if category in self.prompts_by_category and self.prompts_by_category[category]:
                    category_prompts = self.prompts_by_category[category]
                    # 确定从此类别中选择多少个提示词
                    count = min(initial_per_category, len(category_prompts))
                    if count > 0:
                        # 随机选择不重复的提示词
                        selected_from_category = random.sample(category_prompts, count)
                        selected_prompts.extend(selected_from_category)
            
            # 计算仍然需要多少提示词
            remaining = num_prompts - len(selected_prompts)
            
            # 第二轮：公平地分配剩余的提示词（轮流从每个类别中选择）
            if remaining > 0:
                # 创建一个类别的循环队列
                category_cycle = available_categories.copy()
                random.shuffle(category_cycle)  # 随机排序类别
                
                # 跟踪每个类别已经选择的提示词
                selected_by_category = {cat: [] for cat in available_categories}
                for prompt in selected_prompts:
                    if prompt.category in selected_by_category:
                        selected_by_category[prompt.category].append(prompt)
                
                # 轮流从每个类别中选择提示词
                while remaining > 0 and category_cycle:
                    # 获取下一个类别
                    category = category_cycle.pop(0)
                    
                    # 获取该类别中未选择的提示词
                    category_prompts = self.prompts_by_category.get(category, [])
                    unused_prompts = [p for p in category_prompts 
                                     if p not in selected_by_category.get(category, [])]
                    
                    # 如果该类别还有未使用的提示词
                    if unused_prompts:
                        # 随机选择一个
                        prompt = random.choice(unused_prompts)
                        selected_prompts.append(prompt)
                        selected_by_category[category].append(prompt)
                        remaining -= 1
                        
                        # 将该类别放回队列尾部
                        if unused_prompts[1:] and remaining > 0:  # 如果该类别还有更多提示词可供选择
                            category_cycle.append(category)
                
                # 如果还需要更多提示词且所有类别都已用完，则重新开始允许重复
                while remaining > 0:
                    # 从所有类别中选择一个随机提示词
                    all_available = []
                    for category in available_categories:
                        all_available.extend(self.prompts_by_category.get(category, []))
                    
                    # 如果没有可用的提示词，跳出循环
                    if not all_available:
                        break
                    
                    # 随机选择一个提示词
                    prompt = random.choice(all_available)
                    selected_prompts.append(prompt)
                    remaining -= 1
        
        # 打乱顺序，让不同类别的提示词混合在一起
        random.shuffle(selected_prompts)
        
        # 确保不超过请求数量
        return selected_prompts[:num_prompts]
    
    def extract_model_parameter(self,model_name: str) -> str | None:
            """
            从模型名称中提取参数量的数字，支持乘法表达式（如7x8B → 56）
            
            Args:
                model_name (str): 模型名称字符串，如 "Mixtral-8x7B"
                
            Returns:
                str| None: 解析后的数值，无匹配时返回 None
            """
            # 匹配所有可能的数字组合（含乘法表达式），例如 "7x8" 或 "3.5"
            pattern = r'(\d+\.?\d*(?:[xX]\d+\.?\d*)*)(?=[BM])'
            matches = re.findall(pattern, model_name, flags=re.IGNORECASE)
            
            if not matches:
                return None
            
            # 取最后一个匹配项（如 "model-3B-8x7B" 取 "8x7"）
            last_match = matches[-1]
            
            # 分割乘法因子并计算乘积
            factors = re.split(r'[xX]', last_match)
            try:
                result = 1
                for factor in factors:
                    num = float(factor) if '.' in factor else int(factor)
                    result *= num
                return f'{result}B' if result != 1 else None
            except (ValueError, TypeError):
                return None

    def _check_and_unload_model(self, target_model: str):
        """
        执行目标模型的卸载操作，确保性能测试的准确性
        使用多种策略强制卸载目标模型
        
        参数:
            target_model: 目标测试模型名称
            
        返回:
            bool: 卸载是否成功
        """
        
        def find_small_model(models: list, target_model: str) -> str | None:
            """
            优先选择参数量小或名称包含小模型关键词的非目标模型
            
            Args:
                models: 模型列表，每个元素是包含"name"字段的字典
                target_model: 需要排除的目标模型名称
            
            Returns:
                找到的小模型名称，若无合适模型则返回None
            """
            small_keywords = ["tiny", "phi", "small", "mini"]  # 小模型关键词（不区分大小写）
            candidate_models = []
            
            # 第一轮：收集所有非目标模型及其参数量
            for model_info in models:
                model_name = model_info.get("name", "")
                if not model_name or model_name == target_model:
                    continue
                
                # 提取参数量（支持乘法表达式如8x7B=56）
                param = extract_model_parameter(model_name)
                
                # 收集候选模型信息
                candidate_models.append({
                    "name": model_name,
                    "param": param if param is not None else float('inf'),  # 无参数量标记为极大值
                    "has_keyword": any(kw in model_name.lower() for kw in small_keywords),
                    "name_length": len(model_name)
                })
            
            if not candidate_models:
                return None
            
            # 排序优先级：
            # 1. 有参数量且值最小 > 无参数量但有关键词 > 其他
            # 2. 参数量相同时：优先有关键词的模型
            # 3. 都无参数量时：优先名称更短的模型
            candidate_models.sort(key=lambda x: (
                x["param"],                         # 参数量从小到大
                -x["has_keyword"],                   # 有关键词优先
                x["name_length"] if x["param"] == float('inf') else 0  # 无参数量时按名称长度
            ))
            
            best_model = candidate_models[0]["name"]
            print(f"找到最小候选模型: {best_model} (参数量: {candidate_models[0]['param']}B)")
            return best_model
        
        try:


            # # 策略1: 直接使用API卸载模型
            # print(f"策略1: 尝试直接卸载模型 {target_model}")
            # unload_success = self.client.unload_model(target_model)
            # if unload_success:
            #     print(f"成功通过API直接卸载模型 {target_model}")
            #     time.sleep(0.5)  # 等待半秒确保卸载完成
            #     return True
                
            # # 策略2: 多次尝试卸载
            # print(f"策略2: 尝试多次卸载模型 {target_model}")
            # for i in range(3):  # 最多尝试3次
            #     unload_success = self.client.unload_model(target_model)
            #     if unload_success:
            #         print(f"第{i+1}次尝试成功卸载模型 {target_model}")
            #         time.sleep(0.5)
            #         return True
            #     time.sleep(0.2)  # 短暂等待后重试
            
            # 策略3: 获取其他小模型并加载它们，迫使目标模型卸载
            print(f"策略3: 尝试通过加载其他模型间接卸载 {target_model}")
            models = self.client.get_models()
            
            if not models:
                print("没有可用模型，卸载策略无法继续")
                return False
            
            # 找到一个小的非目标测试模型
            other_model = find_small_model(models, target_model)
            if other_model and other_model != target_model:
                # print(f"尝试加载小模型 {other_model} 来间接卸载 {target_model}")
                
                # 发送简短请求，尝试加载其他模型
                test_result, response_time = self.test_model(other_model)
                # print(f"测试结果: {'成功' if test_result else '失败'}, 响应时间: {response_time:.4f}秒")
                
                if test_result:
                    # print(f"成功加载小模型 {other_model}，目标模型 {target_model} 可能已被卸载")                    
                    # 即使无法确认，也假定卸载成功
                    return True
                else:
                    print(f"加载小模型 {other_model} 失败，无法卸载 {target_model}")
            else:
                print(f"没有找到合适的小模型用于卸载 {target_model}")
            
            # 如果所有策略都失败，尝试最后一次卸载
            print(f"所有策略失败，最后一次尝试卸载 {target_model}")
            unload_success = self.client.unload_model(target_model)
            if unload_success:
                print(f"最后一次尝试成功卸载模型 {target_model}")
                return True
                
            # 如果所有策略都失败，提示用户
            print(f"无法卸载模型 {target_model}，请尝试重启Ollama服务")
            return False
            
        except Exception as e:
            print(f"尝试卸载模型过程中出错: {e}")
            return False

    def test_model(self, target_model):
        """
        测试模型是否加载成功，并测量响应时间
        
        Args:
            target_model (str): 要测试的模型名称
            
        Returns:
            tuple: (success: bool, response_time: float) 
                success - 是否成功收到预期响应
                response_time - 从发送请求到接收到第一个token的时间(秒)
        """
        system = "return pong if you receive ping."
        test_prompt = "ping"
        max_retries = 5
        retry_delay = 1
        timeout = 10  # 10秒超时
        
        # print(f"开始测试模型: {target_model}")
        for attempt in range(max_retries):
            try:
                # 使用非流式模式，更加可靠地获取响应
                start_time = time.time()
                
                # 创建一个自定义的生成器函数来处理超时
                def get_response_with_timeout():
                    response_gen = self.client.generate_completion(
                        target_model, test_prompt, system, stream=False
                    )
                    
                    # 设置超时
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"请求超时，超过{timeout}秒")
                    
                    # 在Windows上不使用signal
                    if platform.system() != "Windows":
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout)
                    
                    try:
                        # 获取第一个响应
                        response = next(response_gen)
                        return response
                    finally:
                        # 恢复旧的信号处理器
                        if platform.system() != "Windows":
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                
                # Windows上使用简单的线程超时
                if platform.system() == "Windows":
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(next, self.client.generate_completion(
                            target_model, test_prompt, system, stream=False
                        ))
                        response = future.result(timeout=timeout)
                else:
                    response = get_response_with_timeout()
                
                first_token_time = time.time() - start_time
                
                # 检查响应中是否有错误字段
                if isinstance(response, dict) and "error" in response:
                    error_msg = response.get("error", "未知错误")
                    print(f"模型响应包含错误: {error_msg}")
                    raise Exception(f"API返回错误: {error_msg}")
                
                # 提取响应内容
                response_content = ""
                if isinstance(response, dict):
                    # 尝试从不同的字段获取内容
                    if "response" in response:
                        response_content = response.get("response", "")
                    elif "content" in response:
                        response_content = response.get("content", "")
                    elif "completion" in response:
                        response_content = response.get("completion", "")
                    else:
                        # 找不到内容字段，转换整个响应为字符串
                        response_content = str(response)
                else:
                    response_content = str(response)
                
                # print(f"{target_model}模型响应: {response_content[:50]}... 时间: {first_token_time:.2f}秒")
                
                # 检查响应是否包含"pong"
                if "pong" in response_content.lower():
                    print(f"{target_model} 成功加载")
                    return (True, first_token_time)
                else:
                    print(f"{target_model} 模型响应不包含'pong': {response_content[:100]}")
                    
            except TimeoutError as e:
                print(f"请求超时: {e}")
            except StopIteration:
                print(f"模型未返回任何响应")
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
            
            # 尝试下一次
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
        
        # 所有尝试都失败
        print(f"测试模型 {target_model} 失败: 无法收到有效响应")
        return (False, 0.0)

    def custom_round(self,x):
            integer_part = int(x)
            decimal_part = x - integer_part
            return integer_part + (1 if decimal_part > 0.5 else 0)

    def generate_html_report(self, report: PerformanceReport) -> str:
        """
        生成HTML格式的测试报告
        
        参数:
            report: 性能测试报告对象
            
        返回:
            HTML格式的报告内容
        """
        # 获取系统信息
        system_info = report.system_info
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama模型性能测试报告 - {report.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ color: #3498db; margin-top: 20px; }}
        h3 {{ color: #2c3e50; margin-top: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        .result-excellent {{ color: #27ae60; }}
        .result-good {{ color: #2980b9; }}
        .result-fair {{ color: #f39c12; }}
        .result-poor {{ color: #c0392b; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .chart {{ margin: 30px 0; }}
        .prompt-response {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #3498db; }}
        .prompt {{ color: #2c3e50; font-weight: bold; }}
        .response {{ color: #7f8c8d; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f5f5f5; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Ollama模型性能测试报告</h1>
        
        <h2>测试平台信息</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>操作系统</td><td>{getattr(system_info, "os_name", "未知")} {getattr(system_info, "os_version", "未知")}</td></tr>
            <tr><td>CPU</td><td>{getattr(system_info, "cpu_brand", "未知")} ({getattr(system_info, "cpu_cores", 0)} 核)</td></tr>
            <tr><td>内存</td><td>{self.custom_round(getattr(system_info, "memory_total", 0)):.2f} GB</td></tr>"""
        
        # 添加GPU信息（如果有）
        if hasattr(system_info, "has_gpu") and system_info.has_gpu:
            if hasattr(system_info, "gpu_info"):
                gpu_info = system_info.gpu_info
                for gpu in gpu_info:
                    html += f"""
            <tr><td>GPU {gpu.get("index", 0)}</td><td>{gpu.get("name", "未知")}</td></tr>
            <tr><td>显存 {gpu.get("index", 0)}</td><td>{self.custom_round(gpu.get("memory_total", 0)):.2f} GB</td></tr>"""
        else:
            html += """
            <tr><td>GPU</td><td>无法识别</td></tr>
            <tr><td>显存</td><td>无法识别</td></tr>"""
        
        html += f"""
            <tr><td>Python版本</td><td>{getattr(system_info, "python_version", "未知")}</td></tr>
            <tr><td>Ollama版本</td><td>{getattr(system_info, "ollama_version", "未知")}</td></tr>
        </table>
        
        <div class="summary">
            <h2>测试概述</h2>
            <p><strong>模型名称:</strong> {report.model_name}</p>
            <p><strong>模型参数量:</strong> {report.test_params.get("model_params", "未填写")}</p>
            <p><strong>编码精度:</strong> {report.test_params.get("model_precision", "未填写")}</p>"""
        
        # 添加模型温度和上下文窗口参数
        if "model_temperature" in report.test_params:
            html += f"""
            <p><strong>模型温度:</strong> {report.test_params.get("model_temperature", "未填写")}</p>"""
            
        if "context_window" in report.test_params:
            html += f"""
            <p><strong>上下文窗口:</strong> {report.test_params.get("context_window", "未填写")} tokens</p>"""
            
        html += f"""
            <p><strong>测试日期:</strong> {report.test_params.get("test_date", "未知")}</p>
            <p><strong>性能评级:</strong> <span class="{
                "result-excellent" if report.performance_rating == "极佳" else
                "result-good" if report.performance_rating == "优秀" else
                "result-fair" if report.performance_rating == "良好" else
                "result-poor"
            }">{report.performance_rating}</span></p>
            <p><strong>平均首个token延迟:</strong> {report.avg_first_token_latency:.2f} ms</p>"""
        
        # 添加模型载入时间（如果有），并转换为秒
        if "model_load_time" in report.test_params:
            model_load_time_s = report.test_params.get("model_load_time", 0)
            html += f"""
            <p><strong>模型载入时间:</strong> {model_load_time_s:.2f} s</p>"""
            
        html += f"""
            <p><strong>平均生成速度:</strong> {int(report.avg_tokens_per_second)} tokens/s</p>
            <p><strong>测试轮次:</strong> {report.test_params.get("rounds", 0)}</p>
            <p><strong>并发用户数:</strong> {report.test_params.get("concurrent_users", 1)}</p>"""
            
        # 添加并发信息（如果有）
        if "parallel_info" in report.test_params:
            html += f"""
            <p><strong>并发配置:</strong> {report.test_params.get("parallel_info")}</p>"""
                
        html += """
        </div>
        
        <h2>性能评级标准</h2>
        <table>
            <tr><th>评级</th><th>生成速度</th></tr>
            <tr><td>极佳</td><td>&gt; 60 tokens/s</td></tr>
            <tr><td>优秀</td><td>&gt; 30 tokens/s</td></tr>
            <tr><td>良好</td><td>&gt; 10 tokens/s</td></tr>
            <tr><td>较差</td><td>&lt;= 10 tokens/s</td></tr>
        </table>
        
        <h2>详细测试结果</h2>
        <table>
            <tr>
                <th>轮次</th>
                <th>提示词</th>
                <th>首个token延迟 (ms)</th>
                <th>生成速度 (tokens/s)</th>
                <th>总token数</th>
                <th>总时间 (ms)</th>
            </tr>"""
        
        # 添加详细测试结果
        for i, result in enumerate(report.results):
            # 获取显示的轮次编号
            display_round = i
            
            if result.error:
                html += f"""
            <tr>
                <td>{display_round}</td>
                <td>{result.prompt}</td>
                <td colspan="4">错误: {result.error}</td>
            </tr>"""
            else:
                # 为第0轮添加说明
                first_token_latency_display = f"{result.first_token_latency:.2f}"
                if display_round == 0:
                    first_token_latency_display += " (包含模型加载时间)"
                
                html += f"""
            <tr>
                <td>{display_round}</td>
                <td>{result.prompt}</td>
                <td>{first_token_latency_display}</td>
                <td>{result.tokens_per_second:.2f}</td>
                <td>{result.total_tokens}</td>
                <td>{result.total_time:.2f}</td>
            </tr>"""
        
        html += """
        </table>
        
        <h2>提示词和回复内容</h2>"""
        
        # 添加提示词和模型回复
        for i, result in enumerate(report.results):
            display_round = i
            if not result.error:
                html += f"""
        <div class="prompt-response">
            <h3>测试 {display_round}: {result.prompt}</h3>
            <div class="prompt">
                <strong>提示词:</strong>
                <pre>{result.prompt_content}</pre>
            </div>
            <div class="response">
                <strong>模型回复:</strong>
                <pre>{result.response_text}</pre>
            </div>
        </div>"""
        
        html += f"""
        <div style="margin-top: 40px; text-align: center; color: #777; font-size: 0.8em;">
            <p>由Ollama模型监控测试工具生成 | 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html

    def generate_text_report(self, report: PerformanceReport) -> str:
        """
        生成纯文本格式的测试报告
        
        参数:
            report: 性能测试报告对象
            
        返回:
            文本格式的报告内容
        """
        # 获取系统信息
        system_info = report.system_info
        
        text = """Ollama模型性能测试报告
==========================

测试平台信息
--------------------------
操作系统: {os_name} {os_version}
CPU: {cpu_brand} ({cpu_cores} 核)
内存: {memory_total:.2f} GB""".format(
            os_name=getattr(system_info, "os_name", "未知"),
            os_version=getattr(system_info, "os_version", "未知"),
            cpu_brand=getattr(system_info, "cpu_brand", "未知"),
            cpu_cores=getattr(system_info, "cpu_cores", 0),
            memory_total=self.custom_round(getattr(system_info, "memory_total", 0))
        )
        
        # 添加GPU信息（如果有）
        if hasattr(system_info, "has_gpu") and system_info.has_gpu:
            gpu_info = system_info.gpu_info if hasattr(system_info, "gpu_info") else []
            for gpu in gpu_info:
                text += f"""
GPU {gpu.get("index", 0)}: {gpu.get("name", "未知")}
显存 {gpu.get("index", 0)}: {self.custom_round(gpu.get("memory_total", 0)):.2f} GB"""
                
        text += f"""
Python版本: {getattr(system_info, "python_version", "未知")}
Ollama版本: {getattr(system_info, "ollama_version", "未知")}

测试概述
--------------------------
模型名称: {report.model_name}
模型参数量: {report.test_params.get("model_params", "未填写")}
编码精度: {report.test_params.get("model_precision", "未填写")}"""

        # 添加模型温度和上下文窗口参数
        if "model_temperature" in report.test_params:
            text += f"""
模型温度: {report.test_params.get("model_temperature", "未填写")}"""
            
        if "context_window" in report.test_params:
            text += f"""
上下文窗口: {report.test_params.get("context_window", "未填写")} tokens"""

        text += f"""
测试日期: {report.test_params.get("test_date", "未知")}
性能评级: {report.performance_rating}
平均首个token延迟: {report.avg_first_token_latency:.2f} ms"""

        # 添加模型载入时间（如果有），并转换为秒
        if "model_load_time" in report.test_params:
            model_load_time_s = report.test_params.get("model_load_time", 0)
            text += f"""
模型载入时间: {model_load_time_s:.2f} s"""
            
        text += f"""
平均生成速度: {int(report.avg_tokens_per_second)} tokens/s
测试轮次: {report.test_params.get("rounds", 0)}
并发用户数: {report.test_params.get("concurrent_users", 1)}"""
            
        # 添加并发信息（如果有）
        if "parallel_info" in report.test_params:
            text += f"""
并发配置: {report.test_params.get("parallel_info")}"""
                
        text += """

性能评级标准
--------------------------
极佳: 生成速度 > 60 tokens/s
优秀: 生成速度 > 30 tokens/s
良好: 生成速度 > 10 tokens/s
较差: 生成速度 <= 10 tokens/s

详细测试结果
--------------------------"""
        
        # 添加详细测试结果
        for i, result in enumerate(report.results):
            # 获取显示的轮次编号
            display_round = i
            
            if result.error:
                text += f"""
测试 {display_round}:
  提示词: {result.prompt}
  错误: {result.error}"""
            else:
                # 为第0轮添加说明
                first_token_latency_display = f"{result.first_token_latency:.2f} ms"
                if display_round == 0:
                    first_token_latency_display += " (包含模型加载时间)"
                
                text += f"""
测试 {display_round}:
  提示词: {result.prompt}
  首个token延迟: {first_token_latency_display}
  生成速度: {result.tokens_per_second:.2f} tokens/s
  总token数: {result.total_tokens}
  总时间: {result.total_time:.2f} ms"""
        
        text += """

提示词和回复内容
--------------------------"""

        # 添加提示词和模型回复
        for i, result in enumerate(report.results):
            display_round = i
            if not result.error:
                text += f"""
测试 {display_round}: {result.prompt}
提示词:
{result.prompt_content}

模型回复:
{result.response_text}
--------------------------"""
        
        text += f"""

由Ollama模型监控测试工具生成 | 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
        
        return text 

    def test_case(self, model: str, prompt: str, prompt_type: str = "未分类", system: str = "", client = None, timeout: int = 600) -> TestResult:
        """
        执行单次测试任务
        
        参数:
            model: 模型名称
            prompt: 提示词内容
            prompt_type: 提示词类型/分类
            system: 系统提示词
            client: 指定的客户端实例
            timeout: 超时时间（秒）
            
        返回:
            TestResult: 测试结果对象
        """
        try:
            # 如果未指定客户端，使用当前的客户端
            if client is None:
                client = self.client
                
            # 设置超时控制
            import time
            import threading
            
            # 开始测试
            start_time = time.time()
            first_token_time = None
            total_tokens = 0
            response_text = ""
            token_count_from_api = 0  # API返回的token计数
            eval_duration_from_api = 0  # API返回的评估时间
            
            # 创建取消事件标志
            cancel_event = threading.Event()
            
            # 设置超时定时器
            timer = threading.Timer(timeout, cancel_event.set)
            timer.daemon = True  # 设为守护线程，程序结束时自动终止
            timer.start()
            
            try:
                # 发送请求并计算首个token延迟
                chunks = []  # 收集所有数据块
                # 记录请求发送时间
                request_sent_time = time.time()
                for i, chunk in enumerate(client.generate_completion(model, prompt, system, cancel_event=cancel_event)):
                    # 检查是否发生超时
                    if cancel_event.is_set():
                        raise TimeoutError(f"测试执行超时（超过{timeout}秒）")
                        
                    # 保存块数据用于分析
                    chunks.append(chunk)
                    
                    # 记录第一个token时间 - 确保只记录一次
                    if i == 0 and first_token_time is None:
                        if "response" in chunk:
                            first_token_time = time.time()
                            # 调试信息 - 真实的首token延迟
                            real_latency = (first_token_time - request_sent_time) * 1000
                        
                    # 累加响应文本
                    if "response" in chunk:
                        response = chunk.get("response", "")
                        response_text += response
                        total_tokens += 1
                    
                    # 从API提取更准确的指标（最后一个块通常包含总结信息）
                    if "eval_count" in chunk:
                        token_count_from_api = chunk.get("eval_count", 0)
                    if "eval_duration" in chunk:
                        eval_duration_from_api = chunk.get("eval_duration", 0)
            finally:
                # 取消超时定时器
                timer.cancel()
                    
            end_time = time.time()
            
            # 使用API提供的token计数（如果有）
            if token_count_from_api > 0:
                total_tokens = token_count_from_api
            
            # 计算指标，优先使用请求发送时间计算首token延迟
            total_time = (end_time - start_time) * 1000  # 毫秒
            
            # 首token延迟应该是从请求发送到收到第一个token的时间
            # 而不是包括测试准备等时间
            if first_token_time:
                first_token_latency = (first_token_time - request_sent_time) * 1000
            else:
                first_token_latency = 0
            
            # 计算生成时间和速度
            if eval_duration_from_api > 0:
                # 如果API提供了评估时间，使用API时间
                generation_time = eval_duration_from_api / 1000000000  # 纳秒转秒
            else:
                # 否则使用我们自己测量的时间 - 从收到第一个token到结束
                generation_time = (end_time - first_token_time) if first_token_time else 0
            
            # 计算token生成速度 - 排除首token延迟时间，只计算生成阶段的速度
            if total_tokens > 1 and generation_time > 0:  # 至少有2个token才能计算生成速度
                # 减去第一个token，因为已经在首token延迟中计算过
                actual_generation_tokens = total_tokens - 1
                tokens_per_second = actual_generation_tokens / generation_time
            elif total_tokens == 1 and generation_time > 0:
                # 只有一个token的情况
                tokens_per_second = 1 / generation_time
            else:
                tokens_per_second = 0
            
            # 日志记录调试信息
            print(f"测试详情 - 类型:{prompt_type}, 首Token:{first_token_latency:.2f}ms, 速度:{tokens_per_second:.2f}tokens/s 总耗时:{total_time:.2f}ms")
            
            # 创建显示名和截断提示词
            display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
            
            return TestResult(
                prompt=display_prompt,
                prompt_content=prompt,
                prompt_type=prompt_type,
                response_text=response_text,
                first_token_latency=first_token_latency,
                tokens_per_second=tokens_per_second,
                total_tokens=total_tokens,
                total_time=total_time
            )
            
        except Exception as e:
            error_message = f"测试错误: {str(e)}"
            print(error_message)
            # 创建含错误信息的测试结果
            truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
            return TestResult(
                prompt=truncated_prompt,
                prompt_content=prompt,
                prompt_type=prompt_type,
                response_text=error_message,  # 将错误信息放入响应文本
                first_token_latency=0,
                tokens_per_second=0,
                total_tokens=0,
                total_time=0,
                error=str(e)
            ) 
        


if __name__ == "__main__":
    tester = PerformanceTester()
    tester.extract_model_parameter("qwq:32b")