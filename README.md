# AgentMemoryBench: LLM 智能体终身学习基准测试

用于评估 LLM 智能体在终身学习场景中记忆机制的研究基准。

## 项目简介

本项目评估 LLM 智能体如何学习和保留两种类型的记忆：
- **系统记忆**：任务工作流和执行步骤
- **个人记忆**：用户偏好和习惯

支持 4 种学习范式：在线学习、离线学习、迁移学习和重放学习。

## 项目结构

```
.
├── configs/                    # 配置文件
│   ├── assignment/            # 实验配置
│   │   └── default.yaml       # 主实验配置文件
│   ├── tasks/                 # 任务特定配置（5+1个任务）
│   └── llmapi/                # LLM API 设置
│       ├── api.yaml           # 配置大模型api网站以及密钥
│       ├── agent.yaml         # 配置要在api网站中使用的大模型名称
│       ├── evaluate_agent.yaml# 配置locomo LLM as judge的大模型api网站以及密钥
│       └── evaluate_api.yaml  # 配置locomo LLM as judge要在api网站中使用的大模型名称
│
├── data/                       # 6种任务的数据
│   ├── dbbench/               # 数据库操作（SQL）
│   ├── os_interaction/        # 操作系统命令（Shell）
│   ├── knowledgegraph/        # 知识图谱查询（SPARQL）
│   ├── alfworld/              # 具身AI任务
│   ├── webshop/               # 电商任务
│   └── locomo/                # 长对话记忆
│
├── memory/                     # 已经实现的5种记忆机制
│   ├── zero_shot/             # 基线（无记忆）
│   ├── streamICL/             # 基于RAG的检索
│   ├── awmPro/                # 系统记忆（工作流）
│   ├── mem0/                  # 个人记忆（偏好）
│   └── MEMs/                  # 多记忆系统（提出方法）
│
├── execution/                  # 4种执行引擎
│   ├── single_agent/          # 单智能体执行
│   └── multi_agent_*/         # 多智能体策略
│       ├── voting/            # 多数投票
│       ├── round_robin/       # 轮询执行
│       └── bandit/            # 多臂老虎机
│
├── src/                        # 核心实现
│   ├── runner/                # 主入口
│   │   └── main.py            # 实验运行器
│   ├── client/                # 客户端调度
│   ├── server/                # 后端任务服务器（Docker）
│   └── utils/                 # 分析工具
│
├── extra/                      # Docker 编排
│   └── docker-compose.yml     # 服务定义
│
├── outputs/                    # 实验结果
│   └── [时间戳]/              # 按实验分组
│       └── [任务名]/          # 按任务分组
│           └── [索引].json    # 单个样本结果
│
└── requirements.txt            # Python 依赖
```

## Quick Start

### 1. Data installation

从网站 `https://www.dropbox.com/scl/fi/ai9pm3wgs8gt09gwdav81/virtuoso_db.zip` 下载 virtuoso_db.zip 并解压，然后根据 virtuoso_db文件夹 的路径，将 `xxx/virtuoso_db:/database` 配置至 `extra\docker-compose.yml` 第111行代码

### 2. Requirements installation

```bash
# 创建虚拟环境
conda create -n memoryBench python=3.9

# 激活虚拟环境
conda activate memoryBench

# Install dependencies
pip install -r requirements.txt

# Start backend services (Docker required)
cd extra

docker-compose build local-os-default

docker-compose build local-os-packages

docker-compose build local-os-ubuntu

docker-compose up
```

### 3. Configuration API

！！！最好使用硅基流动的api网站，这样就不必改 `configs\llmapi\agent.yaml`、`configs\llmapi\evaluate_agent.yaml` 文件了！！！

Edit `configs/llmapi/api.yaml` to set your LLM API credentials:

```yaml
base_url: "https://api.siliconflow.cn/v1"
headers:
  Content-Type: application/json
  Authorization: "Bearer YOUR_API_KEY"

...
```

Edit `configs\llmapi\evaluate_api.yaml` to set your LLM API credentials:

```yaml
base_url: "https://api.siliconflow.cn/v1"
headers:
  Content-Type: application/json
  Authorization: "Bearer YOUR_API_KEY"

...
```

### 4. How to create a new memory mechanism?

创建新的记忆机制需要两个步骤：

#### 步骤 1：在 memory 文件夹中实现记忆机制类

在 `memory/` 目录下创建新文件夹（例如 `my_memory/`），并实现继承自 `MemoryMechanism` 的类：

```python
# memory/my_memory/my_memory.py
from __future__ import annotations
from typing import List, Dict, Any
import yaml
from ..base import MemoryMechanism

class MyMemory(MemoryMechanism):
    """你的记忆机制实现"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 初始化你的记忆存储

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在调用 LLM 之前，使用记忆增强 messages

        Args:
            task: 任务名称（如 "dbbench-std", "os-std"）
            messages: 原始消息列表 [{"role": "user", "content": "..."}, ...]

        Returns:
            增强后的消息列表（例如插入历史经验、few-shot 示例等）
        """
        # 从记忆中检索相关经验
        # 将经验注入到 messages 中
        return messages  # 返回增强后的 messages

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """
        在样本执行结束后，更新记忆存储

        Args:
            task: 任务名称
            history: 完整的对话历史（包括 user/assistant/system 消息）
            result: 执行结果（包含 reward、status 等信息）
        """
        # 根据 history 和 result 更新你的记忆存储
        pass

def load_my_memory_from_yaml(config_path: str) -> MyMemory:
    """从 YAML 配置文件加载记忆机制"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return MyMemory(config)
```

同时创建配置文件 `memory/my_memory/my_memory.yaml`：

```yaml
# memory/my_memory/my_memory.yaml
name: my_memory
description: "我的记忆机制描述"

# 你的配置参数
param1: value1
param2: value2
```

#### 步骤 2：在 src/runner/builders.py 中注册

在 `src/runner/builders.py` 的 `build_memory_from_config()` 函数中添加新的分支：

```python
def build_memory_from_config(cfg: ExperimentConfig):
    mem_cfg = cfg.memory_mechanism or {}
    name = mem_cfg.get("name", "zero_shot")
    config_path = mem_cfg.get("config_path")

    # ... 现有的 if/elif 分支 ...

    elif name == "my_memory":  # 添加你的记忆机制
        if not config_path:
            config_path = ROOT_DIR / "memory" / "my_memory" / "my_memory.yaml"
        else:
            config_path = ROOT_DIR / config_path
        from memory.my_memory.my_memory import load_my_memory_from_yaml
        return load_my_memory_from_yaml(str(config_path))

    else:
        raise NotImplementedError(f"Memory mechanism '{name}' not implemented yet")
```

#### 使用新的记忆机制

在 `configs/assignment/default.yaml` 中配置：

```yaml
memory_mechanism:
  name: my_memory
  config_path: memory/my_memory/my_memory.yaml
```

### 5. Configuration Experiment

！！！实验主要在 `configs\assignment\default.yaml` 中配置，初步计划只需要师兄帮忙跑dualRAG在online模式下，seed=1时的DB、OS、KG、ALF、locomo-0、...、locomo-9单任务，注意不用跑WebShop任务！！！

```yaml
# Lifelong Learning Benchmark Configuration
# 配置要测试的任务、记忆机制、执行方法和实验参数

# ===== 任务配置 =====
# 指定要测试的任务列表（5个system memory任务+2个user memory任务，共7个任务）
tasks:

  # 一次选中一个即可！！！

  # system memory任务
  # - name: dbbench-std
  #   config_path: configs/tasks/dbbench.yaml
  # - name: os-std
  #   config_path: configs/tasks/os.yaml
  # - name: kg-std
  #   config_path: configs/tasks/kg.yaml
  # - name: alfworld-std
  #   config_path: configs/tasks/alfworld.yaml
  # - name: webshop-std
  #   config_path: configs/tasks/webshop.yaml

  # user memory任务
  # - name: locomo-0
  #   config_path: configs/tasks/locomo-0.yaml
  # - name: locomo-1
  #   config_path: configs/tasks/locomo-1.yaml
  # - name: locomo-2
  #   config_path: configs/tasks/locomo-2.yaml
  # - name: locomo-3
  #   config_path: configs/tasks/locomo-3.yaml
  # - name: locomo-4
  #   config_path: configs/tasks/locomo-4.yaml
  # - name: locomo-5
  #   config_path: configs/tasks/locomo-5.yaml
  # - name: locomo-6
  #   config_path: configs/tasks/locomo-6.yaml
  # - name: locomo-7
  #   config_path: configs/tasks/locomo-7.yaml
  # - name: locomo-8
  #   config_path: configs/tasks/locomo-8.yaml
  # - name: locomo-9
  #   config_path: configs/tasks/locomo-9.yaml

# ===== 记忆机制配置 =====
# 从 memory 文件夹中选择记忆机制
memory_mechanism:
  name: which_one_would_you_like  # 可选: zero_shot, streamICL, mem0 ,awmPro , mems, awmPro
  config_path: memory/streamICL/streamICL.yaml

# ===== 执行方法配置 =====
# 从 execution 文件夹中选择执行方法
execution_method:
  name: single_agent  # 当前版本仅支持 single_agent
  config_path: execution/single_agent/single_agent.yaml

# ===== 实验参数 =====
experiment:
  # 训练模式: online (在线学习) 或 offline (离线学习) 或 replay (重放学习) 或 transfer (迁移学习) 
  training_mode: online  # online | offline | replay | transfer
  
  ...
  
  # 数据打乱: 是否打乱任务顺序，可以设置随机种子
  shuffle:
    enabled: True  # True | False
    seed: 1  # 整数，如果 enabled 为 true 时使用
  

# ===== 输出配置 =====
output: "outputs/{TIMESTAMP}"
```

### 6. Run Experiments

```bash
# Run with default configuration
python src/runner/main.py
```

### 7. View Results

分析实验结果的脚本共有两个，分别是 `src\utils\analyze_results_for_system_memory.py` 与 `src\utils\analyze_results_for_personal_memory.py`:
- 对于DB、OS、KG、ALF、WebShop任务，使用前者，运行代码 `python -m src.utils.analyze_results_for_system_memory outputs\2026-02-01_20-27-15\os-std`
- 对于Locomo任务，使用后者，运行代码 `python -m src.utils.analyze_results_for_system_memory outputs\2026-02-01_21-22-31\locomo-0`
