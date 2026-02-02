# MemoryBench: It is Imperative for Us to Reconsider Agent Memory

## 1. 引言

### 简单引入
LLM-based Agents, which are LLMs equipped with capabilities such as reasoning, planning, perception, memory, and tool-use [1, 2, 3, 6],  have rapidly evolved from simple chatbots into capable systems that can write code [29], control browsers [30], and perform advanced question answering [28]. 在赋能智能体的各种组件中，tool-use和reasoning等能力已逐渐内化到 LLM 参数中 ，而planning和perception则通过实时的分解目标和渐近元数据披露等方式增强问题解决能力 [3]。然而，这些组件都有一个共同的局限性：它们并未改变智能体的静态属性 [10] ——智能体无法从历史交互或经验中学习 [11]，也无法适配用户的长期偏好 [13]。相比之下，记忆（memory）组件的出现，彻底将智能体从无记忆、无法从过往成败中吸收经验的静态助手，转变为可随环境动态进化的智能伙伴[3, 4, 5]。

Memory 使智能体能够从过去的交互中积累知识、记住用户偏好，并随着时间的推移持续提升性能。这种动态学习能力将记忆与其他智能体组件区分开来，代表了向真正自适应 AI 系统的范式转变。记忆主要分为两类核心类型 ——system memory 与 personal memory，这一分类从记忆的服务对象与内容来源出发，构成了智能体动态进化的基础 [3, 5]。其中，system memory 是智能体在执行复杂任务（如代码编写、浏览器控制）过程中生成的中间输出与执行流程记录，通过存储任务推理轨迹、工具使用经验等，持续强化系统的问题解决与推理能力；personal memory 则聚焦于人类交互过程中的输入与反馈，专门存储用户偏好、对话历史、个人事件等个性化信息，为智能体提供贴合用户需求的定制化响应 [3]。

### 现有研究局限性
尽管记忆的重要性不言而喻，现有研究仍面临显著局限性：
基准层面的问题：
- 缺乏在线评估能力：现有基准大多将记忆视为静态组件，仅支持离线评估，无法观察学习过程。它们通常在单一时间点测试记忆系统性能（如 AgentBench [27] 、 Locomo[17]、KnowMeBench [23]），这就像只通过期末考试成绩评估学生，而不是观察整个学期的学习曲线，但实际上agent memory就像人类记忆————不是一次性写入，而是不断重构、强化、遗忘的过程
- 未全面考虑系统记忆与个人记忆：现有 benchmark 假设任务边界清晰，所以他们要么专注于系统记忆（如 MemoryBench [28]、Evo-memory [10]、StreamBench [6]、LifelongAgentBench [7]），要么专注于个人记忆（如 HaluMem[22]、 KnowMeBench [23]、 Locomo [16]、LongMemEval [17]、MDR [15]），但未能同时全面覆盖两种类型的记忆任务。
- 缺少迁移学习和重放场景的评估维度：从Offline模式到Online模式，实现了从“只看最终成绩，看不到学习曲线”到“记录每次进步，观察知识积累过程”的进步。而迁移学习和重放，这些维度对于理解智能体在学习过程中的知识泛化和知识保留的能力至关重要，但现有基准偏偏缺乏这些评估维度。
- 缺乏统一、易复现的框架：不同基准使用不同的实现方式，使得方法对比和结果复现变得困难——如AWM [11]需要先对两个Benchmark——WebArena [30]和Mind2Web [31]进行online模式的改造，然后才能测评提出的记忆方案；而像MemGen [8] 这样优秀的通用记忆解决方案，却缺乏合适的benchmark进行memory实时表现、知识保留、知识泛化等能力的测评.

方法论层面的问题：
- 上下文工程与记忆机制：部分工作通过上下文工程方法（如 streamICL [6] 、experience replay [7]）激活预训练中的语义先验和格式适配能力实现性能提升 [25, 26]，但它们缺乏对案例的提炼和抽象过程：只是存储和检索原始对话历史以实现简单的记忆功能，却没有对历史轨迹和交互过程进行总结、提取关键信息和抽象规律。真正的记忆机制应该像人类记忆一样，能够从多个案例中提炼出可复用的经验、规则和模式，而非仅仅依赖原始案例的相似性匹配。
- 随着动态场景需求的凸显，部分RAG相关工作也开始尝试融入动态特性[32]。值得注意的是，当RAG的检索系统逐渐向动态化演进时，其与智能体记忆（memory）的功能界限正变得日益模糊[4]，典型代表如DualRAG [33]、DynamicRAG [34]等。但从方法论层面来看，这类方法并未明确界定动态检索与记忆机制的融合逻辑，也未形成兼顾记忆的提炼、抽象与长期保留特性，以及动态检索高效性的统一框架，未能真正实现两者的深度协同，难以有效支撑智能体记忆能力的系统性提升。
- 单一记忆类型的设计局限：现有记忆方法往往专注于单一类型的记忆（系统记忆或个人记忆），缺乏能够同时处理两种记忆类型的统一框架。专注于系统记忆的方法（如 [10, 11, 12, 18, 19, 20, 21, 24]）强调从任务案例中学习经验，但未能解决长期人机对话中的因果推理、跨会话推理、时间推理和用户偏好记忆等能力 [13, 14, 15]；而专注于个人记忆的方法则缺乏对任务执行经验的积累和复用。但现实场景中往往不存在这种清晰的任务边界。这种设计局限导致在混合场景下，专门优化的记忆方法（如 AWM [11]、mem0 [13]）的表现甚至不如简单朴素实现的记忆方法（如 streamICL [6] 等），凸显了单一记忆设计在真实应用场景中的根本缺陷

### 为啥解决不了

现有工作未能解决上述问题，根本原因在于benchmark的固有缺陷。现有基准在场景覆盖与记忆类型支持上存在显著局限性 —— 仅支持离线评估、仅覆盖单一记忆类型，这使得方法设计阶段无需考量混合场景的复杂性，自然难以适配真实应用需求。

具体来看，MemoryBench [28] 虽涵盖 online、offline 等多种运行模式，但所有数据集均局限于纯文本任务，未涉及编码（coding）、具身交互、网页浏览（web browse）等工具调用相关场景，且仅聚焦 system memory 能力测评，缺乏对 personal memory 的考量；LifelongAgentBench 仅纳入 system memory 与 online 模式，既未覆盖迁移学习、重放学习两类关键场景，可扩展性也较差；Evo-Memory [10] 尽管支持多种测评模式，但同样仅考察 agent 的 system memory 能力，无法在 “模拟现实”“任务边界模糊”“流式输入” 等近似真实的复杂场景中，评测记忆机制的综合适配能力。

这种局限性还导致专注于系统记忆的方法在设计时假设任务类型明确 [11, 12] ，无需考虑用户偏好；专注于个人记忆的方法假设交互场景单一，无需考虑任务执行经验 [13, 14] 。这种 “设计 - 评估” 的闭环导致方法无法应对真实场景中的复杂性 —— 智能体在实际应用中需要同时处理系统任务和个人交互，但现有方法缺乏有效的机制来协调系统记忆和个人记忆。而真正贴合真实场景的解决路径，本质上分为两类：要么构建能够同时容纳系统任务经验与个人交互偏好的混合记忆架构，通过明确的优先级调度、冲突消解机制协调两类记忆的调用；要么像 MemGen [8] 那样，提取脱离具体任务与交互场景的通用记忆表示，使其能够跨场景适配系统任务执行与个人化交互需求。

但现有评估范式的局限性使得这两类路径均难以落地验证：即便部分方法（如 MemGen [8]）已尝试探索通用记忆表示，也因缺乏能够覆盖 “任务 - 交互混合”“场景动态变化” 的 memory benchmark，无法充分验证其在近似真实的复杂场景中的有效性；而针对多记忆类型协调的方法，更是因现有基准仅支持单一记忆类型测评，难以开展针对性的设计与优化，最终导致两类潜在解决方案均无法形成 “设计 - 评估 - 迭代” 的完整闭环。

### 我们的贡献
为了解决这些局限性，我们提出了 MemoryBench，一个支持系统记忆和个人记忆评估，以及在线、离线、迁移以及重放学习模式的基准，以及MEMs，受人类前额叶皮层协调大脑中不同记忆系统启发的multi-memorys system——MEMs。我们的核心贡献如下：

1. 统一基准 MemoryBench：
我们设计了一个统一的评估框架，首次同时支持系统记忆和个人记忆的评估，以及在线、离线、重放和迁移学习四种模式。MemoryBench 包含六个数据集，涵盖多种环境类型，能够全面评估智能体在不同场景下的记忆能力。这六个数据集按照环境类型可分为四类：

- **Code-grounded Environments（代码基础环境）**：包含三个数据集，要求智能体通过代码接口与环境交互。**Database (DB) [28]** 评估智能体通过 SQL 操作数据库的能力；**Operating System (OS) [28]** 评估智能体在终端环境中执行 Shell 命令的能力；**Knowledge Graph (KG) [44]** 评估智能体在大型知识图谱中通过 SPARQL 查询进行推理的能力。这些任务要求智能体从历史成功案例中学习代码模式和查询策略，属于系统记忆任务。

- **Embody-grounded Environments（具身基础环境）**：包含 **ALFWorld [32]** 数据集，评估智能体在具身家庭环境中的常识推理和任务执行能力。智能体需要完成诸如"将平底锅放在餐桌上"等任务，从历史任务执行中学习操作模式和空间推理策略，属于系统记忆任务。

- **Web-grounded Environments（网络基础环境）**：包含 **Web Shopping [45]** 数据集，评估智能体在模拟电商环境中进行在线购物的能力。智能体需要从历史购物经验中学习商品搜索和选择模式，属于系统记忆任务。

- **Dialogue-grounded Environments（对话基础环境）**：包含 **LoCoMo [16]** 数据集，评估智能体在长期对话中记住用户偏好和对话历史的能力。该数据集包含 10 个扩展对话，每个对话平均包含约 600 轮对话和 26000 个 token，要求智能体在跨会话的对话中记住用户偏好、关键事实和对话历史，属于个人记忆任务。

这一设计打破了现有基准的局限性：
- **四种评估模式**：与现有基准仅在单一时间点测试（offline）不同，我们的基准支持在线学习（online）模式，能够观察智能体在整个学习过程中的表现变化，记录知识积累、重构和遗忘的动态过程。这使得我们能够评估记忆系统如何随时间演进，而非仅仅关注最终性能。从 Offline 模式到 Online 模式，实现了从"只看最终成绩，看不到学习曲线"到"记录每次进步，观察知识积累过程"的进步。此外，我们还支持重放（replay）和迁移（transfer）学习模式：重放模式通过周期性测试来评估智能体对已学知识的保留能力，迁移模式评估智能体将在一类任务上学到的知识应用到另一类任务的能力。这些评估维度对于理解智能体如何适应和保留知识至关重要，但现有基准恰恰缺乏这些维度。
- **混合记忆任务**：我们同时集成了系统记忆任务（如 SQL 编写、操作系统代码开发）和个人记忆任务（如长期对话中的用户偏好记忆），打破了现有基准只覆盖单一记忆类型的局限。更重要的是，我们支持跨任务学习（cross-task）和任务随机排列（shuffle），能够模拟真实场景中任务边界不清晰的情况。具体而言，当采用 'multi-task + online + cross-task + shuffle' 的组合配置时，我们的基准以长程对话为载体，任务类型在交互过程中随机切换，涵盖 SQL 编写、操作系统代码开发、日常自然语言交互等异构任务。这种设计精准模拟了实际应用中智能体面临的"任务分布离散化、交互过程持续化"的复杂工作场景，使得在基准上表现良好的方法能够在真实场景中同样有效。

1. Multi-Memorys methodology——MEMs：
针对现有方法无法在混合场景下有效协调系统记忆和个人记忆的问题，我们提出了多记忆集成（multi-memorys）方法。该方法的核心创新在于引入了一个轻量的"元认知"层——触发模型（trigger model），实现了从"多个独立存储系统"到"统一认知系统"的范式转变：
- 统一认知框架：与现有方法将系统记忆和个人记忆视为两个独立存储系统（类似两个独立的图书馆）不同，我们将记忆视为一个统一的认知系统，由多个记忆源组成。每个记忆源具有明确的职责边界：**系统记忆**（$M_{sys}$）负责存储任务执行相关的可复用模式，包括工作流模式（如 SQL 查询模式、Shell 命令序列、对话引导流程等）和任务执行策略，这些信息具有跨任务的通用性；**个人记忆**（$M_{pers}$）负责存储用户相关的个性化信息，包括用户偏好（如话题偏好、交互风格偏好）、关键事实（如用户背景信息、历史对话中的关键事件）和对话上下文，这些信息具有用户特异性。触发模型（轻量小模型）作为"元认知"层，根据交互轨迹的内容特征（如是否包含任务执行步骤、是否包含用户偏好表达等）智能地决定应该更新哪些记忆源（可以同时更新多个记忆源），类似于前额叶皮层协调人类大脑中不同记忆系统（工作记忆、长期记忆、情景记忆）的方式。这种设计使得智能体能够根据当前上下文动态整合不同来源的记忆，实现真正的多记忆协同。此外，每个记忆源内部还配备了记忆提取和记忆演化两个轻量小模型：对于系统记忆，我们在AWM [11] 的基础上进行了改进——记忆提取（workflow induction）负责从原始交互轨迹中提取可复用的工作流模式，记忆演化（workflow management）负责通过向量相似度搜索和 LLM 判断对工作流进行增删改查操作，实现记忆的去冗余、冲突解决和持续优化；对于个人记忆，我们采用 Mem0 [13]作为记忆源，其内部已集成了记忆提取和演化机制。
- 系统性对比与验证：通过将现有上下文工程方法（iterative context compression (icc)和 streamICL [6]）和单一记忆方法（如 AWM [11]、Mem0 [13]）在我们的统一基准上进行系统性对比，我们明确剖析了单一记忆设计在"任务边界模糊""在线学习"场景下的核心缺陷。实验结果表明，单一记忆类型的方法在混合场景下表现不佳，甚至有的单一记忆方法在单一场景下表现也不佳（如AWM [11]，我们也系统分析了原因，见消融实验部分），而我们的 ME 方法能够有效协调不同记忆源，实现真正意义上的自进化。

## 2. 文献综述

在[3, 5]中，将agent memory分为了system memory和personal memory两大类：
- system_memory: The system generates a series of intermediate output results during task execution. System memory helps strengthen the system's reasoning ability by storing task execution workflows and experiences.
- personal_memory: The system receives human input and feedback. Personal memory helps the system improve its understanding of user behavior and enhances its personalization capabilities by storing user preferences and conversation history.

### 2.1 Benchmark 层面的相关工作

System memory benchmark 专注于评估智能体从过往任务案例中提炼经验、实现信息长期内化的能力。这类基准的核心目标是评估智能体如何通过记忆提升后续任务的成功率，适配 stream 场景的持续优化。

Evo-memory [10] 是专门为评估 system memory 能力而设计的基准，支持在线学习模式，能够观察智能体在整个学习过程中的表现变化。WebArena [31]、Mind2Web [33]、HotpotQA [34]、ALFWorld [32]，AgentBench [28]等基准虽然主要用于评估智能体的任务执行能力，但也被广泛用于评估 system memory 方法（如 AWM [11]、ExpeL [12]、ExpRag [10]、ReMem [10]）。这些基准涵盖了多种任务类型，包括代码编写、网页操作、知识图谱查询、多跳问答和具身智能等，为 system memory 方法提供了丰富的评估场景。然而，使用这些基准需要自行实现online改造，并且无法观察学习过程，只能覆盖单一类型的记忆任务，为确保评估的标准化和可比性，我们的 system memory 任务（DB、OS、KG、ALFWorld、WebShop）复用了 AgentBench [28] 的任务定义、环境配置和评估指标。在此基础上，我们对这些任务进行了在线学习模式的改造（详见3.2.3节），通过带种子的随机重排和顺序化串行执行，使其能够评估智能体在持续学习过程中的记忆能力演化。StreamBench [6] 和 LifelongAgentBench [7] 是持续学习领域的代表性基准，它们支持在线学习模式，能够评估智能体在流式场景下的持续改进能力。MemoryBench [40] 是专门为 LLM 系统设计的 memory 和持续学习评估基准，提供了用户反馈模拟框架和涵盖多个领域、语言和任务类型的综合基准，用于评估 LLM 系统在服务时间内从累积用户反馈中学习的能力。MemoryBench 包含了多个数据集，既包括 system memory 类型的任务（如文本理解和生成任务），也包括 personal memory 数据集（如 LoCoMo [16]）。然而，MemoryBench 专注于文本任务，不涉及工具调用，严格意义上并不能算是agent benchmark。

Personal memory benchmark 专注于评估智能体记住用户偏好、优化对话体验的能力。这类基准的核心价值是评估智能体如何通过记忆让交互更贴合用户需求（如记住用户 dietary 偏好、行程安排、个人属性等）。

Locomo [16]评估智能体在长期对话中记住用户偏好的能力，包含多个会话和跨会话的推理任务。Locomo 数据集包含多个会话，每个会话包含多个对话轮次，要求智能体在后续会话中记住之前会话中提到的用户偏好。然而，Locomo 仅支持离线评估，无法观察学习过程。LongMemEval [17]评估智能体在超长对话中的记忆能力，关注用户偏好的长期保留。该基准测试智能体在超长对话序列中记住早期提到的信息的能力，特别关注时间推理和跨会话推理。同样，LongMemEval 仅支持离线评估。DMR [15]是由 MemGPT 提出的基准，主要用于评估用户关键事实的维护能力。DMR 评估智能体如何管理用户关键事实和维护滚动对话历史，且其也仅支持离线评估。

与现有基准相比，OnlineMemoryBench 支持多种记忆机制（zero_shot、experience_replay [7]、stream_icl [6]、agent_workflow_memory [11]、mem0 [13] 、context_compression、mems 等）和多种执行方法（single_agent、multi_agent_vote [41]、multi_agent_poll [6]、multi_agent_bandit），支持混合场景评估（可以同时运行 system memory 和 personal memory 任务），并提供了统一的评估框架，使得不同方法可以在相同条件下进行公平对比。更重要的是，OnlineMemoryBench 采用了高度解耦的框架设计：记忆机制、执行方法和任务分发评估都是独立的模块，研究者可以轻松地添加新的method mechanism、execution method或task。框架支持多种类型的数据集集成：既支持**需要后端环境支持的任务**（如 ALFWorld [32] 的 Docker 具身环境、WebShop [45] 的模拟电商后端），也支持**纯文本对话任务**（如 LoCoMo [16] 的长期对话记忆评估）。研究者无需修改核心框架代码即可添加新任务，这种设计使得基准具有良好的可扩展性，能够快速适应新的研究需求。

**表 1：Memory Benchmark 对比分析（这个表晚上回寝室我再检查）**

| Benchmark | 是否为测评memory能力的benchmark | 是否支持online模式 | 是否支持offline模式 | 是否支持replay模式 | 是否支持transfer模式 | 是否支持Code-grounded环境评估 | 是否支持Embody-grounded环境评估 | 是否支持Web-grounded环境评估 | 是否支持Dialogue-grounded环境评估 | 是否支持同时测评system memory和personal memory |
|-----------|:----------------------------:|:----------------:|:-----------------:|:----------------:|:-----------------:|:---------------------------:|:---------------------------:|:---------------------------:|:-------------------------------:|:-------------------------------------------:|
| OnlineMemoryBench (ours) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AgentBench [28] | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| StreamBench [6] | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| LifelongAgentBench [7] | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| MemoryBench | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| Evo-memory [10] | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ALFWorld [32] | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Locomo [16] | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| LongMemEval [17] | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| DMR [15] | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |

### 2.2 方法论层面的相关工作

记忆机制方法可以根据其关注点分为两大类：系统记忆方法和个人记忆方法。

**系统记忆方法**专注于从任务执行经验中提炼可复用的知识。AWM (Agent Workflow Memory) [11] 通过记录和检索工作流历史来提升任务执行效率；ExpeL [12] 将经验抽象为可复用的策略；Evo-memory [10] 提出了自进化记忆机制，能够从任务案例中学习并持续改进。虽然这些方法的共同特点是强调从任务执行经验中提取通用规律，但往往假设任务类型明确、边界清晰，难以处理混合场景下的复杂性。此外，MemGen [8] 和 G-memory [9] 尝试提取更通用的记忆表示：MemGen 使用无固定结构的机器原生记忆（"潜在 token 序列"），而 G-memory 采用三层图结构（洞察图 + 查询图 + 交互图）来建模协作相关记忆。然而，这些方法在评测记忆能力时，均只在仅能测评 system memory 能力的 benchmark 上进行了评估（如 ALFWorld [32]、BigCodeBench [46]、HotpotQA [34]、SciWorld [47] 等），并没有评估它们在混合场景下的表现。

**个人记忆方法**专注于长期对话中的用户偏好记忆和上下文理解。Mem0 [13] 提供了可扩展的长期记忆系统，能够记住用户偏好和关键事实；MemGPT [15] 采用分层内存架构来管理用户关键事实和对话历史；ZEP [14] 使用时序知识图谱来维护用户记忆。这些方法关注跨会话推理、时间推理和用户偏好记忆，但缺乏对任务执行经验的积累和复用能力。这些方法通常在 personal memory benchmark（如 Locomo [16]、LongMemEval [17]、DMR [15]）上进行评估，但这些基准也仅局限于 personal memory，无法评估这些方法在系统任务执行场景下的表现。

图3：awm错误处理locomo、mem0错误处理db的图

## 3. Benchmarks and methodologies
### 3.1 Problem Formulation

#### 3.1.1 Single task trajectory definition
我们形式化定义智能体记忆评估问题如下：考虑一个基于 LLM 的智能体在执行一个 $task$（而非一类任务），其中智能体在环境状态空间 $S$ 中执行动作。环境状态按照受控随机转移模型 $\Psi$ 演化：
$$s_{t+1} \sim \Psi(s_{t+1} | s_t, a_t)$$
其中 $a_t$ 表示智能体在时间步 $t$ 执行的动作。

在每个时间步 $t$，智能体接收观察：
$$o_t = O(s_t, h_t, Q)$$
其中 $h_t$ 表示智能体在时间步 $t$ 可见的交互历史（包括先前消息、中间工具输出、部分推理轨迹等，一般是当前正在进行 $task$ 的历史交互轨迹），$Q$ 表示任务规范（如用户指令、目标描述或外部约束）。

智能体遵循策略选择动作：
$$a_t = \pi(o_t, m_t, Q)$$

其中 $m_t$ 是从记忆系统 $M_t$ 中检索得到的记忆信号，定义为：
$$m_t = R(M_t, o_t, Q)$$
这里 $R$ 是检索算子，根据当前观察和任务规范检索相关记忆内容，返回格式化的记忆信号（如文本片段或结构化摘要）供 LLM 策略直接使用。


#### 3.1.2 记忆系统生命周期
记忆系统 $M_t \in \mathcal{M}$（其中 $\mathcal{M}$ 表示允许的记忆配置空间）通过三个生命周期算子动态演化。虽然记忆表示为统一状态 $M_t$，但这三个算子（形成 $F$、演化 $E$、检索 $R$）并不需要在每个时间步都被调用。不同的记忆效果源于不同的时间调用模式。

- **记忆形成（Formation）**：在时间步 $t$，智能体产生信息产物 $\phi_t$（如工具输出、推理轨迹、部分计划、自我评估或环境反馈），形成算子 $F$ 将这些产物选择性转换为记忆候选：
$$M_{t+1}^{form} = F(M_t, \phi_t)$$
形成算子提取具有未来潜在效用的信息，而非存储整个交互历史的逐字记录。记忆形成可以从原始观察的最小累积到可复用模式的复杂提取和精炼。例如，一个简单的记忆形成过程可以是：
$$M_{t+1}^{form} = M_t \cup \{o_t\}$$
即直接将当前观察 $o_t$ 添加到现有记忆 $M_t$ 中。更复杂的形成过程可能涉及从多个案例中提炼可复用的经验、规则和模式。

- **记忆演化（Evolution）**：新形成的记忆候选通过演化算子 $E$ 整合到现有记忆库中：
$$M_{t+1} = E(M_{t+1}^{form})$$
演化算子可能执行类似数据库操作——增、删、改、查 [13]：去冗余（consolidating redundant entries）、冲突解决（resolving conflicts）、低效用信息丢弃（discarding low-utility information）或记忆重构（restructuring memory for efficient retrieval）等操作。演化后的记忆状态 $M_{t+1}$ 在后续决策步骤和任务中持续存在。大部分记忆策略采用任务级更新模式，即在任务 $k$ 完成时依次调用一次形成算子 $F$ 与演化算子 $E$：
$$M_{k+1}^{form} = F(M^k, \phi^k), \quad M^{k+1} = E(M_{k+1}^{form})$$
其中 $M^k$ 表示任务 $k$ 开始时的记忆状态，$M^{k+1}$ 表示任务 $k$ 完成后的记忆状态，$\phi^k$ 表示任务 $k$ 完成时产生的信息产物（如最终结果、任务反馈、任务完成交互过程等）。

- **记忆检索（Retrieval）**：检索算子 $R$ 根据当前观察和任务规范检索相关记忆。
$$m_t =  R(M_t, o_t, Q),  t>=0$$
检索的时间模式可以灵活变化：同记忆形成和记忆演化，大部分记忆系统只在任务初始化时检索一次和任务结束后分别执行一次记忆形成和演化，在任务 $k$ 内，检索算子可以表示为：
$$m_t = \begin{cases} R(M^k, o_0, Q), & t=0 \\ \perp, & t>0 \end{cases}$$
其中 $\perp$ 表示空检索策略（null retrieval strategy），即 $t>0$ 时不进行记忆检索。


#### 3.1.3 Multi task trajectories definition

系统的一次任务的完整执行产生一个轨迹：
$$\tau = (s_0, o_0, a_0, s_1, o_1, a_1, ..., s_T)$$
其中 $T$ 由任务终止条件或系统特定的停止准则确定。轨迹的每一步反映了环境观察、可选记忆检索、基于 LLM 的计算和驱动下一状态转移的动作执行的交错过程。短期和长期记忆现象源于形成、演化和检索被调用的时间模式，而非离散的架构模块。

**在本工作中，我们假设记忆检索仅在任务初始化时执行一次，记忆形成和演化仅在任务完成时执行一次。** 对于给定的一个任务序列 $\mathcal{T} = \{task_1, task_2, ..., task_n\}$，每个任务 $task_k$ 属于系统记忆任务 $T_{sys}$（如 db、os、kg、webshop、alfworld，具体见3.2.1）或个人记忆任务 $T_{pers}$（如 locomo，具体见3.2.1）。每个任务 $task_k$ 可以形式化为一个输入-输出序列 $\tau_k = \{(x_1^k, y_1^k), \dots, (x_{T_k}^k, y_{T_k}^k)\}$，其中 $(x_t^k, y_t^k)$ 表示任务 $k$ 内时间步 $t$ 的输入和真实输出，$T_k$ 是任务 $k$ 的结束时间。智能体在执行任务 $k$ 的过程中，通过记忆机制 $M$ 从历史轨迹中学习，产生预测轨迹：
$$(x^1, y^1, M^1) \to (x^2, y^2, M^2) \to \dots \to (x^k, \hat{y}^k, M^k)$$
其中 $x^k$ 是任务 $k$ 的输入问题，$\hat{y}^k$ 是任务 $k$ 的完整对话历史（包含 system、user、assistant 和 tool 的交互），$M^k$ 是任务 $k$ 完成后的记忆状态。

### 3.2 Benchmarks

本节详细介绍 OnlineMemoryBench 支持的六个数据集、两类记忆任务的评估指标，以及四种评估模式（online、offline、replay 和 transfer）的形式化定义。此外，我们还讨论了将 LoCoMo [16] 数据集构建为在线评估模式时遇到的技术挑战。

#### 3.2.1 数据集介绍（把方法润完回来写，争取回寝室之前写完）

##### 3.2.1.1 System memory datasets

**Database (DB)**：
- **目标**：评估智能体在真实数据库环境中通过 SQL 操作数据库的能力。数据库分析是智能体应用中的关键任务，但现有研究往往只关注 SQL 翻译或小规模表格问答，缺乏对完整执行流程的评估。
- **相关工作**：现有研究主要关注 SQL 与自然语言之间的翻译（如 [31, 33]）或给定小规模表格的问答任务，但很少有工作评估模型在真实数据库环境中的完整执行流程。
- **OnlineMemoryBench 方法**：我们基于 WebArena [31] 和 Mind2Web [33] 等基准构建了 DBBench 数据集，要求智能体在真实的 SQL 接口和数据库环境中执行查询，涵盖多种查询类型，模拟真实场景中的数据库操作任务。智能体需要从历史成功案例中学习 SQL 编写模式，提升后续任务的成功率。
- **评估指标**：任务成功率（Success Rate, SR）。

**Operating System (OS)**：
- **目标**：评估智能体在终端环境中访问和操作操作系统的能力。将自然语言转换为 Shell 命令并执行是智能体应用中的重要场景，要求智能体理解系统命令语义并正确执行。
- **相关工作**：已有研究尝试将自然语言翻译为 Shell 命令，但大多数工作仅关注翻译准确性，很少有工作在可执行环境中评估模型的实际操作能力和命令执行效果。
- **OnlineMemoryBench 方法**：我们构建了操作系统交互任务，要求智能体在真实的交互式 bash 环境中（如 Ubuntu Docker）执行操作。任务包括确定性问题（如"统计非 /home 目录的用户数量"）和实用目标序列（如"递归设置所有目录文件为只读，排除我的文件"）。智能体需要从历史操作中学习命令模式和系统使用经验，提升后续任务的成功率。
- **评估指标**：任务成功率（Success Rate, SR）。

**Knowledge Graph (KG)**：
- **目标**：评估智能体在大型知识图谱环境中的推理和查询能力。现代知识图谱规模庞大（如 FREEBASE 包含超过 4500 万实体和 30 亿事实），在这种部分可观测环境中操作需要智能体具备广泛的技能，包括处理不完整信息、管理不确定性、理解语言细微差别、规划指令分解以及使用工具与知识图谱接口交互等能力。
- **相关工作**：知识图谱问答已有大量研究，但大多数工作关注小规模图谱或特定领域的查询，缺乏对大规模通用知识图谱的完整评估框架。
- **OnlineMemoryBench 方法**：我们采用知识图谱作为评估智能体决策能力的代表性测试环境。任务形式为问答任务，要求智能体在部分可观测的知识图谱环境中进行多跳推理和复杂查询。智能体需要从历史查询中学习有效的查询模式和推理策略，提升查询准确性和效率。
- **评估指标**：F1 分数。

**Web Shopping (WS)**：
- **目标**：评估智能体在模拟电商环境中进行在线购物的能力。在线购物是现代生活中的重要组成部分，涉及在电商网站上搜索、浏览和选择商品，需要强大的推理和决策能力。
- **相关工作**：WebShop [45] 提供了一个模拟在线购物环境用于评估语言智能体，但原始评估主要针对专门训练的模型，缺乏对通用 LLM 在零样本或少量提示下的评估。
- **OnlineMemoryBench 方法**：我们采用 WebShop 环境，但评估仅使用提示的 LLM（无需额外微调），要求智能体从历史购物经验中学习商品搜索和选择模式，提升购物任务的成功率和效率。
- **评估指标**：任务成功率（Success Rate, SR）。

**House Holding (HH, ALFWorld [32])**：
- **目标**：评估智能体在具身环境中的常识推理和任务执行能力。家庭环境需要强大的常识基础，是语言智能体评估的经典场景。
- **相关工作**：ALFWorld [32] 基于 TextWorld 工具包构建，是评估具身智能体的标准环境。
- **OnlineMemoryBench 方法**：我们使用经典的 ALFWorld 环境评估模型在物理家庭环境中的能力，要求智能体完成诸如"将平底锅放在餐桌上"等任务。智能体需要从历史任务执行中学习操作模式和空间推理策略。
- **评估指标**：任务成功率（Success Rate, SR）。

##### 3.2.1.1 Personal memory datasets

**LoCoMo [16] (Long-term Conversational Memory)**：
- **目标**：评估智能体在长期对话中记住用户偏好和对话历史的能力。长期对话记忆是构建个性化智能体的关键能力，需要智能体能够跨会话记住用户信息并进行推理，实现真正的个性化交互。
- **数据集结构**：LoCoMo 数据集包含 10 个扩展对话，每个对话平均包含约 600 轮对话和 26000 个 token。对话分布在多个会话中，每个对话捕捉两个个体讨论日常经历或过去事件的内容，模拟真实场景中的长期人机交互。
- **问题设置**：每个对话后平均包含 200 个问题，问题具有对应的真实答案。问题类型包括：
  - **Single-hop**：单跳推理问题，直接基于对话内容回答，测试智能体对单次对话信息的记忆能力
  - **Multi-hop**：多跳推理问题，需要结合多个对话片段进行推理，测试智能体对跨对话信息的整合能力
  - **Temporal**：时间推理问题，涉及对话中的时间顺序和时序关系，测试智能体对时间信息的记忆和推理能力
  - **Open-domain**：开放域问题，需要结合常识知识进行回答，测试智能体在记忆基础上进行常识推理的能力
- **OnlineMemoryBench 方法**：我们采用 LoCoMo 数据集评估智能体的长期对话记忆能力，要求智能体在跨会话的对话中记住用户偏好、关键事实和对话历史，并在后续会话中利用这些信息进行推理和回答。智能体需要从历史对话中提取和存储用户相关信息，并在检索时准确回忆相关记忆，实现跨会话的个性化交互。
- **评估指标**：F1 分数和准确率（Accuracy）。

#### 3.2.2 评价指标

对于系统记忆任务（DB、OS、KG、WebShop、ALFWorld），我们使用任务成功率（Success Rate, SR）和任务执行步长（Step）作为主要指标，评估智能体从历史经验中学习并提升任务执行能力的效果。

对于个人记忆任务（LoCoMo），我们使用 F1 分数、BLEU 和 LLM as Judge 作为主要指标，评估智能体记住用户偏好和对话历史的能力。

表二：datasets汇总表（借鉴agentBench）

#### 3.2.3 对数据集进行online改造时遇到的挑战

将现有数据集改造为支持在线学习模式的评估基准面临诸多技术挑战。本节详细描述我们对系统记忆任务和个人记忆任务进行在线改造的方法和遇到的挑战。

**系统记忆任务的在线改造**：系统记忆任务（DB、OS、KG、WebShop、ALFWorld）的数据集结构相对统一，每个样本包含三个核心组件：（1）**系统提示（System Prompt）**，定义任务环境和交互规范，指导智能体如何理解任务并与环境交互；（2）**任务目标（Task Goal）**，描述智能体需要完成的具体任务，包括问题描述和必要的上下文信息；（3）**工具集（Tool Set）**，提供在 Docker 环境中与环境交互的接口和函数定义，使智能体能够执行具体的操作（如 SQL 查询、Shell 命令等）。为了支持在线模式的评估，我们需要将原本可能并行或无序的任务样本转换为顺序执行的序列。具体而言，我们对所有任务样本进行**带种子的随机重排（Seeded Random Shuffling）**，确保不同实验运行之间的可复现性，然后将重排后的任务序列**顺序化串行执行**。这种改造方式使得智能体能够在执行任务序列的过程中逐步积累经验，并通过记忆机制从历史成功案例中学习，从而观察性能随任务执行而提升的学习曲线。

**个人记忆任务的在线改造**：个人记忆任务（LoCoMo）的数据集结构与系统记忆任务存在显著差异。LoCoMo 数据集由 10 个长对话组成，每个对话内部包含约 20 个会话（Session），每个会话具有不同且递进的时间戳，反映了对话的时间演进过程。形式上，我们可以将每个对话的会话序列表示为：

$$\tau_{\text{session}} = \{\text{session}_1, \text{session}_2, \ldots, \text{session}_{20}\}$$

此外，每个对话还包含约 200 个问题（Question），这些问题用于评估智能体对对话内容的记忆能力：

$$\tau_{\text{question}} = \{\text{question}_1, \text{question}_2, \ldots, \text{question}_{200}\}$$

与系统记忆任务不同，LoCoMo 数据集天然适合进行持续学习改造，因为其会话结构本身就体现了时间顺序和知识积累的过程。我们的改造方法基于每个问答对（QA）的答案在对话中的出现位置，将每个问题精确分配到其对应的会话。具体而言，我们分析问题的答案在对话中的时间戳和会话归属信息，确定该问题应该归属于哪个会话，从而构建出混合的任务序列：

$$\tau = \{\text{session}_1, \text{question}_1^{\text{session}_1}, \ldots, \text{session}_2, \text{question}_n^{\text{session}_2}, \ldots\}$$

这种改造方式确保了智能体在回答问题时，只能访问到该问题所属会话及其之前的所有会话内容，从而真实模拟了在线学习场景中知识逐步积累的过程。然而，这种改造也带来了技术挑战：如何准确地将问题分配到正确的会话，以及如何在评估时确保智能体不会"看到未来"的信息，这需要仔细的数据处理和评估流程设计。

#### 3.2.4 评估模式（Evaluation Modes）

OnlineMemoryBench 支持四种评估模式，分别对应不同的学习场景和评估需求：

**Offline 模式**：传统的离线评估模式，智能体在训练集上学习并更新记忆，然后在测试集上评估最终性能。对于 system memory 任务（DB、OS、KG、ALFWorld、WebShop），我们将所有任务打乱后划分为训练集和测试集；对于 personal memory 任务（LoCoMo），我们采用与 Mem0 [13] 相同的配置，将所有 session 作为训练集，所有 question 作为测试集。该模式适用于评估记忆系统的最终性能，但无法观察学习过程。

**Online 模式**：在线学习模式，智能体在执行任务序列的过程中实时更新记忆，我们记录每个任务执行后的性能变化，观察知识积累、重构和遗忘的动态过程。给定任务序列 $\mathcal{T} = \{t^1, t^2, \ldots, t^K\}$，记忆在每一步更新：

$$M^{k+1} = E(M^k, h^k)$$

其中 $E$ 表示记忆演化算子，$h^k$ 表示任务 $t^k$ 的交互历史。该模式能够完整记录智能体的学习曲线，是 OnlineMemoryBench 的核心创新之一。

**Replay 模式**：重放学习模式，通过周期性测试来评估智能体对已学知识的保留能力。我们将任务序列划分为多个阶段，每个阶段包含训练集和测试集。智能体在每个阶段的训练集上学习，然后在测试集上评估知识保留情况。该模式能够评估智能体在持续学习过程中是否会出现灾难性遗忘。

**Transfer 模式**：迁移学习模式，评估智能体将在一类任务上学到的知识应用到另一类任务的能力。智能体首先在源任务集合 $\mathcal{T}_{\text{source}}$ 上学习，然后在目标任务集合 $\mathcal{T}_{\text{target}}$ 上评估迁移效果：

$$\Delta_{\text{transfer}} = P_{\text{transfer}} - P_{\text{baseline}}$$

其中 $P_{\text{transfer}}$ 是迁移后的性能，$P_{\text{baseline}}$ 是在目标任务上运行 zero-shot 方法的基线性能。

## 4. Methodologies

### 4.1 Baseline Methods

我们形式化定义以下基线方法，它们分别代表了不同的记忆机制设计范式。

**Zero-shot**：零样本方法不使用任何记忆机制，智能体仅基于当前观察和任务规范生成动作：
$$a_t = \pi(o_t, \perp, Q)$$
其中 $\perp$ 表示空记忆信号，即不进行记忆检索。该方法反映了 LLM 的基本指令跟随能力，作为性能下界。

**Iterative Context Compression (ICC)**：迭代上下文压缩方法使用 LLM 压缩历史对话来管理上下文长度。在任务 $k$ 执行过程中，给定历史对话序列 $H^k = \{h^1, h^2, ..., h^{k-1}\}$（其中 $h^k$ 表示任务 $k$ 内的完整对话历史），压缩算子 $C_{LLM}$ 将历史长序列压缩为摘要：
$$H^{k,compressed} = C_{\text{LLM}}(H^k, L_{max})$$
如果压缩后的摘要仍然超过 $L_{max}$，则进行递归压缩和截断操作。智能体在完成任务 $k$ 时基于当前观察、任务规范与历史对话序列选择动作：
$$a_t = \pi(o_t, H^{k,compressed}, Q)$$
该方法通过 LLM 压缩避免了上下文窗口溢出，但可能丢失历史细节信息。

**StreamICL [6]**：流式上下文学习方法使用基于向量检索的 RAG（Retrieval-Augmented Generation）系统存储和检索历史成功任务的完整交互轨迹。在任务 $k$ 完成后，如果任务成功，形成算子 $F_{streamICL}$ 将完整轨迹格式化为 chunk $c^k = F_{streamICL}(x^k, \hat{y}^k)$，其中 $x^k$ 是任务 $k$ 的输入问题，$\hat{y}^k$ 是任务 $k$ 的完整对话历史（包含 system、user、assistant 和 tool 的交互）。演化算子 $E_{streamICL}$（由 embedding 模型充当）将 $c^k \cup M_{streamICL}^k$ 编码为 $M_{streamICL}^{k+1}$。在任务执行过程中，检索算子 $R_{stream}$ 根据当前问题 $Q$ 的 embedding 相似度从 $M_{streamICL}^k$ 检索 top-$k$ 个最相似的经验：
$$m_t = R_{stream}(M_{streamICL}^k, Q),  t=0$$

**AWM (Agent Workflow Memory) [11]**：工作流记忆方法通过记录和检索工作流历史来提升任务执行效率。工作流记忆 $M_{awm}$ 使用 RAG 系统存储结构化工作流。在任务 $k$ 执行过程中，检索算子 $R_{awm}$ 根据当前问题 $Q$ 的 embedding 检索 top-$k$ 个最相似的工作流。在任务 $k$ 完成后，如果任务成功（$f^k = 1$），将任务轨迹加入样本缓冲区。缓冲区内满 $I$ 个成功样本触发一次批量归纳，从多个轨迹中提取通用工作流 $W_{\text{batch}}^k = F_{AWM}(\{T^{1}, ..., T^k\})$。再由演化算子 $E_{AWM}$（由 embedding 模型充当）将 $W_{\text{batch}}^k$ 编码为 $M_{AWM}^{k+1}$以覆盖$M_{AWM}^{k}$。

**Mem0 [13]**：Mem0 是一个可扩展的长期记忆系统，专注于用户偏好记忆。记忆系统 $M_{mem0}$ 存储用户相关的结构化记忆条目 $E = \{e_1, e_2, ..., e_n\}$。在任务 $k$ 执行过程中，检索算子 $R_{mem0}$ 使用语义检索、过滤和重排序：
$$m_t = R_{mem0}(M_{mem0}^k, Q),  t=0$$

在任务 $k$ 完成后，如果任务成功（$f^k = 1$），记忆形成过程从当前任务的对话历史 $\phi^k$ 中提取用户相关信息，形成算子 $F_{\text{mem0}}$ 将对话历史转换为记忆候选 $M_{mem0}^{k,form} = F_{\text{mem0}}(M_{mem0}^k, \phi^k)$。演化算子 $E_{\text{mem0}}$对记忆候选进行冲突解决和去重，生成更新后当前任务的$M_{mem0}^{k+1}$。该方法专注于长期维护用户偏好和关键事实，但缺乏对任务执行经验的积累。


### 4.2 MEMs: Multi-Memorys Methodology

MEMs 方法的核心创新在于引入了一个轻量的"元认知"层——触发模型（trigger model），实现了从"多个独立存储系统"到"统一认知系统"的范式转变。MEMs 维护两个记忆源：系统记忆 $M_{sys}$（工作流）和个人记忆 $M_{pers}$（用户偏好）。

**检索阶段**：在任务 $k$ 执行过程中，MEMs 同时检索系统记忆和个人记忆。检索算子 $R_{sys}$ 根据当前问题 $Q$ 的 embedding 从系统记忆 $M_{sys}^k$ 中检索 top-$k$ 个最相似的工作流，检索算子 $R_{pers}$ 从个人记忆 $M_{pers}^k$ 中检索相关用户偏好：
$$m_{sys} = R_{sys}(M_{sys}^k, Q), \quad m_{pers} = R_{pers}(M_{pers}^k, Q)$$
然后将两种记忆按顺序整合并注入到消息序列中（先系统记忆，后个人记忆）：
$$m_t = \text{Concat}(m_{sys}, m_{pers}), \quad t=0$$
智能体基于整合后的记忆信号生成动作：$a_t = \pi(o_t, m_t, Q)$。

**更新阶段**：在任务 $k$ 完成后，触发模型 $T$ 根据交互轨迹 $h^k$ 决定应该更新哪些记忆源：
$$S = T(h^k), \quad S \subseteq \{\text{system\_memory}, \text{personal\_memory}\}$$
其中触发模型基于交互轨迹的内容特征（如是否包含任务执行步骤、用户偏好信息等）智能判断应该激活哪个记忆源。

对于系统记忆更新（当 $\text{system\_memory} \in S$ 时），如果任务成功（$f^k = 1$），形成算子 $F_{sys}$（workflow induction）从交互轨迹 $h^k$ 中提取工作流 $W^k = F_{sys}(h^k)$。演化算子 $E_{sys}$（workflow management）通过向量相似度搜索和 LLM 判断对工作流进行增删改查操作：
$$M_{sys}^{k+1} = \begin{cases} E_{sys}(M_{sys}^k, W^k) & \text{if } \text{system\_memory} \in S \text{ and } f^k = 1 \\ M_{sys}^k & \text{otherwise} \end{cases}$$

对于个人记忆更新（当 $\text{personal\_memory} \in S$ 时），如果任务成功（$f^k = 1$ 或 $\text{reward}^k > 0$），形成算子 $F_{pers}$ 从对话历史 $\phi^k$ 中提取用户相关信息，生成记忆候选 $M_{pers}^{k,form} = F_{pers}(M_{pers}^k, \phi^k)$。演化算子 $E_{pers}$（由 Mem0 Platform 内部机制充当）对记忆候选进行冲突解决和去重：
$$M_{pers}^{k+1} = \begin{cases} E_{pers}(M_{pers}^{k,form}) & \text{if } \text{personal\_memory} \in S \text{ and } (f^k = 1 \text{ or } \text{reward}^k > 0) \\ M_{pers}^k & \text{otherwise} \end{cases}$$

**统一认知框架**：MEMs 将记忆视为一个统一的认知系统，由多个记忆源（系统记忆和个人记忆）组成。触发模型（轻量小模型，如 7B）作为"元认知"层，智能地决定何时激活哪个记忆源，类似于前额叶皮层协调人类大脑中不同记忆系统的方式。这种设计使得智能体能够根据当前上下文动态整合不同来源的记忆，实现真正的多记忆协同。

##参考文献

AGENT MEMORY
AGENT【综述】
[1]The Rise and Potential of Large Language Model Based Agents: A Survey
[2]From System 1 to System 2: A Survey of Reasoning Large Language Models
Agent Memory【综述】
[3]MemoryintheAgeofAIAgents: ASurvey
[4]From HumanMemorytoAIMemory: ASurveyon Memory Mechanisms in the Era of LLMs
[5]Reinventing Clinical Dialogue: Agentic Paradigms for LLM-Enabled Healthcare

Communication
Memory methodology and benchmarks
[6]StreamBench: Towards Benchmarking Continuous Improvement of Language Agents
[7]LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners
[8]MemGen: WeavingGenerativeLatentMemoryfor Self-Evolving Agents
[9]SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation
[10]Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory
[11]AGENT WORKFLOW MEMORY
[12]ExpeL: LLMAgents Are Experiential Learners
[13]Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
[14]ZEP: A TEMPORAL KNOWLEDGE GRAPH ARCHITECTURE FOR AGENT MEMORY
[15]MemGPT:Towards LLMsasOperating Systems
[16]LONGMEMEVAL: BENCHMARKING CHAT ASSIST ANTS ON LONG-TERM INTERACTIVE MEMORY
[17]Evaluating Very Long-Term Conversational Memory of LLM Agents
[18]MEMRL: SELF-EVOLVING AGENTS VIA RUNTIME REINFORCEMENT LEARNING ON EPISODIC MEMORY
[19]Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven 
[20]Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement
[21]ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
[22]HaluMem: Evaluating Hallucinations in Memory Systems of Agents
[23]KnowMe-Bench:BenchmarkingPersonUnderstanding for Lifelong Digital Companions
In-context learning
[24]Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents

[25]Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
[26]LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY

[27]AGENTBENCH: EVALUATING LLMS AS AGENTS
[28]MEMORYBENCH: A BENCHMARK FOR MEMORY ANDCONTINUAL LEARNING IN LLM SYSTEMS
[29]AgentCoder: Multi-Agent Code Generation with Effective Testing and Self-optimisation
[30]WEBARENA: A REALISTIC WEB ENVIRONMENT FOR BUILDING AUTONOMOUS AGENTS
[31]MIND2WEB: Towards a Generalist Agent for the Web

[32]Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers
[33]DualRAG: A Dual-Process Approach to Integrate Reasoning and Retrieval for Multi-Hop Question Answering
[34]DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation