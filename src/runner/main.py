from __future__ import annotations

"""
End-to-end runner for the lifelong-learning benchmark (many memory method + single_agent, multi-turn, tool-calling).

Current version supports:
- assignment.yaml / scheduler / backend / memory.zero_shot / execution.single_agent
- LLM backend via OpenAI-compatible interface, supports tools / tool_calls, calls /interact directly

Run from project root:
    python -m src.runner.main
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from execution.single_agent.single_agent import SingleAgentExecutionEngine
from memory.zero_shot.zero_shot import load_zero_shot_from_yaml
from src.client.scheduler import ScheduleConfig, build_schedule, TaskName, SampleIndex, Schedule
from src.runner.agent import SimpleHTTPChatAgent
from src.runner.backend import BackendClient
from src.runner.builders import build_memory_from_config, build_execution_engine_from_config, ensure_output_dir, build_schedule_from_config
from src.runner.config import ExperimentConfig, load_experiment_config, ROOT_DIR
from src.runner.schedule_utils import (
    load_task_instance, is_locomo_task,
    SESSION_INJECTION_MARKER, REPLAY_TEST_MARKER, REPAIR_GROUP_MARKER
)
from src.server.tasks.locomo.task import convert_session_to_history

# Ĭ�Ϻ�˵�ַ����ͨ��������������
BACKEND_BASE_URL = os.getenv("LLBENCH_BACKEND_URL", "http://localhost:5038/api")


class LocomoSessionWrapper:
    """
    Locomo ����� Session ��װ����ֱ�ӵ��� LLM agent

    �����ڲ���Ҫ���� Session ����������£��� locomo �����ִ������
    """
    def __init__(self, session_id: int, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode: str = "online"):
        from src.server.tasks.locomo.task_base import Session

        # �̳� Session �ĳ�ʼ��
        self.session_id = session_id
        self.id = session_id  # locomo task ������Ҫ session.id
        self.history = []
        self.llm_agent = llm_agent
        self.memory_for_enhance = memory_for_enhance
        self.task_name = task_name
        self.locomo_task_instance = locomo_task_instance
        self.training_mode = training_mode
        self._loop = None
        self._empty_response_retry_limit = 5

    def inject(self, messages):
        """ע����Ϣ�� history"""
        if isinstance(messages, list):
            self.history.extend(messages)
        else:
            self.history.append(messages)

    def sync_action(self, *injection):
        """ֱ�ӵ��� LLM agent������Ҫ���ӵ� Session ����"""
        from src.server.tasks.locomo.task_base import AgentOutput, AgentOutputStatus
        from openai.types.chat import (
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
            ChatCompletionAssistantMessageParam
        )

        # ע����Ϣ
        self.inject(list(injection))

        # �� history ת��Ϊ messages ��ʽ��ֻ���� system, user, assistant��
        messages = []
        for item in self.history:
            if hasattr(item, 'root'):
                msg = item.root
            elif isinstance(item, dict):
                msg = item
            else:
                continue

            # ֻ����������Ϣ���ų� RewardHistoryItem
            if msg.get("role") in ["system", "user", "assistant"]:
                messages.append(msg)

        if self.memory_for_enhance is not None:
            # single_agent
            enhanced_messages = self.memory_for_enhance.use_memory(self.task_name, messages)

            # ����ǿ�����Ϣ���µ� history �У��Ա㱣��ʱ������ǿ���ݣ�
            # ��� messages �� enhanced_messages �Ĳ���
            if enhanced_messages != messages:
                # ���� enhanced_messages���ҳ����޸ĵ���Ϣ�����µ� history
                for idx, (orig_msg, enhanced_msg) in enumerate(zip(messages, enhanced_messages)):
                    # ��� content �Ƿ��޸�
                    if orig_msg.get("content") != enhanced_msg.get("content"):
                        # �ҵ� history �ж�Ӧ����Ϣ������
                        history_idx = 0
                        msg_count = 0
                        for i, item in enumerate(self.history):
                            if hasattr(item, 'root'):
                                msg = item.root
                            elif isinstance(item, dict):
                                msg = item
                            else:
                                continue

                            # ֻ���� system/user/assistant ��Ϣ
                            if msg.get("role") in ["system", "user", "assistant"]:
                                if msg_count == idx:
                                    history_idx = i
                                    break
                                msg_count += 1

                        # ���� history �е���Ϣ
                        if history_idx < len(self.history):
                            role = enhanced_msg.get("role")
                            content = enhanced_msg.get("content", "")
                            if role == "system":
                                self.history[history_idx] = ChatCompletionSystemMessageParam(
                                    role="system",
                                    content=content
                                )
                            elif role == "user":
                                self.history[history_idx] = ChatCompletionUserMessageParam(
                                    role="user",
                                    content=content
                                )
        else:
            enhanced_messages = messages

        # ֱ�ӵ��� LLM agent
        agent = self.llm_agent
        assistant_messages = []
        last_response = None
        for attempt in range(self._empty_response_retry_limit):
            response = agent.inference(enhanced_messages, tools=None)
            last_response = response
            content = response.get("content")
            if content:
                assistant_msg = {
                    "role": "assistant",
                    "content": content
                }
                reasoning_content = response.get("reasoning_content")
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                assistant_messages.append(assistant_msg)
                self.inject(assistant_msg)
                break

            logging.warning(
                "[LocomoSessionWrapper] Empty assistant response for sample %s "
                "(attempt %s/%s); retrying... keys=%s reasoning_present=%s tool_calls_present=%s",
                self.session_id,
                attempt + 1,
                self._empty_response_retry_limit,
                list(response.keys()),
                bool(response.get("reasoning_content")),
                bool(response.get("tool_calls")),
            )
            time.sleep(min(2 * (attempt + 1), 5))

        if not assistant_messages:
            logging.error(
                "[LocomoSessionWrapper] Exhausted retries for sample %s; "
                "recording an empty assistant message to avoid validation failure.",
                self.session_id,
            )
            assistant_msg = {
                "role": "assistant",
                "content": "",
            }
            if last_response and last_response.get("reasoning_content"):
                assistant_msg["reasoning_content"] = last_response["reasoning_content"]
            assistant_messages.append(assistant_msg)
            self.inject(assistant_msg)

        return AgentOutput(
            status=AgentOutputStatus.NORMAL,
            messages=assistant_messages
        )


def validate_training_mode_constraints(exp_cfg: ExperimentConfig) -> tuple[str, bool]:
    """
    ����У��ѵ��ģʽ��Լ������

    Args:
        exp_cfg: ʵ������

    Returns:
        (training_mode, cross_task): ѵ��ģʽ�Ϳ������־

    Raises:
        ValueError: �����ò�����ѵ��ģʽ��Լ��ʱ
    """
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    cross_task = exp_cfg.experiment.get("cross_task", False)
    tasks_cfg = exp_cfg.tasks
    task_names: List[str] = [t["name"] for t in tasks_cfg if "name" in t]

    # ����Ƿ��ж�� locomo ����personal memory ���ݼ�ֻ����һ����
    locomo_tasks = [name for name in task_names if is_locomo_task(name)]
    if len(locomo_tasks) > 1:
        raise ValueError(
            f"Multiple personal memory tasks (locomo) detected: {locomo_tasks}. "
            "Only one personal memory task (locomo-0 - locomo-9) is allowed per run."
        )

    if training_mode == "transfer":
        # transfer ģʽ����Ϊ�������
        transfer_task = exp_cfg.experiment.get("transfer_task")
        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
        if not transfer_task or not transfer_after_task:
            raise ValueError("transfer mode requires both transfer_task and transfer_after_task to be set")

        if transfer_task != transfer_after_task:
            # ���1��������Ǩ�ƣ�transfer_task != transfer_after_task��
            # ���� cross_task=True������ѡ���������񣬲����� locomo ����
            if not cross_task:
                raise ValueError("transfer mode with different tasks requires cross_task=True")
            if len(task_names) != 2:
                raise ValueError(
                    f"transfer mode with different tasks requires exactly 2 tasks, but found {len(task_names)} tasks: {task_names}"
                )
            if locomo_tasks:
                raise ValueError(
                    f"transfer mode with different tasks does not support personal memory tasks (locomo). "
                    f"Found locomo task(s): {locomo_tasks}"
                )
            if transfer_task not in task_names or transfer_after_task not in task_names:
                raise ValueError(
                    f"transfer mode: transfer_task={transfer_task} and transfer_after_task={transfer_after_task} "
                    f"must be in the selected tasks: {task_names}"
                )
        else:
            # ���2��ǰ��Ǩ�ƣ�transfer_task == transfer_after_task��
            # �������������񣨰���locomo��������ֻѡ��һ�����񣬱������� forward_transfer_num
            if len(task_names) != 1:
                raise ValueError(
                    f"transfer mode with same task requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
                )
            if transfer_task not in task_names:
                raise ValueError(
                    f"transfer mode: transfer_task={transfer_task} must be in the selected tasks: {task_names}"
                )
            forward_transfer_num = exp_cfg.experiment.get("forward_transfer_num")
            if forward_transfer_num is None or forward_transfer_num <= 0:
                raise ValueError(
                    f"transfer mode with same task requires forward_transfer_num to be set and > 0. "
                    f"Got: forward_transfer_num={forward_transfer_num}"
                )
    elif training_mode == "replay":
        # replay ģʽ������ cross_task=False������ֻѡ��һ������
        if cross_task:
            raise ValueError("replay mode requires cross_task=False")
        if len(task_names) != 1:
            raise ValueError(
                f"replay mode requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
            )
        # ��� replay �����Ƿ����ã����ڷ� locomo ����
        if not locomo_tasks:
            replay_m = exp_cfg.experiment.get("replay_m")
            replay_n = exp_cfg.experiment.get("replay_n")
            replay_seed = exp_cfg.experiment.get("replay_seed")
            if replay_m is None or replay_n is None or replay_seed is None:
                raise ValueError(
                    f"replay mode requires replay_m, replay_n, and replay_seed to be set. "
                    f"Got: replay_m={replay_m}, replay_n={replay_n}, replay_seed={replay_seed}"
                )
    elif training_mode == "repair":
        # repair ģʽ������ cross_task=False������ֻѡ��һ������
        if cross_task:
            raise ValueError("repair mode requires cross_task=False")
        if len(task_names) != 1:
            raise ValueError(
                f"repair mode requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
            )
        # ��� repair �����Ƿ�����
        if locomo_tasks:
            # locomo ������Ҫ repair_size_locomo �� repair_seed
            repair_size_locomo = exp_cfg.experiment.get("repair_size_locomo")
            repair_seed = exp_cfg.experiment.get("repair_seed")
            if repair_size_locomo is None or repair_seed is None:
                raise ValueError(
                    f"repair mode for locomo tasks requires repair_size_locomo and repair_seed to be set. "
                    f"Got: repair_size_locomo={repair_size_locomo}, repair_seed={repair_seed}"
                )
            if not (0 < repair_size_locomo <= 1):
                raise ValueError(
                    f"repair_size_locomo must be between 0 and 1 (exclusive 0, inclusive 1). "
                    f"Got: repair_size_locomo={repair_size_locomo}"
                )
        else:
            # �� locomo ������Ҫ repair_m, repair_n, repair_seed
            repair_m = exp_cfg.experiment.get("repair_m")
            repair_n = exp_cfg.experiment.get("repair_n")
            repair_seed = exp_cfg.experiment.get("repair_seed")
            if repair_m is None or repair_n is None or repair_seed is None:
                raise ValueError(
                    f"repair mode requires repair_m, repair_n, and repair_seed to be set. "
                    f"Got: repair_m={repair_m}, repair_n={repair_n}, repair_seed={repair_seed}"
                )
    elif training_mode == "offline":
        # offline ģʽ������ cross_task=False������ֻѡ��һ������
        if cross_task:
            raise ValueError("offline mode requires cross_task=False")
        if len(task_names) != 1:
            raise ValueError(
                f"offline mode requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
            )
    else:
        # online ģʽ����֤ cross_task ������������һ����
        if not cross_task:
            # cross_task=False ʱ����ֻ��ѡ��һ�����ݼ�
            if len(task_names) != 1:
                raise ValueError(
                    f"Invalid configuration: cross_task=False requires exactly 1 task, "
                    f"but found {len(task_names)} tasks: {task_names}"
                )
        else:
            # cross_task=True ʱ����ѡ�д���һ�����ݼ�
            if len(task_names) <= 1:
                raise ValueError(
                    f"Invalid configuration: cross_task=True requires more than 1 task, "
                    f"but found {len(task_names)} task(s): {task_names}"
                )

    return training_mode, cross_task


def main() -> None:
    print(f"Using backend base URL: {BACKEND_BASE_URL}")
    backend = BackendClient(BACKEND_BASE_URL)

    # 1) �򵥽������
    try:
        workers = backend.list_workers()
        print("Controller /list_workers OK. Available tasks:")
        print(json.dumps(workers, indent=2))
    except Exception as e:
        print(f"Failed to call /list_workers: {e}")
        print("��ȷ�Ϻ�� Controller ����Ĭ�϶˿� 5038 ��������ͨ�� LLBENCH_BACKEND_URL ���ǵ�ַ��")
        return

    # 2) ��ȡ assignment ����
    exp_cfg = load_experiment_config()

    # 2.1) У��ѵ��ģʽԼ��������У�飩������ȡ training_mode �� cross_task
    training_mode, cross_task = validate_training_mode_constraints(exp_cfg)

    # 2.2) ��ȡ shuffle ����
    shuffle_cfg = exp_cfg.experiment.get("shuffle", {})
    shuffle_enabled = shuffle_cfg.get("enabled", False) if isinstance(shuffle_cfg, dict) else shuffle_cfg

    # 3) ��Ⲣ���� locomo ������Ҫ�ڹ�������֮ǰ��
    locomo_task_instance = None
    locomo_task_name = None
    tasks_cfg = exp_cfg.tasks
    task_names: List[str] = [t["name"] for t in tasks_cfg if "name" in t]

    # ����Ƿ��� locomo ����
    locomo_tasks = [name for name in task_names if is_locomo_task(name)]

    # ����� locomo ���񣬼�����
    if len(locomo_tasks) == 1:
        task_name = locomo_tasks[0]
        locomo_task_instance = load_task_instance(task_name, exp_cfg)
        locomo_task_name = task_name
        if locomo_task_instance is None:
            raise ValueError(f"Failed to load locomo task instance for {task_name}")
        print(f"\n[Locomo Task Detected] {task_name}, sessions: {locomo_task_instance.session_ids}")

    # 4) ����������У�ͳһ��ڣ����������ĵ�����Ϣ��
    schedule_result = build_schedule_from_config(
        exp_cfg, backend,
        locomo_task_instance=locomo_task_instance,
        locomo_task_name=locomo_task_name
    )

    train_schedule = schedule_result["train_schedule"]
    test_schedule = schedule_result["test_schedule"]
    task_to_indices = schedule_result["task_to_indices"]
    replay_info = schedule_result["replay_info"]

    print("\nTasks and available indices:")
    for task, indices in task_to_indices.items():
        print(f"  {task}: {len(indices)} indices -> {indices[:10]}{' ...' if len(indices) > 10 else ''}")

    print(f"\nSchedule summary:")
    print(f"  Train schedule: {len(train_schedule)} samples")
    if test_schedule:
        print(f"  Test schedule: {len(test_schedule)} samples")
    print(f"  First 20 train entries:")
    for pair in train_schedule[:20]:
        print("   ", pair)

    if not train_schedule:
        print("Train schedule is empty; nothing to run.")
        return

    # 4) ���� memory + execution engine
    execution_engine = build_execution_engine_from_config(exp_cfg)

    def build_memory_bundle():
        """��ִ�з�ʽ��ѵ��ģʽ���� memory �� memory_for_enhance�����������л�ʱ���á�"""
        mem = build_memory_from_config(exp_cfg)
        mem_enh = None

        if training_mode == "offline":
            mem_enh = load_zero_shot_from_yaml(str(ROOT_DIR / "memory" / "zero_shot" / "zero_shot.yaml"))
            print(f"Training mode: {training_mode} -> Using zero_shot for use_memory (memory disabled), but still updating memory with {exp_cfg.memory_mechanism.get('name', 'zero_shot')}")
        elif training_mode in ("online", "transfer", "replay", "repair"):
            # online, transfer, replay, repair ģʽ��ʹ�����õļ������
            mem_enh = mem
            print(f"Training mode: {training_mode} -> Using {exp_cfg.memory_mechanism.get('name', 'zero_shot')} for both use_memory and update_memory")
        else:
            raise ValueError(f"Unknown training_mode: {training_mode} (must be 'online', 'offline', 'transfer', 'replay', or 'repair')")

        return mem, mem_enh

    # ��ʼ memory
    memory, memory_for_enhance = build_memory_bundle()

    # 4.1) locomo ����� session ע��ͳһ�� schedule �е� SESSION_INJECTION_MARKER ����
    # ������ offline ģʽ��Ԥע�룬�����ظ�ע��
    # Online/Offline ģʽ�� session ע�붼����ѵ��ѭ����ͨ�� marker ����

    # 5) ���Ŀ¼������ train_size �ָ���� train/test ��Ŀ¼��
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_root = ensure_output_dir(ROOT_DIR / "outputs" / timestamp)

    # ���� training_mode ȷ����Ŀ¼����
    if test_schedule:
        if training_mode == "transfer":
            # transfer ģʽ�������񣩣�ʹ�� transfer_train �� transfer_test Ŀ¼��
            train_output_root = ensure_output_dir(base_output_root / "transfer_train")
            test_output_root = ensure_output_dir(base_output_root / "transfer_test")
        else:
            # offline ģʽ��ʹ�� train �� test Ŀ¼��
            train_output_root = ensure_output_dir(base_output_root / "train")
            test_output_root = ensure_output_dir(base_output_root / "test")
    else:
        if training_mode == "transfer":
            # transfer ģʽ��ǰ��Ǩ�ƣ���ʹ�� transfer_train Ŀ¼��
            train_output_root = ensure_output_dir(base_output_root / "transfer_train")
        else:
            train_output_root = base_output_root
        test_output_root = None

    # Ϊ execution engine ���� LLM agent(s)������ llmapi ���ã�
    if isinstance(execution_engine, SingleAgentExecutionEngine):
        # single_agent: ����һ�� agent
        llm_agent = SimpleHTTPChatAgent(execution_engine.config.agent_name)
    else:
        llm_agent = None

    # 6) ִ��ѵ��������
    last_task_name: TaskName | None = None
    
    # ��¼ִ��˳�����ڻ�����ȷ����ʱ��仯��ͼ��
    # onlineģʽ��������Ŀ¼�±��� execution_order.json
    # offlineģʽ����train��testĿ¼�·ֱ𱣴� execution_order.json
    execution_order_train: Dict[TaskName, List[Dict[str, Any]]] = {}  # {task_name: [execution_record, ...]}
    execution_order_test: Dict[TaskName, List[Dict[str, Any]]] = {}   # {task_name: [execution_record, ...]}
    execution_order_forward_test: Dict[TaskName, List[Dict[str, Any]]] = {}  # transferģʽ��ǰ��Ǩ�Ʋ���

    # Replay ģʽ�����ٵ�ǰ replay ״̬
    current_replay_id = 0
    learned_samples_in_replay: List[SampleIndex] = []  # ��ǰ��ѧϰ������������ȷ�� replay_id��
    current_replay_id_for_test = 1 if (training_mode == "replay" and replay_info) else 0  # ��ǰ����ִ�е� replay �Ĳ��Խ׶Σ�����ȷ����������Ӧ�ñ��浽�ĸ� replay��

    # =========================================================================
    # Repair ģʽ�����Լ���ϵͳ����֪ʶ��ͻ������
    # =========================================================================
    if training_mode == "repair" and replay_info:
        print(f"\n{'='*80}")
        print(f"Running REPAIR mode: {len(replay_info)} repair groups")
        print(f"{'='*80}\n")

        # ��ȡ�������ƣ�repair ģʽֻ֧�ֵ�����
        if len(task_to_indices) != 1:
            raise ValueError(f"repair mode requires exactly 1 task, but got {len(task_to_indices)} tasks")
        actual_task_name = list(task_to_indices.keys())[0]

        # ����ÿ�� repair ��
        for repair_id, repair_group_info in replay_info.items():
            print(f"\n{'='*80}")
            print(f"Processing Repair Group {repair_id}")
            print(f"{'='*80}\n")

            # ��ȡ�� repair �����Ϣ
            if is_locomo_task(actual_task_name):
                # Locomo ����repair_group_info = {"session_id": ..., "all_qa": [...], "reversed_qa": [...]}
                session_id = repair_group_info.get("session_id")
                all_samples = repair_group_info.get("all_qa", [])
                reversed_samples = repair_group_info.get("reversed_qa", [])

                # Locomo ��������ע�� session
                if locomo_task_instance and locomo_task_name:
                    print(f"[Repair {repair_id}] Injecting session {session_id} content into memory...")
                    session_history = locomo_task_instance.get_session_history(session_id)
                    if session_history:
                        if isinstance(memory, dict):
                            for agent_name, agent_mem in memory.items():
                                agent_mem.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        else:
                            memory.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        print(f"  -> Injected session {session_id} ({len(session_history)} dialogues)")
            else:
                # ��ͨ����repair_group_info = {"all_samples": [...], "reversed_samples": [...]}
                all_samples = repair_group_info.get("all_samples", [])
                reversed_samples = repair_group_info.get("reversed_samples", [])

            print(f"[Repair {repair_id}] All samples: {len(all_samples)}, Reversed samples: {len(reversed_samples)}")

            # ���� repair �����Ŀ¼
            repair_dir = base_output_root / f"repair{repair_id}"
            repair_dir.mkdir(parents=True, exist_ok=True)

            # ���� 4 ���׶�
            phases = [
                {"name": "wrongJudge", "use_reversed_rewards": True, "is_test": False},
                {"name": "wrongJudgeTest", "use_reversed_rewards": False, "is_test": True},
                {"name": "rightJudge", "use_reversed_rewards": False, "is_test": False},
                {"name": "rightJudgeTest", "use_reversed_rewards": False, "is_test": True},
            ]

            # ִ�� 4 ���׶�
            for phase in phases:
                phase_name = phase["name"]
                use_reversed_rewards = phase["use_reversed_rewards"]
                is_test_phase = phase["is_test"]

                print(f"\n{'��'*60}")
                print(f"[Repair {repair_id}] Phase: {phase_name}")
                print(f"  - Reversed rewards: {use_reversed_rewards}")
                print(f"  - Test mode: {is_test_phase}")
                print(f"{'��'*60}\n")

                # �����׶����Ŀ¼
                phase_full_dir = repair_dir / f"{phase_name}Full" / actual_task_name
                phase_standard_dir = repair_dir / f"{phase_name}Standard" / actual_task_name
                phase_full_dir.mkdir(parents=True, exist_ok=True)
                phase_standard_dir.mkdir(parents=True, exist_ok=True)

                # ִ���������������浽 Full��
                for sample_idx in all_samples:
                    is_reversed = sample_idx in reversed_samples

                    try:
                        # ���� locomo ����ʹ������ʵ��
                        if is_locomo_task(actual_task_name) and locomo_task_instance and locomo_task_name == actual_task_name:
                            session = LocomoSessionWrapper(sample_idx, llm_agent, memory_for_enhance, actual_task_name, locomo_task_instance, training_mode)
                            task_result = locomo_task_instance.sync_start_sample(sample_idx, session)

                            # ��ȡ messages �� reward
                            messages = []
                            for item in session.history:
                                if hasattr(item, 'root') and isinstance(item.root, dict):
                                    msg = item.root
                                    if msg.get("role") in ["system", "user", "assistant"]:
                                        messages.append(msg)
                                elif isinstance(item, dict) and item.get("role") in ["system", "user", "assistant"]:
                                    messages.append(item)

                            reward = 0
                            for item in session.history:
                                if hasattr(item, 'root') and hasattr(item.root, 'reward'):
                                    reward_item = item.root
                                    if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                        llm_score = reward_item.metrics.get("llm_score")
                                        if llm_score is not None:
                                            reward = float(llm_score)
                                            break
                                    reward = reward_item.reward
                                    break
                                elif isinstance(item, dict) and "reward" in item:
                                    if "metrics" in item and isinstance(item["metrics"], dict):
                                        llm_score = item["metrics"].get("llm_score")
                                        if llm_score is not None:
                                            reward = float(llm_score)
                                            break
                                    reward = item["reward"]
                                    break

                            # ����� history ��û���ҵ������Դ� task_result �л�ȡ
                            if reward == 0 and isinstance(task_result.result, dict):
                                metrics = task_result.result.get("metrics")
                                if isinstance(metrics, dict):
                                    llm_score = metrics.get("llm_score")
                                    if llm_score is not None:
                                        reward = float(llm_score)

                            # ���ԣ���ӡ��תǰ��ֵ
                            original_reward = reward

                            # Ӧ�ý�����ת
                            if use_reversed_rewards and is_reversed:
                                reward = 1 - reward  # ��ת������0->1, 1->0
                                print(f"    [DEBUG] Sample {sample_idx}: original_reward={original_reward:.2f}, is_reversed={is_reversed}, use_reversed_rewards={use_reversed_rewards}, final_reward={reward:.2f}")

                            result = {
                                "reward": reward,
                                "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                                "result": task_result.result
                            }

                        else:
                            # ��ͨ����ʹ�ú��
                            sess = backend.start_sample(actual_task_name, sample_idx)
                            session_id_backend = sess["session_id"]

                            # ��ȡ��ʼ����
                            obs = backend.get_observation(session_id_backend)
                            messages = obs.get("history", [])

                            # ִ������
                            while True:
                                enhanced_msg = execution_engine.run(messages, memory_for_enhance)
                                messages.append(enhanced_msg)

                                step_result = backend.step(session_id_backend, enhanced_msg)
                                if step_result.get("done", False):
                                    break

                                obs = backend.get_observation(session_id_backend)
                                obs_msg = obs.get("observation")
                                if obs_msg:
                                    messages.append(obs_msg)

                            result = backend.get_result(session_id_backend)
                            reward = result.get("reward", 0)

                            # ���ԣ���ӡ��תǰ��ֵ
                            original_reward = reward

                            # Ӧ�ý�����ת
                            if use_reversed_rewards and is_reversed:
                                reward = 1 - reward  # ��ת������0->1, 1->0
                                print(f"    [DEBUG] Sample {sample_idx}: original_reward={original_reward:.2f}, is_reversed={is_reversed}, use_reversed_rewards={use_reversed_rewards}, final_reward={reward:.2f}")

                            result["reward"] = reward

                        # ���� memory�������ǲ��Խ׶Σ�
                        if not is_test_phase:
                            if isinstance(memory, dict):
                                for agent_name, agent_mem in memory.items():
                                    agent_mem.update_memory(actual_task_name, messages, result)
                            else:
                                memory.update_memory(actual_task_name, messages, result)

                        # �������� Full Ŀ¼
                        sample_file_full = phase_full_dir / f"{sample_idx}.json"
                        with open(sample_file_full, 'w', encoding='utf-8') as f:
                            json.dump({"messages": messages, "result": result}, f, ensure_ascii=False, indent=2)

                        # ����� reversed ������Ҳ���浽 Standard Ŀ¼
                        if is_reversed:
                            sample_file_standard = phase_standard_dir / f"{sample_idx}.json"
                            with open(sample_file_standard, 'w', encoding='utf-8') as f:
                                json.dump({"messages": messages, "result": result}, f, ensure_ascii=False, indent=2)

                        status_marker = "[REVERSED]" if is_reversed else ""
                        test_marker = "[TEST]" if is_test_phase else ""
                        print(f"  {status_marker}{test_marker} Sample {sample_idx}: reward={reward:.2f}")

                    except Exception as e:
                        print(f"  [ERROR] Sample {sample_idx} failed: {e}")
                        import traceback
                        traceback.print_exc()

                print(f"\n[Repair {repair_id}] {phase_name} completed:")
                print(f"  - Full: {len(all_samples)} samples saved to {phase_full_dir}")
                print(f"  - Standard: {len(reversed_samples)} samples saved to {phase_standard_dir}")

        print(f"\n{'='*80}")
        print(f"Repair mode execution completed")
        print(f"{'='*80}\n")
        return  # Repair ģʽִ����ϣ�ֱ�ӷ���

    # =========================================================================
    # ����ѵ��/����ִ�У��� repair ģʽ��
    # =========================================================================

    if train_schedule:
        print(f"\n{'='*60}")
        if training_mode == "transfer":
            # Transfer ģʽ��ͳ��ѵ�������Ͳ�������������
            transfer_task = exp_cfg.experiment.get("transfer_task")
            transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
            train_count = sum(1 for task_name, _ in train_schedule if task_name == transfer_task)
            test_count = sum(1 for task_name, _ in train_schedule if task_name == transfer_after_task)
            print(f"Running Transfer mode: {len(train_schedule)} samples (train={train_count}, test={test_count})")
            print(f"  TRAIN task: {transfer_task} ({train_count} samples)")
            print(f"  TEST task: {transfer_after_task} ({test_count} samples)")
        else:
            print(f"Running TRAIN set: {len(train_schedule)} samples")
        print(f"{'='*60}\n")
        
        for idx, (task_name, sample_index) in enumerate(train_schedule, start=1):
            # ���� replay ģʽ�Ĳ����������
            is_replay_test = False
            if task_name == REPLAY_TEST_MARKER:
                # replay ģʽ�Ĳ�����������Ҫ�� task_to_indices �л�ȡʵ�ʵ���������
                if len(task_to_indices) != 1:
                    raise ValueError(f"replay mode: expected 1 task, but got {len(task_to_indices)} tasks")
                actual_task_name = list(task_to_indices.keys())[0]
                task_name = actual_task_name
                is_replay_test = True
                # ʹ�� current_replay_id_for_test ��ȷ����ǰ���ĸ� replay �Ĳ��Խ׶�
                # ���ֵ��ѵ����������ʱ�ᱻ����
                current_replay_id = current_replay_id_for_test
                print(f"[TRAIN {idx}/{len(train_schedule)}] [REPLAY TEST] task={task_name}, index={sample_index} (replay{current_replay_id})")
            else:
                # ѵ��������ȷ����ǰ replay_id������ѵ������������ replay��
                if training_mode == "replay" and replay_info:
                    for rid, info in replay_info.items():
                        if sample_index in info["train"]:
                            # �ҵ�������ѵ����������� replay_id�������µ� replay��
                            # ��Ϊѵ������������ڶ�� replay �� train �б��У��ۻ��ģ�
                            current_replay_id = max(current_replay_id, rid)
                            break
            
            # ���� session ע���ǣ����ڻ�ϵ��ȣ�
            if task_name == SESSION_INJECTION_MARKER:
                session_id = sample_index  # �ڻ�ϵ����У�sample_index �洢���� session_id
                if locomo_task_instance is not None and locomo_task_name:
                    print(f"[TRAIN {idx}/{len(train_schedule)}] [SESSION INJECTION] Injecting session {session_id} content into memory...")
                    session_history = locomo_task_instance.get_session_history(session_id)
                    if session_history:
                        if isinstance(memory, dict):
                            # Multi-agent: �������� agent �� memory
                            for agent_name, agent_mem in memory.items():
                                agent_mem.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        else:
                            memory.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        print(f"  -> Injected session {session_id} ({len(session_history)} dialogues)")
                    else:
                        print(f"  -> Warning: Session {session_id} has no history")
                else:
                    print(f"  -> Warning: SESSION_INJECTION_MARKER found but locomo_task_instance is None")
                continue  # ����ִ�У�������һ������

            # �����������ѧϰ�������л������� memory
            if not cross_task and last_task_name is not None and task_name != last_task_name:
                memory, memory_for_enhance = build_memory_bundle()
                print(f"\n[Memory Reset] cross_task=False, switched task {last_task_name} -> {task_name}, memory rebuilt.\n")
            last_task_name = task_name
            print(f"[TRAIN {idx}/{len(train_schedule)}] task={task_name}, index={sample_index}")

            try:
                # ���� locomo ����ֱ��ʹ������ʵ��������Ҫ���
                if is_locomo_task(task_name) and locomo_task_instance is not None and locomo_task_name == task_name:
                    # ������װ�� Session
                    session = LocomoSessionWrapper(sample_index, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode)
                    
                    # ֱ�ӵ�������ʵ���� sync_start_sample
                    task_result = locomo_task_instance.sync_start_sample(sample_index, session)
                    
                    # �� session.history ����ȡ messages ���ں�������
                    messages = []
                    for item in session.history:
                        if hasattr(item, 'root') and isinstance(item.root, dict):
                            msg = item.root
                            if msg.get("role") in ["system", "user", "assistant"]:
                                messages.append(msg)
                        elif isinstance(item, dict):
                            if item.get("role") in ["system", "user", "assistant"]:
                                messages.append(item)
                    
                    # �� history ����ȡ reward������ previous_sample_utilization �ȼ�����ƣ�
                    # ���� locomo ����reward ���� llm_score ����
                    reward = 0  # Ĭ�� reward Ϊ 0
                    for item in session.history:
                        if hasattr(item, 'root'):
                            # RootModel ���ͣ���� root �Ƿ��� RewardHistoryItem
                            if hasattr(item.root, 'reward'):
                                reward_item = item.root
                                # ���ȴ� metrics �е� llm_score ��ȡ reward
                                if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                    llm_score = reward_item.metrics.get("llm_score")
                                    if llm_score is not None:
                                        reward = float(llm_score)  # llm_score �� 0 �� 1
                                        break
                                # ���û�� metrics��ʹ�� reward �ֶ�
                                reward = reward_item.reward
                                break
                        elif isinstance(item, dict) and "reward" in item:
                            # ������ֵ䣬����Ƿ��� metrics
                            if "metrics" in item and isinstance(item["metrics"], dict):
                                llm_score = item["metrics"].get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item["reward"]
                            break
                        elif hasattr(item, 'reward'):
                            # ֱ���� RewardHistoryItem ʵ��
                            # ���ȴ� metrics �е� llm_score ��ȡ reward
                            if hasattr(item, 'metrics') and isinstance(item.metrics, dict):
                                llm_score = item.metrics.get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item.reward
                            break
                    
                    # ����� history ��û���ҵ������Դ� task_result.result �е� metrics ��ȡ
                    if reward == 0 and isinstance(task_result.result, dict):
                        metrics = task_result.result.get("metrics")
                        if isinstance(metrics, dict):
                            llm_score = metrics.get("llm_score")
                            if llm_score is not None:
                                reward = float(llm_score)
                    
                    # ʹ�� task_result ��Ϊ result������ reward �ֶ��Ա�������ʶ��
                    result = {
                        "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                        "result": task_result.result,
                        "reward": reward,  # ���� reward �ֶΣ����� previous_sample_utilization �ȼ������ʶ��
                    }
                    
                    # ���� memory��ʹ�� session.history��
                    # ���� training_mode �����Ƿ���£�transfer �� replay ģʽ��
                    history = session.history
                    should_update_memory_locomo = True
                    if training_mode == "transfer":
                        transfer_task = exp_cfg.experiment.get("transfer_task")
                        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                        # ֻ���ڿ�����Ǩ�ƣ�transfer_task != transfer_after_task���ҵ�ǰ������ transfer_after_task ʱ���Ų����¼���
                        # ǰ��Ǩ�ƣ�transfer_task == transfer_after_task��ʱ������������Ӧ�ø��¼���
                        if transfer_task != transfer_after_task and task_name == transfer_after_task:
                            should_update_memory_locomo = False
                    elif training_mode == "replay":
                        if is_replay_test:
                            should_update_memory_locomo = False
                    
                    if should_update_memory_locomo:
                        if isinstance(memory, dict):
                            for agent_mem in memory.values():
                                agent_mem.update_memory(task_name, history, result)
                        else:
                            memory.update_memory(task_name, history, result)

                    # Replay ģʽ���������ԣ�ѧϰ���������Ը�������
                    serializable_history = []
                    for item in history:
                        if hasattr(item, 'root'):
                            # RootModel ���ͣ���ȡ root ֵ
                            serializable_history.append(item.root)
                        elif hasattr(item, 'model_dump'):
                            # Pydantic ģ�ͣ�ת��Ϊ�ֵ�
                            # ʹ�� exclude_none=True �ų� None ֵ���� score=None��
                            serializable_history.append(item.model_dump(exclude_none=True))
                        elif isinstance(item, dict):
                            serializable_history.append(item)
                        else:
                            # �������ͣ�����ת��Ϊ�ַ���
                            serializable_history.append(str(item))
                    
                    # ��ȡ agent_name
                    agent_name = "unknown"
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        agent_name = execution_engine.config.agent_name

                    # ȷ�� split��transfer ģʽ�� transfer_after_task �� replay ģʽ�Ĳ�������Ϊ "test"
                    split = "train"
                    if training_mode == "transfer":
                        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                        if task_name == transfer_after_task:
                            split = "test"
                    elif training_mode == "replay":
                        if is_replay_test:
                            split = "test"
                        else:
                            # ѵ�����������ӵ���ѧϰ�б�
                            learned_samples_in_replay.append(sample_index)
                            # ȷ����ǰ replay_id��������ѧϰ������������
                            if replay_info:
                                for rid, info in replay_info.items():
                                    if sample_index in info["train"]:
                                        # �ҵ�������ǰ������ replay��ȡ���� replay_id
                                        current_replay_id = max(current_replay_id, rid)
                    
                    # Replay ģʽ�����浽��Ӧ�� replay �ļ���
                    if training_mode == "replay" and replay_info:
                        if is_replay_test:
                            # �������������浽��ǰ replay �� test �ļ���
                            # ȷ����ǰ replay_id�����ݲ������������� replay��
                            for rid, info in replay_info.items():
                                if sample_index in info["test"]:
                                    current_replay_id = rid
                                    break
                            
                            replay_dir = ensure_output_dir(train_output_root / f"replay{current_replay_id}" / "test")
                            task_dir = ensure_output_dir(replay_dir / task_name)
                            out_path = task_dir / f"{sample_index}.json"
                        else:
                            # ѵ�����������浽��ǰ��֮������ replay �� train �ļ���
                            # �ҵ����а�����ǰ������ replay����ǰ��֮������� replay��
                            target_replays = []
                            for rid, info in replay_info.items():
                                if sample_index in info["train"]:
                                    target_replays.append(rid)
                            
                            # ���浽����Ŀ�� replay �� train �ļ���
                            for rid in target_replays:
                                replay_dir = ensure_output_dir(train_output_root / f"replay{rid}" / "train")
                                task_dir = ensure_output_dir(replay_dir / task_name)
                                out_path = task_dir / f"{sample_index}.json"
                                with out_path.open("w", encoding="utf-8") as f:
                                    json.dump({
                                        "task": task_name,
                                        "index": sample_index,
                                        "split": split,
                                        "status": result["status"],
                                        "result": result["result"],
                                        "history": serializable_history,
                                        "agent_name": agent_name,
                                    }, f, indent=2, ensure_ascii=False)
                            
                            # ��¼ִ��˳��ֻ��¼һ�Σ�ʹ�õ�һ�� replay��
                            if target_replays:
                                if task_name not in execution_order_train:
                                    execution_order_train[task_name] = []
                                execution_order_train[task_name].append({
                                    "task": task_name,
                                    "index": sample_index,
                                    "split": split,
                                    "execution_order": len(execution_order_train[task_name]) + 1,
                                    "timestamp": time.time(),
                                    "status": result["status"],
                                })
                            
                            print(f"  -> Completed: status={result['status']} (saved to replay{target_replays[0]}-{target_replays[-1]}/train)")
                            continue  # ���������ı����߼�
                    else:
                        # �� replay ģʽ�� replay_info Ϊ None��ʹ��ԭ���߼�
                        task_dir = ensure_output_dir(train_output_root / task_name)
                        out_path = task_dir / f"{sample_index}.json"
                    
                    # ��������replay ģʽ�Ĳ������������ replay ģʽ��
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "status": result["status"],
                            "result": result["result"],
                            "history": serializable_history,
                            "agent_name": agent_name,
                        }, f, indent=2, ensure_ascii=False)
                    
                    # ��¼ִ��˳�򣨸��� split ѡ���Ӧ�� execution_order �ֵ䣩
                    # Replay ģʽ�Ĳ���������Ҫ������¼
                    if training_mode == "replay" and is_replay_test:
                        # Replay ģʽ�Ĳ�����������¼����ǰ replay ��ִ��˳��
                        if task_name not in execution_order_test:
                            execution_order_test[task_name] = []
                        execution_order_test[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "execution_order": len(execution_order_test[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": result["status"],
                            "replay_id": current_replay_id,
                        })
                    elif split == "test":
                        if task_name not in execution_order_test:
                            execution_order_test[task_name] = []
                        execution_order_test[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "execution_order": len(execution_order_test[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": result["status"],
                        })
                    else:
                        if task_name not in execution_order_train:
                            execution_order_train[task_name] = []
                        execution_order_train[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "execution_order": len(execution_order_train[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": result["status"],
                        })

                    print(f"  -> Completed: status={result['status']}")

                    # ǰ��Ǩ�Ʋ��ԣ�transfer mode with same task��
                    if training_mode == "transfer":
                        transfer_task = exp_cfg.experiment.get("transfer_task")
                        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                        forward_transfer_num = exp_cfg.experiment.get("forward_transfer_num")

                        # DEBUG: ����ؼ�����
                        print(f"  [Forward Transfer DEBUG] transfer_task={transfer_task}, transfer_after_task={transfer_after_task}, forward_transfer_num={forward_transfer_num}, should_update={should_update_memory_locomo}")

                        # ֻ�е� transfer_task == transfer_after_task �Ҹ����˼���ʱ���Ž���ǰ��Ǩ�Ʋ���
                        if transfer_task == transfer_after_task and should_update_memory_locomo and forward_transfer_num:
                            print(f"  [Forward Transfer] Checking forward test after training on {task_name}[{sample_index}] (forward_num={forward_transfer_num})")
                            # ���� locomo ����Ҳ�� schedule ���������� N ���������� db ����һ�£�
                            # forward_transfer_num ָ���Ǵ����ϵĺ� N ������������ index �ϵ� +N
                            current_position = idx - 1  # idx �� 1 ��ʼ��ת��Ϊ 0-based index

                            # �ӵ�ǰλ��������ҵ� N ���� session ����
                            forward_test_target = None
                            count = 0
                            for i in range(current_position + 1, len(train_schedule)):
                                future_task_name, future_sample_index = train_schedule[i]

                                # ���� session injection marker
                                if future_task_name == SESSION_INJECTION_MARKER:
                                    continue

                                count += 1
                                if count == forward_transfer_num:
                                    forward_test_target = (future_task_name, future_sample_index)
                                    print(f"  [Forward Transfer] Found target at schedule position {i}: {future_task_name}[{future_sample_index}]")
                                    break

                            if not forward_test_target:
                                print(f"  [Forward Transfer] Skipped - not enough future samples (found {count}, need {forward_transfer_num})")

                            # ����ҵ���Ŀ��������ִ��ǰ�����
                            if forward_test_target:
                                test_task_name, test_sample_index = forward_test_target
                                print(f"  [Forward Transfer Test] Testing future sample: {test_task_name}[{test_sample_index}] (forward_num={forward_transfer_num})")

                                try:
                                    # ���� locomo ����ʹ�� LocomoSessionWrapper ִ��ǰ�����
                                    if is_locomo_task(test_task_name) and locomo_task_instance is not None and locomo_task_name == test_task_name:
                                        # ������װ�� Session��ֻ enhance���� update��
                                        test_session = LocomoSessionWrapper(test_sample_index, llm_agent, memory_for_enhance, test_task_name, locomo_task_instance, training_mode)

                                        # ֱ�ӵ�������ʵ���� sync_start_sample
                                        test_task_result = locomo_task_instance.sync_start_sample(test_sample_index, test_session)

                                        # �� test_session.history ����ȡ messages ���ں�������
                                        test_messages = []
                                        for item in test_session.history:
                                            if hasattr(item, 'root') and isinstance(item.root, dict):
                                                msg = item.root
                                                if msg.get("role") in ["system", "user", "assistant"]:
                                                    test_messages.append(msg)
                                            elif isinstance(item, dict):
                                                if item.get("role") in ["system", "user", "assistant"]:
                                                    test_messages.append(item)

                                        # �� history ����ȡ reward
                                        test_reward = 0  # Ĭ�� reward Ϊ 0
                                        for item in test_session.history:
                                            if hasattr(item, 'root'):
                                                # RootModel ���ͣ���� root �Ƿ��� RewardHistoryItem
                                                if hasattr(item.root, 'reward'):
                                                    reward_item = item.root
                                                    # ���ȴ� metrics �е� llm_score ��ȡ reward
                                                    if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                                        llm_score = reward_item.metrics.get("llm_score")
                                                        if llm_score is not None:
                                                            test_reward = float(llm_score)  # llm_score �� 0 �� 1
                                                            break
                                                    # ���û�� metrics��ʹ�� reward �ֶ�
                                                    test_reward = reward_item.reward
                                                    break
                                            elif isinstance(item, dict) and "reward" in item:
                                                # ������ֵ䣬����Ƿ��� metrics
                                                if "metrics" in item and isinstance(item["metrics"], dict):
                                                    llm_score = item["metrics"].get("llm_score")
                                                    if llm_score is not None:
                                                        test_reward = float(llm_score)
                                                        break
                                                test_reward = item["reward"]
                                                break
                                            elif hasattr(item, 'reward'):
                                                # ֱ���� RewardHistoryItem ʵ��
                                                # ���ȴ� metrics �е� llm_score ��ȡ reward
                                                if hasattr(item, 'metrics') and isinstance(item.metrics, dict):
                                                    llm_score = item.metrics.get("llm_score")
                                                    if llm_score is not None:
                                                        test_reward = float(llm_score)
                                                        break
                                                test_reward = item.reward
                                                break

                                        # ����� history ��û���ҵ������Դ� test_task_result.result �е� metrics ��ȡ
                                        if test_reward == 0 and isinstance(test_task_result.result, dict):
                                            metrics = test_task_result.result.get("metrics")
                                            if isinstance(metrics, dict):
                                                llm_score = metrics.get("llm_score")
                                                if llm_score is not None:
                                                    test_reward = float(llm_score)

                                        # ʹ�� test_task_result ��Ϊ result
                                        test_result = {
                                            "status": test_task_result.status.value if hasattr(test_task_result.status, 'value') else str(test_task_result.status),
                                            "result": test_task_result.result,
                                            "reward": test_reward,
                                        }

                                        # �� history ת��Ϊ�����л��ĸ�ʽ
                                        test_history = []
                                        for item in test_session.history:
                                            if hasattr(item, 'root'):
                                                # RootModel ���ͣ���ȡ root ֵ
                                                test_history.append(item.root)
                                            elif hasattr(item, 'model_dump'):
                                                # Pydantic ģ�ͣ�ת��Ϊ�ֵ�
                                                test_history.append(item.model_dump(exclude_none=True))
                                            elif isinstance(item, dict):
                                                test_history.append(item)
                                            else:
                                                # �������ͣ�����ת��Ϊ�ַ���
                                                test_history.append(str(item))
                                    else:
                                        # �� locomo �������������Ӧ�ó����� locomo ��֧��
                                        print(f"  [Forward Transfer Test] -> WARNING: Non-locomo task in locomo branch")
                                        continue

                                    # ����ǰ����Խ����������Ŀ¼
                                    forward_test_dir = ensure_output_dir(base_output_root / "forward_transfer_test" / test_task_name)
                                    test_out_path = forward_test_dir / f"train{sample_index}_test{test_sample_index}.json"

                                    # ȷ�� agent_name �ڽ����
                                    test_agent_name = "unknown"
                                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                                        test_agent_name = execution_engine.config.agent_name

                                    test_output_data = {
                                        "task": test_task_name,
                                        "index": test_sample_index,
                                        "split": "forward_test",
                                        "trained_on_index": sample_index,  # ��¼�����ĸ�����ѵ������Ե�
                                        "forward_num": forward_transfer_num,
                                        "history": test_history,
                                        "result": test_result,
                                        "agent_name": test_agent_name,
                                    }

                                    with test_out_path.open("w", encoding="utf-8") as f:
                                        json.dump(test_output_data, f, ensure_ascii=False, indent=2)

                                    test_status = test_result.get("status", "unknown") if isinstance(test_result, dict) else "unknown"
                                    print(f"  [Forward Transfer Test] -> status={test_status}, saved to {test_out_path.relative_to(ROOT_DIR)}")

                                    # ��¼ǰ��Ǩ�Ʋ��Ե�ִ��˳��
                                    if test_task_name not in execution_order_forward_test:
                                        execution_order_forward_test[test_task_name] = []
                                    execution_order_forward_test[test_task_name].append({
                                        "task": test_task_name,
                                        "index": test_sample_index,
                                        "split": "forward_test",
                                        "trained_on_index": sample_index,
                                        "forward_num": forward_transfer_num,
                                        "execution_order": len(execution_order_forward_test[test_task_name]) + 1,
                                        "timestamp": time.time(),
                                        "status": test_status,
                                    })

                                except Exception as e:
                                    print(f"  [Forward Transfer Test] -> ERROR: {str(e)}")
                                    logging.error(f"Forward transfer test failed for {test_task_name}[{test_sample_index}]: {str(e)}", exc_info=True)

                    continue  # ���������ĺ�˴���
                
                # 6.1 ���� /start_sample����ȡ session_id + ��ʼ messages/tools
                session_id, messages, tools = backend.start_sample(task_name, sample_index)
                print(f"  -> backend returned session_id={session_id}, messages={len(messages)}, tools={len(tools)}")

                # 6.1.1 ���� kg ���񣬹��˵���ʾģ�壬ֻ���� system �����һ�� user ��Ϣ
                if task_name.startswith("kg-") or "kg" in task_name.lower():
                    original_count = len(messages)
                    filtered_messages = []
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    
                    # ������һ�� system ��Ϣ
                    for msg in messages:
                        if msg.get("role") == "system":
                            filtered_messages.append(msg)
                            break
                    
                    # �������һ�� user ��Ϣ�����������⣩
                    if user_messages:
                        filtered_messages.append(user_messages[-1])
                    
                    if filtered_messages:
                        messages = filtered_messages
                        print(f"  -> Filtered kg task messages: {len(messages)} messages (removed {original_count - len(messages)} demo template messages)")

                # 6.2 ͨ�� memory ���Ƹ�д messages
                # offline ģʽ��ʹ�� zero_shot����ʹ�ü��䣩��online ģʽ��ʹ�����õļ������
                # Replay ģʽ��test ������ enhance���� update��ֱ��ʹ�õ�ǰ�ۻ��ļ���

                # ֱ��ʹ�õ�ǰ�� memory_for_enhance�����Խ׶β����¼��䣬����ʹ�õ�ǰ�ۻ��ļ��伴�ɣ�
                test_memory_for_enhance = memory_for_enhance

                # single_agent
                enhanced_messages = test_memory_for_enhance.use_memory(task_name, messages)

                # 6.3 ͨ�� execution engine ִ��
                history, result = execution_engine.run_sample(
                    task=task_name,
                    index=sample_index,
                    session_id=session_id,
                    messages=enhanced_messages,
                    tools=tools,
                    agent_pool=llm_agent,
                    backend_client=backend,
                )

                # 6.3.1 ȷ�� result �м�¼ agent_name������ single_agent Ҳ��¼��
                if isinstance(result, dict):
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        result["agent_name"] = execution_engine.config.agent_name

                # 6.4 ���¼��䣨���� training_mode �����Ƿ���£�
                # Transfer ģʽ��
                #   - ������Ǩ�ƣ�transfer_task != transfer_after_task����transfer_task ���£�transfer_after_task ������
                #   - ǰ��Ǩ�ƣ�transfer_task == transfer_after_task������������������
                # Replay ģʽ��ѵ���������£��������������£�ͨ�� schedule �еı���жϣ�
                should_update_memory = True
                if training_mode == "transfer":
                    transfer_task = exp_cfg.experiment.get("transfer_task")
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    # ֻ���ڿ�����Ǩ���ҵ�ǰ������ transfer_after_task ʱ���Ų����¼���
                    # ǰ��Ǩ��ʱ������������Ӧ�ø��¼���
                    if transfer_task != transfer_after_task and task_name == transfer_after_task:
                        # transfer_after_task �ǲ������񣬲����¼���
                        should_update_memory = False
                    elif task_name == transfer_task:
                        # transfer_task ��ѵ�����񣬸��¼���
                        should_update_memory = True
                elif training_mode == "replay":
                    # replay ģʽ��ͨ�� is_replay_test ��־���ж�
                    if is_replay_test:
                        # ���������������¼���
                        should_update_memory = False
                    else:
                        # ѵ�����������¼���
                        should_update_memory = True

                if should_update_memory:
                    # single_agent: ���� memory
                    memory.update_memory(task_name, history, result)

                # Replay ģʽ���������ԣ�ѧϰ���������Ը�������
                if training_mode == "replay" and should_update_memory and not is_replay_test:
                    print(f"  [Immediate Test] Testing sample {sample_index} immediately after learning...")
                    try:
                        # ��ȡ��ʼmessages�����µ���backend��ȡ�ɾ��ĳ�ʼ״̬��
                        test_session_id, test_messages, test_tools = backend.start_sample(task_name, sample_index)

                        # KG���񣺹�����Ϣ
                        if task_name.startswith("kg-") or "kg" in task_name.lower():
                            test_user_messages = [msg for msg in test_messages if msg.get("role") == "user"]
                            test_filtered_messages = []
                            for msg in test_messages:
                                if msg.get("role") == "system":
                                    test_filtered_messages.append(msg)
                                    break
                            if test_user_messages:
                                test_filtered_messages.append(test_user_messages[-1])
                            if test_filtered_messages:
                                test_messages = test_filtered_messages

                        # ʹ��memory_for_enhance����enhance����update��
                        test_enhanced_messages = memory_for_enhance.use_memory(task_name, test_messages)

                        # ִ�в���
                        test_history, test_result = execution_engine.run_sample(
                            task=task_name,
                            index=sample_index,
                            session_id=test_session_id,
                            messages=test_enhanced_messages,
                            tools=test_tools,
                            agent_pool=llm_agent,
                            backend_client=backend,
                        )

                        # ȷ�� agent_name �ڽ����
                        if isinstance(test_result, dict):
                            if isinstance(execution_engine, SingleAgentExecutionEngine):
                                test_result["agent_name"] = execution_engine.config.agent_name

                        # ת��Ϊ�����л���ʽ
                        test_serializable_history = []
                        for msg in test_history:
                            if isinstance(msg, dict):
                                test_serializable_history.append(msg)
                            elif hasattr(msg, 'model_dump'):
                                test_serializable_history.append(msg.model_dump())
                            else:
                                test_serializable_history.append(str(msg))

                        # ��ȡ agent_name
                        test_agent_name = "unknown"
                        if isinstance(execution_engine, SingleAgentExecutionEngine):
                            test_agent_name = execution_engine.config.agent_name

                        # �����������Խ���� test/ Ŀ¼���� replay1/ ƽ����
                        immediate_test_dir = ensure_output_dir(train_output_root / "test")
                        immediate_test_task_dir = ensure_output_dir(immediate_test_dir / task_name)
                        immediate_test_out_path = immediate_test_task_dir / f"{sample_index}.json"

                        with immediate_test_out_path.open("w", encoding="utf-8") as f:
                            json.dump({
                                "task": task_name,
                                "index": sample_index,
                                "split": "immediate_test",
                                "history": test_serializable_history,
                                "result": test_result,
                                "agent_name": test_agent_name,
                            }, f, ensure_ascii=False, indent=2)

                        # ��¼�� execution_order_test������testĿ¼��execution_order.json��
                        if task_name not in execution_order_test:
                            execution_order_test[task_name] = []
                        test_status = test_result.get("status", "unknown") if isinstance(test_result, dict) else "unknown"
                        execution_order_test[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": "immediate_test",
                            "execution_order": len(execution_order_test[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": test_status,
                        })

                        print(f"  [Immediate Test] -> Completed: status={test_status}")
                    except Exception as e:
                        print(f"  [Immediate Test] -> ERROR: {str(e)}")
                        logging.error(f"Immediate test failed for {task_name}[{sample_index}]: {str(e)}", exc_info=True)

                # 6.4.1 ǰ��Ǩ�Ʋ��ԣ�transfer mode with same task��
                if training_mode == "transfer":
                    transfer_task = exp_cfg.experiment.get("transfer_task")
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    forward_transfer_num = exp_cfg.experiment.get("forward_transfer_num")

                    # ֻ�е� transfer_task == transfer_after_task �Ҹ����˼���ʱ���Ž���ǰ��Ǩ�Ʋ���
                    if transfer_task == transfer_after_task and should_update_memory and forward_transfer_num:
                        print(f"  [Forward Transfer] Checking forward test after training on {task_name}[{sample_index}] (forward_num={forward_transfer_num})")
                        # ����ǰ����Ե�Ŀ������
                        # ��Ҫ�� train_schedule ���ҵ���ǰ������λ�ã�Ȼ�������� forward_transfer_num ���� session ������
                        current_position = idx - 1  # idx �� 1 ��ʼ��ת��Ϊ 0-based index

                        # �ӵ�ǰλ��������ҵ� N ���� session ����
                        forward_test_target = None
                        count = 0
                        for i in range(current_position + 1, len(train_schedule)):
                            future_task_name, future_sample_index = train_schedule[i]

                            # ���� session injection marker
                            if future_task_name == SESSION_INJECTION_MARKER:
                                continue

                            count += 1
                            if count == forward_transfer_num:
                                forward_test_target = (future_task_name, future_sample_index)
                                print(f"  [Forward Transfer] Found target at schedule position {i}: {future_task_name}[{future_sample_index}]")
                                break

                        if not forward_test_target:
                            print(f"  [Forward Transfer] Skipped - not enough future samples (found {count}, need {forward_transfer_num})")

                        # ����ҵ���Ŀ��������ִ��ǰ�����
                        if forward_test_target:
                            test_task_name, test_sample_index = forward_test_target
                            print(f"  [Forward Transfer Test] Testing future sample: {test_task_name}[{test_sample_index}] (forward_num={forward_transfer_num})")

                            try:
                                # ���� db �� system memory ������Ҫ�ֶ����� backend
                                # 1. ���� backend.start_sample ��ȡ session_id, messages, tools
                                test_session_id, test_messages, test_tools = backend.start_sample(test_task_name, test_sample_index)

                                # 2. ͨ�� memory ������ǿ messages��ֻ enhance���� update��
                                test_enhanced_messages = memory_for_enhance.use_memory(test_task_name, test_messages)

                                # 3. ͨ�� execution engine ִ��
                                test_history, test_result = execution_engine.run_sample(
                                    task=test_task_name,
                                    index=test_sample_index,
                                    session_id=test_session_id,
                                    messages=test_enhanced_messages,
                                    tools=test_tools,
                                    agent_pool=llm_agent,
                                    backend_client=backend,
                                )

                                # ����ǰ����Խ����������Ŀ¼
                                forward_test_dir = ensure_output_dir(base_output_root / "forward_transfer_test" / test_task_name)
                                test_out_path = forward_test_dir / f"train{sample_index}_test{test_sample_index}.json"

                                # ȷ�� agent_name �ڽ����
                                test_agent_name = None
                                if isinstance(test_result, dict):
                                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                                        test_result["agent_name"] = execution_engine.config.agent_name
                                    test_agent_name = test_result.get("agent_name")

                                test_output_data = {
                                    "task": test_task_name,
                                    "index": test_sample_index,
                                    "split": "forward_test",
                                    "trained_on_index": sample_index,  # ��¼�����ĸ�����ѵ������Ե�
                                    "forward_num": forward_transfer_num,
                                    "history": test_history,
                                    "result": test_result,
                                }
                                if test_agent_name:
                                    test_output_data["agent_name"] = test_agent_name

                                with test_out_path.open("w", encoding="utf-8") as f:
                                    json.dump(test_output_data, f, ensure_ascii=False, indent=2)

                                test_status = test_result.get("status", "unknown") if isinstance(test_result, dict) else "unknown"
                                print(f"  [Forward Transfer Test] -> status={test_status}, saved to {test_out_path.relative_to(ROOT_DIR)}")

                                # ��¼ǰ��Ǩ�Ʋ��Ե�ִ��˳��
                                if test_task_name not in execution_order_forward_test:
                                    execution_order_forward_test[test_task_name] = []
                                execution_order_forward_test[test_task_name].append({
                                    "task": test_task_name,
                                    "index": test_sample_index,
                                    "split": "forward_test",
                                    "trained_on_index": sample_index,
                                    "forward_num": forward_transfer_num,
                                    "execution_order": len(execution_order_forward_test[test_task_name]) + 1,
                                    "timestamp": time.time(),
                                    "status": test_status,
                                })

                            except Exception as e:
                                print(f"  [Forward Transfer Test] -> ERROR: {str(e)}")
                                logging.error(f"Forward transfer test failed for {test_task_name}[{test_sample_index}]: {str(e)}", exc_info=True)

                # 6.5 ���̣����� training_mode ���� split ��Ŀ¼��
                # ȷ�� agent_name �ڶ��㣨�� result ����ȡ��������ڣ�
                agent_name = None
                if isinstance(result, dict):
                    agent_name = result.get("agent_name")
                
                # ȷ�� split��transfer ģʽ�� transfer_after_task �� replay ģʽ�Ĳ�������Ϊ "test"
                split = "train"
                if training_mode == "transfer":
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    if task_name == transfer_after_task:
                        split = "test"
                elif training_mode == "replay":
                    if is_replay_test:
                        split = "test"
                    else:
                        # ѵ�����������ӵ���ѧϰ�б�
                        learned_samples_in_replay.append(sample_index)
                        # ȷ����ǰ replay_id��������ѧϰ������������
                        if replay_info:
                            for rid, info in replay_info.items():
                                if sample_index in info["train"]:
                                    # �ҵ�������ǰ������ replay��ȡ���� replay_id
                                    current_replay_id = max(current_replay_id, rid)
                        
                        # Replay ģʽ������Ƿ������ĳ�� replay ������ѵ������
                        # ��ɺ���� current_replay_id_for_test������ȷ����������Ӧ�ñ��浽�ĸ� replay
                        if replay_info:
                            # �ҵ���ǰ������������� replay_id
                            current_sample_replay_id = 0
                            for rid, info in replay_info.items():
                                if sample_index in info["train"]:
                                    current_sample_replay_id = max(current_sample_replay_id, rid)
                            
                            # ��鵱ǰ���������� replay ������ѵ�������Ƿ������
                            if current_sample_replay_id > 0:
                                info = replay_info[current_sample_replay_id]
                                train_samples = set(info["train"])
                                # ��� learned_samples_in_replay �Ƿ�����˸� replay ������ѵ������
                                if train_samples.issubset(set(learned_samples_in_replay)):
                                    # �� replay ������ѵ������������ɣ����� current_replay_id_for_test
                                    # ��һ�����Խ׶�Ӧ�ñ��浽��� replay �� test �ļ���
                                    if current_replay_id_for_test < current_sample_replay_id:
                                        current_replay_id_for_test = current_sample_replay_id
                                        print(f"[Replay] Completed all training samples for replay{current_sample_replay_id}, current_replay_id_for_test={current_replay_id_for_test}")
                
                # Replay ģʽ�����浽��Ӧ�� replay �ļ���
                if training_mode == "replay" and replay_info:
                    if is_replay_test:
                        # ����������ֻ���浽��ǰ replay �� test �ļ��У�ʹ�� current_replay_id��
                        # ע�⣺һ�������������ܳ����ڶ�� replay �� test �б��У���ִ��ʱӦ��ֻ���浽��ǰ replay
                        if current_replay_id > 0:
                            replay_dir = ensure_output_dir(train_output_root / f"replay{current_replay_id}" / "test")
                            task_dir = ensure_output_dir(replay_dir / task_name)
                            out_path = task_dir / f"{sample_index}.json"
                            
                            output_data = {
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                                "history": history,
                                "result": result,
                            }
                            if agent_name:
                                output_data["agent_name"] = agent_name
                            
                            with out_path.open("w", encoding="utf-8") as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                            
                            # ��¼ִ��˳��
                            if task_name not in execution_order_test:
                                execution_order_test[task_name] = []
                            execution_order_test[task_name].append({
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                                "execution_order": len(execution_order_test[task_name]) + 1,
                                "timestamp": time.time(),
                                "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                            })
                            
                            print(f"  -> Completed: status={result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'} (saved to replay{current_replay_id}/test)")
                        else:
                            print(f"[Replay] Test sample {sample_index} has invalid current_replay_id={current_replay_id}, skipping save")
                        continue  # ���������ı����߼�
                    else:
                        # ѵ�����������浽��ǰ��֮������ replay �� train �ļ���
                        # �ҵ����а�����ǰ������ replay����ǰ��֮������� replay��
                        target_replays = []
                        for rid, info in replay_info.items():
                            if sample_index in info["train"]:
                                target_replays.append(rid)
                        
                        # ���浽����Ŀ�� replay �� train �ļ���
                        for rid in target_replays:
                            replay_dir = ensure_output_dir(train_output_root / f"replay{rid}" / "train")
                            task_dir = ensure_output_dir(replay_dir / task_name)
                            out_path = task_dir / f"{sample_index}.json"
                            
                            output_data = {
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                            "history": history,
                            "result": result,
                            }
                            if agent_name:
                                output_data["agent_name"] = agent_name
                            
                            with out_path.open("w", encoding="utf-8") as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                        
                        # ��¼ִ��˳��ֻ��¼һ�Σ�ʹ�õ�һ�� replay��
                        if target_replays:
                            if task_name not in execution_order_train:
                                execution_order_train[task_name] = []
                            execution_order_train[task_name].append({
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                                "execution_order": len(execution_order_train[task_name]) + 1,
                                "timestamp": time.time(),
                                "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                            })
                        
                        print(f"  -> Completed: status={result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'} (saved to replay{target_replays[0]}-{target_replays[-1]}/train)")
                        continue  # ���������ı����߼�
                else:
                    # �� replay ģʽ�� replay_info Ϊ None��ʹ��ԭ���߼�
                    task_dir = ensure_output_dir(train_output_root / task_name)
                    out_path = task_dir / f"{sample_index}.json"
                
                # ��������replay ģʽ�Ĳ������������ replay ģʽ��
                output_data = {
                    "task": task_name,
                    "index": sample_index,
                    "split": split,
                            "history": history,
                            "result": result,
                }
                if agent_name:
                    output_data["agent_name"] = agent_name
                
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # ��¼ִ��˳�򣨸��� split ѡ���Ӧ�� execution_order �ֵ䣩
                # Replay ģʽ�Ĳ���������Ҫ������¼
                if training_mode == "replay" and is_replay_test:
                    # Replay ģʽ�Ĳ�����������¼����ǰ replay ��ִ��˳��
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                        "replay_id": current_replay_id,
                    })
                elif split == "test":
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                    })
                else:
                    if task_name not in execution_order_train:
                        execution_order_train[task_name] = []
                    execution_order_train[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_train[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                    })

                # 6.6 ��� agent ��Ϣ
                agent_info = ""
                if isinstance(result, dict) and "agent_name" in result:
                    agent_name = result.get("agent_name", "unknown")
                    agent_info = f", agent={agent_name}"

                print(f"  -> saved to {out_path.relative_to(ROOT_DIR)}{agent_info}\n")

            except Exception as e:
                # ���������쳣����¼���󵫼���������һ������
                error_msg = f"  -> ERROR: Failed to process sample {sample_index} of task {task_name}: {str(e)}"
                print(error_msg)
                logging.error(error_msg, exc_info=True)
                
                # ȷ�� split��transfer ģʽ�� transfer_after_task �� replay ģʽ�Ĳ�������Ϊ "test"
                split = "train"
                if training_mode == "transfer":
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    if task_name == transfer_after_task:
                        split = "test"
                elif training_mode == "replay":
                    if is_replay_test:
                        split = "test"
                
                # ��ѡ�����������Ϣ���ļ�
                task_dir = ensure_output_dir(train_output_root / task_name)
                error_path = task_dir / f"{sample_index}.error.json"
                with error_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                
                # ��¼ִ��˳�򣨼�ʹ����Ҳ��¼������ split ѡ���Ӧ�� execution_order �ֵ䣩
                if split == "test":
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e),
                    })
                else:
                    if task_name not in execution_order_train:
                        execution_order_train[task_name] = []
                    execution_order_train[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_train[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e),
                    })
                
                print(f"  -> error saved to {error_path.relative_to(ROOT_DIR)}\n")
                continue  # ������ǰ������������һ��

            # Ϊ�˸����գ�����֮��Ҳ��΢ͣ��һ�£���ȫ����ִ�У�
            time.sleep(1.0)

    # 7) ִ�в��Լ�������������ڣ�
    if test_schedule:
        print(f"\n{'='*60}")
        print(f"Running TEST set: {len(test_schedule)} samples")
        print(f"{'='*60}\n")
        
        # offline ģʽ������ locomo ���񣬲��Խ׶�Ҳ��Ҫ shuffle��������ã�
        if training_mode == "offline":
            # ����Ƿ��� locomo ����
            locomo_tasks_in_test = [name for name, _ in test_schedule if is_locomo_task(name)]
            if locomo_tasks_in_test and shuffle_enabled:
                import random
                shuffle_seed = exp_cfg.experiment.get("shuffle", {}).get("seed", None)
                if shuffle_seed is not None:
                    random.seed(shuffle_seed)
                random.shuffle(test_schedule)
                print(f"  -> Shuffled {len(test_schedule)} test QAs for locomo task (offline mode)")

        # offline ģʽ�����Լ�ʹ�����õ� memory mechanism ���� use_memory
        if training_mode == "offline":
            print(f"Training mode: {training_mode} -> Test set will use {exp_cfg.memory_mechanism.get('name', 'zero_shot')} for use_memory (memory enabled for testing)")
            # ���¹��� memory_for_enhance��ʹ�����õ� memory mechanism
            # single_agent
            memory_for_enhance = memory

        for idx, (task_name, sample_index) in enumerate(test_schedule, start=1):
            # ���� session ע���ǣ����ڻ�ϵ��ȣ�
            # ע�⣺�� offline ģʽ�� test �׶Σ�session �Ѿ��� train �׶�ע�룬��������Ӧ������
            if task_name == SESSION_INJECTION_MARKER:
                print(f"[TEST {idx}/{len(test_schedule)}] [SESSION INJECTION] Skipping session injection in test phase (offline mode)")
                continue  # ����ִ�У�������һ������

            if not cross_task and last_task_name is not None and task_name != last_task_name:
                memory, memory_for_enhance = build_memory_bundle()
                # offline ģʽ�����Լ�ʹ�����õ� memory mechanism ���� use_memory
                if training_mode == "offline":
                    memory_for_enhance = memory
                print(f"\n[Memory Reset] cross_task=False, switched task {last_task_name} -> {task_name}, memory rebuilt (test split).\n")
            last_task_name = task_name
            print(f"[TEST {idx}/{len(test_schedule)}] task={task_name}, index={sample_index}")

            try:
                # ���� locomo ����ֱ��ʹ������ʵ��������Ҫ���
                if is_locomo_task(task_name) and locomo_task_instance is not None and locomo_task_name == task_name:
                    # ������װ�� Session
                    session = LocomoSessionWrapper(sample_index, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode)
                    
                    # ֱ�ӵ�������ʵ���� sync_start_sample
                    task_result = locomo_task_instance.sync_start_sample(sample_index, session)
                    
                    # �� session.history ����ȡ messages ���ں�������
                    messages = []
                    for item in session.history:
                        if hasattr(item, 'root') and isinstance(item.root, dict):
                            msg = item.root
                            if msg.get("role") in ["system", "user", "assistant"]:
                                messages.append(msg)
                        elif isinstance(item, dict):
                            if item.get("role") in ["system", "user", "assistant"]:
                                messages.append(item)
                    
                    # �� history ����ȡ reward������ previous_sample_utilization �ȼ�����ƣ�
                    # ���� locomo ����reward ���� llm_score ����
                    reward = 0  # Ĭ�� reward Ϊ 0
                    for item in session.history:
                        if hasattr(item, 'root'):
                            # RootModel ���ͣ���� root �Ƿ��� RewardHistoryItem
                            if hasattr(item.root, 'reward'):
                                reward_item = item.root
                                # ���ȴ� metrics �е� llm_score ��ȡ reward
                                if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                    llm_score = reward_item.metrics.get("llm_score")
                                    if llm_score is not None:
                                        reward = float(llm_score)  # llm_score �� 0 �� 1
                                        break
                                # ���û�� metrics��ʹ�� reward �ֶ�
                                reward = reward_item.reward
                                break
                        elif isinstance(item, dict) and "reward" in item:
                            # ������ֵ䣬����Ƿ��� metrics
                            if "metrics" in item and isinstance(item["metrics"], dict):
                                llm_score = item["metrics"].get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item["reward"]
                            break
                        elif hasattr(item, 'reward'):
                            # ֱ���� RewardHistoryItem ʵ��
                            # ���ȴ� metrics �е� llm_score ��ȡ reward
                            if hasattr(item, 'metrics') and isinstance(item.metrics, dict):
                                llm_score = item.metrics.get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item.reward
                            break
                    
                    # ����� history ��û���ҵ������Դ� task_result.result �е� metrics ��ȡ
                    if reward == 0 and isinstance(task_result.result, dict):
                        metrics = task_result.result.get("metrics")
                        if isinstance(metrics, dict):
                            llm_score = metrics.get("llm_score")
                            if llm_score is not None:
                                reward = float(llm_score)
                    
                    # ʹ�� task_result ��Ϊ result������ reward �ֶ��Ա�������ʶ��
                    result = {
                        "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                        "result": task_result.result,
                        "reward": reward,  # ���� reward �ֶΣ����� previous_sample_utilization �ȼ������ʶ��
                    }
                    
                    # ���Լ����� offline ģʽ�²����¼��䣨ֻ���������� online ģʽ�¸��¼���
                    history = session.history
                    if training_mode == "offline":
                        # offline ģʽ�����Լ������¼��䣨ֻ�������ܣ�
                        pass
                    else:
                        # online ģʽ�����Լ�Ҳ���¼���
                        if isinstance(memory, dict):
                            for agent_mem in memory.values():
                                agent_mem.update_memory(task_name, history, result)
                        else:
                            memory.update_memory(task_name, history, result)
                    
                    # ������
                    # �� history ת��Ϊ�����л��ĸ�ʽ
                    serializable_history = []
                    for item in history:
                        if hasattr(item, 'root'):
                            # RootModel ���ͣ���ȡ root ֵ
                            serializable_history.append(item.root)
                        elif hasattr(item, 'model_dump'):
                            # Pydantic ģ�ͣ�ת��Ϊ�ֵ�
                            # ʹ�� exclude_none=True �ų� None ֵ���� score=None��
                            serializable_history.append(item.model_dump(exclude_none=True))
                        elif isinstance(item, dict):
                            serializable_history.append(item)
                        else:
                            # �������ͣ�����ת��Ϊ�ַ���
                            serializable_history.append(str(item))

                    # ��ȡ agent_name
                    agent_name = "unknown"
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        agent_name = execution_engine.config.agent_name

                    task_dir = ensure_output_dir(test_output_root / task_name)
                    out_path = task_dir / f"{sample_index}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump({
                            "task": task_name,
                            "index": sample_index,
                            "split": "test",
                            "status": result["status"],
                            "result": result["result"],
                            "history": serializable_history,
                            "agent_name": agent_name,
                        }, f, indent=2, ensure_ascii=False)
                    
                    # ��¼ִ��˳��
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": "test",
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result["status"],
                    })
                    
                    print(f"  -> Completed: status={result['status']}")
                    continue  # ���������ĺ�˴���
                
                # 7.1 ���� /start_sample����ȡ session_id + ��ʼ messages/tools
                session_id, messages, tools = backend.start_sample(task_name, sample_index)
                print(f"  -> backend returned session_id={session_id}, messages={len(messages)}, tools={len(tools)}")

                # 7.1.1 ���� kg ���񣬹��˵���ʾģ�壬ֻ���� system �����һ�� user ��Ϣ
                if task_name.startswith("kg-") or "kg" in task_name.lower():
                    original_count = len(messages)
                    filtered_messages = []
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    
                    # ������һ�� system ��Ϣ
                    for msg in messages:
                        if msg.get("role") == "system":
                            filtered_messages.append(msg)
                            break
                    
                    # �������һ�� user ��Ϣ�����������⣩
                    if user_messages:
                        filtered_messages.append(user_messages[-1])
                    
                    if filtered_messages:
                        messages = filtered_messages
                        print(f"  -> Filtered kg task messages: {len(messages)} messages (removed {original_count - len(messages)} demo template messages)")

                # 7.2 ͨ�� memory ���Ƹ�д messages�����Լ�Ҳʹ�� memory���������£�
                # single_agent
                enhanced_messages = memory_for_enhance.use_memory(task_name, messages)

                # 7.3 ͨ�� execution engine ִ��
                history, result = execution_engine.run_sample(
                    task=task_name,
                    index=sample_index,
                    session_id=session_id,
                    messages=enhanced_messages,
                    tools=tools,
                    agent_pool=llm_agent,
                    backend_client=backend,
                )

                # 7.3.1 ȷ�� result �м�¼ agent_name������ single_agent Ҳ��¼��
                if isinstance(result, dict):
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        result["agent_name"] = execution_engine.config.agent_name

                # 7.4 ���Լ����� offline ģʽ�²����¼��䣨ֻ���������� online ģʽ�¸��¼���
                if training_mode == "offline":
                    # offline ģʽ�����Լ������¼��䣨ֻ�������ܣ�
                    pass
                else:
                    # online ģʽ�����Լ�Ҳ���¼���
                    memory.update_memory(task_name, history, result)

                # 7.5 ���̵� test Ŀ¼
                # ȷ�� agent_name �ڶ��㣨�� result ����ȡ��������ڣ�
                agent_name = None
                if isinstance(result, dict):
                    agent_name = result.get("agent_name")
                
                task_dir = ensure_output_dir(test_output_root / task_name)
                out_path = task_dir / f"{sample_index}.json"
                output_data = {
                            "task": task_name,
                            "index": sample_index,
                            "split": "test",
                            "history": history,
                            "result": result,
                }
                if agent_name:
                    output_data["agent_name"] = agent_name
                
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # ��¼ִ��˳��
                if task_name not in execution_order_test:
                    execution_order_test[task_name] = []
                execution_order_test[task_name].append({
                    "task": task_name,
                    "index": sample_index,
                    "split": "test",
                    "execution_order": len(execution_order_test[task_name]) + 1,
                    "timestamp": time.time(),
                    "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                })

                # 7.6 ��� agent ��Ϣ
                agent_info = ""
                if isinstance(result, dict) and "agent_name" in result:
                    agent_name = result.get("agent_name", "unknown")
                    agent_info = f", agent={agent_name}"

                print(f"  -> saved to {out_path.relative_to(ROOT_DIR)}{agent_info}\n")

            except Exception as e:
                # ���������쳣����¼���󵫼���������һ������
                error_msg = f"  -> ERROR: Failed to process sample {sample_index} of task {task_name}: {str(e)}"
                print(error_msg)
                logging.error(error_msg, exc_info=True)
                
                # ��ѡ�����������Ϣ���ļ�
                task_dir = ensure_output_dir(test_output_root / task_name)
                error_path = task_dir / f"{sample_index}.error.json"
                with error_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task": task_name,
                            "index": sample_index,
                            "split": "test",
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                
                # ��¼ִ��˳�򣨼�ʹ����Ҳ��¼��
                if task_name not in execution_order_test:
                    execution_order_test[task_name] = []
                execution_order_test[task_name].append({
                    "task": task_name,
                    "index": sample_index,
                    "split": "test",
                    "execution_order": len(execution_order_test[task_name]) + 1,
                    "timestamp": time.time(),
                    "status": "error",
                    "error": str(e),
                })
                
                print(f"  -> error saved to {error_path.relative_to(ROOT_DIR)}\n")
                continue  # ������ǰ������������һ��

            # Ϊ�˸����գ�����֮��Ҳ��΢ͣ��һ�£���ȫ����ִ�У�
            time.sleep(1.0)

    # 8) ����ִ��˳���ļ�
    # onlineģʽ������һ��Ŀ¼��base_output_root������ execution_order.json���������������ִ��˳��
    # offlineģʽ����train��testĿ¼�·ֱ𱣴� execution_order.json���������������ִ��˳��
    # transferģʽ����transfer_train��forward_transfer_testĿ¼�·ֱ𱣴� execution_order.json
    if execution_order_train:
        # �ϲ����������ִ��˳�򣬰� timestamp ����
        all_train_orders = []
        for task_name, order_list in execution_order_train.items():
            all_train_orders.extend(order_list)
        # �� timestamp ����
        all_train_orders.sort(key=lambda x: x.get("timestamp", 0))
        # ���·��� execution_order��ȫ��˳��
        for idx, order_item in enumerate(all_train_orders, start=1):
            order_item["execution_order"] = idx

        if test_schedule:
            # offlineģʽ����trainĿ¼�±��棨������������
            order_path = train_output_root / "execution_order.json"
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(all_train_orders, f, indent=2, ensure_ascii=False)
            print(f"[Execution Order] Saved train execution order: {len(all_train_orders)} samples from {len(execution_order_train)} task(s) -> {order_path.relative_to(ROOT_DIR)}")
        else:
            # online/transferģʽ���ڵ�ǰĿ¼����
            if training_mode == "transfer":
                # transfer ǰ��Ǩ��ģʽ��train_output_root �Ѿ��� transfer_train Ŀ¼
                order_path = train_output_root / "execution_order.json"
            else:
                # onlineģʽ������һ��Ŀ¼��base_output_root������
                order_path = base_output_root / "execution_order.json"
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(all_train_orders, f, indent=2, ensure_ascii=False)
            print(f"[Execution Order] Saved execution order: {len(all_train_orders)} samples from {len(execution_order_train)} task(s) -> {order_path.relative_to(ROOT_DIR)}")
    
    if execution_order_test:
        # offlineģʽ����testĿ¼�±��棨������������
        # replayģʽ���������Ա��浽test/Ŀ¼��replay���Ա��浽���Ե�replayX/test/Ŀ¼
        # �ϲ����������ִ��˳�򣬰� timestamp ����
        all_test_orders = []
        for task_name, order_list in execution_order_test.items():
            all_test_orders.extend(order_list)
        # �� timestamp ����
        all_test_orders.sort(key=lambda x: x.get("timestamp", 0))
        # ���·��� execution_order��ȫ��˳��
        for idx, order_item in enumerate(all_test_orders, start=1):
            order_item["execution_order"] = idx

        # replayģʽ����Ҫ�ֱ𱣴��������Ժ�replay���Ե�execution_order
        if training_mode == "replay":
            # �����������Ժ�replay����
            immediate_test_orders = [o for o in all_test_orders if o.get("split") == "immediate_test"]
            replay_test_orders = [o for o in all_test_orders if o.get("split") == "test" and "replay_id" in o]

            # �����������Ե�execution_order��test/Ŀ¼
            if immediate_test_orders:
                # ���·���execution_order
                for idx, order_item in enumerate(immediate_test_orders, start=1):
                    order_item["execution_order"] = idx
                immediate_test_order_path = train_output_root / "test" / "execution_order.json"
                ensure_output_dir(train_output_root / "test")
                with immediate_test_order_path.open("w", encoding="utf-8") as f:
                    json.dump(immediate_test_orders, f, indent=2, ensure_ascii=False)
                print(f"[Execution Order] Saved immediate test execution order: {len(immediate_test_orders)} samples -> {immediate_test_order_path.relative_to(ROOT_DIR)}")

            # ����replay���Ե�execution_order�����Ե�replayX/test/Ŀ¼
            if replay_test_orders:
                # ��replay_id����
                replay_groups = {}
                for order_item in replay_test_orders:
                    replay_id = order_item.get("replay_id")
                    if replay_id not in replay_groups:
                        replay_groups[replay_id] = []
                    replay_groups[replay_id].append(order_item)

                # Ϊÿ��replay����execution_order
                for replay_id, orders in replay_groups.items():
                    # ���·���execution_order
                    for idx, order_item in enumerate(orders, start=1):
                        order_item["execution_order"] = idx
                    replay_test_order_path = train_output_root / f"replay{replay_id}" / "test" / "execution_order.json"
                    ensure_output_dir(train_output_root / f"replay{replay_id}" / "test")
                    with replay_test_order_path.open("w", encoding="utf-8") as f:
                        json.dump(orders, f, indent=2, ensure_ascii=False)
                    print(f"[Execution Order] Saved replay{replay_id} test execution order: {len(orders)} samples -> {replay_test_order_path.relative_to(ROOT_DIR)}")
        else:
            # ��replayģʽ��ʹ��ԭ���߼�
            # ��� test_output_root Ϊ None��ʹ�� train_output_root
            if test_output_root is not None:
                order_path = test_output_root / "execution_order.json"
            else:
                order_path = train_output_root / "execution_order_test.json"
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(all_test_orders, f, indent=2, ensure_ascii=False)
            print(f"[Execution Order] Saved test execution order: {len(all_test_orders)} samples from {len(execution_order_test)} task(s) -> {order_path.relative_to(ROOT_DIR)}")


    # Transfer ģʽ������ forward_transfer_test ��ִ��˳��
    if execution_order_forward_test:
        # �ϲ����������ִ��˳�򣬰� timestamp ����
        all_forward_test_orders = []
        for task_name, order_list in execution_order_forward_test.items():
            all_forward_test_orders.extend(order_list)
        # �� timestamp ����
        all_forward_test_orders.sort(key=lambda x: x.get("timestamp", 0))
        # ���·��� execution_order��ȫ��˳��
        for idx, order_item in enumerate(all_forward_test_orders, start=1):
            order_item["execution_order"] = idx

        # ���浽 base_output_root / forward_transfer_test Ŀ¼�����ļ�����λ��һ�£�
        forward_test_order_path = base_output_root / "forward_transfer_test" / "execution_order.json"
        ensure_output_dir(base_output_root / "forward_transfer_test")
        with forward_test_order_path.open("w", encoding="utf-8") as f:
            json.dump(all_forward_test_orders, f, indent=2, ensure_ascii=False)
        print(f"[Execution Order] Saved forward transfer test execution order: {len(all_forward_test_orders)} samples from {len(execution_order_forward_test)} task(s) -> {forward_test_order_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
