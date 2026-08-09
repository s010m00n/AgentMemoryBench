from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from memory.base import MemoryMechanism, parse_llm_json_response
from src.utils.message_schema import (
    assert_memory_injection_position,
    enhance_messages_with_memory,
    extract_message_info,
    extract_original_question,
)


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"


def _extract_api_key(headers: Dict[str, Any]) -> str:
    auth_value = str(headers.get("Authorization", "") or "").strip()
    if auth_value.lower().startswith("bearer "):
        return auth_value[7:].strip()
    raise ValueError("Authorization header missing Bearer token in llmapi config")


def _normalize_base_url(url: str) -> str:
    normalized = str(url or "").rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized + "/"


@dataclass
class SkillNudgeConfig:
    selector_model_name: str
    manager_model_name: str
    selector_prompt_path: Path
    manager_prompt_path: Path
    skill_storage_path: Path
    nudge_interval: int
    max_manager_steps: int
    max_selected_skills: int
    max_catalog_size: int
    prompt_template: str
    few_shot_enabled: bool
    few_shot_top_k: int
    few_shot_order: str
    few_shot_embedding_model: str
    few_shot_seed: int
    few_shot_storage_path: Path
    few_shot_success_only: bool
    few_shot_reward_bigger_than_zero: bool
    few_shot_include_reasoning_content: bool
    few_shot_prompt_template: str
    where: str
    session_injection_only: bool
    update_success_only: bool
    update_reward_bigger_than_zero: bool
    request_timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 2.0
    retry_backoff: float = 2.0


class SkillNudgeMemory(MemoryMechanism):
    def __init__(self, config: SkillNudgeConfig) -> None:
        self.config = config
        self.template_title = self.config.prompt_template.split("{skills}")[0].strip()
        self.selector_prompt = self._read_text(self.config.selector_prompt_path)
        self.manager_prompt = self._read_text(self.config.manager_prompt_path)
        self.config.skill_storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.skill_storage_path.exists():
            self.config.skill_storage_path.touch()
        self.config.few_shot_storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.few_shot_storage_path.exists():
            self.config.few_shot_storage_path.touch()

        self.selector_cfg = self._load_llm_config(self.config.selector_model_name)
        self.manager_cfg = self._load_llm_config(self.config.manager_model_name)
        self._samples_since_last_nudge = 0
        self._review_window: List[Dict[str, Any]] = []
        self._few_shot_rag: Optional[Any] = None
        self._init_fewshot_rag_if_needed()

    @staticmethod
    def _print(message: str) -> None:
        print(f"[SkillNudge] {message}")

    @staticmethod
    def _read_text(path: Path) -> str:
        with path.open("r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @staticmethod
    def _load_llm_config(model_name: str) -> Dict[str, Any]:
        agent_cfg_path = LLMAPI_DIR / "function_agent.yaml"
        api_cfg_path = LLMAPI_DIR / "function_api.yaml"

        with agent_cfg_path.open("r", encoding="utf-8") as f:
            agents_cfg = yaml.safe_load(f) or {}
        if model_name not in agents_cfg:
            raise ValueError(f"Model '{model_name}' not found in {agent_cfg_path}")

        with api_cfg_path.open("r", encoding="utf-8") as f:
            api_cfg = yaml.safe_load(f) or {}

        base_params = api_cfg.get("parameters", {}) or {}
        agent_params = (agents_cfg.get(model_name) or {}).get("parameters", {}) or {}

        body = dict(base_params.get("body", {}) or {})
        body.update(agent_params.get("body", {}) or {})

        url = base_params.get("url")
        if not url:
            raise ValueError("URL not found in function_api.yaml")

        headers = dict(base_params.get("headers", {}) or {})
        headers.update(agent_params.get("headers", {}) or {})

        api_key = _extract_api_key(headers)
        base_url = _normalize_base_url(url)
        return {"base_url": base_url, "api_key": api_key, "body": body}

    def _call_llm(
        self,
        cfg: Dict[str, Any],
        messages: List[Dict[str, Any]],
        purpose: str,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        body = {
            **(cfg.get("body", {}) or {}),
            "messages": messages,
            **(request_overrides or {}),
        }

        attempt = 0
        delay = self.config.retry_delay
        while True:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.config.request_timeout,
                )
                response.raise_for_status()
                payload = response.json() if response.content else {}
                choices = payload.get("choices") or []
                if not choices:
                    return None
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("text"):
                            parts.append(str(item["text"]))
                    return "\n".join(parts).strip()
                return str(content or "").strip()
            except Exception as exc:
                attempt += 1
                if self.config.max_retries >= 0 and attempt > self.config.max_retries:
                    LOGGER.warning("[SkillNudge] %s failed after %s attempts: %s", purpose, attempt, exc)
                    return None
                LOGGER.warning(
                    "[SkillNudge] %s failed on attempt %s: %s; retrying in %.2fs",
                    purpose,
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= self.config.retry_backoff

    def _load_skills(self) -> List[Dict[str, Any]]:
        skills: List[Dict[str, Any]] = []
        if not self.config.skill_storage_path.exists():
            return skills

        with self.config.skill_storage_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict) and item.get("id"):
                    skills.append(item)
        return skills

    def _write_skills(self, skills: List[Dict[str, Any]]) -> None:
        with self.config.skill_storage_path.open("w", encoding="utf-8") as f:
            for skill in skills:
                f.write(json.dumps(skill, ensure_ascii=False) + "\n")

    def _load_fewshot_examples(self) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        if not self.config.few_shot_storage_path.exists():
            return examples

        with self.config.few_shot_storage_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict) and item.get("value"):
                    examples.append(item)
        return examples

    def _append_fewshot_record(self, record: Dict[str, Any]) -> None:
        with self.config.few_shot_storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _init_fewshot_rag_if_needed(self) -> None:
        if not self.config.few_shot_enabled:
            return

        try:
            from memory.streamICL.streamICL import RAG
        except Exception as exc:
            self._print(f"few-shot RAG unavailable: {exc}")
            self._few_shot_rag = None
            return

        try:
            self._few_shot_rag = RAG(
                embedding_model=self.config.few_shot_embedding_model,
                top_k=self.config.few_shot_top_k,
                order=self.config.few_shot_order,
                seed=self.config.few_shot_seed,
            )
            examples = self._load_fewshot_examples()
            loaded = 0
            for item in examples:
                key = str(item.get("key", "") or "").strip()
                value = str(item.get("value", "") or "").strip()
                if not key or not value:
                    continue
                self._few_shot_rag.insert(key=key, value=value)
                loaded += 1
            self._print(
                f"few-shot RAG initialized: order={self.config.few_shot_order}, "
                f"top_k={self.config.few_shot_top_k}, loaded={loaded}"
            )
        except Exception as exc:
            self._print(f"few-shot RAG init failed: {exc}")
            self._few_shot_rag = None

    def _build_catalog(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        recent = sorted(
            skills,
            key=lambda x: str(x.get("updated_at", "")),
            reverse=True,
        )
        catalog: List[Dict[str, Any]] = []
        for skill in recent[: self.config.max_catalog_size]:
            catalog.append(
                {
                    "id": str(skill.get("id", "")),
                    "name": str(skill.get("name", "")),
                    "summary": str(skill.get("summary", "")),
                    "polarity": str(skill.get("polarity", "positive")),
                    "tags": list(skill.get("tags", []) or []),
                }
            )
        return catalog

    def _extract_query_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        template_titles = [self.template_title]
        question = extract_original_question(messages, where=self.config.where, template_titles=template_titles)
        if question:
            return str(question).strip()
        return None

    def _format_selected_skills(self, selected_skills: List[Dict[str, Any]]) -> str:
        positive_blocks: List[str] = []
        negative_blocks: List[str] = []
        for skill in selected_skills:
            skill_id = str(skill.get("id", "")).strip()
            name = str(skill.get("name", "")).strip()
            summary = str(skill.get("summary", "")).strip()
            content = str(skill.get("content", "")).strip()
            polarity = str(skill.get("polarity", "positive")).strip().lower()
            tags = ", ".join(str(x) for x in (skill.get("tags", []) or []))

            lines = [f"[{skill_id}] {name}"]
            if summary:
                lines.append(f"When to use: {summary}")
            if tags:
                lines.append(f"Tags: {tags}")
            if content:
                lines.append("Content:")
                lines.append(content)
            block = "\n".join(lines).strip()
            if polarity == "negative":
                negative_blocks.append(block)
            else:
                positive_blocks.append(block)

        sections: List[str] = []
        if positive_blocks:
            sections.append("Relevant positive skills:\n" + "\n\n".join(positive_blocks))
        if negative_blocks:
            sections.append("Relevant pitfalls / negative skills:\n" + "\n\n".join(negative_blocks))
        return "\n\n".join(sections).strip()

    def _build_fewshot_memory_text(self, task: str, query: str) -> str:
        if not self.config.few_shot_enabled:
            self._print(f"use_memory task={task}: few-shot disabled")
            return ""

        if not self._few_shot_rag:
            self._print(f"use_memory task={task}: few-shot RAG not initialized")
            return ""

        selected = self._few_shot_rag.retrieve(query=query, top_k=self.config.few_shot_top_k)
        if not selected:
            self._print(f"use_memory task={task}: few-shot retrieval empty")
            return ""

        example_text = "\n\n\n".join(str(item).strip() for item in selected if str(item).strip())
        if not example_text:
            self._print(f"use_memory task={task}: few-shot formatted empty")
            return ""

        self._print(
            f"use_memory task={task}: few-shot selected={len(selected)}, order={self.config.few_shot_order}, "
            f"example_chars={len(example_text)}"
        )
        return self.config.few_shot_prompt_template.format(examples=example_text)

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enhanced = list(messages) if messages is not None else []
        query = self._extract_query_from_messages(messages)
        if not query:
            self._print(f"use_memory task={task}: no query extracted, skipping retrieval")
            return enhanced

        skills = self._load_skills()
        skill_memory_content = ""
        selected_skills: List[Dict[str, Any]] = []
        if not skills:
            self._print(f"use_memory task={task}: skill library empty")
        else:
            catalog = self._build_catalog(skills)
            self._print(
                f"use_memory task={task}: query_len={len(query)}, skills_total={len(skills)}, "
                f"catalog_size={len(catalog)}"
            )
            selector_user = json.dumps(
                {
                    "task": task,
                    "query": query,
                    "max_selected_skills": self.config.max_selected_skills,
                    "catalog": catalog,
                },
                ensure_ascii=False,
                indent=2,
            )
            selector_messages = [
                {
                    "role": "system",
                    "content": self.selector_prompt.replace(
                        "{max_selected_skills}",
                        str(self.config.max_selected_skills),
                    ),
                },
                {"role": "user", "content": selector_user},
            ]
            response = self._call_llm(
                self.selector_cfg,
                selector_messages,
                purpose="skill selection",
                request_overrides={"response_format": {"type": "json_object"}},
            )
            parsed = parse_llm_json_response(response or "", logger_prefix="SkillNudgeSelector")
            if not isinstance(parsed, dict):
                self._print(f"use_memory task={task}: selector returned invalid JSON, skipping skill injection")
            else:
                selected_ids = parsed.get("selected_skill_ids", []) or []
                if not isinstance(selected_ids, list):
                    self._print(f"use_memory task={task}: selector selected_skill_ids invalid, skipping skill injection")
                else:
                    selected_set = {str(skill_id) for skill_id in selected_ids[: self.config.max_selected_skills]}
                    selected_skills = [skill for skill in skills if str(skill.get("id", "")) in selected_set]
                    self._print(
                        f"use_memory task={task}: selector chose={list(selected_set) if selected_set else []}, "
                        f"matched={len(selected_skills)}"
                    )
                    if selected_skills:
                        skill_text = self._format_selected_skills(selected_skills)
                        if skill_text:
                            skill_memory_content = self.config.prompt_template.format(skills=skill_text)
                        else:
                            self._print(f"use_memory task={task}: selected skills formatted to empty text")
                    else:
                        self._print(f"use_memory task={task}: no matching skills selected")

        fewshot_memory_content = self._build_fewshot_memory_text(task, query)
        memory_parts = [part for part in [skill_memory_content, fewshot_memory_content] if part]
        if not memory_parts:
            self._print(f"use_memory task={task}: no skill memory and no few-shot memory to inject")
            return enhanced

        memory_content = "\n\n".join(memory_parts)
        enhanced = enhance_messages_with_memory(enhanced, memory_content, where=self.config.where)
        assert_memory_injection_position(enhanced, self.config.where)
        self._print(
            f"use_memory task={task}: injected skill_count={len(selected_skills)}, "
            f"skill_chars={len(skill_memory_content)}, fewshot_chars={len(fewshot_memory_content)}, "
            f"total_chars={len(memory_content)}"
        )
        return enhanced

    def _build_trajectory_text(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> str:
        template_titles = [self.template_title]
        original_question = extract_original_question(history, where=self.config.where, template_titles=template_titles)
        status = str(result.get("status", "") or "")
        reward = result.get("reward", 0)
        sample_type = "success" if (status == "completed" or (isinstance(reward, (int, float)) and reward > 0)) else "failure"

        lines: List[str] = [f"Task: {task}", f"Sample Type: {sample_type}", f"Status: {status}", f"Reward: {reward}"]
        if original_question:
            lines.append(f"Query: {str(original_question).strip()}")
        lines.append("Trajectory:")

        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None or role == "system":
                continue
            text = str(content or "").strip()

            if role == "user":
                if text:
                    lines.append(f"User: {text}")
                continue

            if role == "assistant":
                reasoning = str((msg_dict or {}).get("reasoning_content", "") or "").strip()
                if reasoning:
                    lines.append("Assistant Thought:")
                    lines.append(f"<think>{reasoning}</think>")
                tool_calls = (msg_dict or {}).get("tool_calls", []) or []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = function.get("name", "unknown_tool")
                    arguments = function.get("arguments", "{}")
                    lines.append(f"Tool Call: {tool_name}({arguments})")
                if text:
                    lines.append(f"Assistant: {text}")
                continue

            if role == "tool" and text:
                lines.append(f"Tool Result: {text}")

        return "\n".join(lines).strip()

    def _build_window_trajectory_text(self, window_samples: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for idx, sample in enumerate(window_samples, start=1):
            sample_task = str(sample.get("task", "") or "").strip()
            sample_result = sample.get("result", {}) or {}
            history = sample.get("history", []) or []
            trajectory_text = self._build_trajectory_text(sample_task, history, sample_result)
            if not trajectory_text:
                continue
            blocks.append(f"=== Sample {idx} / {len(window_samples)} ===\n{trajectory_text}")
        return "\n\n".join(blocks).strip()

    def _next_skill_id(self, skills: List[Dict[str, Any]]) -> str:
        max_id = 0
        for skill in skills:
            raw = str(skill.get("id", "")).strip()
            if raw.startswith("skill_"):
                try:
                    max_id = max(max_id, int(raw.split("_", 1)[1]))
                except Exception:
                    continue
        return f"skill_{max_id + 1:06d}"

    def _find_skill(self, skills: List[Dict[str, Any]], skill_id: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        for idx, skill in enumerate(skills):
            if str(skill.get("id", "")) == skill_id:
                return idx, skill
        return None

    def _apply_action(self, skills: List[Dict[str, Any]], action: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        action_name = str(action.get("action", "")).strip()
        now = datetime.utcnow().isoformat(timespec="seconds")

        if action_name == "create_skill":
            raw_skill = action.get("skill", {}) or {}
            if not isinstance(raw_skill, dict):
                return skills, "create_skill ignored: invalid skill payload"
            new_skill = {
                "id": self._next_skill_id(skills),
                "name": str(raw_skill.get("name", "")).strip(),
                "summary": str(raw_skill.get("summary", "")).strip(),
                "content": str(raw_skill.get("content", "")).strip(),
                "polarity": str(raw_skill.get("polarity", "positive")).strip().lower() or "positive",
                "tags": [str(x) for x in (raw_skill.get("tags", []) or [])],
                "created_at": now,
                "updated_at": now,
                "usage_count": 0,
            }
            if not new_skill["name"] or not new_skill["summary"] or not new_skill["content"]:
                return skills, "create_skill ignored: missing required fields"
            skills.append(new_skill)
            self._write_skills(skills)
            self._print(
                f"update_memory: created {new_skill['id']} polarity={new_skill['polarity']} "
                f"name={new_skill['name'][:80]}"
            )
            return skills, f"create_skill applied: {new_skill['id']}"

        if action_name == "update_skill":
            skill_id = str(action.get("skill_id", "")).strip()
            found = self._find_skill(skills, skill_id)
            if not found:
                return skills, f"update_skill ignored: {skill_id} not found"
            raw_skill = action.get("skill", {}) or {}
            if not isinstance(raw_skill, dict):
                return skills, "update_skill ignored: invalid skill payload"
            idx, old = found
            updated = dict(old)
            updated.update(
                {
                    "name": str(raw_skill.get("name", old.get("name", ""))).strip(),
                    "summary": str(raw_skill.get("summary", old.get("summary", ""))).strip(),
                    "content": str(raw_skill.get("content", old.get("content", ""))).strip(),
                    "polarity": str(raw_skill.get("polarity", old.get("polarity", "positive"))).strip().lower() or "positive",
                    "tags": [str(x) for x in (raw_skill.get("tags", old.get("tags", [])) or [])],
                    "updated_at": now,
                }
            )
            skills[idx] = updated
            self._write_skills(skills)
            self._print(
                f"update_memory: updated {skill_id} polarity={updated['polarity']} "
                f"name={updated['name'][:80]}"
            )
            return skills, f"update_skill applied: {skill_id}"

        if action_name == "delete_skill":
            skill_id = str(action.get("skill_id", "")).strip()
            found = self._find_skill(skills, skill_id)
            if not found:
                return skills, f"delete_skill ignored: {skill_id} not found"
            idx, _ = found
            del skills[idx]
            self._write_skills(skills)
            self._print(f"update_memory: deleted {skill_id}")
            return skills, f"delete_skill applied: {skill_id}"

        return skills, f"{action_name} ignored"

    def _build_fewshot_question(self, history: List[Dict[str, Any]]) -> str:
        template_titles = [self.template_title]
        question = extract_original_question(history, where=self.config.where, template_titles=template_titles)
        return str(question or "").strip()

    def _build_fewshot_example(self, history: List[Dict[str, Any]]) -> str:
        question = self._build_fewshot_question(history)
        if not question:
            return ""

        answer_lines: List[str] = []
        skip_first_user = True
        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None or role == "system":
                continue

            text = str(content or "")
            if role == "user":
                if skip_first_user:
                    skip_first_user = False
                    continue
                if text.strip():
                    answer_lines.append(f"User: {text.strip()}")
                continue

            if role == "assistant":
                tool_calls = (msg_dict or {}).get("tool_calls", []) or []
                reasoning_content = ""
                if self.config.few_shot_include_reasoning_content:
                    reasoning_content = str((msg_dict or {}).get("reasoning_content", "") or "").strip()
                    reasoning_content = reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content
                think_part = f"<think>{reasoning_content}</think> " if reasoning_content else ""

                if tool_calls:
                    tool_bits: List[str] = []
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        function = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                        tool_name = function.get("name", "unknown")
                        tool_args = function.get("arguments", "{}")
                        tool_bits.append(f"{tool_name}({tool_args})")
                    suffix = " " + " ".join(tool_bits) if tool_bits else ""
                    answer_lines.append(f"Assistant: {think_part}{text.strip()}{suffix}".strip())
                else:
                    answer_lines.append(f"Assistant: {think_part}{text.strip()}".strip())
                continue

            if role == "tool" and text.strip():
                tool_text = text.strip()
                tool_text = tool_text[:500] + "..." if len(tool_text) > 500 else tool_text
                answer_lines.append(f"Tool: {tool_text}")

        answer = "\n".join(line for line in answer_lines if line) if answer_lines else "Completed successfully."
        return f"Question: {question}\n{answer}"

    def _maybe_store_fewshot_example(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        if not self.config.few_shot_enabled:
            return

        status = result.get("status", "")
        reward = result.get("reward", 0)
        is_success = status == "completed" or (isinstance(reward, (int, float)) and reward > 0)
        if self.config.few_shot_success_only and not is_success:
            self._print(f"update_memory task={task}: few-shot skipped by success_only (status={status}, reward={reward})")
            return
        if self.config.few_shot_reward_bigger_than_zero and (not isinstance(reward, (int, float)) or reward <= 0):
            self._print(f"update_memory task={task}: few-shot skipped by reward_bigger_than_zero (reward={reward})")
            return

        example_text = self._build_fewshot_example(history)
        question = self._build_fewshot_question(history)
        if not example_text or not question:
            self._print(f"update_memory task={task}: few-shot example empty, skipping store")
            return

        record = {
            "task": task,
            "key": question,
            "value": example_text,
            "status": status,
            "reward": reward,
            "type": result.get("type", ""),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self._append_fewshot_record(record)
        if self._few_shot_rag:
            try:
                self._few_shot_rag.insert(key=question, value=example_text)
            except Exception as exc:
                self._print(f"update_memory task={task}: failed to insert few-shot example into RAG: {exc}")
        self._print(
            f"update_memory task={task}: stored few-shot example key_len={len(question)}, "
            f"example_chars={len(example_text)}"
        )

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        if self.config.session_injection_only and result.get("type") != "session_injection":
            self._print(f"update_memory task={task}: skipped by session_injection_only")
            return

        self._maybe_store_fewshot_example(task, history, result)

        if self.config.update_success_only:
            status = result.get("status", "")
            reward = result.get("reward", 0)
            is_success = status == "completed" or (isinstance(reward, (int, float)) and reward > 0)
            if not is_success:
                self._print(
                    f"update_memory task={task}: skipped by update_success_only "
                    f"(status={status}, reward={reward})"
                )
                return

        if self.config.update_reward_bigger_than_zero:
            reward = result.get("reward", 0)
            if not isinstance(reward, (int, float)) or reward <= 0:
                self._print(
                    f"update_memory task={task}: skipped by update_reward_bigger_than_zero "
                    f"(reward={reward})"
                )
                return

        self._samples_since_last_nudge += 1
        self._review_window.append(
            {
                "task": task,
                "history": list(history) if history is not None else [],
                "result": dict(result) if isinstance(result, dict) else {"result": result},
            }
        )
        self._print(
            f"update_memory task={task}: nudge counter "
            f"{self._samples_since_last_nudge}/{self.config.nudge_interval}"
        )
        if self.config.nudge_interval > 0 and self._samples_since_last_nudge < self.config.nudge_interval:
            return
        window_samples = list(self._review_window)
        self._samples_since_last_nudge = 0
        self._review_window = []
        self._print(
            f"update_memory task={task}: nudge triggered, starting manager review "
            f"over window_size={len(window_samples)}"
        )

        trajectory_text = self._build_window_trajectory_text(window_samples)
        if not trajectory_text:
            self._print(f"update_memory task={task}: empty window trajectory text, skipping manager review")
            return

        skills = self._load_skills()
        self._print(
            f"update_memory task={task}: trajectory_chars={len(trajectory_text)}, "
            f"window_size={len(window_samples)}, "
            f"skills_total={len(skills)}"
        )
        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": self.manager_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "result": result,
                        "window_size": len(window_samples),
                        "window_sample_results": [
                            {
                                "task": str(sample.get("task", "") or ""),
                                "status": (sample.get("result", {}) or {}).get("status", ""),
                                "reward": (sample.get("result", {}) or {}).get("reward", 0),
                                "type": (sample.get("result", {}) or {}).get("type", ""),
                            }
                            for sample in window_samples
                        ],
                        "trajectory": trajectory_text,
                        "catalog": self._build_catalog(skills),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            },
        ]

        for step_idx in range(max(1, self.config.max_manager_steps)):
            self._print(
                f"update_memory task={task}: manager step {step_idx + 1}/{self.config.max_manager_steps}"
            )
            response = self._call_llm(
                self.manager_cfg,
                conversation,
                purpose="skill manager update",
                request_overrides={"response_format": {"type": "json_object"}},
            )
            parsed = parse_llm_json_response(response or "", logger_prefix="SkillNudgeManager")
            if not isinstance(parsed, dict):
                self._print(f"update_memory task={task}: manager returned invalid JSON, aborting review")
                return

            action_name = str(parsed.get("action", "")).strip()
            reasoning = str(parsed.get("reasoning", "") or "").strip()
            self._print(
                f"update_memory task={task}: manager action={action_name or 'UNKNOWN'}"
                + (f", reasoning={reasoning}" if reasoning else "")
            )
            conversation.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})

            if action_name == "finish":
                self._print(f"update_memory task={task}: manager finished with no further actions")
                return

            if action_name == "view_skill":
                skill_id = str(parsed.get("skill_id", "")).strip()
                found = self._find_skill(skills, skill_id)
                if not found:
                    self._print(f"update_memory task={task}: requested view_skill {skill_id} not found")
                    conversation.append({"role": "user", "content": f"Skill {skill_id} not found. Choose another action."})
                    continue
                _, skill = found
                self._print(
                    f"update_memory task={task}: providing full content for {skill_id} "
                    f"name={str(skill.get('name', ''))[:80]}"
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": json.dumps({"viewed_skill": skill}, ensure_ascii=False, indent=2),
                    }
                )
                continue

            skills, status_message = self._apply_action(skills, parsed)
            self._print(f"update_memory task={task}: {status_message}; skills_total={len(skills)}")
            conversation.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "action_result": status_message,
                            "catalog": self._build_catalog(skills),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                }
            )

        self._print(
            f"update_memory task={task}: reached max_manager_steps={self.config.max_manager_steps}, stopping review"
        )


def load_skill_nudge_from_yaml(config_path: str) -> SkillNudgeMemory:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict) or "skill_nudge" not in raw or not isinstance(raw["skill_nudge"], dict):
        raise ValueError(
            f"SkillNudge config at {config_path} must use a top-level 'skill_nudge' mapping."
        )

    raw = raw["skill_nudge"]
    few_shot_raw = raw.get("few_shot", {}) or {}

    config = SkillNudgeConfig(
        selector_model_name=str(raw.get("selector_model_name", "gpt-4o-mini")),
        manager_model_name=str(raw.get("manager_model_name", "gpt-4o-mini")),
        selector_prompt_path=SkillNudgeMemory._resolve_path(
            str(raw.get("selector_prompt_path", "memory/skill_nudge/prompts/selector.txt"))
        ),
        manager_prompt_path=SkillNudgeMemory._resolve_path(
            str(raw.get("manager_prompt_path", "memory/skill_nudge/prompts/manager.txt"))
        ),
        skill_storage_path=SkillNudgeMemory._resolve_path(
            str(raw.get("skill_storage_path", "memory/skill_nudge/skills.jsonl"))
        ),
        nudge_interval=max(1, int(raw.get("nudge_interval", 5))),
        max_manager_steps=max(1, int(raw.get("max_manager_steps", 4))),
        max_selected_skills=max(1, int(raw.get("max_selected_skills", 3))),
        max_catalog_size=max(1, int(raw.get("max_catalog_size", 200))),
        prompt_template=str(
            raw.get(
                "prompt_template",
                "Here are relevant reusable skills from prior experience.\nUse them only when they truly apply.\n\n{skills}",
            )
        ),
        few_shot_enabled=bool(few_shot_raw.get("enabled", False)),
        few_shot_top_k=max(1, int(few_shot_raw.get("top_k", 4))),
        few_shot_order=str(few_shot_raw.get("order", "random")),
        few_shot_embedding_model=str(
            few_shot_raw.get("embedding_model", "BAAI/bge-base-en-v1.5")
        ),
        few_shot_seed=int(few_shot_raw.get("seed", 42)),
        few_shot_storage_path=SkillNudgeMemory._resolve_path(
            str(few_shot_raw.get("storage_path", "memory/skill_nudge/fewshot_examples.jsonl"))
        ),
        few_shot_success_only=bool(few_shot_raw.get("success_only", True)),
        few_shot_reward_bigger_than_zero=bool(few_shot_raw.get("reward_bigger_than_zero", False)),
        few_shot_include_reasoning_content=bool(few_shot_raw.get("include_reasoning_content", True)),
        few_shot_prompt_template=str(
            few_shot_raw.get(
                "prompt_template",
                "Here are some examples of the task you have completed:\n\n{examples}",
            )
        ),
        where=str(raw.get("where", "front")),
        session_injection_only=bool(raw.get("session_injection_only", False)),
        update_success_only=bool(raw.get("update_success_only", False)),
        update_reward_bigger_than_zero=bool(raw.get("update_reward_bigger_than_zero", False)),
        request_timeout=float(raw.get("request_timeout", 120.0)),
        max_retries=int(raw.get("max_retries", 3)),
        retry_delay=float(raw.get("retry_delay", 2.0)),
        retry_backoff=float(raw.get("retry_backoff", 2.0)),
    )
    return SkillNudgeMemory(config)
