"""
Memory Mechanism Registry - centralized management for registering and loading all memory mechanisms

Naming convention:
- Use snake_case in configuration (e.g. stream_icl, awm, mems)
- The registry maps these names to their actual classes and loader functions
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Dict, Any

from memory.base import MemoryMechanism


# Registry: name -> (loader_function, default_config_path)
_MEMORY_REGISTRY: Dict[str, tuple[Callable[[str], MemoryMechanism], str]] = {}


def _build_lazy_loader(module_path: str, loader_name: str) -> Callable[[str], MemoryMechanism]:
    """
    Build a loader that imports the backing module only when the memory
    mechanism is actually selected.
    """
    def _lazy_loader(config_path: str) -> MemoryMechanism:
        module = importlib.import_module(module_path)
        loader_func = getattr(module, loader_name)
        return loader_func(config_path)

    return _lazy_loader


def register_memory(
    name: str,
    loader_func: Callable[[str], MemoryMechanism],
    default_config_path: str,
) -> None:
    """
    Register a memory mechanism.

    Args:
        name: mechanism name in snake_case (e.g. stream_icl, awm)
        loader_func: loader function that accepts a config_path and returns a MemoryMechanism instance
        default_config_path: default config file path relative to the project root
    """
    _MEMORY_REGISTRY[name] = (loader_func, default_config_path)


def get_memory_loader(name: str) -> tuple[Callable[[str], MemoryMechanism], str]:
    """
    Get the loader function and default config path for a memory mechanism.

    Args:
        name: mechanism name

    Returns:
        (loader_func, default_config_path)

    Raises:
        ValueError: if the mechanism is not registered
    """
    if name not in _MEMORY_REGISTRY:
        available = ", ".join(sorted(_MEMORY_REGISTRY.keys()))
        raise ValueError(
            f"Memory mechanism '{name}' not registered. "
            f"Available mechanisms: {available}"
        )
    return _MEMORY_REGISTRY[name]


def list_available_memories() -> list[str]:
    """Return names of all registered memory mechanisms."""
    return sorted(_MEMORY_REGISTRY.keys())


# ===== Register all memory mechanisms =====

def _register_all_memories():
    """Register all built-in memory mechanisms."""

    # zero_shot
    register_memory(
        name="zero_shot",
        loader_func=_build_lazy_loader("memory.zero_shot.zero_shot", "load_zero_shot_from_yaml"),
        default_config_path="memory/zero_shot/zero_shot.yaml",
    )

    # stream_icl (snake_case)
    register_memory(
        name="stream_icl",
        loader_func=_build_lazy_loader("memory.streamICL.streamICL", "load_stream_icl_from_yaml"),
        default_config_path="memory/streamICL/streamICL.yaml",
    )

    # mem0
    register_memory(
        name="mem0",
        loader_func=_build_lazy_loader("memory.mem0.mem0", "load_mem0_from_yaml"),
        default_config_path="memory/mem0/mem0.yaml",
    )

    # everos_personal
    register_memory(
        name="everos_personal",
        loader_func=_build_lazy_loader("memory.everos_personal.everos_personal", "load_everos_personal_from_yaml"),
        default_config_path="memory/everos_personal/everos_personal.yaml",
    )

    # skill_nudge
    register_memory(
        name="skill_nudge",
        loader_func=_build_lazy_loader("memory.skill_nudge.skill_nudge", "load_skill_nudge_from_yaml"),
        default_config_path="memory/skill_nudge/skill_nudge.yaml",
    )

    # mems (lowercase)
    register_memory(
        name="mems",
        loader_func=_build_lazy_loader("memory.MEMs", "load_mems_from_yaml"),
        default_config_path="memory/MEMs/MEMs.yaml",
    )

    # awm (snake_case)
    register_memory(
        name="awm",
        loader_func=_build_lazy_loader("memory.AWM", "load_awm_from_yaml"),
        default_config_path="memory/AWM/AWM.yaml",
    )

# Auto-register all memory mechanisms on import
_register_all_memories()
