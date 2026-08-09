#!/usr/bin/env python3
"""
Calculate Forgetting Rate (FR) for Replay mode.

Data structure (per replay stage K):
  replayK/train/  — all samples trained up to stage K (cumulative).
                    A sample's FIRST appearance here is its immediate test score.
  replayK/test/   — random subset of historical samples retested at stage K
                    WITHOUT memory update. These are the replay scores.

FR = max(0, (P_immediate - P_replay) / P_immediate) * 100%
FR ∈ [0, 100]: 0 = no forgetting, higher = more forgetting.
Negative raw values (consolidation) are clipped to 0.

Usage:
    python calculate_replay_metrics.py <replay_dir> [metric_type]

Example:
    python calculate_replay_metrics.py outputs/replay/qwen3.5-27B-everOS-locomo0-exp070
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple


def load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def get_score(data: Dict, metric: str) -> Optional[float]:
    if data is None:
        return None
    metrics = data.get("result", {}).get("metrics", {})
    return metrics.get(metric)


def get_stage_index(stage_dir: Path) -> int:
    return int(stage_dir.name.replace("replay", ""))


def scan_samples(folder: Path, metric: str) -> Dict[Tuple, float]:
    """Read all (task, index) -> score from a train/ or test/ subfolder."""
    scores = {}
    if not folder.exists():
        return scores
    for task_dir in folder.iterdir():
        if not task_dir.is_dir():
            continue
        for sample_file in task_dir.glob("*.json"):
            data = load_json(sample_file)
            score = get_score(data, metric)
            if score is not None:
                scores[(task_dir.name, int(sample_file.stem))] = score
    return scores


def calculate_replay_metrics(replay_dir: Path, metric: str = "llm_score") -> Dict:
    replay_dir = Path(replay_dir)

    # 1. Collect and sort all replay stages
    stages = sorted(
        [d for d in replay_dir.iterdir() if d.is_dir() and d.name.startswith("replay")],
        key=get_stage_index
    )
    K = len(stages)
    print(f"Found {K} replay stages in: {replay_dir.name}")

    # 2. Find each sample's immediate score = score in the FIRST train/ it appears in
    immediate: Dict[Tuple, float] = {}   # (task, idx) -> P_immediate
    sample_first_stage: Dict[Tuple, int] = {}  # (task, idx) -> stage index

    for stage in stages:
        k = get_stage_index(stage)
        train_scores = scan_samples(stage / "train", metric)
        for key, score in train_scores.items():
            if key not in immediate:          # first appearance = immediate test
                immediate[key] = score
                sample_first_stage[key] = k

    print(f"Samples with immediate scores: {len(immediate)}")

    # 3. Collect replay scores: replayK/test/ for all stages
    replay: Dict[int, Dict[Tuple, float]] = {}  # stage_k -> {(task, idx): score}
    for stage in stages:
        k = get_stage_index(stage)
        replay[k] = scan_samples(stage / "test", metric)

    # 4. Calculate FR
    # For each sample s first learned at stage j:
    #   FR(s) = mean over k > j of max(0, (P_immediate(s) - P_replay_k(s)) / P_immediate(s)) * 100
    # Overall FR = mean over all samples that have at least one replay observation

    sample_frs = []

    for sample, p_imm in immediate.items():
        if p_imm == 0:
            continue  # skip: can't normalize, and method never learned this sample
        j = sample_first_stage[sample]

        fr_values = []
        for k, replay_scores in replay.items():
            if k <= j:
                continue  # only look at stages AFTER the sample was first learned
            if sample in replay_scores:
                p_rep = replay_scores[sample]
                fr = max(0.0, (p_imm - p_rep) / p_imm) * 100.0
                fr_values.append(fr)

        if fr_values:
            sample_frs.append(sum(fr_values) / len(fr_values))

    overall_fr = sum(sample_frs) / len(sample_frs) if sample_frs else 0.0

    print(f"\nFR computed over {len(sample_frs)} samples with replay observations.")
    print(f"Overall Forgetting Rate (FR): {overall_fr:.2f}%")
    print(f"  FR=0 means no forgetting; higher FR means more forgetting.")

    return {
        "metric_type": metric,
        "num_replay_stages": K,
        "num_samples_with_immediate": len(immediate),
        "num_samples_with_replay": len(sample_frs),
        "forgetting_rate": round(overall_fr, 4),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_replay_metrics.py <replay_dir> [metric_type]")
        print("Metric types: llm_score (default), f1_score, bleu_score")
        sys.exit(1)

    replay_dir = Path(sys.argv[1])
    metric = sys.argv[2] if len(sys.argv) > 2 else "llm_score"

    if not replay_dir.exists():
        print(f"Error: directory does not exist — {replay_dir}")
        sys.exit(1)

    result = calculate_replay_metrics(replay_dir, metric)

    output_file = replay_dir / f"replay_metrics_FR_{metric}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
