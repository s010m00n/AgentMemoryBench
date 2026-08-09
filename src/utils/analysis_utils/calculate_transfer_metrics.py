#!/usr/bin/env python3
"""
Calculate Forward Transfer Gain (FTG) for Intra-Transfer mode.

Data structure (per experiment folder):
  transfer_train/<task>/{j}.json   — baseline score for sample j
                                     (tested WITHOUT any cross-sample transfer)
  forward_transfer_test/<task>/train{i}_test{j}.json
                                   — score when memory is built from sample i,
                                     then tested on sample j (j > i in stream order)

FTG = mean over all (i, j) pairs of:
        score(train_i → test_j) - score(baseline_j)

FTG > 0: positive forward transfer (memory helps future tasks)
FTG = 0: no transfer effect
FTG < 0: negative transfer (memory hurts future tasks)

Usage:
    python calculate_transfer_metrics.py <transfer_dir> [metric_type]

Example:
    python calculate_transfer_metrics.py outputs/intra-transfer/qwen3.5-27B-everOS-locomo0-exp062
"""

import json
import sys
import re
from pathlib import Path
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
    # forward_transfer_test files store score under result.result.metrics
    metrics = data.get("result", {})
    if isinstance(metrics, dict) and "result" in metrics:
        metrics = metrics["result"].get("metrics", {})
    elif isinstance(metrics, dict) and "metrics" in metrics:
        metrics = metrics.get("metrics", {})
    else:
        # transfer_train files store score directly under result.metrics
        metrics = data.get("result", {}).get("metrics", {})
    return metrics.get(metric)


def load_baseline_scores(transfer_train_dir: Path, metric: str) -> Dict[int, float]:
    """Load baseline scores: {sample_index: score} from transfer_train/<task>/"""
    scores = {}
    if not transfer_train_dir.exists():
        return scores
    for task_dir in transfer_train_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for f in task_dir.glob("*.json"):
            try:
                idx = int(f.stem)
            except ValueError:
                continue
            data = load_json(f)
            score = get_score(data, metric)
            if score is not None:
                scores[idx] = score
    return scores


def load_transfer_scores(forward_test_dir: Path, metric: str) -> Dict[Tuple[int, int], float]:
    """Load transfer scores: {(train_i, test_j): score} from forward_transfer_test/<task>/"""
    scores = {}
    if not forward_test_dir.exists():
        return scores
    pattern = re.compile(r'^train(\d+)_test(\d+)\.json$')
    for task_dir in forward_test_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for f in task_dir.glob("*.json"):
            m = pattern.match(f.name)
            if not m:
                continue
            i, j = int(m.group(1)), int(m.group(2))
            data = load_json(f)
            score = get_score(data, metric)
            if score is not None:
                scores[(i, j)] = score
    return scores


def calculate_ftg(transfer_dir: Path, metric: str = "llm_score") -> Dict:
    transfer_dir = Path(transfer_dir)

    train_dir   = transfer_dir / "transfer_train"
    forward_dir = transfer_dir / "forward_transfer_test"

    baseline  = load_baseline_scores(train_dir, metric)
    transfers = load_transfer_scores(forward_dir, metric)

    print(f"Baseline samples loaded : {len(baseline)}")
    print(f"Transfer pairs loaded   : {len(transfers)}")

    gains = []
    missing_baseline = 0

    for (i, j), transfer_score in transfers.items():
        if j not in baseline:
            missing_baseline += 1
            continue
        gain = transfer_score - baseline[j]
        gains.append(gain)

    if missing_baseline:
        print(f"Warning: {missing_baseline} pairs skipped (no baseline for test sample)")

    ftg = sum(gains) / len(gains) if gains else 0.0

    print(f"\nFTG computed over {len(gains)} pairs.")
    print(f"Forward Transfer Gain (FTG): {ftg * 100:.2f}%")
    print(f"  FTG > 0: positive transfer | FTG < 0: negative transfer")

    return {
        "metric_type"          : metric,
        "num_baseline_samples" : len(baseline),
        "num_transfer_pairs"   : len(transfers),
        "num_pairs_used"       : len(gains),
        "forward_transfer_gain": round(ftg, 6),
        "forward_transfer_gain_pct": round(ftg * 100, 4),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_transfer_metrics.py <transfer_dir> [metric_type]")
        print("Metric types: llm_score (default), f1_score, bleu_score")
        sys.exit(1)

    transfer_dir = Path(sys.argv[1])
    metric = sys.argv[2] if len(sys.argv) > 2 else "llm_score"

    if not transfer_dir.exists():
        print(f"Error: directory does not exist — {transfer_dir}")
        sys.exit(1)

    result = calculate_ftg(transfer_dir, metric)

    output_file = transfer_dir / f"transfer_metrics_FTG_{metric}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
