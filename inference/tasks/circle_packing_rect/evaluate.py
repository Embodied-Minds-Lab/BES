
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shinka.core import run_shinka_eval

NUM_CIRCLES = 21

TEMPLATE_TREE_PATH = Path(__file__).parent / "goal_tree_template.json"


def _run_root(results_dir: str) -> Path:
    return Path(results_dir).parent.parent


def _load_goal_tree(run_root: Path) -> Optional[Dict[str, Any]]:
    if (run_root / ".backward_disabled").exists():
        return None
    tree_path = run_root / "goal_tree.json"
    if not tree_path.exists():
        if not TEMPLATE_TREE_PATH.exists():
            return None
        run_root.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(TEMPLATE_TREE_PATH.read_text())
    return json.loads(tree_path.read_text())


def minimum_circumscribing_rectangle(circles: np.ndarray) -> Tuple[float, float]:
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    return float(max_x - min_x), float(max_y - min_y)


def adapted_validate_packing(
    run_output: np.ndarray,
    atol: float = 0.0,
) -> Tuple[bool, Optional[str]]:
    if not isinstance(run_output, np.ndarray):
        run_output = np.array(run_output)

    if run_output.shape != (NUM_CIRCLES, 3):
        return False, (
            f"Invalid shape: got {run_output.shape}, expected ({NUM_CIRCLES}, 3)"
        )

    if not np.all(np.isfinite(run_output)):
        return False, "Non-finite values found in circles array."

    radii = run_output[:, 2]
    if np.any(radii < 0):
        neg = np.where(radii < 0)[0]
        return False, f"Negative radii at indices: {neg}"

    n = NUM_CIRCLES
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((run_output[i, :2] - run_output[j, :2]) ** 2)))
            if dist < radii[i] + radii[j] - atol:
                return False, (
                    f"Circles {i} & {j} overlap: dist={dist:.4f}, "
                    f"r_i+r_j={radii[i]+radii[j]:.4f}"
                )

    width, height = minimum_circumscribing_rectangle(run_output)
    if width + height > (2.0 + atol):
        return False, (
            f"Bounding rect width+height={width+height:.4f} > 2 "
            f"(perimeter {2*(width+height):.4f} > 4)."
        )

    return True, "Valid packing."


def get_circle_packing_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def aggregate_circle_packing_metrics(
    results: List[np.ndarray], results_dir: str
) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    circles = np.asarray(results[0], dtype=float)
    radii_sum = float(np.sum(circles[:, 2]))

    run_root = _run_root(results_dir)
    tree = _load_goal_tree(run_root)
    bw_score: Optional[float] = None
    bw_satisfaction: Dict[str, float] = {}

    if tree is not None and tree.get("children"):
        from shinka.bidirectional_search import (
            verify_tree, recursive_score, flatten_satisfaction
        )
        from bench_hooks import _eval_vars_from_arrays
        verified = verify_tree(tree, _eval_vars_from_arrays(circles))
        bw_score = float(recursive_score(verified))
        bw_satisfaction = flatten_satisfaction(verified)

    public_metrics: Dict[str, Any] = {
        "num_circles": NUM_CIRCLES,
    }
    private_metrics: Dict[str, Any] = {
        "reported_sum_of_radii": radii_sum,
    }
    if bw_score is not None:
        public_metrics["backward_score"] = bw_score
        public_metrics["backward_subgoals_satisfied"] = sum(bw_satisfaction.values())
        public_metrics["backward_subgoals_total"] = len(bw_satisfaction)
        private_metrics["backward_satisfaction"] = bw_satisfaction

    metrics: Dict[str, Any] = {
        "combined_score": float(bw_score) if bw_score is not None else float(radii_sum),
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            circles=circles,
            centers=circles[:, :2],
            radii=circles[:, 2],
            reported_sum=radii_sum,
        )
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator_with_context(r: List[np.ndarray]) -> Dict[str, Any]:
        return aggregate_circle_packing_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="circle_packing21",
        num_runs=1,
        get_experiment_kwargs=get_circle_packing_kwargs,
        validate_fn=adapted_validate_packing,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Circle-packing-rect (n=21) evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'circle_packing21')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
