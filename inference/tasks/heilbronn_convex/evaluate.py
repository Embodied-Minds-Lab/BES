# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
import itertools
import json
import os
import sys
import time
from importlib import __import__
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial import ConvexHull

BENCHMARK = 0.030936889034895654
NUM_POINTS = 13

TEMPLATE_TREE_PATH = Path(__file__).parent / "goal_tree_template.json"


def _run_root(results_dir: str) -> Path:
    """results_dir is `<run_root>/gen_<N>/results/`; two .parent hops -> run_root."""
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


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given its vertices p1, p2, and p3."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


def evaluate(program_path: str, results_dir: Optional[str] = None,
             extra_npz_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        points = None
        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)

            start_time = time.time()
            points = program.heilbronn_convex13()
            eval_time = time.time() - start_time
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape != (NUM_POINTS, 2):
            raise ValueError(
                f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS, 2)}"
            )

        # primary metrics
        min_triangle_area = min(
            triangle_area(p1, p2, p3)
            for p1, p2, p3 in itertools.combinations(points, 3)
        )
        convex_hull_area = float(ConvexHull(points).volume)
        min_area_normalized = float(min_triangle_area / convex_hull_area)
        raw_combined = float(min_area_normalized / BENCHMARK)

        # Save extra.npz to whichever output target was requested.
        extra_save_error = None
        if extra_npz_path is not None:
            extra_path = Path(extra_npz_path)
        elif results_dir is not None:
            extra_path = Path(results_dir) / "extra.npz"
        else:
            extra_path = None
        if extra_path is not None:
            try:
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    str(extra_path),
                    points=np.asarray(points, dtype=float),
                    min_area=np.float64(min_triangle_area),
                    convex_hull_area=np.float64(convex_hull_area),
                    min_area_normalized=np.float64(min_area_normalized),
                    combined_score_raw=np.float64(raw_combined),
                )
                print(f"extra.npz saved to {extra_path}")
            except Exception as e:
                extra_save_error = str(e)
                print(f"Error saving extra.npz: {e}")

        # Backward scoring — only in harness mode (results_dir given, no extra_npz_path).
        bw_score: Optional[float] = None
        bw_satisfaction: Dict[str, float] = {}
        if results_dir is not None and extra_npz_path is None:
            run_root = _run_root(results_dir)
            tree = _load_goal_tree(run_root)
            if tree is not None and tree.get("children"):
                from shinka.bidirectional_search import (
                    flatten_satisfaction,
                    recursive_score,
                    verify_tree,
                )
                from bench_hooks import _eval_vars_from_points

                ev = _eval_vars_from_points(points)
                verified = verify_tree(tree, ev)
                bw_score = float(recursive_score(verified))
                bw_satisfaction = flatten_satisfaction(verified)

        public_metrics: Dict[str, Any] = {
            "min_area_normalized": min_area_normalized,
            "convex_hull_area": convex_hull_area,
            "combined_score_raw": raw_combined,  # raw min_area_norm / BENCHMARK
            "eval_time": float(eval_time),
        }
        private_metrics: Dict[str, Any] = {
            "min_area": float(min_triangle_area),
            "min_area_normalized": min_area_normalized,
        }
        if bw_score is not None:
            public_metrics["backward_score"] = bw_score
            public_metrics["backward_subgoals_satisfied"] = sum(bw_satisfaction.values())
            public_metrics["backward_subgoals_total"] = len(bw_satisfaction)
            private_metrics["backward_satisfaction"] = bw_satisfaction
        if extra_save_error is not None:
            private_metrics["extra_npz_save_error"] = extra_save_error

        return {
            "combined_score": float(bw_score) if bw_score is not None else raw_combined,
            "public": public_metrics,
            "private": private_metrics,
        }
    except Exception as e:
        return {
            "combined_score": 0.0,
            "public": {},
            "private": {},
            "error": str(e),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    result = evaluate(args.program_path, results_dir=args.results_dir)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(result, f)
    correct = "error" not in result
    with open(os.path.join(args.results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": result.get("error")}, f)
