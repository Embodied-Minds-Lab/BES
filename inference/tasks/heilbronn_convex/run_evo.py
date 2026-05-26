#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

REPO_DIR = Path(__file__).parent
TEMPLATE_TREE = REPO_DIR / "goal_tree_template.json"
INITIAL_PROGRAM = REPO_DIR / "initial.py"
IMPROVEMENT_MARGIN = 1e-2   # combined_score scale; baseline programs ~0.3-0.7
STAGNATION_THRESHOLD = 5
BACKWARD_MAX_DEPTH = 2


def _make_backward_callback(run_root: Path):
    import io

    from shinka.bidirectional_search import (
        rebuild_tree_with_new_reference,
        try_expand_if_due,
    )
    from bench_hooks import HOOKS

    tree_path = run_root / "goal_tree.json"
    state_path = run_root / "tree_expander_state.json"
    ref_npz = run_root / "reference_solution.npz"
    db_path = run_root / "programs.sqlite"

    _state: Dict[str, Any] = {
        "last_best": -float("inf"),
        "counter": 0,
        "best_extra_payload": None,
    }

    def _read_extras(jobs):
        for job in jobs:
            extra = Path(job.results_dir) / "extra.npz"
            if not extra.exists():
                continue
            try:
                d = np.load(str(extra))
                payload = {k: d[k] for k in d.files}
                score = float(np.asarray(payload.get("combined_score_raw", 0.0)).reshape(-1)[0])
                yield int(job.generation), payload, score
            except Exception as e:
                print(f"[backward_cb] gen={job.generation} read extra.npz failed: {e}")

    def _tree_has_unexpanded_leaf(tree: Dict[str, Any], max_depth: int) -> bool:
        def walk(n):
            children = n.get("children") or []
            lvl = int(n.get("level", 0))
            if not children and lvl < max_depth:
                return True
            return any(walk(c) for c in children)
        return walk(tree)

    def _do_rebuild(payload: Dict[str, Any], new_score: float, cur_score: float, why: str):
        buf = io.BytesIO()
        np.savez(buf, **payload)
        try:
            last = rebuild_tree_with_new_reference(
                run_root, TEMPLATE_TREE, buf.getvalue(),
                hooks=HOOKS, max_depth=BACKWARD_MAX_DEPTH,
            )
            print(
                f"[backward_cb] rebuild ({why}) new ref combined_raw={new_score:.6f} "
                f"(was {cur_score:.6f}); expanded={last}"
            )
        except Exception as e:
            print(f"[backward_cb] rebuild ({why}) failed: {type(e).__name__}: {e}")

    def _do_expand() -> bool:
        try:
            expanded = try_expand_if_due(
                tree_path, db_path, ref_npz, state_path,
                hooks=HOOKS, interval=1, max_depth=BACKWARD_MAX_DEPTH,
            )
            if expanded:
                print(f"[backward_cb] expanded node: {expanded}")
                return True
            print(f"[backward_cb] expand: nothing to do")
            return False
        except Exception as e:
            print(f"[backward_cb] expand failed: {type(e).__name__}: {e}")
            return False

    async def _callback(jobs, runner):
        for gen, payload, new_score in sorted(_read_extras(jobs), key=lambda t: t[0]):
            if new_score > _state["last_best"] + IMPROVEMENT_MARGIN:
                _state["counter"] = 0
                _state["last_best"] = new_score
                _state["best_extra_payload"] = payload
                print(
                    f"[backward_cb] gen={gen} combined_raw={new_score:.4f} "
                    f"(IMPROVED, counter reset)"
                )
            else:
                _state["counter"] += 1
                if new_score > _state["last_best"]:
                    _state["last_best"] = new_score
                    _state["best_extra_payload"] = payload
                print(
                    f"[backward_cb] gen={gen} combined_raw={new_score:.4f} "
                    f"(no-improve, counter={_state['counter']}/{STAGNATION_THRESHOLD})"
                )

        if _state["counter"] < STAGNATION_THRESHOLD:
            return
        if _state["best_extra_payload"] is None:
            return

        tree_obj = None
        if tree_path.exists():
            try:
                tree_obj = json.loads(tree_path.read_text())
            except Exception:
                tree_obj = None
        cur_ref_score = -float("inf")
        if ref_npz.exists():
            try:
                cur_ref_score = float(
                    np.asarray(np.load(str(ref_npz))["combined_score_raw"]).reshape(-1)[0]
                )
            except Exception:
                pass

        no_real_tree = tree_obj is None or not tree_obj.get("children")
        if no_real_tree:
            _do_rebuild(_state["best_extra_payload"], _state["last_best"],
                        cur_ref_score, why="bootstrap")
        elif _state["last_best"] > cur_ref_score + IMPROVEMENT_MARGIN:
            _do_rebuild(_state["best_extra_payload"], _state["last_best"],
                        cur_ref_score, why="ref-outdated")
        elif _tree_has_unexpanded_leaf(tree_obj, BACKWARD_MAX_DEPTH):
            if not _do_expand():
                print(f"[backward_cb] expand exhausted — forcing rebuild")
                _do_rebuild(_state["best_extra_payload"], _state["last_best"],
                            cur_ref_score, why="expand-exhausted")
        else:
            print(f"[backward_cb] stagnation hit but tree at max depth + ref fresh; nop")

        _state["counter"] = 0

    return _callback


def _seed_rolling_reference(run_root: Path, initial_program: Path) -> None:
    """Write <run_root>/reference_solution.npz from `initial.py`'s output, using
    evaluate.py in seeding mode (extra_npz_path) to avoid touching the goal tree."""
    ref_npz = run_root / "reference_solution.npz"
    if ref_npz.exists():
        return
    spec = importlib.util.spec_from_file_location("_seed_evaluate", str(REPO_DIR / "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_root.mkdir(parents=True, exist_ok=True)
    result = mod.evaluate(str(initial_program), extra_npz_path=str(ref_npz))
    if not ref_npz.exists():
        raise RuntimeError(
            f"_seed_rolling_reference: evaluate() did not write reference_solution.npz "
            f"(error: {result.get('error')})"
        )
    print(f"[seed_ref] wrote {ref_npz}")


TASK_SYS_MSG = """SETTING:
You are an expert computational geometer and optimization specialist with deep expertise in the Heilbronn triangle problem - a fundamental challenge in discrete geometry first posed by Hans Heilbronn in 1957.
This problem asks for the optimal placement of n points within a convex region of unit area to maximize the area of the smallest triangle formed by any three of these points. 
Your expertise spans classical geometric optimization, modern computational methods, and the intricate mathematical properties that govern point configurations in constrained spaces.

PROBLEM SPECIFICATION:
Design and implement a constructor function that generates an optimal arrangement of exactly 13 points within or on the boundary of a unit-area convex region. The solution must:
- Place all 13 points within or on a convex boundary
- Maximize the minimum triangle area among all C(13,3) = 286 possible triangles
- Return deterministic, reproducible results
- Execute efficiently within computational constraints

PERFORMANCE METRICS:
1. **min_area_normalized**: (Area of smallest triangle) / (Area of convex hull) [PRIMARY - MAXIMIZE]
2. **combined_score**: min_area_normalized / 0.030936889034895654 [BENCHMARK COMPARISON - TARGET > 1.0]
3. **eval_time**: Execution time in seconds [EFFICIENCY - secondary priority]

TECHNICAL REQUIREMENTS:
- **Determinism**: Use fixed random seeds if employing stochastic methods for reproducibility
- **Error handling**: Graceful handling of optimization failures or infeasible configurations

MATHEMATICAL CONTEXT & THEORETICAL BACKGROUND:
- **PROBLEM COMPLEXITY**: The Heilbronn problem is among the most challenging in discrete geometry, with optimal configurations rigorously known only for n ≤ 4 points
- **ASYMPTOTIC BEHAVIOR**: For large n, the optimal value approaches O(1/n²) with logarithmic corrections, but the exact constant remains unknown
- **GEOMETRIC CONSTRAINTS**: Points must balance competing objectives:
  * Interior points can form larger triangles but create crowding
  * Boundary points avoid area penalties but limit triangle formation
  * Edge cases arise when three points become nearly collinear
- **SYMMETRY CONSIDERATIONS**: Optimal configurations often exhibit rotational symmetries (particularly 3-fold due to triangular geometry)
- **SCALING INVARIANCE**: The problem is scale-invariant; solutions can be normalized to any convex region
- **CRITICAL GEOMETRIC PROPERTIES**:
  * Delaunay triangulation properties and angle optimization
  * Voronoi diagram regularity as indicator of point distribution quality
  * Relationship between circumradius and triangle area
  * Connection to sphere packing and energy minimization principles

ADVANCED OPTIMIZATION STRATEGIES:
- **MULTI-SCALE APPROACH**: Coarse global search → fine local refinement with adaptive step sizes
- **CONSTRAINT HANDLING**: Penalty methods, barrier functions, or projection operators for convexity
- **INITIALIZATION STRATEGIES**:
  * Perturbed regular grids (triangular, square, hexagonal lattices)
  * Random points with force-based relaxation
  * Symmetry-constrained configurations (3-fold, 6-fold rotational)
  * Hybrid boundary/interior distributions
  * Low-discrepancy sequences (Sobol, Halton) for uniform coverage
- **OBJECTIVE FUNCTION DESIGN**:
  * Smooth approximations to min() function (LogSumExp, p-norms with p→∞)
  * Barrier methods for boundary constraints
  * Multi-objective formulations balancing multiple triangle areas
  * Weighted combinations of smallest k triangle areas
- **ADVANCED TECHNIQUES**:
  * Riemannian optimization on manifolds
  * Variational methods treating point density as continuous field
  * Machine learning-guided search using learned geometric priors
  * Topological optimization considering point connectivity graphs
  * Continuation methods with parameter homotopy

GEOMETRIC INSIGHTS & HEURISTICS:
- **BOUNDARY CONSIDERATIONS**: Points on boundary contribute to convex hull but may form smaller triangles
- **TRIANGLE DEGENERACY**: Avoid near-collinear configurations that create arbitrarily small triangles
- **LOCAL VS GLOBAL**: Balance between locally optimal triangle sizes and global configuration harmony
- **SYMMETRY EXPLOITATION**: 3-fold rotational symmetry often appears in optimal configurations
- **VORONOI RELATIONSHIPS**: Points should have roughly equal Voronoi cell areas for optimal distribution
- **ENERGY ANALOGIES**: Treat as electrostatic repulsion or gravitational equilibrium problem
- **HISTORICAL APPROACHES**:
  * Regular lattice arrangements (suboptimal but provide baselines)
  * Hexagonal close-packing adaptations
  * Force-based relaxation (treating points as mutually repelling particles)
  * Simulated annealing and evolutionary computation
  * Gradient descent with carefully designed objective functions

VALIDATION FRAMEWORK:
- **Geometric constraint verification**:
  * Point count validation: Exactly 13 points required
  * Convexity check: All points within or on boundary of convex hull
- **Data integrity checks**:
  * Coordinate bounds: All coordinates are finite real numbers
  * Point uniqueness: No duplicate points (within numerical tolerance)
  * Geometric consistency: Points form valid geometric configuration
- **Solution quality assessment**:
  * Local optimality testing through small perturbations
  * Symmetry analysis: Detection of rotational/reflectional symmetries
  * Distribution quality: Voronoi cell area variance, nearest neighbor statistics
  * Convergence verification: For iterative methods, check convergence criteria
- **Determinism verification**:
  * Multiple execution consistency: Same results across multiple runs
  * Seed effectiveness: Proper random seed implementation
  * Platform independence: Results stable across different computing environments
"""


COMPLEX_PATCH_TYPES = [
    "diff",
    "diff_ablate",
    "full",
    "cross_combine",
    "cross_translocate",
    "cross",
]
COMPLEX_PATCH_PROBS = [0.40, 0.20, 0.25, 0.05, 0.05, 0.05]


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from bench_hooks import HOOKS as _HOOKS_FOR_RANKING
    from shinka.database.score import set_raw_metric_key, set_bucket_precision
    set_raw_metric_key(_HOOKS_FOR_RANKING.raw_metric_key)
    set_bucket_precision(5e-4)
    print(f"[score] adaptive bucket interp using RAW_KEY="
          f"'{_HOOKS_FOR_RANKING.raw_metric_key}', bucket=5e-4")

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id and config["evo_config"].get("results_dir"):
        suffixed = f"{config['evo_config']['results_dir']}_{job_id}"
        print(f"[results_dir] {config['evo_config']['results_dir']} → {suffixed}")
        config["evo_config"]["results_dir"] = suffixed

    if config.get("complex_patch_actions", {}).get("enabled", False):
        config["evo_config"]["patch_types"] = list(COMPLEX_PATCH_TYPES)
        config["evo_config"]["patch_type_probs"] = list(COMPLEX_PATCH_PROBS)
        print(
            f"[complex_patch_actions] ON: patch_types="
            f"{COMPLEX_PATCH_TYPES} probs={COMPLEX_PATCH_PROBS}"
        )

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(eval_program_path="evaluate.py", time="00:06:00")
    db_config = DatabaseConfig(**config["db_config"])

    run_root = Path(evo_config.results_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    backward_marker = run_root / ".backward_disabled"
    backward_enabled = config.get("backward_search", {}).get("enabled", True)

    backward_callback = None
    if backward_enabled:
        if backward_marker.exists():
            backward_marker.unlink()
        decompose_model = config.get("backward_search", {}).get("decompose_model")
        if decompose_model:
            from bench_hooks import HOOKS as _BWHOOKS
            _BWHOOKS.model = decompose_model
            print(f"[backward_search] decompose_model overridden to '{decompose_model}'")
        _seed_rolling_reference(run_root, INITIAL_PROGRAM)
        backward_callback = _make_backward_callback(run_root)
        print(
            f"[backward_search] enabled (lazy bootstrap: triggers after "
            f"{STAGNATION_THRESHOLD} consecutive forward gens with "
            f"improvement <= {IMPROVEMENT_MARGIN})"
        )
    else:
        backward_marker.touch()
        print(
            f"[backward_search] OFF — wrote {backward_marker}; evaluate.py will "
            f"score by raw combined_score and skip tree expansion."
        )

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
        backward_callback=backward_callback,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    main(args.config_path)
