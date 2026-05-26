#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

REPO_DIR = Path(__file__).parent
TEMPLATE_TREE = REPO_DIR / "goal_tree_template.json"
INITIAL_PROGRAM = REPO_DIR / "initial.py"
IMPROVEMENT_MARGIN = 1e-2  # raw_sum_r gain must exceed this to count as "improvement"
STAGNATION_THRESHOLD = 5    # consecutive non-improving forward gens to trigger backward
BACKWARD_MAX_DEPTH = 2      # root + L1 + L2


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

    _state: Dict[str, Any] = {"last_best": -float("inf"), "counter": 0,
                               "best_centers": None, "best_radii": None}

    def _read_extras(jobs):
        """Yield (gen, centers, radii, sum_r) for each job whose extra.npz parses."""
        for job in jobs:
            extra = Path(job.results_dir) / "extra.npz"
            if not extra.exists():
                continue
            try:
                d = np.load(str(extra))
                c = np.asarray(d["centers"], dtype=float)
                r = np.asarray(d["radii"], dtype=float)
                yield int(job.generation), c, r, float(r.sum())
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

    def _do_rebuild(centers, radii, new_sum, cur_sum, why):
        buf = io.BytesIO()
        np.savez(buf, centers=centers, radii=radii)
        try:
            last = rebuild_tree_with_new_reference(
                run_root,
                TEMPLATE_TREE,
                buf.getvalue(),
                hooks=HOOKS,
                max_depth=BACKWARD_MAX_DEPTH,
            )
            print(
                f"[backward_cb] rebuild ({why}) new ref sum_r={new_sum:.6f} "
                f"(was {cur_sum:.6f}); expanded={last}"
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
        # 1. Walk the just-completed jobs in generation order; update stagnation counter.
        for gen, centers, radii, new_sum in sorted(_read_extras(jobs)):
            if new_sum > _state["last_best"] + IMPROVEMENT_MARGIN:
                _state["counter"] = 0
                _state["last_best"] = new_sum
                _state["best_centers"] = centers
                _state["best_radii"] = radii
                print(
                    f"[backward_cb] gen={gen} sum_r={new_sum:.4f} "
                    f"(IMPROVED, counter reset)"
                )
            else:
                _state["counter"] += 1
                if new_sum > _state["last_best"]:
                    _state["last_best"] = new_sum
                    _state["best_centers"] = centers
                    _state["best_radii"] = radii
                print(
                    f"[backward_cb] gen={gen} sum_r={new_sum:.4f} "
                    f"(no-improve, counter={_state['counter']}/{STAGNATION_THRESHOLD})"
                )

        # 2. Trigger backward action if stagnation reached threshold.
        if _state["counter"] < STAGNATION_THRESHOLD:
            return
        if _state["best_centers"] is None:
            # Nothing valid to use as a ref yet — wait for next batch.
            return

        # Decision tree
        tree_obj = None
        if tree_path.exists():
            try:
                tree_obj = json.loads(tree_path.read_text())
            except Exception:
                tree_obj = None
        cur_ref_sum = -float("inf")
        if ref_npz.exists():
            try:
                cur_ref_sum = float(np.load(str(ref_npz))["radii"].sum())
            except Exception:
                pass

        no_real_tree = tree_obj is None or not tree_obj.get("children")
        if no_real_tree:
            _do_rebuild(_state["best_centers"], _state["best_radii"],
                        _state["last_best"], cur_ref_sum, why="bootstrap")
        elif _state["last_best"] > cur_ref_sum + IMPROVEMENT_MARGIN:
            _do_rebuild(_state["best_centers"], _state["best_radii"],
                        _state["last_best"], cur_ref_sum, why="ref-outdated")
        elif _tree_has_unexpanded_leaf(tree_obj, BACKWARD_MAX_DEPTH):
            if not _do_expand():
                print(f"[backward_cb] expand exhausted — forcing rebuild")
                _do_rebuild(_state["best_centers"], _state["best_radii"],
                            _state["last_best"], cur_ref_sum, why="expand-exhausted")
        else:
            print(f"[backward_cb] stagnation hit but tree at max depth + ref fresh; nop")

        _state["counter"] = 0

    return _callback


def _seed_rolling_reference(run_root: Path, initial_program: Path) -> None:
    ref_npz = run_root / "reference_solution.npz"
    if ref_npz.exists():
        return
    spec = importlib.util.spec_from_file_location("_seed_initial", str(initial_program))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    centers, radii, _ = mod.run_packing()
    np.savez(
        str(ref_npz),
        centers=np.asarray(centers, dtype=float),
        radii=np.asarray(radii, dtype=float),
    )

search_task_sys_msg = """You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.636.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles
2. A pure hexagonal arrangement may not be optimal due to edge effects
3. The densest known circle packings often use a hybrid approach
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
5. Consider strategic placement of circles at square corners and edges
6. Adjusting the pattern to place larger circles at the center and smaller at the edges
7. The math literature suggests special arrangements for specific values of n
8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints

Be creative and try to find a new solution better than the best known result."""


COMPLEX_PATCH_TYPES = [
    "diff",
    "diff_ablate",
    "full",
    "cross_combine",
    "cross_translocate",
    "cross",
]
COMPLEX_PATCH_PROBS = [0.50, 0.05, 0.30, 0.05, 0.05, 0.05]


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from bench_hooks import HOOKS as _HOOKS_FOR_RANKING
    from shinka.database.score import set_raw_metric_key
    set_raw_metric_key(_HOOKS_FOR_RANKING.raw_metric_key)
    print(f"[score] adaptive bucket interp using RAW_KEY="
          f"'{_HOOKS_FOR_RANKING.raw_metric_key}'")

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg

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
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="00:05:00",
    )
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
            f"score by raw reported_sum and skip tree expansion."
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
