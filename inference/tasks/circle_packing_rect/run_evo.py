#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

REPO_DIR = Path(__file__).parent
TEMPLATE_TREE = REPO_DIR / "goal_tree_template.json"
INITIAL_PROGRAM = REPO_DIR / "initial.py"

IMPROVEMENT_MARGIN = 1e-2  # raw_sum_r gain must exceed this to count as "improvement"
STAGNATION_THRESHOLD = 5    # consecutive non-improving forward gens to trigger backward
BACKWARD_MAX_DEPTH = 2      # root + L1 + L2

TASK_SYS_MSG = """SETTING:
You are an expert computational geometer and optimization specialist with deep expertise in circle packing problems, geometric optimization algorithms, and constraint satisfaction.
Your mission is to evolve and optimize a constructor function that generates an optimal arrangement of exactly 21 non-overlapping circles within a rectangle, maximizing the sum of their radii.

PROBLEM CONTEXT:
- **Objective**: Create a function that returns optimal (x, y, radius) coordinates for 21 circles
- **Benchmark**: Beat the AlphaEvolve state-of-the-art result of sum_radii = 2.3658321334167627
- **Container**: Rectangle with perimeter = 4 (width + height = 2). You may choose optimal width/height ratio
- **Constraints**:
  * All circles must be fully contained within rectangle boundaries
  * No circle overlaps (distance between centers >= sum of their radii)
  * Exactly 21 circles required
  * All radii must be positive

PERFORMANCE METRICS:
1. **sum_radii**: Total sum of all 21 circle radii (PRIMARY OBJECTIVE - maximize)
2. **combined_score**: sum_radii / 2.3658321334167627 (progress toward beating benchmark)
3. **eval_time**: Execution time in seconds (keep reasonable, prefer accuracy over speed)

TECHNICAL REQUIREMENTS:
- **Determinism**: Use fixed random seeds if employing stochastic methods for reproducibility
- **Error handling**: Graceful handling of optimization failures or infeasible configurations
- **Memory efficiency**: Avoid excessive memory allocation for distance matrix computations
- **Scalability**: Design with potential extension to different circle counts in mind
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


# ---------------------------------------------------------------------------
# Backward archive helpers
# ---------------------------------------------------------------------------

def _archive_dir(run_root: Path) -> Path:
    d = run_root / "backward_history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_archive_seq(run_root: Path) -> int:
    d = _archive_dir(run_root)
    used = []
    for child in d.iterdir():
        if not child.is_dir():
            continue
        try:
            used.append(int(child.name.split("_", 1)[0]))
        except Exception:
            continue
    return (max(used) + 1) if used else 0


def _snapshot_archive_pre(run_root: Path, action: str, gen: int, meta_extra: Dict[str, Any]) -> Path:
    seq = _next_archive_seq(run_root)
    name = f"{seq:03d}_{action}_g{gen}"
    d = _archive_dir(run_root) / name
    d.mkdir(parents=True, exist_ok=True)

    tree_path = run_root / "goal_tree.json"
    ref_path = run_root / "reference_solution.npz"
    if tree_path.exists():
        shutil.copy2(tree_path, d / "tree_before.json")
    if ref_path.exists():
        shutil.copy2(ref_path, d / "reference_before.npz")

    meta = {
        "seq": seq,
        "action": action,
        "triggered_at_gen": gen,
        "ts": time.time(),
        **meta_extra,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def _snapshot_archive_post(archive_dir: Path, run_root: Path, post_extra: Dict[str, Any]) -> None:
    tree_path = run_root / "goal_tree.json"
    ref_path = run_root / "reference_solution.npz"
    if tree_path.exists():
        shutil.copy2(tree_path, archive_dir / "tree_after.json")
    if ref_path.exists():
        shutil.copy2(ref_path, archive_dir / "reference_after.npz")
    meta_path = archive_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta.update(post_extra)
    meta_path.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Backward callback (stagnation-driven, decision-tree action selection)
# ---------------------------------------------------------------------------

def _make_backward_callback(run_root: Path):
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
        "best_circles": None,
    }

    def _read_extras(jobs):
        for job in jobs:
            extra = Path(job.results_dir) / "extra.npz"
            if not extra.exists():
                continue
            try:
                d = np.load(str(extra))
                if "circles" in d.files:
                    circles = np.asarray(d["circles"], dtype=float)
                elif "centers" in d.files and "radii" in d.files:
                    circles = np.column_stack([d["centers"], d["radii"]]).astype(float)
                else:
                    continue
                yield int(job.generation), circles, float(circles[:, 2].sum())
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

    def _do_rebuild(circles, new_sum, cur_sum, why, gen) -> bool:
        # archive snapshot BEFORE rebuilds tree+ref
        adir = _snapshot_archive_pre(
            run_root, action=f"rebuild_{why}", gen=gen,
            meta_extra={"prev_ref_sum_r": cur_sum, "new_ref_sum_r": new_sum},
        )
        buf = io.BytesIO()
        np.savez(buf,
                 circles=circles,
                 centers=circles[:, :2],
                 radii=circles[:, 2])
        try:
            last = rebuild_tree_with_new_reference(
                run_root,
                TEMPLATE_TREE,
                buf.getvalue(),
                hooks=HOOKS,
                max_depth=BACKWARD_MAX_DEPTH,
            )
            print(
                f"[backward_cb] rebuild ({why}) gen={gen} "
                f"new ref sum_r={new_sum:.6f} (was {cur_sum:.6f}); expanded={last}"
            )
            _snapshot_archive_post(adir, run_root, {"expanded_node": last, "ok": True})
            return True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[backward_cb] rebuild ({why}) failed gen={gen}: {err}")
            _snapshot_archive_post(adir, run_root, {"ok": False, "error": err})
            return False

    def _do_expand(gen) -> bool:
        adir = _snapshot_archive_pre(run_root, action="expand", gen=gen, meta_extra={})
        try:
            expanded = try_expand_if_due(
                tree_path, db_path, ref_npz, state_path,
                hooks=HOOKS, interval=1, max_depth=BACKWARD_MAX_DEPTH,
            )
            if expanded:
                print(f"[backward_cb] expanded node: {expanded}")
                _snapshot_archive_post(adir, run_root, {"expanded_node": expanded, "ok": True})
                return True
            print(f"[backward_cb] expand: nothing to do")
            _snapshot_archive_post(adir, run_root, {"expanded_node": None, "ok": True,
                                                    "note": "nothing to expand"})
            return False
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[backward_cb] expand failed: {err}")
            _snapshot_archive_post(adir, run_root, {"ok": False, "error": err})
            return False

    async def _callback(jobs, runner):
        # Walk just-completed jobs in generation order; update stagnation counter.
        for gen, circles, new_sum in sorted(_read_extras(jobs)):
            if new_sum > _state["last_best"] + IMPROVEMENT_MARGIN:
                _state["counter"] = 0
                _state["last_best"] = new_sum
                _state["best_circles"] = circles
                print(f"[backward_cb] gen={gen} sum_r={new_sum:.4f} (IMPROVED, counter reset)")
            else:
                _state["counter"] += 1
                if new_sum > _state["last_best"]:
                    _state["last_best"] = new_sum
                    _state["best_circles"] = circles
                print(
                    f"[backward_cb] gen={gen} sum_r={new_sum:.4f} "
                    f"(no-improve, counter={_state['counter']}/{STAGNATION_THRESHOLD})"
                )

        if _state["counter"] < STAGNATION_THRESHOLD:
            return
        if _state["best_circles"] is None:
            return  # no valid candidate yet

        latest_gen = max((int(j.generation) for j in jobs), default=-1)

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
                try:
                    cur_ref_sum = float(np.load(str(ref_npz))["circles"][:, 2].sum())
                except Exception:
                    pass

        no_real_tree = tree_obj is None or not tree_obj.get("children")
        if no_real_tree:
            _do_rebuild(_state["best_circles"], _state["last_best"], cur_ref_sum,
                        why="bootstrap", gen=latest_gen)
        elif _state["last_best"] > cur_ref_sum + IMPROVEMENT_MARGIN:
            _do_rebuild(_state["best_circles"], _state["last_best"], cur_ref_sum,
                        why="ref-outdated", gen=latest_gen)
        elif _tree_has_unexpanded_leaf(tree_obj, BACKWARD_MAX_DEPTH):
            if not _do_expand(latest_gen):
                print(f"[backward_cb] expand exhausted — forcing rebuild")
                ok = _do_rebuild(
                    _state["best_circles"], _state["last_best"], cur_ref_sum,
                    why="expand-exhausted", gen=latest_gen,
                )
                if ok:
                    runner.queued_patch_types.append("diff_ablate")
                    print(
                        f"[backward_cb] queued next forward patch_type=diff_ablate "
                        f"(pivot trigger; queue={runner.queued_patch_types})"
                    )
        else:
            print(f"[backward_cb] stagnation hit but tree at max depth + ref fresh; nop")

        _state["counter"] = 0

    return _callback


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from bench_hooks import HOOKS as _HOOKS_FOR_RANKING
    from shinka.database.score import set_raw_metric_key
    set_raw_metric_key(_HOOKS_FOR_RANKING.raw_metric_key)
    print(f"[score] adaptive bucket interp using RAW_KEY="
          f"'{_HOOKS_FOR_RANKING.raw_metric_key}'")

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG

    # Suffix results_dir with SLURM_JOB_ID so concurrent / repeat runs don't collide.
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
        backward_callback = _make_backward_callback(run_root)
        print(
            f"[backward_search] enabled (lazy bootstrap: triggers after "
            f"{STAGNATION_THRESHOLD} consecutive forward gens with "
            f"improvement <= {IMPROVEMENT_MARGIN}). "
            f"Every backward action is archived to {run_root}/backward_history/."
        )
    else:
        backward_marker.touch()
        print(
            f"[backward_search] OFF — wrote {backward_marker}; evaluate.py will "
            f"score by raw radii_sum/BENCHMARK and skip tree expansion."
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
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    main(p.parse_args().config_path)
