"""Dynamic goal-tree expansion for bidirectional search (problem-agnostic).

Every `interval` generations, pick one leaf that has NEVER been fully satisfied
(self_score >= 1 - EPS) by any historical correct program, ask the LLM to decompose it
into 2-4 subgoals, validate each new subgoal against the reference, and append the
validated ones as children.

Monotonicity guarantee: because we only add children to never-fully-satisfied leaves,
`recursive_score` for every historical program is weakly non-decreasing across tree
mutations.

Concurrency: uses an fcntl flock on `<tree>.lock` + a JSON state file to ensure at
most one expansion per generation boundary under `max_evaluation_jobs > 1`.

Benchmark integration: the caller supplies a `BenchHooks` object that plugs in
problem-specific data loading and prompt construction.
"""
from __future__ import annotations

import fcntl
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .decompose import decompose_subgoals
from .goal_tree import (
    SAT_EPS,
    flatten_satisfaction,
    recursive_score,
    verify_tree,
)

DEFAULT_INTERVAL = 5
DEFAULT_MAX_DEPTH = 3
SAT_THRESHOLD = 1.0 - SAT_EPS


@dataclass
class BenchHooks:
    """Benchmark-specific callbacks for bidirectional search.

    - prompt_template: Python `str.format`-style template (the full decomposition prompt).
    - build_prompt_kwargs(node, ref_eval_vars, elite_eval_vars=None) -> dict: kwargs to fill
      the template. `elite_eval_vars` (a list of eval_vars dicts for top-K archive elites,
      best-first) is passed only when `load_elite_eval_vars` is defined.
    - load_reference_eval_vars(reference_src) -> dict: load eval_vars from a benchmark's
      canonical reference file (e.g. a .npz).
    - load_gen_eval_vars(gen_dir) -> Optional[dict]: extract eval_vars from a past
      generation's results directory (for re-scoring history). Return None if missing.
    - load_elite_eval_vars(run_root) -> List[dict]: optional. If set, expand_once gathers
      the top-K elite programs (by combined_score, correct=1) from <run_root>/programs.sqlite
      and passes their eval_vars list to build_prompt_kwargs as `elite_eval_vars=`. Used to
      decompose against the search frontier rather than a single reference.
    - model: LLM model id for decomposition.
    - raw_metric_key: name of the per-benchmark scalar in private_metrics that
      shinka.database.score uses as the dominant ranking key (raw bucket; bw_score
      acts as intra-bucket sub-rank). Each benchmark must declare this so
    """
    prompt_template: str
    build_prompt_kwargs: Callable[..., Dict[str, Any]]
    load_reference_eval_vars: Callable[[Path], Dict[str, Any]]
    load_gen_eval_vars: Callable[[Path], Optional[Dict[str, Any]]]
    model: str = "gemini-3-pro-preview"
    load_elite_eval_vars: Optional[Callable[[Path], List[Dict[str, Any]]]] = None
    raw_metric_key: str = "reported_sum_of_radii"


def count_satisfactions(db_path: Path) -> Dict[str, int]:
    """Return `{node_id: #correct programs that fully satisfied it}` across all history.

    With dense scores, "satisfied" means self_score >= SAT_THRESHOLD (≈ 1.0). Nodes with
    only partial credit across all programs are treated as not-yet-satisfied so they
    remain candidates for further decomposition.
    """
    counts: Dict[str, int] = {}
    if not Path(db_path).exists():
        return counts
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT private_metrics FROM programs "
            "WHERE correct=1 AND private_metrics IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    for (priv_json,) in rows:
        try:
            priv = json.loads(priv_json)
        except Exception:
            continue
        sat = priv.get("backward_satisfaction") or {}
        for node_id, val in sat.items():
            try:
                score = float(val)
            except Exception:
                score = 1.0 if bool(val) else 0.0
            if score >= SAT_THRESHOLD:
                counts[node_id] = counts.get(node_id, 0) + 1
    return counts


def _iter_leaves(node: Dict[str, Any], depth: int = 0):
    children = node.get("children") or []
    if not children:
        yield depth, node
    for c in children:
        yield from _iter_leaves(c, depth + 1)


def pick_never_satisfied_leaf(
    tree: Dict[str, Any],
    counts: Dict[str, int],
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Optional[Dict[str, Any]]:
    """Pick the shallowest never-satisfied leaf that can still be decomposed.

    A leaf is eligible iff its `level` is strictly less than `max_depth`. The root
    is at level 0, so with `max_depth=3` we may expand levels 0/1/2; leaves already
    at level 3 are frozen.
    """
    candidates: List[tuple] = []
    for _, node in _iter_leaves(tree):
        if counts.get(node["id"], 0) > 0:
            continue
        if int(node.get("level", 0)) >= max_depth:
            continue
        candidates.append((int(node.get("level", 0)), node["id"], node))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


def _max_generation(db_path: Path) -> Optional[int]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT MAX(generation) FROM programs WHERE correct=1"
        ).fetchone()
    finally:
        conn.close()
    return None if row is None or row[0] is None else int(row[0])


def expand_once(
    tree_path: Path,
    db_path: Path,
    reference_src: Path,
    hooks: BenchHooks,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Optional[Dict[str, Any]]:
    """Expand one never-fully-satisfied leaf. Caller must hold the file lock.

    Returns the expanded node (with new `children` populated) or None if no candidate
    or all LLM-generated subgoals failed reference validation. Leaves whose level
    has reached `max_depth` are never picked.
    """
    tree = json.loads(tree_path.read_text())
    counts = count_satisfactions(db_path)
    target = pick_never_satisfied_leaf(tree, counts, max_depth=max_depth)
    if target is None:
        return None

    ref_eval_vars = hooks.load_reference_eval_vars(Path(reference_src))
    elite_eval_vars = None
    if hooks.load_elite_eval_vars is not None:
        try:
            elite_eval_vars = hooks.load_elite_eval_vars(Path(reference_src).parent)
        except Exception as e:
            print(f"[expand_once] load_elite_eval_vars failed: {type(e).__name__}: {e}")
            elite_eval_vars = None
    prompt_kwargs = hooks.build_prompt_kwargs(
        target, ref_eval_vars, elite_eval_vars=elite_eval_vars
    )
    subs = decompose_subgoals(hooks.model, hooks.prompt_template, prompt_kwargs)

    new_level = int(target.get("level", 0)) + 1
    kept: List[Dict[str, Any]] = [
        {
            "id": f"{target['id']}.L{new_level}_{j}",
            "level": new_level,
            "kind": s.get("kind", "reference"),
            "description": s["description"],
            "verify_code": s["verify_code"],
            "expected_result": s["expected_result"],
            "children": [],
        }
        for j, s in enumerate(subs)
    ]
    if not kept:
        return None

    def _find(n, tid):
        if n.get("id") == tid:
            return n
        for c in n.get("children") or []:
            r = _find(c, tid)
            if r is not None:
                return r
        return None

    tgt = _find(tree, target["id"])
    assert tgt is not None, f"internal: lost node {target['id']}"
    tgt["children"] = kept
    tree_path.write_text(json.dumps(tree, indent=2))

    rescore_all_programs(tree_path, db_path, tree_path.parent, hooks)
    return tgt


def rescore_all_programs(
    tree_path: Path, db_path: Path, run_root: Path, hooks: BenchHooks
) -> int:
    """Re-run verify_tree + recursive_score for every correct program using the
    current tree, and UPDATE combined_score + private_metrics + public_metrics in
    SQLite. Returns the number of rows updated.
    """
    if not Path(db_path).exists():
        return 0
    tree = json.loads(tree_path.read_text())
    conn = sqlite3.connect(str(db_path))
    updated = 0
    try:
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='programs'"
        ).fetchone()
        if not has_table:
            return 0
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT id, generation, private_metrics, public_metrics FROM programs WHERE correct=1"
        ).fetchall()
        for pid, gen, priv_json, pub_json in rows:
            gen_dir = run_root / f"gen_{gen}"
            eval_vars = hooks.load_gen_eval_vars(gen_dir)
            if eval_vars is None:
                continue
            verified = verify_tree(tree, eval_vars)
            new_score = float(recursive_score(verified))
            new_sat = flatten_satisfaction(verified)
            priv = json.loads(priv_json) if priv_json else {}
            pub = json.loads(pub_json) if pub_json else {}
            priv["backward_satisfaction"] = new_sat
            pub["backward_score"] = new_score
            pub["backward_subgoals_satisfied"] = sum(new_sat.values())
            pub["backward_subgoals_total"] = len(new_sat)
            conn.execute(
                "UPDATE programs SET combined_score=?, private_metrics=?, public_metrics=? WHERE id=?",
                (new_score, json.dumps(priv), json.dumps(pub), pid),
            )
            updated += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return updated


def bootstrap(
    run_root: Path,
    template_path: Path,
    reference_src: Path,
    hooks: BenchHooks,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Optional[str]:
    """One-time setup before the first forward generation.

    1. Seed <run_root>/goal_tree.json from `template_path` if absent.
    2. Seed <run_root>/reference_solution.npz (or whatever the benchmark calls it; this
       module writes bytes to `<run_root>/reference_solution.npz` for consistency with
       later evaluations) if absent.
    3. Run expand_once to decompose root into L1 subgoals.

    Returns the expanded node id, or None if decomposition failed.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    tree_path = run_root / "goal_tree.json"
    ref_npz = run_root / "reference_solution.npz"
    if not tree_path.exists():
        tree_path.write_text(Path(template_path).read_text())
    if not ref_npz.exists():
        ref_npz.write_bytes(Path(reference_src).read_bytes())

    db_path = run_root / "programs.sqlite"  # may not exist yet

    lock_path = str(tree_path) + ".lock"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            expanded = expand_once(
                tree_path, db_path, ref_npz, hooks, max_depth=max_depth
            )
            return expanded["id"] if expanded else None
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def rebuild_tree_with_new_reference(
    run_root: Path,
    template_path: Path,
    new_reference_npz_bytes: bytes,
    hooks: BenchHooks,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Optional[str]:
    """Replace the rolling reference and rebuild the goal tree from scratch.

    Triggered when the benchmark side decides the rolling reference should advance
    (e.g., forward search produced a strictly better candidate). Acquires the same
    file lock as bootstrap. Steps:
      1. overwrite <run_root>/reference_solution.npz with the supplied bytes
      2. reset <run_root>/goal_tree.json to the template
      3. delete <run_root>/tree_expander_state.json so the next periodic call fires
      4. call expand_once once to grow root → L1 against the new reference
         (rescore_all_programs is invoked inside expand_once)

    Returns the id of the expanded root, or None if decomposition produced nothing.
    """
    run_root = Path(run_root)
    tree_path = run_root / "goal_tree.json"
    state_path = run_root / "tree_expander_state.json"
    ref_npz = run_root / "reference_solution.npz"
    db_path = run_root / "programs.sqlite"

    lock_path = str(tree_path) + ".lock"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            ref_npz.write_bytes(new_reference_npz_bytes)
            tree_path.write_text(Path(template_path).read_text())
            if state_path.exists():
                state_path.unlink()
            expanded = expand_once(
                tree_path, db_path, ref_npz, hooks, max_depth=max_depth
            )
            return expanded["id"] if expanded else None
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def try_expand_if_due(
    tree_path: Path,
    db_path: Path,
    reference_npz: Path,
    state_path: Path,
    hooks: BenchHooks,
    interval: int = DEFAULT_INTERVAL,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Optional[str]:
    """File-locked wrapper. Returns expanded node_id, or None if not due / no candidate."""
    tree_path = Path(tree_path)
    db_path = Path(db_path)
    reference_npz = Path(reference_npz)
    state_path = Path(state_path)

    current_gen = _max_generation(db_path)
    if current_gen is None:
        return None

    lock_path = str(tree_path) + ".lock"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            state: Dict[str, Any] = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text())
                except Exception:
                    state = {}
            last = int(state.get("last_expanded_gen", -interval))
            if current_gen - last < interval:
                return None

            if not reference_npz.exists():
                return None

            expanded = expand_once(
                tree_path, db_path, reference_npz, hooks, max_depth=max_depth
            )
            state["last_expanded_gen"] = current_gen
            history = state.setdefault("history", [])
            history.append({
                "gen": current_gen,
                "expanded_node": expanded["id"] if expanded else None,
                "num_new_children": len(expanded["children"]) if expanded else 0,
            })
            state_path.write_text(json.dumps(state, indent=2))
            return expanded["id"] if expanded else None
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
