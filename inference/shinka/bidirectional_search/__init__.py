"""Bidirectional search: dynamic goal-tree expansion + dense recursive scoring.

Use this to run backward (decompose objective → subgoals) search interleaved with
forward (LLM-evolve programs) search, scoring programs by how densely they satisfy
a benchmark's goal tree.

Benchmark integration: implement a `BenchHooks` object (prompt template + callbacks
for loading `eval_vars`) and pass it to `bootstrap` / `try_expand_if_due`.

Public API:
  - goal_tree: verify_node, verify_tree, recursive_score, flatten_satisfaction
  - decompose: decompose_subgoals, build_goal_tree
  - expander: BenchHooks, bootstrap, try_expand_if_due, expand_once,
              count_satisfactions, rescore_all_programs,
              rebuild_tree_with_new_reference
"""
from .decompose import (
    build_goal_tree,
    decompose_subgoals,
)
from .expander import (
    BenchHooks,
    DEFAULT_INTERVAL,
    DEFAULT_MAX_DEPTH,
    SAT_THRESHOLD,
    bootstrap,
    count_satisfactions,
    expand_once,
    pick_never_satisfied_leaf,
    rebuild_tree_with_new_reference,
    rescore_all_programs,
    try_expand_if_due,
)
from .goal_tree import (
    CHILD_W,
    SAT_EPS,
    SELF_W,
    flatten_satisfaction,
    recursive_score,
    verify_node,
    verify_tree,
)

__all__ = [
    # goal_tree
    "verify_node",
    "verify_tree",
    "recursive_score",
    "flatten_satisfaction",
    "SELF_W",
    "CHILD_W",
    "SAT_EPS",
    # decompose
    "decompose_subgoals",
    "build_goal_tree",
    # expander
    "BenchHooks",
    "bootstrap",
    "try_expand_if_due",
    "expand_once",
    "count_satisfactions",
    "pick_never_satisfied_leaf",
    "rebuild_tree_with_new_reference",
    "rescore_all_programs",
    "DEFAULT_INTERVAL",
    "DEFAULT_MAX_DEPTH",
    "SAT_THRESHOLD",
]
