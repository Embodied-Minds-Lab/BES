"""Generic goal-tree dense scoring (problem-agnostic).

A goal tree is a JSON-friendly nested dict:
    {"id": str, "description": str, "verify_code": str, "children": [...], "self_score": float}

`verify_code` is a Python expression evaluated against a benchmark-supplied `eval_vars`
namespace (plus a small set of always-available builtins like `np`, `math`, `len`).
It MAY return:
  - a bool (True → 1.0, False → 0.0), or
  - a float in [0,1] for a dense partial-credit score (e.g. `min(1.0, count/target)`).
Anything outside [0,1] is clipped. Failures (exceptions / empty code) → 0.0.

`recursive_score` blends self with children:
  - if self >= 1 - SAT_EPS: short-circuit to 1.0 (parent satisfied);
  - elif leaf: return self;
  - else: SELF_W * self + CHILD_W * mean(child scores).
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

SELF_W = 0.3
CHILD_W = 0.7
SAT_EPS = 1e-6  # self_score >= 1 - SAT_EPS counts as fully satisfied

_BUILTINS: Dict[str, Any] = {
    "np": np,
    "math": math,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "all": all,
    "any": any,
}


def _make_ns(eval_vars: Dict[str, Any]) -> Dict[str, Any]:
    ns = dict(_BUILTINS)
    ns.update(eval_vars)
    return ns


def verify_node(node: Dict[str, Any], eval_vars: Dict[str, Any]) -> float:
    """Return a dense score in [0,1]. bool → {0.0, 1.0}; floats are clipped; errors → 0.0."""
    code = node.get("verify_code", "")
    if not code:
        return 0.0
    try:
        ns = _make_ns(eval_vars)
        result = eval(code, ns)
    except Exception:
        return 0.0
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    try:
        v = float(result)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def verify_tree(tree: Dict[str, Any], eval_vars: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of `tree` with `self_score` set on every node."""
    out = {**tree}
    out["self_score"] = verify_node(tree, eval_vars)
    out["children"] = [verify_tree(c, eval_vars) for c in tree.get("children", [])]
    return out


def recursive_score(node: Dict[str, Any]) -> float:
    self_s = float(node.get("self_score", 0.0))
    if self_s >= 1.0 - SAT_EPS:
        return 1.0
    children = node.get("children", [])
    if not children:
        return self_s
    child_mean = sum(recursive_score(c) for c in children) / len(children)
    return SELF_W * self_s + CHILD_W * child_mean


def flatten_satisfaction(node: Dict[str, Any]) -> Dict[str, float]:
    """Return {node_id: self_score in [0,1]} for every node in the tree."""
    out = {node["id"]: float(node.get("self_score", 0.0))}
    for c in node.get("children", []):
        out.update(flatten_satisfaction(c))
    return out
