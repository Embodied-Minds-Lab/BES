from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shinka.bidirectional_search import BenchHooks

ELITE_K = 5

DECOMPOSE_PROMPT = """Decompose a circle-packing goal into smaller verifiable subgoals.

## Problem
Pack n=26 non-overlapping circles in [0,1]² to achieve sum_of_radii > 2.636 (STRICTLY EXCEED best-known).
Every candidate already satisfies validity (in-square, non-overlap, r>=0). Do NOT use those as subgoals.

## Elite reference layouts (top {n_elites} from current archive — these define the search frontier)
Look ACROSS all of them to identify:
(a) structural properties EVERY elite has → strong reference-kind subgoals
(b) properties where elites still VARY → indicates room for improvement, target with aspirational subgoals
(c) properties NO elite has yet but a hypothetical sum_r > best solution would have → aspirational

{elite_blocks}

## Parent goal to decompose
{goal_desc}

## Two kinds of subgoals (BOTH required)
Produce 3–4 subgoals, MIXING the two kinds below:

A. `kind="reference"` (1–2 subgoals): structural properties that EVERY elite above already
   has and that naive layouts demonstrably LACK (naive grid ≈ 2.4 or concentric ring ≈ 2.2).
   MUST be True on every elite AND False on a uniform-radii ring/grid. Generic properties
   that any symmetric layout has (e.g. "geometric center near (0.5,0.5)", "mirror symmetry")
   are BAD — naive rings satisfy them too. Good shapes: "max radius >= some_value", "at
   least K circles with r >= some_value", "at least one pair of tangent circles with
   combined radius > some_value". Pick the threshold so EVERY elite passes.

B. `kind="aspirational"` (2–3 subgoals): concrete structural ideas for HOW to push sum_r
   further. These typically evaluate FALSE on most or all elites — that is the point: they
   describe what an exceed-target solution would have but the current elites still lack.
   Each must ALSO be False on naive layouts (uniform grid ≈ 2.4 or ring of 26 equal circles
   ≈ 2.2); otherwise a naive layout trivially passes despite a poor sum_r. Good shapes:
   - the largest circle is strictly larger than the max max_radius across elites
     (naive baselines: grid r≈0.083, ring r≈0.12);
   - more circles above some "large" radius cutoff than any elite achieves;
   - tighter local packing: many pairs with center-distance < r_i + r_j + epsilon (tangent clusters);
   - a structural motif (hexagonal core, nested ring, dense corner clusters) requiring large radii.
   AVOID pure symmetry/centroid predicates — naive baselines satisfy them trivially.
   Set thresholds slightly beyond what the best elite exhibits — a layout that matches the
   elites still FAILS, but a layout that exceeds them can pass.

## Rules (both kinds)
- A strict sub-property of the parent — never a rephrasing of the parent itself.
- Pick subgoals from DIFFERENT categories: radius distribution · boundary usage · spatial
  coverage · symmetry · local geometry. Don't write variants of one idea.

## verify_code: return a DENSE score in [0,1], not just bool
`verify_code` should evaluate to a float in [0,1] representing partial credit — NOT a bool.
Use the form `min(1.0, <actual> / <target>)` so that progress toward the goal earns credit.
A bool is accepted (True→1.0, False→0.0) but wastes gradient; prefer dense.

CRITICAL FORMAT REQUIREMENT — **single Python expression only**:
  `verify_code` is evaluated via Python's `eval()`. It MUST be a single expression
  (no semicolons, no multi-line `x = …; y = …; result` chains, no `def`/`for`/`if`
  statements, no newline-separated assignments). If you need intermediate values,
  inline them or use a one-shot generator/comprehension.

  GOOD (single expression):
    `min(1.0, np.sum(radii >= 0.11) / 9)`
    `min(1.0, sum(1 for i in range(n) for j in range(i+1,n)
                   if np.linalg.norm(centers[i]-centers[j]) <= radii[i]+radii[j]+1e-3) / 30)`

  BAD (will silently score 0 — DO NOT EMIT):
    `d = np.linalg.norm(...); cnt = (d <= ...).sum(); min(1.0, cnt/30)`   ← semicolons
    `tol = 1e-3
     touch = ...
     min(1.0, touch.sum()/4)`                                              ← newlines

## Output
Each subgoal:
- `kind`: "reference" or "aspirational".
- `description`: short sentence; for aspirational, briefly say WHY it pushes.
- `verify_code`: Python expression returning float in [0,1] (or bool). Uses `centers` ((26,2)),
  `radii` ((26,)), `n=26`, `sum_r=float(radii.sum())`, `np`. Must run without error.
- `expected_result`: typical value across the elite reference layouts (1.0 if every elite
  meets it; a fraction < 1.0 if elites only partially meet it).

Output ONLY a JSON array:

```json
[
  {{"kind": "reference",    "description": "...", "verify_code": "...", "expected_result": "..."}},
  {{"kind": "aspirational", "description": "...", "verify_code": "...", "expected_result": "..."}}
]
```
"""


def _eval_vars_from_arrays(centers: np.ndarray, radii: np.ndarray) -> Dict[str, Any]:
    return {
        "centers": centers,
        "radii": radii,
        "x": centers[:, 0],
        "y": centers[:, 1],
        "n": int(radii.shape[0]),
        "sum_r": float(radii.sum()),
    }


def _ref_table(centers: np.ndarray, radii: np.ndarray) -> str:
    lines = []
    for i, ((x, y), r) in enumerate(zip(centers, radii)):
        lines.append(f"  i={i:2d}  x={x:.4f}  y={y:.4f}  r={r:.4f}")
    return "\n".join(lines)


def _format_elite_blocks(elites: List[Dict[str, Any]]) -> str:
    """Render a list of elite eval_vars dicts as numbered blocks for the prompt."""
    blocks: List[str] = []
    for rank, ev in enumerate(elites, start=1):
        header = f"=== elite #{rank} (sum_r = {ev['sum_r']:.4f}, rank {rank}) ==="
        blocks.append(header + "\n" + _ref_table(ev["centers"], ev["radii"]))
    return "\n\n".join(blocks)


def load_reference_eval_vars(npz_path: Path) -> Dict[str, Any]:
    d = np.load(str(npz_path))
    return _eval_vars_from_arrays(d["centers"], d["radii"])


def load_gen_eval_vars(gen_dir: Path) -> Optional[Dict[str, Any]]:
    npz = Path(gen_dir) / "results" / "extra.npz"
    if not npz.exists():
        return None
    try:
        d = np.load(str(npz))
        return _eval_vars_from_arrays(d["centers"], d["radii"])
    except Exception:
        return None


def load_elite_eval_vars(run_root: Path, k: int = ELITE_K) -> List[Dict[str, Any]]:
    """Return up to k top-scoring CORRECT programs' eval_vars, best-first.

    Pulls (id, generation, combined_score) from <run_root>/programs.sqlite, then loads each
    program's gen_<N>/results/extra.npz to recover (centers, radii). Skips any gen whose
    extra.npz is missing or corrupt. Returns [] if the DB is missing or has no eligible
    rows yet.
    """
    db_path = Path(run_root) / "programs.sqlite"
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=5
        )
        cur = con.cursor()
        cur.execute(
            "SELECT generation, combined_score FROM programs "
            "WHERE correct=1 AND combined_score IS NOT NULL "
            "ORDER BY combined_score DESC LIMIT ?",
            (k * 4,),
        )
        rows = cur.fetchall()
    except Exception:
        return []
    finally:
        try:
            con.close()
        except Exception:
            pass

    out: List[Dict[str, Any]] = []
    for gen, _score in rows:
        gen_dir = Path(run_root) / f"gen_{int(gen)}"
        ev = load_gen_eval_vars(gen_dir)
        if ev is None:
            continue
        out.append(ev)
        if len(out) >= k:
            break
    return out


def build_prompt_kwargs(
    node: Dict[str, Any],
    ref_eval_vars: Dict[str, Any],
    elite_eval_vars: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    elites = elite_eval_vars or [ref_eval_vars]
    return {
        "n_elites": len(elites),
        "elite_blocks": _format_elite_blocks(elites),
        "goal_desc": node["description"],
    }


HOOKS = BenchHooks(
    prompt_template=DECOMPOSE_PROMPT,
    build_prompt_kwargs=build_prompt_kwargs,
    load_reference_eval_vars=load_reference_eval_vars,
    load_gen_eval_vars=load_gen_eval_vars,
    load_elite_eval_vars=load_elite_eval_vars,
    model="gpt-5",
    raw_metric_key="reported_sum_of_radii",
)
