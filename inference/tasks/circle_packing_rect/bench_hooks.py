from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shinka.bidirectional_search import BenchHooks

ELITE_K = 5
TARGET_SUM_R = 2.366

DECOMPOSE_PROMPT = """Decompose a circle-packing goal into smaller verifiable subgoals.

## Problem
Pack n=21 non-overlapping circles inside a rectangle of perimeter 4 (width + height = 2;
the program chooses the width/height ratio) to achieve sum_of_radii > 2.366
(STRICTLY EXCEED the AlphaEvolve best-known result). Every candidate already satisfies
validity (in-rect, non-overlap, r>=0). Do NOT use those as subgoals.

## Elite reference layouts (top {n_elites} from current archive — these define the search frontier)
Each block lists the rectangle dimensions (w, h) chosen by that elite, then 21 rows
of (x, y, r). Different elites may have different w/h ratios — when you propose a
subgoal, prefer thresholds that work across ratios (or normalize by w / h / area /
perimeter as appropriate).

Look ACROSS all of them to identify:
(a) structural properties EVERY elite has → strong reference-kind subgoals
(b) properties where elites still VARY → indicates room for improvement, target with aspirational subgoals
(c) properties NO elite has yet but a hypothetical sum_r > 2.366 solution would have → aspirational

{elite_blocks}

## Parent goal to decompose
{goal_desc}

## Two kinds of subgoals (BOTH required)
Produce 3–4 subgoals, MIXING the two kinds below:

A. `kind="reference"` (1–2 subgoals): structural properties that EVERY elite above already
   has and that naive layouts demonstrably LACK (a uniform 4×5 grid in a 1×1 square with
   r≈0.1 gives sum_r ≈ 2.0; a single row of 21 equal circles touches sum_r ≈ 2.1 only when
   the rect is extremely elongated). MUST be True on every elite AND False on those naive
   baselines. Generic properties that any symmetric layout has (e.g. "centroid near rect
   center", "mirror symmetry") are BAD — naive layouts satisfy them too. Good shapes:
   "max radius >= some_value", "at least K circles with r >= some_value", "at least one
   pair of tangent circles with combined radius > some_value", "rect aspect ratio in
   [a, b]". Pick the threshold so EVERY elite passes.

B. `kind="aspirational"` (2–3 subgoals): concrete structural ideas for HOW to push sum_r
   further. These typically evaluate FALSE on most or all elites — that is the point: they
   describe what an exceed-target solution would have but the current elites still lack.
   Each must ALSO be False on naive baselines (uniform grid ≈ 2.0 or single-row ring ≈ 2.1);
   otherwise a naive layout trivially passes despite a poor sum_r. Good shapes:
   - the largest circle is strictly larger than the max max_radius across elites;
   - more circles above some "large" radius cutoff than any elite achieves;
   - tighter local packing: many pairs with center-distance < r_i + r_j + epsilon (tangent clusters);
   - a structural motif (hexagonal core, nested ring, dense corner clusters, or rect
     elongation toward a particular ratio) requiring large radii.
   AVOID pure symmetry/centroid predicates — naive baselines satisfy them trivially.
   Set thresholds slightly beyond what the best elite exhibits — a layout that matches the
   elites still FAILS, but a layout that exceeds them can pass.

## Rules (both kinds)
- A strict sub-property of the parent — never a rephrasing of the parent itself.
- Pick subgoals from DIFFERENT categories: radius distribution · boundary usage · spatial
  coverage · symmetry · local geometry · rect aspect ratio. Don't write variants of one idea.

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
- `verify_code`: Python expression returning float in [0,1] (or bool). Available variables:
  `circles` ((21,3) — rows are (x,y,r)), `centers` ((21,2)), `radii` ((21,)), `x`, `y`,
  `n=21`, `sum_r=float(radii.sum())`, `w` (rect width), `h` (rect height), `np`. Must run
  without error.
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


def _rect_dims(circles: np.ndarray) -> tuple[float, float]:
    """Width and height of the minimum circumscribing rectangle of all circles."""
    if circles.size == 0:
        return 0.0, 0.0
    min_x = float(np.min(circles[:, 0] - circles[:, 2]))
    max_x = float(np.max(circles[:, 0] + circles[:, 2]))
    min_y = float(np.min(circles[:, 1] - circles[:, 2]))
    max_y = float(np.max(circles[:, 1] + circles[:, 2]))
    return max_x - min_x, max_y - min_y


def _eval_vars_from_arrays(circles: np.ndarray) -> Dict[str, Any]:
    circles = np.asarray(circles, dtype=float)
    centers = circles[:, :2]
    radii = circles[:, 2]
    w, h = _rect_dims(circles)
    return {
        "circles": circles,
        "centers": centers,
        "radii": radii,
        "x": centers[:, 0],
        "y": centers[:, 1],
        "n": int(radii.shape[0]),
        "sum_r": float(radii.sum()),
        "w": w,
        "h": h,
    }


def _ref_table(circles: np.ndarray) -> str:
    w, h = _rect_dims(circles)
    lines = [f"  rect: w={w:.4f}  h={h:.4f}"]
    for i, (x, y, r) in enumerate(circles):
        lines.append(f"  i={i:2d}  x={x:.4f}  y={y:.4f}  r={r:.4f}")
    return "\n".join(lines)


def _format_elite_blocks(elites: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for rank, ev in enumerate(elites, start=1):
        header = f"=== elite #{rank} (sum_r = {ev['sum_r']:.4f}, rank {rank}) ==="
        blocks.append(header + "\n" + _ref_table(ev["circles"]))
    return "\n\n".join(blocks)


def load_reference_eval_vars(npz_path: Path) -> Dict[str, Any]:
    d = np.load(str(npz_path))
    if "circles" in d.files:
        circles = d["circles"]
    elif "centers" in d.files and "radii" in d.files:
        circles = np.column_stack([d["centers"], d["radii"]])
    else:
        raise ValueError(
            f"reference npz {npz_path} must have either 'circles' or 'centers'+'radii' keys; "
            f"got {d.files}"
        )
    return _eval_vars_from_arrays(np.asarray(circles, dtype=float))


def load_gen_eval_vars(gen_dir: Path) -> Optional[Dict[str, Any]]:
    npz = Path(gen_dir) / "results" / "extra.npz"
    if not npz.exists():
        return None
    try:
        return load_reference_eval_vars(npz)
    except Exception:
        return None


def load_elite_eval_vars(run_root: Path, k: int = ELITE_K) -> List[Dict[str, Any]]:
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
