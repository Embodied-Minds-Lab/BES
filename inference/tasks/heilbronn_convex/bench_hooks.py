from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import ConvexHull

from shinka.bidirectional_search import BenchHooks

NUM_POINTS = 13
BENCHMARK = 0.030936889034895654
ELITE_K = 5


DECOMPOSE_PROMPT = """Decompose a Heilbronn-convex (n=13) goal into smaller verifiable subgoals.

## Problem
Place n=13 points in 2D to maximize min_area = the area of the smallest triangle
formed by any 3 of the 13 points, normalized by the area of the convex hull of
the 13 points (so the score is scale-invariant; only the configuration shape
matters). Scoring: min_area_normalized = min_area / convex_hull_area ; beat
benchmark min_area_normalized >= 0.030936889 (AlphaEvolve), i.e.
combined_score = min_area_normalized / BENCHMARK >= 1.0 (STRICTLY EXCEED).

The convex region is NOT fixed — each candidate program picks its own 13 points
and the convex hull is computed from those points. So shape-of-region is part
of the search; degenerate point sets (all collinear, all clustered) still get
a well-defined score (zero or near-zero).

Every candidate already satisfies validity (returns a (13,2) finite array).
Do NOT use that as a subgoal.

## Elite reference layouts (top {n_elites} from current archive — these define the search frontier)
Look ACROSS all of them to identify:
(a) structural properties EVERY elite has → strong reference-kind subgoals
(b) properties where elites still VARY → indicates room for improvement, target with aspirational subgoals
(c) properties NO elite has yet but a hypothetical >BENCHMARK solution would have → aspirational

{elite_blocks}

## Parent goal to decompose
{goal_desc}

## Two kinds of subgoals (BOTH required)
Produce 3-4 subgoals, MIXING the two kinds below:

A. `kind="reference"` (1-2 subgoals): structural properties EVERY elite above already
   has and that naive baselines demonstrably LACK. Naive baselines:
     - NAIVE_RANDOM (uniform random in [0,1]^2): min_area_normalized ~ 1e-4 to 5e-3
     - NAIVE_GRID (lattice or square-grid 13 points): min_area_normalized ~ 1e-2, bounded
     - NAIVE_REGULAR (a regular 13-gon, closed-form): min_area_normalized ~ 0.018 (≈ 0.6× benchmark)
     - NAIVE_COLLAPSED (all 13 points clustered or near-collinear): score ≈ 0
   Good shapes:
     - "k-th smallest triangle area >= some_threshold" (anti-degeneracy)
     - "diameter of point set >= some_value × sqrt(convex_hull_area)" (avoid collapse)
     - "smallest pairwise distance >= some_threshold" (no near-duplicates)
   Pick the threshold so EVERY elite passes AND every naive — including regular 13-gon
   — clearly fails. Generic properties that any symmetric/closed-form construction
   satisfies are BAD: regular 13-gon already scores ~0.6× benchmark, so a subgoal it
   trivially passes can't push the search past that ceiling.

B. `kind="aspirational"` (2-3 subgoals): concrete structural ideas for HOW to push
   min_area_normalized further. These typically evaluate FALSE on most or all elites
   — that is the point: they describe what an exceed-benchmark solution would have
   but elites still lack. Each must ALSO fail on naive baselines (including regular
   13-gon — see scores above). Good shapes:
     - "the 2nd-smallest triangle area > 1.3 × min_area" (push the floor up — fewer
        near-degenerate triangle pairs);
     - "no triple of points within ε of collinear (no near-degenerate triangles
        beyond the very smallest)";
     - "convex_hull_area / (diameter^2) >= some_value" (round, balanced hull rather
        than thin/elongated);
     - "max triangle area / min triangle area <= some_ratio" (uniformity of the
        triangle area distribution).
   Set thresholds slightly stronger than the best elite exhibits — a layout matching
   only the elites still FAILS, and one matching only the regular 13-gon also fails.

## Rules (both kinds)
- A strict sub-property of the parent — never a rephrasing of the parent itself.
- Pick subgoals from DIFFERENT categories: triangle-area distribution · convex hull
  utilization · spatial spread · near-collinearity avoidance. Don't write variants
  of one idea.

## verify_code: return a DENSE score in [0,1], not just bool
`verify_code` should evaluate to a float in [0,1] representing partial credit -- NOT
a bool. Use the form `min(1.0, <actual> / <target>)` so progress toward the goal
earns credit. A bool is accepted (True -> 1.0, False -> 0.0) but wastes gradient.

CRITICAL FORMAT REQUIREMENT — **single Python expression only**:
  `verify_code` is evaluated via Python's `eval()`. It MUST be a single expression
  (no semicolons, no multi-line `x = ...; y = ...; result` chains, no `def`/`for`/`if`
  statements, no newline-separated assignments). If you need intermediate values,
  inline them or use a one-shot generator/comprehension. For ratios involving
  triangle_areas[0] or min_area, use a hull-fraction denominator floor so
  degenerate (0/0) inputs score near 0 rather than 1.

  GOOD (single expression):
    `min(1.0, np.sum(np.diff(np.sort(triangle_areas)[:5])) / 0.005)`
    `min(1.0, triangle_areas[1] / max(triangle_areas[0], 1e-3 * convex_hull_area + 1e-12) / 1.3)`

  BAD (will silently misbehave — DO NOT EMIT):
    `s = np.sort(triangle_areas); min(1.0, s[1]/s[0]/1.3)`     ← semicolons
    `min(1.0, triangle_areas[1] / triangle_areas[0])`           ← 0/0 → 1 on collapsed
    `tol = 0.05
     m = min_area
     min(1.0, ...)`                                             ← newlines

## Available namespace for verify_code

Arrays / scalars (all derived from the candidate's 13 points):
  `points`            : (13, 2) np.ndarray
  `x`, `y`            : points[:,0], points[:,1]
  `n`                 : 13 (int)
  `min_area`          : float, smallest triangle area among C(13,3)=286 triangles
  `max_area`          : float, largest triangle area
  `triangle_areas`    : (286,) np.ndarray of all triangle areas, sorted ascending
  `convex_hull_area`  : float, area of the convex hull of the 13 points
  `min_area_normalized` : min_area / convex_hull_area (the primary metric)
  `combined_score`    : min_area_normalized / BENCHMARK (target ≥ 1.0)
  `pairwise_min`      : float, smallest pairwise distance
  `pairwise_max`      : float, largest pairwise distance (≈ diameter)
  `centroid`          : (2,) array, mean of points
  `n_on_hull`         : int, number of vertices on convex hull
  `BENCHMARK`         : 0.030936889 (AlphaEvolve target)

Plus `np`. NO scipy / itertools / other imports inside verify_code. Must run without error.

## Output
Each subgoal:
- `kind`: "reference" or "aspirational".
- `description`: short sentence; for aspirational, briefly say WHY it pushes
  min_area_normalized beyond {ref_min_area_norm:.4f}.
- `verify_code`: Python expression returning float in [0,1] (or bool). Uses the
  namespace listed above. Must run without error.
- `expected_result`: typical value across the elite reference layouts (1.0 if every
  elite meets it; a fraction < 1.0 if elites only partially meet it).

Output ONLY a JSON array:

```json
[
  {{"kind": "reference",    "description": "...", "verify_code": "...", "expected_result": "..."}},
  {{"kind": "aspirational", "description": "...", "verify_code": "...", "expected_result": "..."}}
]
```
""".replace("{NUM_POINTS}", str(NUM_POINTS))


def _all_triangle_areas(points: np.ndarray) -> np.ndarray:
    """Vectorized areas of all C(n,3) triangles, sorted ascending."""
    n = points.shape[0]
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    two_area = np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return np.sort(0.5 * two_area)


def _pairwise_extremes(points: np.ndarray):
    d = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    m = dist + np.eye(dist.shape[0]) * 1e9
    return float(m.min()), float(dist.max())


def _convex_hull_area(points: np.ndarray) -> float:
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def _n_on_hull(points: np.ndarray) -> int:
    try:
        return int(len(ConvexHull(points).vertices))
    except Exception:
        return 0


def _eval_vars_from_points(points: np.ndarray) -> Dict[str, Any]:
    points = np.asarray(points, dtype=float)
    areas = _all_triangle_areas(points) if points.shape[0] >= 3 else np.zeros(0)
    pmin, pmax = _pairwise_extremes(points) if points.shape[0] >= 2 else (0.0, 0.0)
    hull_area = _convex_hull_area(points)
    min_a = float(areas.min()) if areas.size else 0.0
    max_a = float(areas.max()) if areas.size else 0.0
    min_norm = (min_a / hull_area) if hull_area > 0 else 0.0
    return {
        "points": points,
        "x": points[:, 0],
        "y": points[:, 1],
        "n": int(points.shape[0]),
        "min_area": min_a,
        "max_area": max_a,
        "triangle_areas": areas,
        "convex_hull_area": hull_area,
        "min_area_normalized": min_norm,
        "combined_score": min_norm / BENCHMARK if BENCHMARK > 0 else 0.0,
        "pairwise_min": pmin,
        "pairwise_max": pmax,
        "centroid": np.mean(points, axis=0) if points.size else np.zeros(2),
        "n_on_hull": _n_on_hull(points),
        "BENCHMARK": BENCHMARK,
    }


def _ref_table(points: np.ndarray) -> str:
    lines = []
    for i, (px, py) in enumerate(points):
        lines.append(f"  i={i:2d}  x={px:.4f}  y={py:.4f}")
    return "\n".join(lines)


def _format_elite_blocks(elites: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for rank, ev in enumerate(elites, start=1):
        header = (
            f"=== elite #{rank} (min_area_norm = {ev['min_area_normalized']:.6f}, "
            f"combined_score = {ev['combined_score']:.4f}, rank {rank}) ==="
        )
        body = _ref_table(ev["points"])
        body += (
            f"\n  hull_area={ev['convex_hull_area']:.4f}  "
            f"diameter={ev['pairwise_max']:.4f}  "
            f"n_on_hull={ev['n_on_hull']}  "
            f"min_area={ev['min_area']:.6f}  "
            f"max_area={ev['max_area']:.4f}"
        )
        blocks.append(header + "\n" + body)
    return "\n\n".join(blocks)


def load_reference_eval_vars(npz_path: Path) -> Dict[str, Any]:
    d = np.load(str(npz_path))
    return _eval_vars_from_points(d["points"])


def load_gen_eval_vars(gen_dir: Path) -> Optional[Dict[str, Any]]:
    npz = Path(gen_dir) / "results" / "extra.npz"
    if not npz.exists():
        return None
    try:
        d = np.load(str(npz))
        return _eval_vars_from_points(d["points"])
    except Exception:
        return None


def load_elite_eval_vars(run_root: Path, k: int = ELITE_K) -> List[Dict[str, Any]]:
    """Top-k by combined_score from <run_root>/programs.sqlite."""
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
        "ref_min_area_norm": float(ref_eval_vars["min_area_normalized"]),
    }


HOOKS = BenchHooks(
    prompt_template=DECOMPOSE_PROMPT,
    build_prompt_kwargs=build_prompt_kwargs,
    load_reference_eval_vars=load_reference_eval_vars,
    load_gen_eval_vars=load_gen_eval_vars,
    load_elite_eval_vars=load_elite_eval_vars,
    model="gpt-5",
    raw_metric_key="min_area_normalized",
)
