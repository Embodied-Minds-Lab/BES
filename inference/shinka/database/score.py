"""Effective-score helper used by every parent/archive/inspiration ranking site.

There is also a small ``wrap_rows()`` adapter for callers that work in raw
sqlite row dicts (no Program objects available) — it parses the metric JSON
fields and returns a list of attribute-accessible objects suitable for the
ranking helpers below.


Algorithm (adaptive bucket interpolation):
- raw is the dominant ranking key, bucketed at fixed precision (1e-6 by default).
- Within a bucket, backward_score acts as an intra-bucket sub-rank.
- The bw component is scaled by `gap_k * (1 - safety)` where `gap_k` is the
  distance from this bucket to the next-higher bucket. The top bucket reuses
  the gap of the second-highest bucket. This guarantees that bw can never
  push a program to or past the next-higher bucket.
- If raw is missing for any program, or bw is missing for everyone, falls
  back to the stored `combined_score` on each program.

The set passed in defines the bucket landscape — callers should pass the
specific set being ranked (e.g. the eligible programs in an island, the
candidates being compared against the archive, etc.).
"""

import json
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

RAW_KEY = "reported_sum_of_radii"
BW_KEY = "backward_score"
BUCKET_PRECISION = 1e-2
BW_SAFETY = 0.01


def set_raw_metric_key(key: str) -> None:
    """Override the private_metrics field that adaptive bucket interpolation uses
    as the dominant ranking key.
    """
    global RAW_KEY
    RAW_KEY = key


def get_raw_metric_key() -> str:
    return RAW_KEY


def set_bucket_precision(precision: float) -> None:
    """Override the bucket width used by adaptive bucket interpolation."""
    global BUCKET_PRECISION
    BUCKET_PRECISION = precision


def get_bucket_precision() -> float:
    return BUCKET_PRECISION


def _extract_raw_bw(program: Any) -> Tuple[Optional[float], Optional[float]]:
    """Pull (raw, bw) out of a Program-like object's metric dicts."""
    priv = getattr(program, "private_metrics", None)
    pub = getattr(program, "public_metrics", None)
    raw = priv.get(RAW_KEY) if isinstance(priv, dict) else None
    bw = pub.get(BW_KEY) if isinstance(pub, dict) else None
    return raw, bw


def compute_effective_scores(programs: List[Any]) -> List[float]:
    """Return one effective score per input program, in the same order.

    Falls back to `program.combined_score or 0.0` when raw/bw are unavailable.
    """
    if not programs:
        return []

    raws_bws = [_extract_raw_bw(p) for p in programs]
    have_raw_for_all = all(rb[0] is not None for rb in raws_bws)
    have_any_bw = any(rb[1] is not None for rb in raws_bws)

    if not (have_raw_for_all and have_any_bw):
        return [(getattr(p, "combined_score", 0.0) or 0.0) for p in programs]

    bucket_keys = [
        round(raw / BUCKET_PRECISION) * BUCKET_PRECISION for raw, _ in raws_bws
    ]
    sorted_buckets = sorted(set(bucket_keys), reverse=True)  # s_1 > s_2 > ...

    gaps = {}
    if len(sorted_buckets) >= 2:
        gap_top = sorted_buckets[0] - sorted_buckets[1]
        gaps[sorted_buckets[0]] = gap_top  # top bucket reuses second bucket's gap
        for k in range(1, len(sorted_buckets)):
            gaps[sorted_buckets[k]] = sorted_buckets[k - 1] - sorted_buckets[k]
    else:
        gaps[sorted_buckets[0]] = 0.0  # only one bucket: bw has no room

    return [
        bk + (bw if bw is not None else 0.0) * gaps[bk] * (1.0 - BW_SAFETY)
        for (raw, bw), bk in zip(raws_bws, bucket_keys)
    ]


def effective_score_map(programs: List[Any]) -> dict:
    """Return {program.id: effective_score} for the given set."""
    scores = compute_effective_scores(programs)
    return {getattr(p, "id", None): s for p, s in zip(programs, scores)}


def sort_by_effective(
    programs: List[Any], reverse: bool = True
) -> List[Any]:
    """Return programs sorted by their effective score (within this set)."""
    if not programs:
        return []
    scores = compute_effective_scores(programs)
    paired = list(zip(scores, range(len(programs)), programs))
    paired.sort(key=lambda t: t[0], reverse=reverse)
    return [p for _, _, p in paired]


def effective_score(program: Any, peers: List[Any]) -> float:
    """Effective score for one program in the context of `peers` (must include it)."""
    scores = compute_effective_scores(peers)
    for p, s in zip(peers, scores):
        if getattr(p, "id", None) == getattr(program, "id", None):
            return s
    return getattr(program, "combined_score", 0.0) or 0.0


def _parse_json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def wrap_rows(rows: List[Any]) -> List[SimpleNamespace]:
    """Wrap raw sqlite row dicts (or sqlite3.Row) into objects with attribute
    access on the fields the score helpers need: id, combined_score,
    public_metrics (dict), private_metrics (dict)."""
    wrapped: List[SimpleNamespace] = []
    for row in rows:
        d = dict(row) if not isinstance(row, dict) else row
        wrapped.append(
            SimpleNamespace(
                id=d.get("id"),
                combined_score=d.get("combined_score"),
                public_metrics=_parse_json_dict(d.get("public_metrics")),
                private_metrics=_parse_json_dict(d.get("private_metrics")),
            )
        )
    return wrapped
