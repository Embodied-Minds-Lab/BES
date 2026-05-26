"""Reward function for Knights and Knaves logic puzzles.

Ground truth format: '{"Alice": 1, "Bob": 0}' (1=knight, 0=knave)
Model output: should contain a JSON object like {"Alice": 1, "Bob": 0}

Robust extraction handles: markdown code blocks, trailing commas,
single quotes, text before/after JSON, multiple attempts.
"""

import json
import re


def _try_parse_obj(s):
    """Try json.loads with fixups, return dict or None."""
    # Try as-is
    for candidate in [
        s,
        re.sub(r",\s*}", "}", s),                    # trailing comma
        s.replace("'", '"'),                           # single quotes
        re.sub(r",\s*}", "}", s.replace("'", '"')),   # both
    ]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def extract_last_json_obj(text):
    """Robustly extract the last JSON object from LLM output."""
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # Try from each '{' position, last match wins
    best = None
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            break
        search_from = start + 1
        while search_from < len(text):
            end = text.find("}", search_from)
            if end == -1:
                break
            candidate = text[start:end + 1]
            result = _try_parse_obj(candidate)
            if result is not None:
                best = result
                # Don't break — keep searching for later (last) JSON
            search_from = end + 1
        idx = start + 1
    return best


def compute_score(solution_str, ground_truth):
    """Return 1.0 if all identities match, 0.0 otherwise."""
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    pred = extract_last_json_obj(solution_str)
    if pred is None:
        return 0.0

    # Normalize: lowercase keys, int values
    gt_norm = {k.lower(): int(v) for k, v in gt.items()}
    try:
        pred_norm = {k.lower(): int(v) for k, v in pred.items()}
    except (ValueError, TypeError):
        return 0.0

    # Every person in gt must match
    for name, expected in gt_norm.items():
        if pred_norm.get(name) != expected:
            return 0.0
    return 1.0
