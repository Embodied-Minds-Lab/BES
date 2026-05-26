"""Generic LLM-driven goal-tree decomposition (problem-agnostic).

The benchmark supplies:
  - a Python `str.format`-style prompt template,
  - a function that turns a target node + reference `eval_vars` into prompt kwargs,
  - the reference `eval_vars` used by the LLM as context for the prompt.

This module handles:
  - calling the LLM (any provider supported by shinka.llm),
  - parsing a ```json [...]``` array out of the response,
  - building the resulting subgoals into a tree (free-form: every subgoal the LLM
    proposes is kept; a syntactically broken verify_code silently scores 0 via
    `verify_node` rather than blocking the tree).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.query import query as llm_query


def _call_llm(model: str, prompt: str, max_tokens: int = 16384) -> str:
    """Provider-agnostic single-shot text query for goal-tree decomposition.

    Uses the shared `shinka.llm.query.query` dispatcher, so any provider supported by
    the rest of the LLM stack (gemini, openai/gpt-5, anthropic, etc.) works as long
    as the model name is recognized. Reasoning models (gpt-5 etc.) get a `medium`
    effort budget by default; non-reasoning models ignore that field.
    """
    kwargs = sample_model_kwargs(
        model_names=[model],
        temperatures=[0.3],
        reasoning_efforts=["high"],
        max_tokens=[max_tokens],
    )
    kwargs.pop("model_name", None)  # passed explicitly below
    result = llm_query(
        model_name=model,
        msg=prompt,
        system_msg="",
        msg_history=[],
        output_model=None,
        **kwargs,
    )
    if result is None or not getattr(result, "content", None):
        raise ValueError(f"LLM ({model}) returned empty response for decompose prompt")
    return str(result.content)


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
        raw = m.group(1) if m else None
    if raw is None:
        raise ValueError(f"No JSON array found in LLM output:\n{text[:500]}")
    return json.loads(raw)


def decompose_subgoals(
    model: str,
    prompt_template: str,
    prompt_kwargs: Dict[str, Any],
    max_tokens: int = 16384,
) -> List[Dict[str, Any]]:
    """Render the prompt, call the LLM, return a list of parsed subgoal dicts.

    Each returned dict has: kind ("reference"/"aspirational"), description, verify_code,
    expected_result. Unknown kinds fall back to "reference".
    """
    prompt = prompt_template.format(**prompt_kwargs)
    text = _call_llm(model, prompt, max_tokens=max_tokens)
    subs = _extract_json_array(text)
    out: List[Dict[str, Any]] = []
    for s in subs:
        if not all(k in s for k in ("description", "verify_code", "expected_result")):
            continue
        kind = str(s.get("kind", "reference")).strip().lower()
        if kind not in ("reference", "aspirational"):
            kind = "reference"
        out.append({
            "kind": kind,
            "description": str(s["description"]).strip(),
            "verify_code": str(s["verify_code"]).strip(),
            "expected_result": str(s["expected_result"]).strip(),
        })
    return out


def build_goal_tree(
    root_node: Dict[str, Any],
    ref_eval_vars: Dict[str, Any],
    prompt_template: str,
    build_prompt_kwargs: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    out_path: Path,
    model: str = "gemini-3-pro-preview",
    max_depth: int = 2,
) -> Dict[str, Any]:
    """Recursively decompose `root_node` up to `max_depth` levels, writing JSON to `out_path`.

    `build_prompt_kwargs(node, ref_eval_vars)` returns the kwargs for prompt_template.format.
    Free-form: every subgoal the LLM proposes is kept (no reference-based filtering).
    """
    def expand(node: Dict[str, Any], depth: int):
        if depth >= max_depth:
            return
        prompt_kwargs = build_prompt_kwargs(node, ref_eval_vars)
        subs = decompose_subgoals(model, prompt_template, prompt_kwargs)
        kept: List[Dict[str, Any]] = [
            {
                "id": f"{node['id']}.L{depth+1}_{j}",
                "level": depth + 1,
                "kind": s.get("kind", "reference"),
                "description": s["description"],
                "verify_code": s["verify_code"],
                "expected_result": s["expected_result"],
                "children": [],
            }
            for j, s in enumerate(subs)
        ]
        node["children"] = kept
        for c in kept:
            expand(c, depth + 1)

    expand(root_node, 0)
    Path(out_path).write_text(json.dumps(root_node, indent=2))
    return root_node
