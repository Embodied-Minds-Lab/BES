import asyncio
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KK-specific utilities
# ---------------------------------------------------------------------------

ANSWER_SUFFIX = (
    'Please think step by step, by considering whether each person is lying '
    'and if that leads to contradiction. At the end, output your final answer '
    'as a JSON object where keys are names and values are 1 for knight or 0 '
    'for knave. For example: {"Alice": 1, "Bob": 0}'
)


def extract_last_json_obj(text: str) -> dict | None:
    """Extract the last JSON object from text. Returns dict or None."""
    # Find all potential JSON objects (between { and })
    results = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = text[start:i + 1]
                    # Fix single quotes
                    candidate = candidate.replace("'", '"')
                    # Remove trailing commas
                    candidate = re.sub(r',\s*}', '}', candidate)
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        results.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1
    return results[-1] if results else None


def kk_compute_score(response_text: str, ground_truth_json: str) -> float:
    """Score a KK response against ground truth. Returns 0.0 or 1.0."""
    try:
        pred = extract_last_json_obj(response_text)
        if pred is None:
            return 0.0
        gt = json.loads(ground_truth_json)
        # Normalize keys to lowercase for comparison
        pred_norm = {k.strip().lower(): int(v) for k, v in pred.items()}
        gt_norm = {k.strip().lower(): int(v) for k, v in gt.items()}
        return 1.0 if pred_norm == gt_norm else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Prompts (adapted for KK)
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
Given this logic puzzle and its known solution, break the goal into simple, \
concrete subgoals that a verifier can easily check.

IMPORTANT: Each subgoal should be a simple factual claim with a short, \
concrete expected answer — NOT a reasoning process. The answer should be \
something you can directly check in a model's response.

Good example subgoals:
- description: "Determine Alice's identity", answer: "Alice is a knave"
- description: "Determine Bob's identity", answer: "Bob is a knight"
- description: "Show that Alice cannot be a knight", answer: "Contradiction found when assuming Alice is a knight"

Bad example subgoals (too complex):
- description: "Define truth conditions", answer: "Alice is a knight iff..."
- description: "Evaluate the logical implications", answer: "If Alice is a knight (L=1) then..."

## Problem
{problem}

## Solution
{answer}

## Goal to decompose
{goal}

## Expected result for this goal
{goal_answer}

Output 2-4 simple, checkable subgoals:

```json
[
  {{"description": "short description", "answer": "short concrete answer"}},
  ...
]
```"""


VERIFY_PROMPT = """\
You are a judge for a Knights and Knaves logic puzzle.

## Problem
{problem}

## Model's Current Reasoning
{reasoning}

## Sub-Goal
{goal}

## Sub-Goal's Expected Answer
{goal_answer}

Has the sub-goal been achieved in the model's current reasoning?
Please think step by step, and answer with exactly 'YES' or 'NO'."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SearchNode:
    node_id: int
    paragraphs: list[str]
    score: float = 0.0
    is_terminal: bool = False
    is_complete: bool = False
    parent_id: int | None = None
    depth: int = 0
    operation: str = ""

    @property
    def text(self) -> str:
        return "".join(self.paragraphs)

    @property
    def answer(self) -> dict | None:
        return extract_last_json_obj(self.text)


@dataclass
class GoalNode:
    goal_id: int
    description: str
    answer: str = ""
    parent_id: int | None = None
    children_ids: list[int] = field(default_factory=list)
    depth: int = 0
    deprecated: bool = False


class GoalTree:
    def __init__(self):
        self.goals: dict[int, GoalNode] = {}
        self.root_id: int = 0
        self._next_id: int = 0

    def add_root(self, description: str) -> int:
        self._next_id += 1
        self.goals[self._next_id] = GoalNode(
            goal_id=self._next_id, description=description, depth=0,
        )
        self.root_id = self._next_id
        return self._next_id

    def add_subgoal(self, parent_id: int, description: str, answer: str = "") -> int:
        self._next_id += 1
        parent = self.goals[parent_id]
        goal = GoalNode(
            goal_id=self._next_id, description=description, answer=answer,
            parent_id=parent_id, depth=parent.depth + 1,
        )
        self.goals[self._next_id] = goal
        parent.children_ids.append(self._next_id)
        return self._next_id

    def leaf_goals(self) -> list[GoalNode]:
        return [g for g in self.goals.values() if not g.children_ids]

    def non_root_goals(self) -> list[GoalNode]:
        return [g for g in self.goals.values() if g.goal_id != self.root_id]


class RequestType(Enum):
    EXPAND = auto()
    EXPAND_FINISH = auto()
    DECOMPOSE = auto()
    VERIFY = auto()


@dataclass
class LLMRequest:
    model_name: str  # "main" or "verifier"
    request_type: RequestType
    prompt: str
    params: dict  # sampling params dict
    parent_node_id: int = 0
    decompose_goal_id: int = 0
    verify_node_id: int = 0
    verify_goal_id: int = 0
    validator: Any = None


# ---------------------------------------------------------------------------
# Standalone scoring functions
# ---------------------------------------------------------------------------


def _extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from LLM output."""
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    def _try_parse(s):
        def _fix_backslashes(t):
            out, i = [], 0
            while i < len(t):
                if t[i] == '\\':
                    if i + 1 < len(t) and t[i + 1] in '\\"\/bfnrtu':
                        out.append(t[i]); out.append(t[i + 1]); i += 2
                    else:
                        out.append('\\'); out.append('\\'); i += 1
                else:
                    out.append(t[i]); i += 1
            return ''.join(out)
        s_fixed = _fix_backslashes(s)
        for candidate in [s, s_fixed,
                          re.sub(r",\s*([}\]])", r"\1", s),
                          re.sub(r",\s*([}\]])", r"\1", s_fixed)]:
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    idx = 0
    while idx < len(text):
        start = text.find("[", idx)
        if start == -1:
            break
        search_from = start + 1
        while search_from < len(text):
            end = text.find("]", search_from)
            if end == -1:
                break
            candidate = text[start:end + 1]
            result = _try_parse(candidate)
            if result is not None:
                return result
            search_from = end + 1
        idx = start + 1
    return None


def _compute_recursive_score(node, goal_id, goal_tree, sat_cache, ground_truth):
    goal = goal_tree.goals[goal_id]
    self_score = sat_cache.get((node.node_id, goal_id), 0.0)
    if self_score >= 1.0:
        return 1.0
    if not goal.children_ids:
        return self_score
    child_scores = [
        _compute_recursive_score(node, cid, goal_tree, sat_cache, ground_truth)
        for cid in goal.children_ids
    ]
    return 0.7 * self_score + 0.3 * sum(child_scores) / len(child_scores)


def _get_pareto_front(candidates, goal_tree, sat_cache):
    all_goal_ids = [g.goal_id for g in goal_tree.goals.values()]

    def get_scores(node):
        return tuple(sat_cache.get((node.node_id, gid), 0.0) for gid in all_goal_ids)

    def dominates(a, b):
        return all(ai >= bi for ai, bi in zip(a, b)) and any(
            ai > bi for ai, bi in zip(a, b)
        )

    scored = [(n, get_scores(n)) for n in candidates]
    pareto = []
    for i, (ni, si) in enumerate(scored):
        dominated = False
        for j, (nj, sj) in enumerate(scored):
            if i == j:
                continue
            if dominates(sj, si):
                dominated = True
                break
        if not dominated:
            pareto.append(ni)
    return pareto if pareto else candidates[:1]


def _boltzmann_select(nodes, temperature):
    if len(nodes) == 1 or temperature <= 0:
        return max(nodes, key=lambda n: n.score)
    scores = [n.score / temperature for n in nodes]
    ceiling = max(scores)
    weights = [math.exp(s - ceiling) for s in scores]
    return random.choices(nodes, weights=weights, k=1)[0]


def _exploration_select(pool, all_nodes, temperature):
    if len(pool) == 1:
        return pool[0]
    child_count = {}
    for n in all_nodes:
        if n.parent_id is not None:
            child_count[n.parent_id] = child_count.get(n.parent_id, 0) + 1
    scores = []
    for n in pool:
        n_children = child_count.get(n.node_id, 0)
        unexpanded_bonus = 1.0 if n_children == 0 else 0.0
        scores.append(n.score * 10 + unexpanded_bonus)
    if temperature <= 0:
        return pool[max(range(len(pool)), key=lambda i: scores[i])]
    scaled = [s / max(temperature, 0.01) for s in scores]
    ceiling = max(scaled)
    weights = [math.exp(s - ceiling) for s in scaled]
    return random.choices(pool, weights=weights, k=1)[0]


def _select_combine_pair(pareto_nodes, goal_tree, sat_cache, temperature):
    if len(pareto_nodes) < 2:
        return None
    leaves = goal_tree.leaf_goals()
    leaf_ids = [g.goal_id for g in leaves if g.goal_id != goal_tree.root_id]
    if not leaf_ids:
        return None
    pairs, scores = [], []
    for i in range(len(pareto_nodes)):
        for j in range(i + 1, len(pareto_nodes)):
            a, b = pareto_nodes[i], pareto_nodes[j]
            u = sum(
                max(
                    sat_cache.get((a.node_id, gid), 0.0),
                    sat_cache.get((b.node_id, gid), 0.0),
                )
                for gid in leaf_ids
            ) / len(leaf_ids)
            pairs.append((a, b))
            scores.append(u * 10)
    if not pairs:
        return None
    if temperature <= 0:
        return pairs[max(range(len(pairs)), key=lambda i: scores[i])]
    scaled = [s / temperature for s in scores]
    ceiling = max(scaled)
    weights = [math.exp(s - ceiling) for s in scaled]
    return random.choices(pairs, weights=weights, k=1)[0]


def _pick_best(nodes):
    tc = [n for n in nodes if n.is_terminal and n.is_complete]
    if tc:
        return max(tc, key=lambda n: n.score)
    c = [n for n in nodes if n.is_complete]
    if c:
        return max(c, key=lambda n: n.score)
    s = [n for n in nodes if n.paragraphs]
    return max(s, key=lambda n: n.score) if s else None


# ---------------------------------------------------------------------------
# SearchState: full state machine for one KK problem
# ---------------------------------------------------------------------------


class SearchState:
    MAIN_MODEL = "main"
    VERIFIER_MODEL = "verifier"

    def __init__(
        self,
        problem_id: str,
        problem: str,
        ground_truth: str,
        tokenizer,
        budget: int = 20,
        max_paragraph_tokens: int = 4096,
        max_combine_tokens: int = 8192,
        initial_temp: float = 2.0,
        final_temp: float = 1.0,
        decompose_interval: int = 40,
        max_goal_depth: int = 3,
        gen_temperature: float = 0.6,
        gen_top_p: float = 0.95,
        gen_top_k: int = 20,
        gen_presence_penalty: float = 0.0,
    ):
        self.problem_id = problem_id
        self.problem = problem
        self.ground_truth = ground_truth
        self.tokenizer = tokenizer
        self.budget = budget
        self.decompose_interval = decompose_interval
        self.max_goal_depth = max_goal_depth
        self.initial_temp = initial_temp
        self.final_temp = final_temp

        # Sampling params for main model (local vLLM)
        # Stop at \n\n — called k times per expand (k random in [1,10])
        self.expand_params = {
            "max_tokens": max_paragraph_tokens,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
            "top_k": gen_top_k,
            "presence_penalty": gen_presence_penalty,
            "stop": ["\n\n"],
            "include_stop_str_in_output": True,
        }
        self.finish_params = {
            "max_tokens": max_paragraph_tokens,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
            "top_k": gen_top_k,
            "presence_penalty": gen_presence_penalty,
        }
        self.free_gen_params = {
            "max_tokens": max_combine_tokens,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
            "top_k": gen_top_k,
            "presence_penalty": gen_presence_penalty,
        }
        # Params for verifier/decomposer (Gemini API)
        self.decompose_params = {
            "max_tokens": 8192,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
        }
        self.verify_params = {
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
        }

        # State
        self.nodes: list[SearchNode] = []
        self.goal_tree = GoalTree()
        self.sat_cache: dict[tuple[int, int], float] = {}
        self.step_log: list[dict] = []
        self._next_node_id = 0
        self.expand_count = 0
        self.step = 0
        # Multi-paragraph expand state
        self._expand_remaining = 0        # how many more paragraphs to collect
        self._expand_parent_id = None     # original parent node id
        self._expand_collected = []       # paragraphs collected so far
        self.done = False
        self.result_answer = ""
        self.result_text = ""
        self._phase = "init_decompose"
        self._pending_decompose_step = False
        self._verify_queue: list[tuple[int, int]] = []
        self._verify_queued: set[tuple[int, int]] = set()
        self._finish_queue: list[int] = []
        self._pending_verify_count: dict[int, int] = {}
        self._auto_finished: set[int] = set()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def _enqueue_verify(self, node_id: int, goal_id: int) -> bool:
        key = (node_id, goal_id)
        if key in self.sat_cache or key in self._verify_queued:
            return False
        self._verify_queue.append(key)
        self._verify_queued.add(key)
        return True

    def _add_node(self, n: SearchNode) -> SearchNode:
        self._next_node_id += 1
        n.node_id = self._next_node_id
        self.nodes.append(n)
        return n

    # ------------------------------------------------------------------
    # Prompt building (KK-adapted)
    # ------------------------------------------------------------------

    def _continue_prompt(self, paragraphs: list[str]) -> str:
        """Build prompt for continuing from existing paragraphs."""
        user_content = self.problem + "\n\n" + ANSWER_SUFFIX
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt += "".join(paragraphs)
        return prompt

    def _finish_prompt(self, node: SearchNode) -> str:
        """Prompt to generate the final answer section."""
        user_content = self.problem + "\n\n" + ANSWER_SUFFIX
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = node.text
        if not text.rstrip().endswith("### Final Answer"):
            text += "\n\n### Final Answer\n\n"
        prompt += text
        return prompt

    def _verify_prompt(self, node: SearchNode, goal: GoalNode) -> str:
        """Build verification prompt for Gemini."""
        if goal.goal_id == self.goal_tree.root_id:
            # For root goal, send only the answer part
            text = node.text
            marker = "### Final Answer"
            if marker in text:
                reasoning = text.split(marker, 1)[1].strip()
            else:
                reasoning = text
        else:
            reasoning = node.text
        return VERIFY_PROMPT.format(
            problem=self.problem,
            goal=goal.description,
            goal_answer=goal.answer or "N/A",
            reasoning=reasoning,
        )

    def _try_root_verify_rule_based(self, node) -> bool:
        """Rule-based check for root goal using kk_logic."""
        if not node.is_terminal or not node.answer:
            return False
        answer_json = json.dumps(node.answer)
        score = kk_compute_score(answer_json, self.ground_truth)
        if score >= 1.0:
            root_key = (node.node_id, self.goal_tree.root_id)
            self.sat_cache[root_key] = 1.0
            node.score = 1.0
            self.result_answer = json.dumps(node.answer)
            self.result_text = node.text
            self.done = True
            return True
        return False

    # ------------------------------------------------------------------
    # next_request: state machine
    # ------------------------------------------------------------------

    def next_request(self) -> LLMRequest | None:
        if self.done:
            return None

        # Continue multi-paragraph expand (highest priority, before verify)
        if self._expand_remaining > 0 and self._expand_parent_id is not None:
            parent = next((n for n in self.nodes if n.node_id == self._expand_parent_id), None)
            if parent:
                # Build prompt from parent paragraphs + already collected
                virtual_paras = parent.paragraphs + self._expand_collected
                return LLMRequest(
                    model_name=self.MAIN_MODEL,
                    request_type=RequestType.EXPAND,
                    prompt=self._continue_prompt(virtual_paras),
                    params=self.expand_params,
                    parent_node_id=self._expand_parent_id,
                )
            self._expand_remaining = 0
            self._expand_parent_id = None
            self._expand_collected = []

        # Verify queue (highest priority)
        if self._verify_queue:
            node_id, goal_id = self._verify_queue.pop(0)
            self._verify_queued.discard((node_id, goal_id))
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            goal = self.goal_tree.goals.get(goal_id)
            if node and goal and node.paragraphs:
                return LLMRequest(
                    model_name=self.VERIFIER_MODEL,
                    request_type=RequestType.VERIFY,
                    prompt=self._verify_prompt(node, goal),
                    params=self.verify_params,
                    verify_node_id=node_id,
                    verify_goal_id=goal_id,
                )

        # Finish queue
        if self._finish_queue:
            node_id = self._finish_queue.pop(0)
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if not node:
                pass
            elif node_id in self._auto_finished:
                return LLMRequest(
                    model_name=self.MAIN_MODEL,
                    request_type=RequestType.EXPAND,
                    prompt=self._continue_prompt(node.paragraphs),
                    params=self.free_gen_params,
                    parent_node_id=node.node_id,
                )
            elif node.is_terminal and not node.is_complete:
                return LLMRequest(
                    model_name=self.MAIN_MODEL,
                    request_type=RequestType.EXPAND_FINISH,
                    prompt=self._finish_prompt(node),
                    params=self.finish_params,
                    parent_node_id=node.node_id,
                )

        # Init decompose
        if self._phase == "init_decompose":
            root_id = self.goal_tree.add_root(
                f"Solve this Knights and Knaves puzzle and find the correct assignment: {self.problem}"
            )
            self.goal_tree.goals[root_id].answer = self.ground_truth
            req = self._make_decompose_request()
            if req:
                self._phase = "wait_init_decompose"
                return req
            self._phase = "first_expand"

        if self._phase == "first_expand":
            # Root node: empty (model will generate opening + first step)
            self._add_node(
                SearchNode(node_id=0, paragraphs=[""], operation="root")
            )
            self._phase = "search"

        while self._phase == "search":
            if self.step >= self.budget - 1:
                self._finish()
                return None

            if self._pending_verify_count:
                return None

            if self._pending_decompose_step:
                req = self._make_decompose_request()
                if req:
                    self._pending_decompose_step = False
                    return req
                self._pending_decompose_step = False

            candidates = [n for n in self.nodes if not n.is_terminal]
            if not candidates:
                self._finish()
                return None

            t = self.step / max(self.budget - 2, 1)
            temp = self.initial_temp + (self.final_temp - self.initial_temp) * t

            scored_candidates = [n for n in candidates if n.paragraphs]
            if scored_candidates:
                pareto = _get_pareto_front(scored_candidates, self.goal_tree, self.sat_cache)
            else:
                pareto = []

            has_subgoals = len(self.goal_tree.goals) > 1
            two_paths = has_subgoals and len(scored_candidates) >= 2
            multi_para = [n for n in scored_candidates if len(n.paragraphs) > 2]

            roll = random.random()

            # Concatemerization (10%)
            if roll < 0.10 and two_paths:
                pair = _select_combine_pair(
                    scored_candidates, self.goal_tree, self.sat_cache, temp
                )
                if pair:
                    child = self._merge_nodes(pair[0], pair[1])
                    if child:
                        self._register_child(child, "combine",
                                             combine_ids=(pair[0].node_id, pair[1].node_id))
                        self.step += 1
                        continue

            # Deletion (5%)
            elif roll < 0.15 and multi_para:
                _w = [math.exp(n.score * 10 / max(temp, 0.01)) for n in multi_para]
                source = random.choices(multi_para, weights=_w, k=1)[0]
                child, del_info = self._deletion(source)
                if child:
                    self._register_child(child, "deletion",
                                         combine_ids=(source.node_id, source.node_id))
                    self.step += 1
                    continue

            # Translocation (7.5%)
            elif roll < 0.225 and len(multi_para) >= 2:
                _w = [math.exp(n.score * 10 / max(temp, 0.01)) for n in multi_para]
                a = random.choices(multi_para, weights=_w, k=1)[0]
                remaining = [n for n in multi_para if n.node_id != a.node_id]
                if remaining:
                    _w2 = [math.exp(n.score * 10 / max(temp, 0.01)) for n in remaining]
                    b = random.choices(remaining, weights=_w2, k=1)[0]
                else:
                    b = a
                child, trans_info = self._translocation(a, b)
                if child:
                    self._register_child(child, "translocation",
                                         combine_ids=(a.node_id, b.node_id))
                    self.step += 1
                    continue

            # Crossing over (7.5%)
            elif roll < 0.30 and len(multi_para) >= 2:
                _w = [math.exp(n.score * 10 / max(temp, 0.01)) for n in multi_para]
                a = random.choices(multi_para, weights=_w, k=1)[0]
                remaining = [n for n in multi_para if n.node_id != a.node_id]
                if remaining:
                    _w2 = [math.exp(n.score * 10 / max(temp, 0.01)) for n in remaining]
                    b = random.choices(remaining, weights=_w2, k=1)[0]
                else:
                    b = a
                child, cross_info = self._crossing_over(a, b)
                if child:
                    self._register_child(child, "crossing_over",
                                         combine_ids=(a.node_id, b.node_id))
                    self.step += 1
                    continue

            # Amplification (70% or fallback) — multi-paragraph expand
            expand_pool = candidates
            selected = _exploration_select(expand_pool, self.nodes, temp)
            # Random k in [1, 10]: generate k paragraphs in sequence into one node
            k = random.randint(1, 10)
            self._expand_remaining = k - 1  # first result decrements by 1
            self._expand_parent_id = selected.node_id
            self._expand_collected = []
            return LLMRequest(
                model_name=self.MAIN_MODEL,
                request_type=RequestType.EXPAND,
                prompt=self._continue_prompt(selected.paragraphs),
                params=self.expand_params,
                parent_node_id=selected.node_id,
            )

        return None

    # ------------------------------------------------------------------
    # Mutations (DNA-inspired, no LLM call)
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_prefix_len(a_paras, b_paras):
        shared = 0
        for i in range(min(len(a_paras), len(b_paras))):
            if a_paras[i] == b_paras[i]:
                shared += 1
            else:
                break
        return shared

    @staticmethod
    def _valid_pair(a, b):
        shared = 0
        for i in range(min(len(a.paragraphs), len(b.paragraphs))):
            if a.paragraphs[i] == b.paragraphs[i]:
                shared += 1
            else:
                break
        return bool(a.paragraphs[shared:]) and bool(b.paragraphs[shared:])

    def _merge_nodes(self, a: SearchNode, b: SearchNode) -> SearchNode | None:
        """Concatemerization: shared prefix + suffix_a + suffix_b."""
        if not self._valid_pair(a, b):
            return None
        shared = self._shared_prefix_len(a.paragraphs, b.paragraphs)
        suffix_a = a.paragraphs[shared:]
        suffix_b = b.paragraphs[shared:]
        merged = a.paragraphs[:shared] + suffix_a + suffix_b
        if not merged:
            return None
        full_text = "".join(merged)
        return SearchNode(
            node_id=0,
            paragraphs=merged,
            parent_id=a.node_id,
            is_terminal="### Final Answer" in full_text,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=max(a.depth, b.depth) + 1,
            operation="combine",
        )

    def _deletion(self, node: SearchNode) -> tuple[SearchNode | None, dict]:
        paras = node.paragraphs[:]
        if len(paras) <= 2:
            return None, {}
        idx = random.randint(1, len(paras) - 2)
        deleted_text = paras[idx][:100]
        del paras[idx]
        full_text = "".join(paras)
        child = SearchNode(
            node_id=0,
            paragraphs=paras,
            parent_id=node.node_id,
            is_terminal="### Final Answer" in full_text,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=node.depth + 1,
            operation="deletion",
        )
        return child, {"source_node": node.node_id, "deleted_index": idx,
                       "deleted_preview": deleted_text}

    def _translocation(self, a: SearchNode, b: SearchNode) -> tuple[SearchNode | None, dict]:
        if not self._valid_pair(a, b):
            return None, {}
        shared = self._shared_prefix_len(a.paragraphs, b.paragraphs)
        suffix_a = a.paragraphs[shared:]
        suffix_b = b.paragraphs[shared:]
        if not suffix_a or not suffix_b:
            return None, {}
        a_idx = random.randint(0, len(suffix_a) - 1)
        b_idx = random.randint(0, len(suffix_b) - 1)
        new_suffix = suffix_a[:a_idx] + [suffix_b[b_idx]] + suffix_a[a_idx + 1:]
        paras = a.paragraphs[:shared] + new_suffix
        full_text = "".join(paras)
        child = SearchNode(
            node_id=0,
            paragraphs=paras,
            parent_id=a.node_id,
            is_terminal="### Final Answer" in full_text,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=max(a.depth, b.depth) + 1,
            operation="translocation",
        )
        return child, {"source_a": a.node_id, "source_b": b.node_id,
                       "shared_prefix": shared}

    def _crossing_over(self, a: SearchNode, b: SearchNode) -> tuple[SearchNode | None, dict]:
        if not self._valid_pair(a, b):
            return None, {}
        shared = self._shared_prefix_len(a.paragraphs, b.paragraphs)
        suffix_a = a.paragraphs[shared:]
        suffix_b = b.paragraphs[shared:]
        if not suffix_a or not suffix_b:
            return None, {}
        m = random.randint(0, len(suffix_a) - 1)
        n = random.randint(1, len(suffix_b))
        paras = a.paragraphs[:shared] + suffix_a[:m] + suffix_b[-n:]
        if not paras:
            return None, {}
        full_text = "".join(paras)
        child = SearchNode(
            node_id=0,
            paragraphs=paras,
            parent_id=a.node_id,
            is_terminal="### Final Answer" in full_text,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=max(a.depth, b.depth) + 1,
            operation="crossing_over",
        )
        return child, {"source_a": a.node_id, "source_b": b.node_id,
                       "shared_prefix": shared, "a_take": m, "b_take": n}

    def _register_child(self, child: SearchNode, action: str,
                        combine_ids: tuple[int, int] | None = None):
        child = self._add_node(child)
        if child.is_terminal and not child.is_complete:
            self._finish_queue.append(child.node_id)
        verify_count = sum(
            self._enqueue_verify(child.node_id, goal.goal_id)
            for goal in self.goal_tree.non_root_goals()
        )
        if child.is_terminal and not self._try_root_verify_rule_based(child):
            if self._enqueue_verify(child.node_id, self.goal_tree.root_id):
                verify_count += 1
        if verify_count > 0:
            self._pending_verify_count[child.node_id] = verify_count
        if not self.done:
            child.score = _compute_recursive_score(
                child, self.goal_tree.root_id, self.goal_tree,
                self.sat_cache, self.ground_truth,
            )
        log_entry = {
            "step": self.step + 1, "action": action,
            "new_node_id": child.node_id, "parent_node_id": child.parent_id,
            "score": child.score, "terminal": child.is_terminal,
            "complete": child.is_complete, "n_nodes": len(self.nodes),
        }
        if combine_ids:
            log_entry["combine_node_ids"] = list(combine_ids)
        self.step_log.append(log_entry)

    # ------------------------------------------------------------------
    # process_result
    # ------------------------------------------------------------------

    def process_result(self, text: str, request: LLMRequest):
        if request.request_type == RequestType.VERIFY:
            self._process_verify(text, request.verify_node_id, request.verify_goal_id)
            return

        if request.request_type == RequestType.DECOMPOSE:
            n_new = 0
            if text:
                n_new = self._process_decompose(text, request.decompose_goal_id)
            if n_new == 0:
                self.step_log.append({
                    "action": "decompose_failed",
                    "goal_id": request.decompose_goal_id,
                })
            if self._phase == "wait_init_decompose":
                self._phase = "first_expand"
            return

        if request.request_type == RequestType.EXPAND_FINISH:
            self._process_finish(text, request.parent_node_id)
            return

        if request.request_type != RequestType.EXPAND:
            self.step += 1
            return

        # --- EXPAND: multi-paragraph collection ---
        # Collect this paragraph
        if text and text.strip():
            self._expand_collected.append(text)

        is_terminal = "### Final Answer" in (text or "")
        self._expand_remaining = max(0, self._expand_remaining - 1)

        # Keep collecting if more remaining and not terminal
        if self._expand_remaining > 0 and not is_terminal:
            return  # don't build node yet, next_request will issue another EXPAND

        # Done collecting — build one node with all paragraphs
        self._finalize_multi_expand(self._expand_parent_id or request.parent_node_id)

    def _finalize_multi_expand(self, parent_id):
        """Build one node from all collected paragraphs, register it, increment step."""
        collected = self._expand_collected
        self._expand_collected = []
        self._expand_remaining = 0
        self._expand_parent_id = None

        if not collected:
            self.step += 1
            return

        parent = next((n for n in self.nodes if n.node_id == parent_id), None)
        if not parent:
            self.step += 1
            return

        paras = parent.paragraphs + collected
        full_text = "".join(paras)
        child = SearchNode(
            node_id=0,
            paragraphs=paras,
            parent_id=parent_id,
            is_terminal="### Final Answer" in full_text,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=parent.depth + 1,
            operation="expand",
        )

        self.expand_count += 1
        if self.expand_count % self.decompose_interval == 0:
            self._pending_decompose_step = True

        child = self._add_node(child)
        if child.is_terminal and not child.is_complete:
            self._finish_queue.append(child.node_id)

        verify_count = sum(
            self._enqueue_verify(child.node_id, goal.goal_id)
            for goal in self.goal_tree.non_root_goals()
        )
        if child.is_terminal and not self._try_root_verify_rule_based(child):
            if self._enqueue_verify(child.node_id, self.goal_tree.root_id):
                verify_count += 1
        if verify_count > 0:
            self._pending_verify_count[child.node_id] = verify_count

        if not self.done:
            child.score = _compute_recursive_score(
                child, self.goal_tree.root_id, self.goal_tree,
                self.sat_cache, self.ground_truth,
            )

        self.step_log.append({
            "step": self.step + 1,
            "action": "expand",
            "new_node_id": child.node_id,
            "parent_node_id": parent_id,
            "score": child.score,
            "terminal": child.is_terminal,
            "complete": child.is_complete,
            "n_paragraphs_added": len(collected),
            "n_nodes": len(self.nodes),
        })
        self.step += 1

    def _build_expand_node(self, text, parent_id):
        """Build child node by appending one paragraph."""
        parent = next((n for n in self.nodes if n.node_id == parent_id), None)
        if not parent:
            return None
        if not text or not text.strip():
            return None
        paras = parent.paragraphs + [text]
        full_text = "".join(paras)
        return SearchNode(
            node_id=0,
            paragraphs=paras,
            parent_id=parent_id,
            is_terminal="### Final Answer" in full_text,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=parent.depth + 1,
            operation="expand",
        )

    def _process_finish(self, text, node_id):
        text = text.strip()
        if not text:
            return
        parent = next((n for n in self.nodes if n.node_id == node_id), None)
        if not parent:
            return
        paras = parent.paragraphs + [text]
        full_text = "".join(paras)
        child = SearchNode(
            node_id=0,
            paragraphs=paras,
            parent_id=node_id,
            is_terminal=True,
            is_complete=extract_last_json_obj(full_text) is not None,
            depth=parent.depth + 1,
            operation="expand",
        )
        child = self._add_node(child)
        if not self._try_root_verify_rule_based(child):
            verify_count = sum(
                self._enqueue_verify(child.node_id, goal.goal_id)
                for goal in self.goal_tree.non_root_goals()
            )
            if self._enqueue_verify(child.node_id, self.goal_tree.root_id):
                verify_count += 1
            if verify_count > 0:
                self._pending_verify_count[child.node_id] = verify_count
            child.score = _compute_recursive_score(
                child, self.goal_tree.root_id, self.goal_tree,
                self.sat_cache, self.ground_truth,
            )

    def _process_verify(self, text, node_id, goal_id):
        score = 0.0
        matches = re.findall(r"\b(YES|NO)\b", text, re.IGNORECASE)
        if matches:
            score = 1.0 if matches[-1].upper() == "YES" else 0.0
        self.sat_cache[(node_id, goal_id)] = score

        if score >= 1.0 and goal_id == self.goal_tree.root_id:
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if node:
                node.score = 1.0
                self.result_answer = json.dumps(node.answer) if node.answer else ""
                self.result_text = node.text
                self.done = True
                return

        if score >= 1.0:
            goal = self.goal_tree.goals.get(goal_id)
            if goal:
                goal.deprecated = True

        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if node:
            node.score = _compute_recursive_score(
                node, self.goal_tree.root_id, self.goal_tree,
                self.sat_cache, self.ground_truth,
            )

        # Auto-finish
        if node_id in self._pending_verify_count:
            self._pending_verify_count[node_id] -= 1
            if self._pending_verify_count[node_id] <= 0:
                del self._pending_verify_count[node_id]
                if (node and node.paragraphs
                        and not node.is_terminal
                        and node_id not in self._auto_finished):
                    all_goal_ids = [g.goal_id for g in self.goal_tree.goals.values()]
                    pushed = False
                    for gid in all_goal_ids:
                        node_sat = self.sat_cache.get((node_id, gid), 0.0)
                        if node_sat <= 0:
                            continue
                        best_others = max(
                            (self.sat_cache.get((n.node_id, gid), 0.0)
                             for n in self.nodes if n.node_id != node_id and n.paragraphs),
                            default=0.0
                        )
                        if node_sat > best_others:
                            pushed = True
                            break
                    if pushed:
                        self._auto_finished.add(node_id)
                        self._finish_queue.append(node_id)

    def _process_decompose(self, text, goal_id) -> int:
        new_goal_ids = []
        subgoals = _extract_json_array(text)
        if subgoals:
            for sg in subgoals:
                if isinstance(sg, dict):
                    desc = str(sg.get("description", "")).strip()
                    ans = str(sg.get("answer", "")).strip()
                    if desc:
                        gid = self.goal_tree.add_subgoal(goal_id, desc, ans)
                        new_goal_ids.append(gid)
        if not new_goal_ids:
            blocks = re.findall(
                r"<subgoal>\s*<description>(.*?)</description>\s*<answer>(.*?)</answer>\s*</subgoal>",
                text, re.DOTALL,
            )
            for desc, ans in blocks:
                if desc.strip():
                    gid = self.goal_tree.add_subgoal(goal_id, desc.strip(), ans.strip())
                    new_goal_ids.append(gid)
        for node in self.nodes:
            if not node.paragraphs:
                continue
            for gid in new_goal_ids:
                self._enqueue_verify(node.node_id, gid)
        return len(new_goal_ids)

    def _make_decompose_request(self):
        leaves = self.goal_tree.leaf_goals()
        eligible = []
        for leaf in leaves:
            if leaf.deprecated or leaf.depth >= self.max_goal_depth:
                continue
            fully_satisfied = any(
                self.sat_cache.get((n.node_id, leaf.goal_id), 0.0) >= 1.0
                for n in self.nodes if n.paragraphs
            )
            if not fully_satisfied:
                eligible.append(leaf)
        if not eligible:
            return None
        target = random.choice(eligible)
        content = DECOMPOSE_PROMPT.format(
            problem=self.problem,
            answer=self.ground_truth,
            goal=target.description,
            goal_answer=target.answer or self.ground_truth,
        )

        def _validate_decompose(text):
            if _extract_json_array(text):
                return True
            if re.search(r"<subgoal>.*?<description>", text, re.DOTALL):
                return True
            return False

        return LLMRequest(
            model_name=self.VERIFIER_MODEL,  # Gemini does decompose too
            request_type=RequestType.DECOMPOSE,
            prompt=content,  # Plain text for Gemini chat API
            params=self.decompose_params,
            decompose_goal_id=target.goal_id,
            validator=_validate_decompose,
        )

    def _finish(self):
        best = _pick_best(self.nodes)
        self.result_answer = json.dumps(best.answer) if best and best.answer else ""
        self.result_text = best.text if best else ""
        self.done = True

    # ------------------------------------------------------------------
    # Response selection for GRPO
    # ------------------------------------------------------------------

    def select_top_n(self, n: int) -> list[str]:
        """Select n best diverse complete responses for GRPO training."""
        # Prefer complete terminal nodes
        complete = sorted(
            [nd for nd in self.nodes if nd.is_terminal and nd.is_complete],
            key=lambda nd: nd.score, reverse=True,
        )
        # Deduplicate by answer
        seen_answers = set()
        diverse = []
        for nd in complete:
            ans_key = json.dumps(nd.answer, sort_keys=True) if nd.answer else ""
            if ans_key not in seen_answers:
                seen_answers.add(ans_key)
                diverse.append(nd.text)
            if len(diverse) >= n:
                break

        # If not enough, add remaining complete (even duplicate answers)
        if len(diverse) < n:
            for nd in complete:
                if nd.text not in diverse:
                    diverse.append(nd.text)
                if len(diverse) >= n:
                    break

        # If still not enough, add best incomplete nodes
        if len(diverse) < n:
            incomplete = sorted(
                [nd for nd in self.nodes if nd.paragraphs and nd.text not in diverse],
                key=lambda nd: nd.score, reverse=True,
            )
            for nd in incomplete:
                diverse.append(nd.text)
                if len(diverse) >= n:
                    break

        # Pad with empty if truly nothing (shouldn't happen)
        while len(diverse) < n:
            diverse.append("")

        return diverse[:n]


# ---------------------------------------------------------------------------
# KKSearchRolloutManager: batch scheduler for GRPO integration
# ---------------------------------------------------------------------------


class KKSearchRolloutManager:
    def __init__(
        self,
        vllm_generate_fn,
        tokenizer,
        verifier_base_url: str,
        verifier_api_key: str,
        verifier_model: str = "gemini-3-flash-preview",
        search_budget: int = 20,
        search_kwargs: dict | None = None,
    ):
        """
        Args:
            vllm_generate_fn: async callable(prompt: str, params: dict) -> str
                Function to generate text from the local vLLM policy model.
            tokenizer: the policy model's tokenizer
            verifier_base_url: Gemini API base URL
            verifier_api_key: Gemini API key
            verifier_model: Gemini model name
            search_budget: search steps per prompt
            search_kwargs: additional kwargs for SearchState
        """
        self.vllm_generate_fn = vllm_generate_fn
        self.tokenizer = tokenizer
        self.verifier_base_url = verifier_base_url
        self.verifier_api_key = verifier_api_key
        self.verifier_model = verifier_model
        self.search_budget = search_budget
        self.search_kwargs = search_kwargs or {}

    async def _send_verifier_request(self, client, prompt: str, params: dict,
                                     validator=None, max_retries=3) -> str:
        """Send request to Gemini API with retry."""
        last_text = None
        for attempt in range(max_retries):
            try:
                r = await client.chat.completions.create(
                    model=self.verifier_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=params.get("max_tokens", 4096),
                    temperature=params.get("temperature", 1.0),
                    top_p=params.get("top_p", 0.95),
                )
                text = r.choices[0].message.content
                last_text = text
                if validator and not validator(text):
                    if attempt < max_retries - 1:
                        continue
                    return text  # return last attempt even if validation fails
                return text
            except Exception as e:
                last_text = f"API error: {e}"
                if attempt == max_retries - 1:
                    logger.warning(f"Verifier request failed after {max_retries} retries: {e}")
                    return last_text
                await asyncio.sleep(2 ** attempt)
        return last_text or ""

    async def run_search_batch(
        self,
        problems: list[dict],
        n: int,
    ) -> dict[str, list[str]]:
        """Run search for a batch of problems concurrently.

        Args:
            problems: list of {"problem_id": str, "problem": str, "ground_truth": str}
            n: number of responses to select per problem

        Returns:
            dict mapping problem_id to list of n response texts
        """
        import httpx
        from openai import AsyncOpenAI

        # Init async Gemini client
        gemini_client = AsyncOpenAI(
            base_url=self.verifier_base_url,
            api_key=self.verifier_api_key,
            timeout=httpx.Timeout(timeout=300, connect=10.0),
        )

        # Init search states
        states: dict[str, SearchState] = {}
        for prob in problems:
            states[prob["problem_id"]] = SearchState(
                problem_id=prob["problem_id"],
                problem=prob["problem"],
                ground_truth=prob["ground_truth"],
                tokenizer=self.tokenizer,
                budget=self.search_budget,
                **self.search_kwargs,
            )

        busy_states: set[str] = set()
        in_flight: dict[asyncio.Task, tuple[str, LLMRequest]] = {}

        async def _send_main_request(prompt, params):
            """Send expand request to local vLLM."""
            return await self.vllm_generate_fn(prompt, params)

        def _collect_requests():
            for pid, state in states.items():
                if pid in busy_states or state.done:
                    continue
                while True:
                    req = state.next_request()
                    if not req:
                        break
                    if req.model_name == SearchState.MAIN_MODEL:
                        task = asyncio.create_task(_send_main_request(req.prompt, req.params))
                    else:
                        task = asyncio.create_task(
                            self._send_verifier_request(
                                gemini_client, req.prompt, req.params, req.validator
                            )
                        )
                    in_flight[task] = (pid, req)
                    if req.request_type not in (RequestType.VERIFY, RequestType.EXPAND_FINISH):
                        busy_states.add(pid)
                        break

        _collect_requests()

        while in_flight:
            done_tasks, _ = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                pid, req = in_flight.pop(task)
                has_other = any(p == pid for p, _ in in_flight.values())
                if not has_other:
                    busy_states.discard(pid)

                try:
                    text = task.result()
                except Exception as e:
                    logger.warning(f"Request error for {pid}: {e}")
                    text = None

                state = states[pid]
                if text is not None and not state.done:
                    state.process_result(text, req)
                elif not state.done and req.request_type == RequestType.DECOMPOSE:
                    if state._phase == "wait_init_decompose":
                        state._phase = "first_expand"

            _collect_requests()

        # Select top-n responses per problem
        results = {}
        for pid, state in states.items():
            if not state.done:
                state._finish()
            results[pid] = state.select_top_n(n)
            logger.info(
                f"Search done: {pid}, nodes={len(state.nodes)}, "
                f"steps={state.step}, complete={sum(1 for nd in state.nodes if nd.is_complete)}"
            )

        return results

    def format_as_dataproto(
        self,
        search_results: dict[str, list[str]],
        gen_batch,
        n: int,
    ):
        """Format search results as DataProto matching verl's rollout output.

        Args:
            search_results: dict mapping problem_id to list of n response texts
            gen_batch: original DataProto batch (before repeat)
            n: number of responses per prompt

        Returns:
            DataProto with prompts, responses, input_ids, attention_mask,
            position_ids, response_mask — same format as AgentLoopManager output.
        """
        from tensordict import TensorDict
        from verl.protocol import DataProto

        batch_size = len(gen_batch)
        tokenizer = self.tokenizer

        # Get prompt token ids from the original batch
        prompt_ids_list = gen_batch.batch["input_ids"]  # (B, prompt_len)
        prompt_attn_list = gen_batch.batch["attention_mask"]  # (B, prompt_len)

        # Get problem IDs to match search results
        # uid is assigned later; we use index order to match
        uids = gen_batch.non_tensor_batch.get("uid", np.arange(batch_size).astype(str))

        # Determine max response length across all search results
        all_response_ids = []
        all_prompt_ids = []

        for i in range(batch_size):
            pid = str(i)  # problem index as ID
            responses = search_results.get(pid, [""] * n)
            prompt = prompt_ids_list[i]  # (prompt_len,)
            prompt_mask = prompt_attn_list[i]

            for resp_text in responses:
                # Tokenize response
                resp_tokens = tokenizer.encode(resp_text, add_special_tokens=False)
                all_response_ids.append(resp_tokens)
                all_prompt_ids.append((prompt, prompt_mask))

        # Pad responses to same length
        max_resp_len = max(len(r) for r in all_response_ids) if all_response_ids else 1
        max_resp_len = max(max_resp_len, 1)

        prompt_length = prompt_ids_list.shape[1]
        total_len = prompt_length + max_resp_len
        total_samples = batch_size * n

        # Build tensors
        all_input_ids = torch.zeros(total_samples, total_len, dtype=torch.long)
        all_attention_mask = torch.zeros(total_samples, total_len, dtype=torch.long)
        all_position_ids = torch.zeros(total_samples, total_len, dtype=torch.long)
        all_prompts = torch.zeros(total_samples, prompt_length, dtype=torch.long)
        all_responses = torch.zeros(total_samples, max_resp_len, dtype=torch.long)
        all_response_mask = torch.zeros(total_samples, max_resp_len, dtype=torch.long)

        for idx in range(total_samples):
            prompt_tokens, prompt_mask = all_prompt_ids[idx]
            resp_tokens = all_response_ids[idx]
            resp_len = len(resp_tokens)

            # Prompts (left-padded, same as original)
            all_prompts[idx] = prompt_tokens
            all_input_ids[idx, :prompt_length] = prompt_tokens
            all_attention_mask[idx, :prompt_length] = prompt_mask

            # Responses (right-padded)
            if resp_len > 0:
                actual_len = min(resp_len, max_resp_len)
                resp_tensor = torch.tensor(resp_tokens[:actual_len], dtype=torch.long)
                all_responses[idx, :actual_len] = resp_tensor
                all_response_mask[idx, :actual_len] = 1
                all_input_ids[idx, prompt_length:prompt_length + actual_len] = resp_tensor
                all_attention_mask[idx, prompt_length:prompt_length + actual_len] = 1

            # Position IDs: cumsum of attention_mask - 1
            all_position_ids[idx] = torch.cumsum(all_attention_mask[idx], dim=0) - 1
            all_position_ids[idx] = torch.clamp(all_position_ids[idx], min=0)

        batch_dict = TensorDict(
            {
                "prompts": all_prompts,
                "responses": all_responses,
                "response_mask": all_response_mask,
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "position_ids": all_position_ids,
            },
            batch_size=total_samples,
        )

        # Build non_tensor_batch: repeat each entry n times (interleaved)
        non_tensor_batch = {}
        for key, val in gen_batch.non_tensor_batch.items():
            non_tensor_batch[key] = np.repeat(val, n, axis=0)

        # Add multi_modal_inputs placeholder
        if "multi_modal_inputs" not in non_tensor_batch:
            non_tensor_batch["multi_modal_inputs"] = np.array(
                [{}] * total_samples, dtype=object
            )

        output = DataProto(batch=batch_dict, non_tensor_batch=non_tensor_batch)
        output.meta_info = {"timing": {}}
        return output
