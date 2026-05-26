import asyncio
import json
import os
import random
import re
import requests
import string
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import torch

from verl import DataProto

from .tensor_helper import TensorHelper, TensorConfig

# ---------------------------------------------------------------------------
# Section 1: search-state helpers
# ---------------------------------------------------------------------------

GRPO_N = 8

DECOMP_SYS = (
    "You decompose multi-hop questions into atomic sub-questions. Output a "
    "JSON array of strings; each string is one atomic sub-question. Use '#N' "
    "to refer to the answer of the N-th prior sub-question. No prose."
)
DECOMP_USER_TMPL = (
    'Examples:\n'
    'Q: Who founded the company that produces the Model S car?\n'
    'A: ["Which company produces the Model S car?", "Who founded #1?"]\n\n'
    'Q: In what year did the director of the film that won Best Picture in 2018 die?\n'
    'A: ["Which film won Best Picture in 2018?", "Who directed #1?", "In what year did #2 die?"]\n\n'
    'Q: {Q}\nA: '
)

GENERIC_PLACEHOLDER = "[X]"
_STOPWORDS = {"the", "a", "an", "of", "in", "to", "and", "or", "for", "on",
              "at", "is", "are", "was", "were", "by", "with", "as", "from"}

STOP_STRINGS = ["</search>", "</answer>"]
FINALIZE_MAX_TRIES = 10  # cap on free-finalize retries per max_turns node


def normalize_subq_for_match(subq):
    return re.sub(r"#\d+", GENERIC_PLACEHOLDER, subq)


def gen_search_abstractions(search, max_w=3, enabled=True):
    yield search
    if not enabled:
        return
    seen = {search}
    tokens = search.split()
    for w in range(1, max_w + 1):
        for i in range(len(tokens) - w + 1):
            span = tokens[i:i + w]
            if all(t.lower().strip(string.punctuation) in _STOPWORDS for t in span):
                continue
            new = " ".join(tokens[:i] + [GENERIC_PLACEHOLDER] + tokens[i + w:])
            if new not in seen:
                seen.add(new)
                yield new


from verl.utils.reward_score.qa_em_format import (
    normalize_answer,    # noqa: F401  (re-exported for tests)
    em_check,
)


def em(pred, gold):
    """Binary 0/1 EM, matching qa_em_format.em_check on a single gold."""
    return float(em_check(pred, [gold]))


TAG_RE = re.compile(r"<(think|search|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def parse_actions(text):
    out = []
    for m in TAG_RE.finditer(text):
        out.append((m.group(1).lower(), m.group(2).strip(), m.group(0)))
    return out


def _retriever_batch_search(search_url: str, queries: List[str], topk: int = 3) -> list:
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    raw = requests.post(search_url, json=payload).json()["result"]
    return [_retriever_passages2string(r) for r in raw]


def _retriever_passages2string(retrieval_result) -> str:
    if retrieval_result is None:
        return ""
    out = ""
    if len(retrieval_result) > 0 and isinstance(retrieval_result[0], dict):
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            out += f"Doc {idx+1}(Title: {title}) {text}\n"
    else:
        for tmp in retrieval_result:
            out += tmp
    return out


class BackwardClient:

    def __init__(self, base_url, model="meta-llama/Llama-3.1-8B-Instruct",
                 max_tokens=200, temperature=0.3, top_p=0.9, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def decompose(self, question):
        msgs = [
            {"role": "system", "content": DECOMP_SYS},
            {"role": "user", "content": DECOMP_USER_TMPL.format(Q=question)},
        ]
        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=body, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception:
            return [question]
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if not m:
            return [question]
        try:
            import json as _json
            arr = _json.loads(m.group(0))
            if isinstance(arr, list) and arr and all(isinstance(x, str) for x in arr):
                return arr
        except Exception:
            pass
        return [question]

    def decompose_batch(self, questions):
        if not questions:
            return []
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(64, len(questions))) as ex:
            return list(ex.map(self.decompose, questions))


# ---------------------------------------------------------------------------
# Section 4: Search node + per-question state
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    id: int
    turns: tuple
    score: float = 0.0
    terminal: bool = False
    final_answer: str = ""
    parent_id: Optional[int] = None
    # All parents. () for root, 1-tuple for expand/delete, 2-tuple for
    # combine/translocate/crossover (= (A.id, B.id)).
    parent_ids: tuple = ()
    operation: str = "expand"
    # depth==max_turns nodes can no longer be expanded (search budget
    # used) but CAN still serve as mutation parents. They get a "free
    # finalize" generation (no stop tokens, strict [think,answer] parse,
    # capped at FINALIZE_MAX_TRIES retries).
    needs_finalize: bool = False
    finalize_done: bool = False
    finalize_tries: int = 0
    subq_max_sim: list = None
    verify_score: float = 0.0

    @property
    def depth(self):
        return sum(1 for t in self.turns
                   if t.get("role") == "agent" and t.get("kind") == "search")


class _BidirState:
    """Per-question BES search state. coverage_mode forced to 'sequential'
    and span_abstract on by default.

    The driver outside calls next_request() to get the next prompt to feed
    to vLLM; once vLLM returns text, calls process_result(text, req).
    """

    def __init__(self, question, gold_targets, prompt_ids, tokenizer,
                 emb_model, sim_threshold=0.6, budget=30, max_turns=8,
                 max_obs_length=500, q_seed=0,
                 init_temp_sel=1.5, final_temp_sel=0.3, no_prog_limit=50,
                 span_abstract=True,
                 early_stop_on_correct=False,
                 think_prefix=False,
                 lambda_deg_bonus=0.1,
                 k_parallel=1):
        self.question = question
        # gold_targets: list of acceptable answer strings (golden_answers)
        self.gold_targets = list(gold_targets) if gold_targets else []
        self.prompt_ids = list(prompt_ids)
        self.tokenizer = tokenizer
        self.emb_model = emb_model
        self.sim_threshold = sim_threshold
        self.budget = budget
        self.max_turns = max_turns
        self.max_obs_length = max_obs_length
        self.init_temp_sel = init_temp_sel
        self.final_temp_sel = final_temp_sel
        self.lambda_deg_bonus = lambda_deg_bonus
        self.k_parallel = k_parallel
        self.no_prog_limit = no_prog_limit
        self.span_abstract = span_abstract
        self.early_stop_on_correct = early_stop_on_correct
        self.think_prefix = think_prefix

        self.rng = random.Random(q_seed)
        self.nodes: List[_Node] = []
        self.next_id = 0
        self._search_cands_cache: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self.subq_norms: List[str] = []
        self.subq_norm_embeds = None
        self.llm_subqs: List[str] = []

        self.budget_used = 0
        self.no_progress = 0
        self.done = False

        self.failed_attempts: List[Dict] = []
        self.mutation_log: List[Dict] = []

        self.add(turns=(), parent_id=None, operation="root")

    # -- subq decompose result wired in by the driver --
    def set_subqs(self, subqs):
        self.llm_subqs = list(subqs)
        self.subq_norms = [normalize_subq_for_match(sq) for sq in subqs]
        if self.subq_norms and self.emb_model is not None:
            self.subq_norm_embeds = self.emb_model.encode(
                self.subq_norms, normalize_embeddings=True, show_progress_bar=False)
        for n in self.nodes:
            n.score = self._score_node(n)

    # -- search candidate cache for symmetric span abstraction --
    def _get_search_cands(self, search):
        if search not in self._search_cands_cache:
            cands = list(gen_search_abstractions(search, enabled=self.span_abstract))
            embs = self.emb_model.encode(
                cands, normalize_embeddings=True, show_progress_bar=False)
            self._search_cands_cache[search] = (cands, embs)
        return self._search_cands_cache[search]

    def add(self, turns, parent_id, operation, parent_ids=None):
        turns = tuple(turns)
        for n in self.nodes:
            if n.turns == turns:
                return None
        if parent_ids is None:
            parent_ids = () if parent_id is None else (parent_id,)
        node = _Node(id=self.next_id, turns=turns, parent_id=parent_id,
                     parent_ids=tuple(parent_ids), operation=operation)
        for t in turns:
            if t.get("role") == "agent" and t.get("kind") == "answer":
                node.terminal = True
                node.final_answer = t.get("content", "")
                break
        if (not node.terminal) and node.depth >= self.max_turns:
            node.needs_finalize = True
        node.score = self._score_node(node)
        self.nodes.append(node)
        self.next_id += 1
        return node

    def _score_node(self, node):
        searches = [t["content"] for t in node.turns
                    if t.get("role") == "agent" and t.get("kind") == "search"]
        n_subq = len(self.subq_norms) if self.subq_norms else 0
        if self.subq_norm_embeds is None or len(searches) == 0:
            avg_cov = 0.0
            node.subq_max_sim = [0.0] * n_subq
        else:
            max_sims = np.full(n_subq, -1.0, dtype=np.float32)
            for s in searches:
                _, cand_embs = self._get_search_cands(s)
                sims = self.subq_norm_embeds @ cand_embs.T
                max_sims = np.maximum(max_sims, sims.max(axis=1))
            covered = max_sims >= self.sim_threshold
            seq = 0
            for c in covered:
                if c: seq += 1
                else: break
            avg_cov = seq / n_subq
            node.subq_max_sim = [float(x) for x in max_sims]
        if node.terminal and node.final_answer and self.gold_targets:
            root_ok = max((em(node.final_answer, g) for g in self.gold_targets), default=0.0)
        else:
            root_ok = 0.0
        node.verify_score = float(root_ok)
        # Early-stop signal for training: once we've found a correct answer,
        # stop searching this question (caller will pad with chain rollouts).
        if self.early_stop_on_correct and root_ok >= 1.0:
            self.done = True
        return 0.7 * root_ok + 0.3 * avg_cov

    # -- prompt construction --
    def _normalize_response(self, text):
        return ("<think>" + text) if self.think_prefix else text

    def _build_request_ids(self, turns):
        rolling = "".join(t["raw"] for t in turns) if turns else ""
        if self.think_prefix:
            rolling += "<think>"
        if not rolling:
            return list(self.prompt_ids)
        roll_ids = self.tokenizer.encode(rolling, add_special_tokens=False)
        return list(self.prompt_ids) + roll_ids

    def _render_text_for_log(self, turns):
        """Decode-friendly text used only for logging/visualization."""
        return self.tokenizer.decode(self.prompt_ids, skip_special_tokens=False) \
               + "".join(t["raw"] for t in turns)

    # -- pool / Boltzmann --
    def _node_degree(self, node):
        """deg(n) = number of children of n in the pool (any operation that
        lists n in parent_ids)."""
        nid = node.id
        return sum(1 for n in self.nodes if nid in (n.parent_ids or ()))

    def _unary_score(self, node):
        """s(n) + λ if node has no children, else s(n)."""
        return node.score + (self.lambda_deg_bonus
                             if self._node_degree(node) == 0 else 0.0)

    def _temp_sel(self):
        t = self.budget_used / max(self.budget, 1)
        return self.init_temp_sel - (self.init_temp_sel - self.final_temp_sel) * t

    def _boltzmann_idx(self, scores, temp):
        if not scores:
            return None
        if temp <= 0:
            return scores.index(max(scores))
        m = max(scores)
        weights = [np.exp((s - m) / max(temp, 1e-6)) for s in scores]
        total = sum(weights)
        if total <= 0:
            return self.rng.randrange(len(scores))
        r = self.rng.random() * total
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if cum >= r:
                return i
        return len(scores) - 1

    # -- mutations --
    def _step_starts(self, turns):
        starts = [i for i, t in enumerate(turns)
                  if t.get("role") == "agent" and t.get("kind") == "think"]
        starts.append(len(turns))
        return starts

    def _mut_combine(self, a, b):
        common = 0
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                common = i + 1
            else:
                break
        if common == 0 or common >= len(a) or common >= len(b):
            return None
        return tuple(a) + tuple(b[common:])

    def _mut_delete(self, a):
        """Remove one interior (think, search, info) triple. Requires at least 3 triples."""
        starts = self._step_starts(a)
        triples = []
        for k in range(len(starts) - 1):
            i, j = starts[k], starts[k + 1]
            if (j - i) == 3 and \
               a[i + 1].get("role") == "agent" and a[i + 1].get("kind") == "search" and \
               a[i + 2].get("role") == "obs":
                triples.append((i, j))
        if len(triples) < 3:
            return None
        idx = self.rng.randint(1, len(triples) - 2)   # interior, 0-indexed
        start, end = triples[idx]
        return tuple(a[:start]) + tuple(a[end:])

    def _mut_translocate(self, a, b):
        """Replace one (think, search, info) triple in a with one drawn from b (after shared prefix)."""
        s = 0
        for k in range(min(len(a), len(b))):
            if a[k] == b[k]:
                s = k + 1
            else:
                break
        a_starts = self._step_starts(a)
        b_starts = self._step_starts(b)
        a_triples = []   # (start, end) ranges in σ_a (start ≥ s)
        for k in range(len(a_starts) - 1):
            i, j = a_starts[k], a_starts[k + 1]
            if i < s: continue
            if (j - i) == 3 and \
               a[i + 1].get("role") == "agent" and a[i + 1].get("kind") == "search" and \
               a[i + 2].get("role") == "obs":
                a_triples.append((i, j))
        b_triples = []   # full triples (turns) from σ_b (start ≥ s)
        for k in range(len(b_starts) - 1):
            i, j = b_starts[k], b_starts[k + 1]
            if i < s: continue
            if (j - i) == 3 and \
               b[i + 1].get("role") == "agent" and b[i + 1].get("kind") == "search" and \
               b[i + 2].get("role") == "obs":
                b_triples.append(tuple(b[i:j]))
        if not a_triples or not b_triples:
            return None
        r_start, r_end = self.rng.choice(a_triples)
        b_triple = self.rng.choice(b_triples)
        return tuple(a[:r_start]) + b_triple + tuple(a[r_end:])

    def _mut_crossover(self, a, b):
        """Crossover: shared prefix + a's suffix up to cut i + b's suffix from cut j."""
        s = 0
        for k in range(min(len(a), len(b))):
            if a[k] == b[k]:
                s = k + 1
            else:
                break
        a_cuts = [i for i, t in enumerate(a)
                  if i >= s and t.get("role") == "agent" and t.get("kind") == "think"]
        a_cuts.append(len(a))   # i = m_a (take all of σ_a)
        b_cuts = [i for i, t in enumerate(b)
                  if i >= s and t.get("role") == "agent" and t.get("kind") == "think"]
        if not b_cuts:
            return None
        ca = self.rng.choice(a_cuts)
        cb = self.rng.choice(b_cuts)
        return tuple(a[:ca]) + tuple(b[cb:])

    # -- joint Boltzmann pair selection (binary mutations) --
    def _node_covered(self, node):
        if self.subq_norm_embeds is None:
            return None
        n_subq = len(self.subq_norms)
        searches = [t["content"] for t in node.turns
                    if t.get("role") == "agent" and t.get("kind") == "search"]
        if not searches:
            return np.zeros(n_subq, dtype=bool)
        max_sims = np.full(n_subq, -1.0, dtype=np.float32)
        for s in searches:
            _, cand_embs = self._get_search_cands(s)
            sims = self.subq_norm_embeds @ cand_embs.T
            max_sims = np.maximum(max_sims, sims.max(axis=1))
        return max_sims >= self.sim_threshold

    def _node_verify(self, node):
        if not (node.terminal and node.final_answer and self.gold_targets):
            return 0.0
        return max((em(node.final_answer, g) for g in self.gold_targets), default=0.0)

    def _joint_score(self, a, b):
        cov_a = self._node_covered(a)
        cov_b = self._node_covered(b)
        if cov_a is None or cov_b is None:
            avg_joint = 0.0
        else:
            joint = np.logical_or(cov_a, cov_b)
            seq = 0
            for c in joint:
                if c: seq += 1
                else: break
            avg_joint = seq / len(joint) if len(joint) else 0.0
        v = max(self._node_verify(a), self._node_verify(b))
        return 0.7 * v + 0.3 * avg_joint

    def _select_pair_boltzmann(self, pool, temp):
        if len(pool) < 2:
            return None
        pairs, scores = [], []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                pairs.append((pool[i], pool[j]))
                scores.append(self._joint_score(pool[i], pool[j]))
        if not pairs:
            return None
        idx = self._boltzmann_idx(scores, temp)
        if idx is None:
            return None
        return pairs[idx]

    @staticmethod
    def _truncate_to_max_search_turns(turns, max_search):
        out = []
        n_search = 0
        for t in turns:
            if (t.get("role") == "agent" and t.get("kind") == "search"
                    and n_search >= max_search):
                if (out and out[-1].get("role") == "agent"
                        and out[-1].get("kind") == "think"):
                    out.pop()
                break
            out.append(t)
            if t.get("role") == "agent" and t.get("kind") == "search":
                n_search += 1
        return tuple(out)

    def _try_mutation_typed(self, mt):
        pool = [n for n in self.nodes[1:] if not n.terminal]

        def _log(outcome, **extra):
            self.mutation_log.append({
                "mutation_type": mt,
                "budget_used": self.budget_used,
                "outcome": outcome,
                **extra,
            })

        if not pool:
            _log("rejected_empty_pool")
            return False
        new = None
        pids = ()
        if mt == "delete":
            scores = [self._unary_score(n) for n in pool]
            a = pool[self._boltzmann_idx(scores, self._temp_sel())]
            new = self._mut_delete(a.turns)
            pids = (a.id,)
        else:
            if len(pool) < 2:
                _log("rejected_pool_too_small")
                return False
            pair = self._select_pair_boltzmann(pool, self._temp_sel())
            if pair is None:
                _log("rejected_no_pair")
                return False
            a, b = pair
            if mt == "combine":
                new = self._mut_combine(a.turns, b.turns)
            elif mt == "translocate":
                new = self._mut_translocate(a.turns, b.turns)
            elif mt == "crossover":
                new = self._mut_crossover(a.turns, b.turns)
            pids = (a.id, b.id)
        if new is None:
            _log("rejected_mutation_returned_none", parent_ids=list(pids))
            return False
        n_search = sum(1 for t in new
                       if t.get("role") == "agent" and t.get("kind") == "search")
        truncated = False
        if n_search > self.max_turns:
            new = self._truncate_to_max_search_turns(new, self.max_turns)
            truncated = True
            if not new:
                _log("rejected_truncate_to_zero", parent_ids=list(pids))
                return False
        added = self.add(new, parent_id=pids[0], operation=mt,
                         parent_ids=pids)
        if added is None:
            _log("rejected_duplicate", parent_ids=list(pids), truncated=truncated)
            return False
        self.budget_used += 1
        self.no_progress = 0
        _log("success", parent_ids=list(pids), new_node_id=added.id,
             truncated=truncated, child_score=added.score,
             child_terminal=added.terminal,
             child_final_answer=added.final_answer or "")
        return True

    def _eligible_pool(self):
        return [n for n in self.nodes if not n.terminal and n.depth < self.max_turns]

    def _check_budget(self):
        if self.budget_used >= self.budget:
            self.done = True
            return False
        return True

    def next_request(self):
        if self.done:
            return None
        if not self._check_budget():
            return None
        fin_pool = [n for n in self.nodes
                    if n.needs_finalize and not n.finalize_done
                    and not n.terminal]
        if fin_pool:
            scores = [self._unary_score(n) for n in fin_pool]
            parent = fin_pool[self._boltzmann_idx(scores, self._temp_sel())]
            request_ids = self._build_request_ids(parent.turns)
            return {"type": "finalize",
                    "prompt_ids": request_ids,
                    "parent_id": parent.id,
                    "stop": list(STOP_STRINGS)}
        for _ in range(self.budget * 2):
            if not self._check_budget():
                return None
            roll = self.rng.random()
            if roll < 0.30:
                if roll < 0.10:    mt = "combine"
                elif roll < 0.15:  mt = "delete"
                elif roll < 0.225: mt = "translocate"
                else:              mt = "crossover"
                if self._try_mutation_typed(mt):
                    continue
                self.no_progress += 1
                if self.no_progress >= self.no_prog_limit:
                    self.done = True; return None
                continue

            eligible = self._eligible_pool()
            if not eligible:
                for fb in ("combine", "translocate", "crossover", "delete"):
                    if self._try_mutation_typed(fb):
                        break
                else:
                    self.no_progress += 1
                    if self.no_progress >= self.no_prog_limit:
                        self.done = True; return None
                continue

            scores = [self._unary_score(n) for n in eligible]
            idx = self._boltzmann_idx(scores, self._temp_sel())
            parent = eligible[idx]
            request_ids = self._build_request_ids(parent.turns)
            return {"type": "expand",
                    "prompt_ids": request_ids,
                    "parent_id": parent.id,
                    "stop": list(STOP_STRINGS)}

        self.done = True
        return None

    def next_request_k(self, K: int) -> List[Dict]:
        reqs = []
        for _ in range(K):
            if self.done:
                break
            req = self.next_request()
            if req is None:
                break
            reqs.append(req)
        return reqs

    @staticmethod
    def peek_search_query(text):
        actions = parse_actions(text)
        if (len(actions) == 2 and actions[0][0] == "think"
                and actions[1][0] == "search"
                and actions[0][1] and actions[1][1]):
            return actions[1][1]
        return None

    def process_result(self, text, token_ids, req, info_map=None):
        if req.get("type") == "finalize":
            self._handle_finalize(text, token_ids, req["parent_id"])
            return
        self.budget_used += 1
        text = self._normalize_response(text)
        actions = parse_actions(text)
        bad = (
            len(actions) != 2
            or actions[0][0] != "think"
            or actions[1][0] not in ("search", "answer")
            or not actions[0][1]
            or not actions[1][1]
        )
        if bad:
            self.failed_attempts.append({
                "round": self.budget_used,
                "parent_id": req.get("parent_id"),
                "reason": "format_mismatch",
                "n_tags": len(actions),
                "tag_kinds": [a[0] for a in actions],
                "raw_response": text,
            })
            self.no_progress += 1
            if self.no_progress >= self.no_prog_limit:
                self.done = True
            return

        k_think, c_think, r_think = actions[0]
        k_act, c_act, r_act = actions[1]

        parent_id = req["parent_id"]
        parent = self.nodes[parent_id]
        new_turns = list(parent.turns)
        block_ids = list(token_ids)
        new_turns.append({"role": "agent", "kind": "think",
                          "raw": r_think.strip(), "content": c_think,
                          "_block_token_ids": block_ids})
        new_turns.append({"role": "agent", "kind": k_act,
                          "raw": r_act.strip(), "content": c_act,
                          "_block_token_ids": block_ids})
        if k_act == "search":
            info = (info_map or {}).get(c_act, "")
            obs_raw = f"\n\n<information>{info.strip()}</information>\n\n"
            obs_ids = self.tokenizer.encode(obs_raw, add_special_tokens=False)
            if len(obs_ids) > self.max_obs_length:
                obs_ids = obs_ids[:self.max_obs_length]
                obs_raw = self.tokenizer.decode(obs_ids, skip_special_tokens=False)
            # Cache obs_ids alongside obs_raw so build doesn't re-encode.
            new_turns.append({"role": "obs", "kind": "info",
                              "raw": obs_raw, "content": info,
                              "_obs_token_ids": list(obs_ids)})
        added = self.add(tuple(new_turns), parent_id=parent_id, operation="expand")
        if added is None:
            self.failed_attempts.append({
                "round": self.budget_used,
                "parent_id": parent_id,
                "reason": "duplicate",
                "action_kind": k_act,
                "action_content": c_act,
                "raw_response": text,
            })
            self.no_progress += 1
            if self.no_progress >= self.no_prog_limit:
                self.done = True
        else:
            self.no_progress = 0

    def _handle_finalize(self, text, token_ids, parent_id):
        self.budget_used += 1
        text = self._normalize_response(text)
        parent = self.nodes[parent_id]
        parent.finalize_tries += 1
        actions = parse_actions(text)
        bad = (
            len(actions) != 2
            or actions[0][0] != "think"
            or actions[1][0] != "answer"
            or not actions[0][1]
            or not actions[1][1]
        )
        if bad:
            self.failed_attempts.append({
                "round": self.budget_used,
                "parent_id": parent_id,
                "reason": "finalize_bad_format",
                "n_tags": len(actions),
                "tag_kinds": [a[0] for a in actions],
                "try_idx": parent.finalize_tries,
                "raw_response": text,
            })
            if parent.finalize_tries >= FINALIZE_MAX_TRIES:
                parent.finalize_done = True
            self.no_progress += 1
            if self.no_progress >= self.no_prog_limit:
                self.done = True
            return
        _, c_think, r_think = actions[0]
        _, c_ans, r_ans = actions[1]
        block_ids = list(token_ids)
        new_turns = list(parent.turns) + [
            {"role": "agent", "kind": "think",
             "raw": r_think.strip(), "content": c_think,
             "_block_token_ids": block_ids},
            {"role": "agent", "kind": "answer",
             "raw": r_ans.strip(), "content": c_ans,
             "_block_token_ids": block_ids},
        ]
        added = self.add(tuple(new_turns), parent_id=parent_id,
                         operation="finalize")
        if added is None:
            self.failed_attempts.append({
                "round": self.budget_used,
                "parent_id": parent_id,
                "reason": "finalize_duplicate",
                "action_kind": "answer",
                "action_content": c_ans,
                "try_idx": parent.finalize_tries,
                "raw_response": text,
            })
            if parent.finalize_tries >= FINALIZE_MAX_TRIES:
                parent.finalize_done = True
            self.no_progress += 1
            if self.no_progress >= self.no_prog_limit:
                self.done = True
            return
        parent.finalize_done = True
        self.no_progress = 0


class _ChainRollout:

    def __init__(self, prompt_ids, tokenizer, max_turns, max_obs_length,
                 question_idx, gold_targets, think_prefix=False):
        self.prompt_ids = list(prompt_ids)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_obs_length = max_obs_length
        self.question_idx = question_idx
        self.gold_targets = list(gold_targets) if gold_targets else []
        self.turns: List[Dict] = []
        self.n_actions = 0          # search + answer (chain semantics)
        self.done = False
        self.terminal = False       # True iff trajectory ended on <answer>
        self.final_answer = ""
        self.score = 0.0            # 1.0 if EM correct, else 0.0
        self.think_prefix = think_prefix
        self.forced_round_done = False

    def _normalize_response(self, text):
        return ("<think>" + text) if self.think_prefix else text

    def _build_request_ids(self):
        rolling = "".join(t["raw"] for t in self.turns) if self.turns else ""
        if self.think_prefix:
            rolling += "<think>"
        if not rolling:
            return list(self.prompt_ids)
        roll_ids = self.tokenizer.encode(rolling, add_special_tokens=False)
        return list(self.prompt_ids) + roll_ids

    def next_request(self):
        if self.done:
            return None
        if self.n_actions >= self.max_turns:
            if self.forced_round_done:
                self.done = True
                return None
            return {"type": "chain_force_answer",
                    "prompt_ids": self._build_request_ids(),
                    "stop": list(STOP_STRINGS)}
        return {"type": "chain_step",
                "prompt_ids": self._build_request_ids(),
                "stop": list(STOP_STRINGS)}

    def process_result(self, text, token_ids, info_map=None, is_force_round=False):
        text = self._normalize_response(text)
        actions = parse_actions(text)
        first_act = next(((k, c, r) for (k, c, r) in actions
                          if k in ("search", "answer")), None)
        if first_act is None:
            # Invalid response (no action) — count as wasted round and end
            self.done = True
            if is_force_round:
                self.forced_round_done = True
            return
        block_ids = list(token_ids)
        # Take any preceding <think> for context (zero or one)
        first_think = next(((k, c, r) for (k, c, r) in actions if k == "think"), None)
        if first_think is not None and (
                actions.index(first_think) < actions.index(first_act)):
            _, c_th, r_th = first_think
            if c_th:
                self.turns.append({"role": "agent", "kind": "think",
                                   "raw": r_th.strip(), "content": c_th,
                                   "_block_token_ids": block_ids})
        kind, content, raw = first_act
        self.turns.append({"role": "agent", "kind": kind,
                           "raw": raw.strip(), "content": content,
                           "_block_token_ids": block_ids})
        self.n_actions += 1
        if kind == "answer":
            self.terminal = True
            self.final_answer = content
            self.done = True
            if self.gold_targets:
                self.score = float(max(
                    (em(content, g) for g in self.gold_targets), default=0.0))
            if is_force_round:
                self.forced_round_done = True
            return
        if kind == "search":
            info = (info_map or {}).get(content, "")
            obs_raw = f"\n\n<information>{info.strip()}</information>\n\n"
            ids = self.tokenizer.encode(obs_raw, add_special_tokens=False)
            if len(ids) > self.max_obs_length:
                ids = ids[:self.max_obs_length]
                obs_raw = self.tokenizer.decode(ids, skip_special_tokens=False)
            self.turns.append({"role": "obs", "kind": "info",
                               "raw": obs_raw, "content": info,
                               "_obs_token_ids": list(ids)})
            if is_force_round:
                self.forced_round_done = True
                self.done = True
            elif self.n_actions >= self.max_turns:
                pass

    @staticmethod
    def peek_search_query(text):
        actions = parse_actions(text)
        first_search = next(((k, c, r) for (k, c, r) in actions
                             if k == "search"), None)
        if first_search is None:
            return None
        _, c, _ = first_search
        return c if c else None


@dataclass
class GenerationBidirectionalConfig:
    max_turns: int = 8
    max_start_length: int = 2048
    max_prompt_length: int = 4096
    max_response_length: int = 4096
    max_obs_length: int = 500
    num_gpus: int = 2
    # Search
    budget: int = 30
    sim_threshold: float = 0.6
    no_span_abstract: bool = False
    # Retrieval
    search_url: str = ""
    topk: int = 3
    # Backward (decompose) server
    backward_url: str = "http://localhost:8236/v1"
    backward_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str = "cpu"
    # GRPO group size
    grpo_n: int = 8
    k_parallel: int = 1
    think_prefix: bool = False




# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _async_decompose_one(session, base_url, model, question,
                               max_tokens=200, temperature=0.3, top_p=0.9,
                               timeout=120):
    msgs = [
        {"role": "system", "content": DECOMP_SYS},
        {"role": "user", "content": DECOMP_USER_TMPL.format(Q=question)},
    ]
    body = {"model": model, "messages": msgs,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    try:
        async with session.post(f"{base_url.rstrip('/')}/chat/completions",
                                json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            text = (await r.json())["choices"][0]["message"]["content"]
    except Exception:
        return [question]
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return [question]
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list) and arr and all(isinstance(x, str) for x in arr):
            return arr
    except Exception:
        pass
    return [question]


async def _async_decompose_batch(base_url, model, questions):
    if not questions:
        return []
    connector = aiohttp.TCPConnector(limit=max(64, len(questions)))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_async_decompose_one(session, base_url, model, q) for q in questions]
        return await asyncio.gather(*tasks)


async def _async_retrieve(session, search_url, query, topk, timeout=120):
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    try:
        async with session.post(search_url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            obj = await r.json()
        return _retriever_passages2string(obj["result"][0])
    except Exception as e:
        return f"(retrieval failed: {e})"


class _BatchedRetriever:
    """Aggregates concurrent retrieve queries into one HTTP POST.

    Each state/chain task awaits `submit(query)`. The batcher loop drains
    the queue up to `max_batch` (or `max_wait_ms` deadline), POSTs one
    `{"queries": [...], "topk": K}` to the retriever, then resolves all
    futures with their respective formatted-passages strings.
    """

    def __init__(self, http_session, search_url, topk,
                 max_batch=128, max_wait_ms=15, timeout=180):
        self.session = http_session
        self.search_url = search_url
        self.topk = topk
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.timeout = timeout
        self.queue: Optional[asyncio.Queue] = None
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self):
        self.queue = asyncio.Queue()
        self._loop_task = asyncio.create_task(self._batcher_loop())

    async def stop(self):
        await self.queue.put(None)
        if self._loop_task is not None:
            await self._loop_task

    async def submit(self, query):
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((query, future))
        return await future

    async def _batcher_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            first = await self.queue.get()
            if first is None:    # sentinel
                return
            batch = [first]
            deadline = loop.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    await self._dispatch(batch)
                    return
                batch.append(item)
            await self._dispatch(batch)

    async def _dispatch(self, batch):
        queries = [q for q, _ in batch]
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        try:
            async with self.session.post(
                    self.search_url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)) as r:
                obj = await r.json()
        except Exception as e:
            err = f"(retrieval failed: {e})"
            for _, future in batch:
                if not future.done():
                    future.set_result(err)
            return
        results = obj.get("result", [])
        for (_, future), result in zip(batch, results):
            if future.done():
                continue
            try:
                future.set_result(_retriever_passages2string(result))
            except Exception as e:
                future.set_result(f"(retrieval parse failed: {e})")


class _BatchedDispatcher:
    def __init__(self, generate_sync_fn, max_batch=128, max_wait_ms=15):
        self.generate_sync_fn = generate_sync_fn
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.queue: Optional[asyncio.Queue] = None
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self):
        self.queue = asyncio.Queue()
        self._loop_task = asyncio.create_task(self._batcher_loop())

    async def stop(self):
        await self.queue.put(None)
        if self._loop_task is not None:
            await self._loop_task

    async def submit(self, prompt_ids, stop_strings):
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((tuple(stop_strings), list(prompt_ids), future))
        return await future

    async def _batcher_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            first = await self.queue.get()
            if first is None:  # sentinel
                return
            stop_key = first[0]
            batch = [first]
            deadline = loop.time() + self.max_wait_ms / 1000
            # Greedily drain the queue up to max_batch / deadline
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    # Sentinel arrived mid-batch — process current batch then return.
                    await self._dispatch(batch)
                    return
                if item[0] != stop_key:
                    await self._dispatch(batch)
                    stop_key = item[0]
                    batch = [item]
                    deadline = loop.time() + self.max_wait_ms / 1000
                else:
                    batch.append(item)
            await self._dispatch(batch)

    async def _dispatch(self, batch):
        prompt_ids_list = [item[1] for item in batch]
        stop_strings = list(batch[0][0])
        # generate_sync_fn returns list of (text, token_ids) tuples
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, self.generate_sync_fn, prompt_ids_list, stop_strings)
        except Exception as e:
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)
            return
        for (_, _, future), result in zip(batch, results):
            if not future.done():
                future.set_result(result)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class LLMGenerationBidirectionalAsyncManager:
    """Async cross-question driver for BES bidirectional search. Each State
    (and chain pad) runs as its own asyncio task behind a batched dispatcher.
    """

    _emb_cache: Dict[str, object] = {}

    def __init__(self, tokenizer, actor_rollout_wg,
                 config: GenerationBidirectionalConfig,
                 is_validation: bool = False):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
        ))

        self.search_url = config.search_url
        self.topk = config.topk
        self.backward_url = config.backward_url
        self.backward_model = config.backward_model
        self.emb_model = self._get_embedder(config.embedder_name,
                                             config.embedder_device)

    @classmethod
    def _get_embedder(cls, name, device):
        if name in cls._emb_cache:
            return cls._emb_cache[name]
        if device == "cuda":
            try:
                import torch as _t
                if not _t.cuda.is_available():
                    print("[BES] CUDA not visible to driver, falling back to CPU embedder",
                          flush=True)
                    device = "cpu"
            except Exception:
                device = "cpu"
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(name, device=device)
            print(f"[BES] embedder loaded on {device}", flush=True)
        except Exception as e:
            print(f"[BES] embedder load failed: {e}; coverage signal off",
                  flush=True)
            m = None
        cls._emb_cache[name] = m
        return m

    def _generate_sync_batched(self, prompt_ids_list, stop_strings):
        if not prompt_ids_list:
            return []
        tok = self.tokenizer
        pad_id = tok.pad_token_id
        max_len = max(len(p) for p in prompt_ids_list)
        max_len = min(max_len, self.config.max_prompt_length)
        input_ids, attention_mask = [], []
        for p in prompt_ids_list:
            p = p[-max_len:]
            n = len(p)
            pad_n = max_len - n
            input_ids.append([pad_id] * pad_n + list(p))
            attention_mask.append([0] * pad_n + [1] * n)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        position_ids_t = torch.cumsum(attention_mask_t, dim=-1) - 1
        position_ids_t.masked_fill_(attention_mask_t == 0, 0)
        proto = DataProto.from_dict({
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "position_ids": position_ids_t,
        })
        proto.meta_info = {
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
        }
        out_proto = self._generate_with_gpu_padding(proto)
        resp_ids = out_proto.batch["responses"]
        texts = self.tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
        out = []
        for i, t in enumerate(texts):
            ids = resp_ids[i].tolist()
            while ids and ids[-1] == pad_id:
                ids.pop()
            for s in stop_strings:
                if s in t:
                    cut_text = t.split(s)[0] + s
                    if cut_text != t:
                        cut_ids = self.tokenizer.encode(cut_text,
                                                       add_special_tokens=False)
                        if 0 < len(cut_ids) <= len(ids):
                            ids = ids[:len(cut_ids)]
                    t = cut_text
                    break
            out.append((t, ids))
        return out

    def _generate_with_gpu_padding(self, active_batch):
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            pad_seq = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_seq], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
        padded_output.batch = trimmed_batch
        return padded_output

    def _extract_question_text(self, prompt_ids):
        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        m = re.search(r"Question:\s*(.+?)\s*$", text, re.DOTALL)
        q = m.group(1).strip() if m else text.strip()
        q = re.sub(r"\s*(assistant|user|system)\s*$", "", q, flags=re.IGNORECASE)
        return q.strip()

    def _extract_gold_targets(self, gen_batch, idx):
        rm = gen_batch.non_tensor_batch.get("reward_model", None)
        if rm is None:
            return []
        item = rm[idx]
        gt = item.get("ground_truth", {}) if isinstance(item, dict) else {}
        if isinstance(gt, dict):
            tgt = gt.get("target", [])
        else:
            tgt = gt
        if hasattr(tgt, "tolist"):
            tgt = tgt.tolist()
        return [str(x) for x in tgt]

    # ------------------------------------------------------------------
    # Async per-state loop (BES search OR chain rollout)
    # ------------------------------------------------------------------
    async def _state_loop(self, state, dispatcher, retriever):
        K = max(1, getattr(state, "k_parallel", 1))
        if not hasattr(state, "_bes_events"):
            state._bes_events = []

        async def one_call(req):
            """Submit 1 LLM call + (if expand) retrieve. Returns
            (text, token_ids, info_map, req, t_req_start)."""
            t_req = time.time()
            text, token_ids = await dispatcher.submit(req["prompt_ids"], req["stop"])
            info_map = {}
            if req.get("type") == "expand":
                norm_for_peek = state._normalize_response(text)
                sq = _BidirState.peek_search_query(norm_for_peek)
                if sq:
                    info = await retriever.submit(sq)
                    info_map = {sq: info}
            return text, token_ids, info_map, req, t_req

        while not state.done:
            reqs = state.next_request_k(K)
            if not reqs:
                break

            # Fire K calls in parallel.
            results = await asyncio.gather(*[one_call(r) for r in reqs])

            for text, token_ids, info_map, req, t_req in results:
                # --- pre-process snapshot for event log ---
                n_nodes_before = len(state.nodes)
                n_failed_before = len(state.failed_attempts)
                n_mut_log_before = len(state.mutation_log)
                req_type = req.get("type", "expand")
                parent_id = req.get("parent_id", -1)
                parent_depth = -1; parent_score = 0.0
                if req_type in ("expand", "finalize") and 0 <= parent_id < n_nodes_before:
                    p = state.nodes[parent_id]
                    parent_depth = p.depth; parent_score = p.score
                mutation_type = req.get("mutation_type", req.get("mt", ""))
                mutation_parents = req.get("parent_ids", [])

                # --- parse status (post-normalize) ---
                norm = state._normalize_response(text)
                actions = parse_actions(norm)
                if (len(actions) == 2 and actions[0][0] == "think"
                        and actions[1][0] in ("search", "answer")
                        and actions[0][1] and actions[1][1]):
                    action_kind = actions[1][0]
                    action_content = actions[1][1]
                    parse_status = "ok"
                else:
                    action_kind = None; action_content = ""
                    parse_status = (f"bad ({len(actions)} tags, "
                                    f"first={actions[0][0] if actions else 'none'})")

                try:
                    state.process_result(text, token_ids, req, info_map=info_map)
                    proc_err = ""
                except Exception as e:
                    import traceback
                    print(f"[BES ERROR] state proc: {e}", flush=True)
                    print(traceback.format_exc()[:500], flush=True)
                    proc_err = str(e)[:200]
                    state.done = True

                # --- post-process snapshot ---
                new_nodes = state.nodes[n_nodes_before:]
                new_failed = state.failed_attempts[n_failed_before:]
                new_mut_log = state.mutation_log[n_mut_log_before:]
                event = {
                    "t": round(time.time() - t_req, 3),
                    "request_type": req_type,
                    "parent_id": int(parent_id),
                    "parent_depth": int(parent_depth),
                    "parent_score": float(parent_score),
                    "mutation_type": str(mutation_type) if mutation_type else "",
                    "mutation_parents": list(mutation_parents),
                    "raw_response": text[:1500],
                    "parse_status": parse_status,
                    "action_kind": action_kind,
                    "action_content": action_content,
                    "retrieval_info": (info_map.get(action_content, "")[:1500]
                                        if action_kind == "search" else ""),
                    "n_new_nodes": len(new_nodes),
                    "new_node_ids": [int(n.id) for n in new_nodes],
                    "new_terminals": [
                        {"id": int(n.id), "answer": n.final_answer, "score": float(n.score),
                         "depth": int(n.depth)}
                        for n in new_nodes if n.terminal and n.final_answer
                    ],
                    "new_failed_attempts": [dict(f) for f in new_failed],
                    "new_mutation_log": [dict(m) for m in new_mut_log],
                    "process_error": proc_err,
                }
                state._bes_events.append(event)
                if state.done:
                    break

    async def _chain_loop(self, cr, dispatcher, retriever):
        """Drive one _ChainRollout to completion via async dispatcher."""
        if not hasattr(cr, "_bes_events"):
            cr._bes_events = []
        while not cr.done:
            req = cr.next_request()
            if req is None:
                break
            t_req = time.time()
            text, token_ids = await dispatcher.submit(req["prompt_ids"], req["stop"])
            info_map = {}
            is_force = (req.get("type") == "chain_force_answer")
            if not is_force:
                norm_for_peek = cr._normalize_response(text)
                sq = _ChainRollout.peek_search_query(norm_for_peek)
                if sq:
                    info = await retriever.submit(sq)
                    info_map = {sq: info}

            # --- pre-process snapshot ---
            n_turns_before = len(cr.turns)
            n_actions_before = cr.n_actions
            terminal_before = cr.terminal

            # --- parse status (lenient, like _ChainRollout.process_result) ---
            norm = cr._normalize_response(text)
            actions_parsed = parse_actions(norm)
            first_act = next(((k, c, r) for (k, c, r) in actions_parsed
                              if k in ("search", "answer")), None)
            if first_act is None:
                action_kind = None; action_content = ""
                parse_status = "bad (no action)"
            else:
                action_kind = first_act[0]
                action_content = first_act[1]
                parse_status = "ok"

            try:
                cr.process_result(text, token_ids, info_map=info_map,
                                  is_force_round=is_force)
                proc_err = ""
            except Exception as e:
                proc_err = str(e)[:200]
                cr.done = True

            event = {
                "t": round(time.time() - t_req, 3),
                "request_type": ("chain_force_answer" if is_force else "chain_step"),
                "n_actions_before": int(n_actions_before),
                "n_actions_after": int(cr.n_actions),
                "raw_response": text[:1500],
                "parse_status": parse_status,
                "action_kind": action_kind,
                "action_content": action_content,
                "retrieval_info": (info_map.get(action_content, "")[:1500]
                                    if action_kind == "search" else ""),
                "n_new_turns": len(cr.turns) - n_turns_before,
                "became_terminal": (cr.terminal and not terminal_before),
                "final_answer": cr.final_answer if cr.terminal else "",
                "process_error": proc_err,
            }
            cr._bes_events.append(event)

    async def _state_with_pads(self, sidx, state, per_q_prompt_ids, gold_lists,
                                chain_rollouts, dispatcher, retriever):
        """Drive a State to done, then spawn + drive chain pads to fill the
        GRPO group up to G."""
        await self._state_loop(state, dispatcher, retriever)
        G = self.config.grpo_n
        terms = [n for n in state.nodes if n.terminal and n.final_answer]
        seen, unique = set(), []
        for nd in sorted(terms, key=lambda x: x.score, reverse=True):
            k = normalize_answer(nd.final_answer)
            if k in seen: continue
            seen.add(k); unique.append(nd)
        n_pad = max(0, G - len(unique))
        for _ in range(n_pad):
            cr = _ChainRollout(
                prompt_ids=per_q_prompt_ids[sidx],
                tokenizer=self.tokenizer,
                max_turns=self.config.max_turns,
                max_obs_length=self.config.max_obs_length,
                question_idx=sidx,
                gold_targets=gold_lists[sidx],
                think_prefix=self.config.think_prefix,
            )
            chain_rollouts[sidx].append(cr)
        if chain_rollouts[sidx]:
            await asyncio.gather(*[
                self._chain_loop(cr, dispatcher, retriever)
                for cr in chain_rollouts[sidx]
            ])

    def _pick_grpo_group(self, state, gold_targets):
        G = self.config.grpo_n
        terminals = [n for n in state.nodes if n.terminal and n.final_answer]
        seen, unique = set(), []
        for nd in sorted(terminals, key=lambda x: x.score, reverse=True):
            key = normalize_answer(nd.final_answer)
            if key in seen: continue
            seen.add(key); unique.append(nd)
        if len(unique) <= G:
            return unique
        if gold_targets:
            def _is_correct(n):
                return max((em(n.final_answer, g) for g in gold_targets),
                           default=0.0) >= 1.0
            correct = [nd for nd in unique if _is_correct(nd)]
            wrong = [nd for nd in unique if not _is_correct(nd)]
            if correct:
                return correct[:1] + wrong[:G - 1]
            return wrong[:G]
        return unique[:G]

    def _build_response_tensors_for_terminal(self, node):
        return self._build_response_tensors(node.turns)

    def _build_response_tensors_for_chain(self, cr):
        return self._build_response_tensors(cr.turns)

    def _build_response_tensors(self, turns):
        response_ids, info_mask, turns_mask = [], [], []
        agent_turn_idx = 0
        emitted_block_ptrs = set()
        for t in turns:
            if t.get("role") == "agent":
                ids = t.get("_block_token_ids")
                if ids is not None:
                    if id(ids) in emitted_block_ptrs:
                        continue
                    emitted_block_ptrs.add(id(ids))
                else:
                    ids = self.tokenizer.encode(t["raw"], add_special_tokens=False)
                if not ids:
                    continue
                agent_turn_idx += 1
                response_ids.extend(ids)
                info_mask.extend([1] * len(ids))
                turns_mask.extend([agent_turn_idx] * len(ids))
            else:
                ids = t.get("_obs_token_ids")
                if ids is None:
                    ids = self.tokenizer.encode(t["raw"], add_special_tokens=False)
                if not ids:
                    continue
                response_ids.extend(ids)
                info_mask.extend([0] * len(ids))
                turns_mask.extend([0] * len(ids))
        return response_ids, info_mask, turns_mask

    # ------------------------------------------------------------------
    # Top-level entry
    # ------------------------------------------------------------------
    def run_llm_loop_bidirectional(self, gen_batch, initial_input_ids):
        return asyncio.run(self._run_async(gen_batch, initial_input_ids))

    async def _run_async(self, gen_batch, initial_input_ids):

        bsz = gen_batch.batch["input_ids"].shape[0]
        G = self.config.grpo_n

        # --- pull questions, golds, prompt ids ---
        questions = []
        gold_lists = []
        per_q_prompt_ids: List[List[int]] = []
        for i in range(bsz):
            q = self._extract_question_text(gen_batch.batch["input_ids"][i])
            questions.append(q)
            gold_lists.append(self._extract_gold_targets(gen_batch, i))
            ids = gen_batch.batch["input_ids"][i].tolist()
            attn = gen_batch.batch["attention_mask"][i].tolist()
            cleaned = [t for t, a in zip(ids, attn) if a == 1]
            per_q_prompt_ids.append(cleaned)

        # --- async backward decompose (all questions concurrent) ---
        t_decomp = time.time()
        decomps = await _async_decompose_batch(
            self.backward_url, self.backward_model, questions)
        print(f"[BES] async decompose: {len(questions)} questions in "
              f"{time.time()-t_decomp:.1f}s", flush=True)

        # --- build _BidirState per question ---
        states: List[_BidirState] = []
        for i in range(bsz):
            q_seed = (hash(questions[i]) ^ i) & 0xFFFFFFFF
            s = _BidirState(
                question=questions[i], gold_targets=gold_lists[i],
                prompt_ids=per_q_prompt_ids[i],
                tokenizer=self.tokenizer, emb_model=self.emb_model,
                sim_threshold=self.config.sim_threshold,
                budget=self.config.budget, max_turns=self.config.max_turns,
                max_obs_length=self.config.max_obs_length,
                q_seed=q_seed,
                span_abstract=(not self.config.no_span_abstract),
                early_stop_on_correct=True,
                think_prefix=self.config.think_prefix,
                k_parallel=getattr(self.config, "k_parallel", 1))
            s.set_subqs(decomps[i])
            states.append(s)

        # --- driver: dispatcher + retriever + state tasks running concurrently ---
        dispatcher = _BatchedDispatcher(
            self._generate_sync_batched,
            max_batch=max(512, bsz * 4),
            max_wait_ms=80,
        )
        await dispatcher.start()

        chain_rollouts: List[List[_ChainRollout]] = [[] for _ in range(bsz)]
        connector = aiohttp.TCPConnector(limit=max(64, bsz * 2))
        t_drive = time.time()
        try:
            async with aiohttp.ClientSession(connector=connector) as http_session:
                retriever = _BatchedRetriever(
                    http_session, self.search_url, self.topk,
                    max_batch=max(512, bsz * 4),
                    max_wait_ms=60,    # longer than dispatcher (retrieve is faster)
                )
                await retriever.start()
                try:
                    await asyncio.gather(*[
                        self._state_with_pads(i, states[i], per_q_prompt_ids,
                                              gold_lists, chain_rollouts,
                                              dispatcher, retriever)
                        for i in range(bsz)
                    ])
                finally:
                    await retriever.stop()
        finally:
            await dispatcher.stop()
        print(f"[BES] async drive: {bsz} Q × G={G} traj in "
              f"{time.time()-t_drive:.1f}s", flush=True)

        # --- per question: pick G terminals; build per-traj tensors ---
        all_response_ids: List[List[int]] = []
        all_info_masks: List[List[int]] = []
        all_turns_masks: List[List[int]] = []
        for i, s in enumerate(states):
            picks = self._pick_grpo_group(s, gold_lists[i])
            chain_pads = chain_rollouts[i]
            for k in range(G):
                if k < len(picks):
                    rid, im, tm = self._build_response_tensors_for_terminal(picks[k])
                elif (k - len(picks)) < len(chain_pads):
                    cr = chain_pads[k - len(picks)]
                    rid, im, tm = self._build_response_tensors_for_chain(cr)
                else:
                    rid, im, tm = [], [], []
                all_response_ids.append(rid)
                all_info_masks.append(im)
                all_turns_masks.append(tm)

        # --- pad / stack into tensors ---
        max_resp = self.config.max_response_length
        pad_id = self.tokenizer.pad_token_id

        def _pad(seq, fill):
            seq = seq[:max_resp]
            return seq + [fill] * (max_resp - len(seq))

        response_ids_t = torch.tensor(
            [_pad(s, pad_id) for s in all_response_ids], dtype=torch.long)
        info_mask_t = torch.tensor(
            [_pad(s, 0) for s in all_info_masks], dtype=torch.long)
        turns_mask_t = torch.tensor(
            [_pad(s, 0) for s in all_turns_masks], dtype=torch.long)

        final_output = gen_batch.repeat(repeat_times=G, interleave=True)
        prompts_t = final_output.batch["prompts"].long()

        responses_with_info_mask = response_ids_t.clone()
        responses_with_info_mask[info_mask_t == 0] = pad_id

        final_output.batch["responses"] = response_ids_t
        final_output.batch["prompts"] = prompts_t
        final_output.batch["input_ids"] = torch.cat(
            [prompts_t, response_ids_t], dim=1)
        final_output.batch["attention_mask"] = torch.cat([
            self.tensor_fn.create_attention_mask(prompts_t),
            self.tensor_fn.create_attention_mask(response_ids_t),
        ], dim=1)
        final_output.batch["info_mask"] = torch.cat([
            self.tensor_fn.create_attention_mask(prompts_t),
            self.tensor_fn.create_attention_mask(responses_with_info_mask),
        ], dim=1)
        final_output.batch["position_ids"] = self.tensor_fn.create_position_ids(
            final_output.batch["attention_mask"])
        final_output.batch["turns_mask"] = turns_mask_t

        try:
            from verl.utils.reward_score.qa_em_format import compute_score_em
        except Exception:
            compute_score_em = None
        token_level_scores_t = torch.zeros_like(response_ids_t, dtype=torch.float32)
        if compute_score_em is not None:
            structure_format_score = 0.2
            final_format_score = 0.1
            retrieval_score = 0.0
            for traj_i, rid in enumerate(all_response_ids):
                if not rid:
                    continue
                pidx = traj_i // G
                try:
                    prompt_text = self.tokenizer.decode(
                        per_q_prompt_ids[pidx], skip_special_tokens=False)
                    resp_text = self.tokenizer.decode(rid, skip_special_tokens=False)
                except Exception:
                    continue
                solution_str = prompt_text + resp_text
                ground_truth = {"target": gold_lists[pidx]}
                try:
                    score = compute_score_em(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        structure_format_score=structure_format_score,
                        final_format_score=final_format_score,
                        retrieval_score=retrieval_score,
                        format_score=0.0, score=1.0,
                        data_source="musique")
                except Exception as e:
                    print(f"[BES score] traj {traj_i}: {e}", flush=True)
                    score = 0.0
                last_valid = min(len(rid), token_level_scores_t.shape[1]) - 1
                if last_valid >= 0:
                    token_level_scores_t[traj_i, last_valid] = float(score)
        final_output.batch["token_level_scores"] = token_level_scores_t

        # Stats for the [BES finalize] summary block right below
        _scores_per_traj = token_level_scores_t.sum(dim=-1)   # 1 nz per row
        _sc_mean = float(_scores_per_traj.mean().item()) if _scores_per_traj.numel() else 0.0
        _sc_max  = float(_scores_per_traj.max().item())  if _scores_per_traj.numel() else 0.0
        _sc_min  = float(_scores_per_traj.min().item())  if _scores_per_traj.numel() else 0.0
        _sc_pos  = int((_scores_per_traj > 0).sum().item())

        try:
            from verl.utils.reward_score.qa_em_format import (
                is_valid_sequence as _ivs,
                extract_solution as _extract,
            )
        except Exception:
            _ivs, _extract = None, None
        n_total = bsz * G
        n_bes = n_chain = n_empty = 0
        n_have_answer_tag = 0       # response itself contains <answer>...</answer>
        n_valid_format = 0          # is_valid_sequence(prompt+response) == True
        n_extract_ok = 0            # extract_solution(prompt+response) is not None
        # build summary rows aligned to all_response_ids order (= picks then chain_pads per Q)
        summary_rows = []
        idx = 0
        for i, s in enumerate(states):
            picks = self._pick_grpo_group(s, gold_lists[i])
            chain_pads = chain_rollouts[i]
            # decode prompt once per Q (stripped of pads) for the is_valid check
            try:
                prompt_text = self.tokenizer.decode(
                    per_q_prompt_ids[i], skip_special_tokens=False)
            except Exception:
                prompt_text = ""
            for k in range(G):
                rid = all_response_ids[idx]
                if not rid:
                    src = "empty"; n_empty += 1
                    decoded = ""
                elif k < len(picks):
                    src = "BES"; n_bes += 1
                    decoded = self.tokenizer.decode(rid, skip_special_tokens=False)
                else:
                    src = "chain_pad"; n_chain += 1
                    decoded = self.tokenizer.decode(rid, skip_special_tokens=False)
                has_ans = ("<answer>" in decoded) and ("</answer>" in decoded)
                if has_ans:
                    n_have_answer_tag += 1
                solution = prompt_text + decoded
                if _ivs is not None:
                    ok, _reason = _ivs(solution)
                    if ok: n_valid_format += 1
                if _extract is not None:
                    ans = _extract(solution)
                    if ans is not None: n_extract_ok += 1
                summary_rows.append((i, k, src, len(rid), has_ans, decoded))
                idx += 1
        print(f"[BES finalize] total={n_total}  BES={n_bes} chain_pad={n_chain} empty={n_empty}",
              flush=True)
        print(f"[BES finalize]   have_<answer>_in_response={n_have_answer_tag}/{n_total}  "
              f"is_valid_format={n_valid_format}/{n_total}  "
              f"extract_solution_ok={n_extract_ok}/{n_total}", flush=True)
        print(f"[BES finalize]   token_level_score: mean={_sc_mean:.4f} "
              f"max={_sc_max:.3f} min={_sc_min:.3f} positive={_sc_pos}/{n_total}",
              flush=True)
        # print 3 sample decoded responses (one BES, one chain_pad, one no-answer)
        sample_bes = next((r for r in summary_rows if r[2] == "BES"), None)
        sample_pad = next((r for r in summary_rows if r[2] == "chain_pad"), None)
        sample_no_ans = next((r for r in summary_rows if not r[4]), None)
        for label, r in [("BES", sample_bes), ("CHAIN_PAD", sample_pad), ("NO_ANSWER", sample_no_ans)]:
            if r is None: continue
            i, k, src, ln, has_ans, dec = r
            head = dec[:400].replace("\n", " ")
            tail = dec[-400:].replace("\n", " ") if len(dec) > 400 else ""
            print(f"[BES finalize] sample_{label} (Q{i} k{k} src={src} len={ln} has_ans={has_ans}):",
                  flush=True)
            print(f"  HEAD: {head}", flush=True)
            if tail and tail != head:
                print(f"  TAIL: {tail}", flush=True)

        debug_log_dir = os.environ.get("BES_SEARCH_DEBUG_DIR", "")
        if debug_log_dir:
            try:
                os.makedirs(debug_log_dir, exist_ok=True)
                rollout_log = {
                    "questions": questions,
                    "gold_targets_per_q": gold_lists,
                    "decomps_per_q": decomps,
                    "config": {
                        "budget": self.config.budget,
                        "grpo_n": G,
                        "sim_threshold": self.config.sim_threshold,
                        "max_turns": self.config.max_turns,
                        "version": "bes_async",
                    },
                    "finalize_summary": {
                        "total": n_total, "BES": n_bes, "chain_pad": n_chain, "empty": n_empty,
                        "have_answer_tag": n_have_answer_tag,
                        "is_valid_format": n_valid_format,
                        "extract_solution_ok": n_extract_ok,
                    },
                }
                per_q = []
                idx = 0
                for i, s in enumerate(states):
                    picks = self._pick_grpo_group(s, gold_lists[i])
                    pads = chain_rollouts[i]
                    terms = [n for n in s.nodes if n.terminal and n.final_answer]
                    grpo_group = []
                    for k in range(G):
                        rid = all_response_ids[idx]; im = all_info_masks[idx]
                        agent_tok = sum(im); info_tok = max(0, len(im) - agent_tok)
                        decoded = self.tokenizer.decode(rid, skip_special_tokens=False) if rid else ""
                        if not rid:
                            grpo_group.append({"rank": k, "source": "empty"})
                        elif k < len(picks):
                            n = picks[k]
                            grpo_group.append({
                                "rank": k, "source": "BES", "id": n.id,
                                "final_answer": n.final_answer,
                                "em": float(max((em(n.final_answer, g)
                                                 for g in gold_lists[i]), default=0.0)),
                                "response_ids_len": len(rid),
                                "agent_token_count": int(agent_tok),
                                "info_token_count": int(info_tok),
                                "response_decoded": decoded,
                            })
                        else:
                            cr = pads[k - len(picks)]
                            grpo_group.append({
                                "rank": k, "source": "chain_pad", "id": None,
                                "final_answer": cr.final_answer, "em": cr.score,
                                "n_actions": cr.n_actions, "terminal": cr.terminal,
                                "response_ids_len": len(rid),
                                "agent_token_count": int(agent_tok),
                                "info_token_count": int(info_tok),
                                "response_decoded": decoded,
                            })
                        idx += 1

                    def _node_record(n):
                        searches = [t["content"] for t in n.turns
                                    if t.get("role") == "agent" and t.get("kind") == "search"]
                        last_obs = ""
                        for t in reversed(n.turns):
                            if t.get("role") == "obs":
                                last_obs = t.get("content", ""); break
                        em_v = (float(max((em(n.final_answer, g)
                                           for g in gold_lists[i]), default=0.0))
                                if n.terminal and n.final_answer else 0.0)
                        return {
                            "id": int(n.id),
                            "parent_id": int(n.parent_id) if n.parent_id is not None else -1,
                            "parent_ids": [int(x) for x in n.parent_ids],
                            "depth": int(n.depth),
                            "score": float(n.score),
                            "operation": n.operation,
                            "terminal": bool(n.terminal),
                            "final_answer": n.final_answer or "",
                            "em": em_v,
                            "verify_score": float(getattr(n, "verify_score", 0.0)),
                            "subq_max_sim": list(getattr(n, "subq_max_sim", []) or []),
                            "search_queries": searches,
                            "last_retrieval": last_obs,
                            "trajectory_text": "".join(t["raw"] for t in n.turns),
                        }

                    per_q.append({
                        "question": questions[i],
                        "gold_targets": gold_lists[i],
                        "decomp": s.llm_subqs,
                        "n_total_nodes": len(s.nodes),
                        "n_terminals": len(terms),
                        "n_chain_pads": len(pads),
                        # Full tree
                        "all_nodes": [_node_record(n) for n in s.nodes],
                        # Terminal summary (faster than scanning all_nodes)
                        "all_terminals": [{
                            "id": int(n.id), "score": float(n.score),
                            "final_answer": n.final_answer, "depth": int(n.depth),
                            "em": float(max((em(n.final_answer, g)
                                             for g in gold_lists[i]),
                                            default=0.0)),
                            "trajectory_text": "".join(t["raw"] for t in n.turns),
                        } for n in terms],
                        # Chain pad full state + their per-step driver events
                        "chain_pads": [{
                            "n_actions": int(cr.n_actions),
                            "terminal": bool(cr.terminal),
                            "final_answer": cr.final_answer,
                            "em": float(cr.score),
                            "trajectory_text": "".join(t["raw"] for t in cr.turns),
                            "bes_events": list(getattr(cr, "_bes_events", [])),
                        } for cr in pads],
                        # State.mutation_log: every mutation attempt + result
                        "mutation_log": list(s.mutation_log),
                        # State.failed_attempts: failed parse / rejected expansions
                        "failed_attempts": [
                            {**f, "raw_response": str(f.get("raw_response", ""))[:1500]}
                            for f in s.failed_attempts
                        ],
                        # Driver-level event timeline (one entry per dispatcher round)
                        "bes_events": list(getattr(s, "_bes_events", [])),
                        # GRPO selection (G slots: BES picks + chain pads)
                        "selected_for_grpo": grpo_group,
                    })
                rollout_log["per_question"] = per_q

                rounds = []
                for sidx, st in enumerate(states):
                    for k, ev in enumerate(getattr(st, "_bes_events", []) or []):
                        while len(rounds) <= k:
                            rounds.append({"round": k, "entries": []})
                        entry = dict(ev)
                        entry["state_idx"] = sidx
                        rounds[k]["entries"].append(entry)
                    for cpi, cr in enumerate(chain_rollouts[sidx]):
                        for k, ev in enumerate(getattr(cr, "_bes_events", []) or []):
                            while len(rounds) <= k:
                                rounds.append({"round": k, "entries": []})
                            entry = dict(ev)
                            entry["state_idx"] = sidx
                            entry["parent_id"] = -1
                            entry["chain_pad_idx"] = cpi
                            rounds[k]["entries"].append(entry)
                rollout_log["rounds"] = rounds

                ts = int(time.time())
                log_path = os.path.join(debug_log_dir, f"rollout_log_{ts}.json")
                with open(log_path, "w") as f:
                    json.dump(rollout_log, f, ensure_ascii=False, indent=2,
                              default=lambda o: float(o) if hasattr(o, "item") else str(o))
                print(f"[BES rollout log] wrote {log_path}", flush=True)
            except Exception as e:
                print(f"[BES rollout log] write failed: {e}", flush=True)


        turns_stats = []
        active_mask = []
        valid_search_stats = []
        valid_action_stats = []
        for _r in summary_rows:
            _i, _k, _src, _ln, _has_ans, _dec = _r
            _n_search = _dec.count('<search>')
            _n_actions = _n_search + (1 if _has_ans else 0)
            turns_stats.append(max(1, _n_actions))
            active_mask.append(0 if _has_ans else 1)  # 0 = finished
            valid_search_stats.append(_n_search)
            valid_action_stats.append(_n_actions)
        final_output.meta_info['turns_stats'] = turns_stats
        final_output.meta_info['active_mask'] = active_mask
        final_output.meta_info['valid_action_stats'] = valid_action_stats
        final_output.meta_info['valid_search_stats'] = valid_search_stats

        return final_output
