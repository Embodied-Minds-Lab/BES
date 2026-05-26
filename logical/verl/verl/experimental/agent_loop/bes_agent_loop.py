import json
import logging
import math
import os
import random
import re
import sys
import itertools
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__name__)

GRPO_N = 8  # fixed group size returned per prompt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_last_json_obj(text):
    results = []
    depth = 0; start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    c = text[start:i + 1].replace("'", '"')
                    c = re.sub(r',\s*}', '}', c)
                    obj = json.loads(c)
                    if isinstance(obj, dict): results.append(obj)
                except: pass
                start = -1
    return results[-1] if results else None


def _kk_compute_score(response_text, ground_truth_json):
    try:
        pred = _extract_last_json_obj(response_text)
        if pred is None: return 0.0
        gt = json.loads(ground_truth_json)
        p = {k.strip().lower(): int(v) for k, v in pred.items()}
        g = {k.strip().lower(): int(v) for k, v in gt.items()}
        return 1.0 if p == g else 0.0
    except: return 0.0


# ---------------------------------------------------------------------------
# Stepwise verification (inline, no external dependency)
# ---------------------------------------------------------------------------

def _test_satisfiability(statement, assignments):
    if statement[0] == 'telling-truth': return assignments[statement[1]]
    if statement[0] == 'lying': return not assignments[statement[1]]
    if statement[0] == 'not': return not _test_satisfiability(statement[1], assignments)
    if statement[0] == 'and':
        return all(_test_satisfiability(statement[i], assignments) for i in range(1, len(statement)))
    if statement[0] == 'or':
        return any(_test_satisfiability(statement[i], assignments) for i in range(1, len(statement)))
    if statement[0] == '->':
        return (not _test_satisfiability(statement[1], assignments)) or _test_satisfiability(statement[2], assignments)
    if statement[0] == '<=>':
        return _test_satisfiability(statement[1], assignments) == _test_satisfiability(statement[2], assignments)
    raise KeyError(f'Unknown: {statement}')


def _can_be_falsified(stmts, assignments):
    n_people = len(stmts)
    remap = [i for i, x in enumerate(assignments) if x is None]
    for p_idx in range(n_people):
        if assignments[p_idx] is None: continue
        p_statement = stmts[p_idx]
        if not assignments[p_idx]: p_statement = ('not', p_statement)
        has_solution = False
        for proposal in itertools.product([True, False], repeat=len(remap)):
            new_a = list(assignments)
            for i, x in zip(remap, proposal): new_a[i] = x
            if _test_satisfiability(p_statement, new_a):
                has_solution = True; break
        if not has_solution: return False
    return True


def _verify_response(names, statements, solution, text):
    n_people = len(names)
    n_correct = n_wrong = 0
    ok_pats = [r"No contradiction", r"consistent", r"No issues", r"holds without", r"does not contradict"]
    contra_pats = [
        r"(\w+)\s+cannot\s+be\s+(?:a\s+)?(knight|knave)",
        r"(\w+)\s+being\s+(?:a\s+)?(knight|knave)\s+is\s+impossible",
        r"If\s+(\w+)\s+were\s+(?:a\s+)?(knight|knave)",
    ]
    name_map = {n.lower(): i for i, n in enumerate(names)}

    for para in text.split("\n\n"):
        para = para.strip()
        if not para or para.startswith("Let") or para.startswith("I ") or para.startswith("To "):
            continue
        if "### Final Answer" in para: continue

        is_ok = any(re.search(p, para, re.IGNORECASE) for p in ok_pats)
        is_contra = any(re.search(p, para, re.IGNORECASE) for p in contra_pats)

        if is_ok:
            m = re.match(r"Assume\s+(.+?)\.", para)
            if not m: continue
            parts = re.findall(r"(\w+)\s+is\s+(?:a\s+)?(knight|knave)", m.group(1), re.IGNORECASE)
            test_a = [None] * n_people
            for nm, role in parts:
                if nm.lower() in name_map:
                    test_a[name_map[nm.lower()]] = (role.lower() == "knight")
            if _can_be_falsified(statements, test_a): n_correct += 1
            else: n_wrong += 1

        elif is_contra:
            m_assume = re.match(r"Assume\s+(.+?)\.", para)
            base = [None] * n_people
            if m_assume:
                for nm, role in re.findall(r"(\w+)\s+is\s+(?:a\s+)?(knight|knave)", m_assume.group(1), re.IGNORECASE):
                    if nm.lower() in name_map:
                        base[name_map[nm.lower()]] = (role.lower() == "knight")
            for p in contra_pats:
                mc = re.search(p, para, re.IGNORECASE)
                if mc:
                    nm, role = mc.group(1), mc.group(2)
                    if nm.lower() in name_map:
                        trial = list(base)
                        trial[name_map[nm.lower()]] = (role.lower() == "knight")
                        if not _can_be_falsified(statements, trial): n_correct += 1
                        else: n_wrong += 1
                    break

    return n_correct, n_wrong


# ---------------------------------------------------------------------------
# Goal tree generation + verification
# ---------------------------------------------------------------------------

def _generate_goal_tree(names, solution):
    tree = {"id": "L1", "level": 1, "description": "Solve the puzzle completely",
            "verify_type": "final_answer", "children": [], "satisfied": False}
    for name, sol in zip(names, solution):
        role = "knight" if sol else "knave"
        opposite = "knave" if sol else "knight"
        l2 = {"id": f"L2_{name}", "level": 2, "person": name, "identity": role,
               "description": f"Determine {name} is a {role}",
               "verify_type": "identity_check", "children": [], "satisfied": False}
        l3s = [
            {"id": f"L3_{name}_try_{opposite}", "level": 3,
             "description": f"Assume {name} is a {opposite}",
             "verify_type": "assume_present",
             "verify_params": {"person": name, "role": opposite}, "satisfied": False},
            {"id": f"L3_{name}_contra_{opposite}", "level": 3,
             "description": f"Show {name} cannot be a {opposite}",
             "verify_type": "contradiction_found",
             "verify_params": {"person": name, "role": opposite}, "satisfied": False},
            {"id": f"L3_{name}_try_{role}", "level": 3,
             "description": f"Assume {name} is a {role}",
             "verify_type": "assume_present",
             "verify_params": {"person": name, "role": role}, "satisfied": False},
            {"id": f"L3_{name}_confirm_{role}", "level": 3,
             "description": f"Confirm {name} as {role} is consistent",
             "verify_type": "no_contradiction",
             "verify_params": {"person": name, "role": role}, "satisfied": False},
        ]
        for other_name, other_sol in zip(names, solution):
            if other_name == name: continue
            other_role = "knight" if other_sol else "knave"
            l3s.append({"id": f"L3_{name}+{other_name}", "level": 3,
                "description": f"Assume {name}={role} and {other_name}={other_role}",
                "verify_type": "multi_assume_present",
                "verify_params": {"persons": {name: role, other_name: other_role}}, "satisfied": False})
        l2["children"] = l3s
        tree["children"].append(l2)
    return tree


def _verify_against_tree(text, tree, ground_truth, cache=None):
    c = cache if cache is not None else {}
    if tree["id"] in c:
        tree["satisfied"] = c[tree["id"]]
    elif _kk_compute_score(text, ground_truth) >= 1.0:
        tree["satisfied"] = True
    for l2 in tree["children"]:
        if l2["id"] in c:
            l2["satisfied"] = c[l2["id"]]
        else:
            person, identity = l2["person"], l2["identity"]
            pat = re.compile(
                rf'Assume\s+.*?{re.escape(person)}\s+is\s+(?:a\s+)?{re.escape(identity)}.*?'
                rf'(?:No contradiction|No issues|consistent|holds without)',
                re.IGNORECASE | re.DOTALL)
            if pat.search(text): l2["satisfied"] = True
            if "### Final Answer" in text and re.search(
                rf'{re.escape(person)}\s+is\s+(?:a\s+)?{re.escape(identity)}', text, re.IGNORECASE):
                l2["satisfied"] = True
        for l3 in l2["children"]:
            if l3["id"] in c:
                l3["satisfied"] = c[l3["id"]]
                continue
            vt = l3["verify_type"]
            vp = l3.get("verify_params", {})
            if vt == "assume_present":
                if re.search(rf'Assume\s+.*?{re.escape(vp["person"])}\s+is\s+(?:a\s+)?{re.escape(vp["role"])}',
                             text, re.IGNORECASE):
                    l3["satisfied"] = True
            elif vt == "contradiction_found":
                pats = [rf'{re.escape(vp["person"])}\s+cannot\s+be\s+(?:a\s+)?{re.escape(vp["role"])}',
                        rf'{re.escape(vp["person"])}\s+being\s+(?:a\s+)?{re.escape(vp["role"])}\s+is\s+impossible',
                        rf'If\s+{re.escape(vp["person"])}\s+were\s+(?:a\s+)?{re.escape(vp["role"])}.*?contradict']
                if any(re.search(p, text, re.IGNORECASE) for p in pats):
                    l3["satisfied"] = True
            elif vt == "no_contradiction":
                pat2 = re.compile(
                    rf'Assume\s+.*?{re.escape(vp["person"])}\s+is\s+(?:a\s+)?{re.escape(vp["role"])}.*?'
                    rf'(?:No contradiction|No issues|consistent|holds without)',
                    re.IGNORECASE | re.DOTALL)
                if pat2.search(text): l3["satisfied"] = True
            elif vt == "multi_assume_present":
                if all(re.search(rf'Assume\s+.*?{re.escape(p)}\s+is\s+(?:a\s+)?{re.escape(r)}',
                                 text, re.IGNORECASE) for p, r in vp["persons"].items()):
                    l3["satisfied"] = True


def _recursive_score(node):
    if node["satisfied"]: return 1.0
    children = node.get("children", [])
    if not children: return 0.0
    return 0.7 * sum(_recursive_score(c) for c in children) / len(children)


def _flatten_satisfaction(tree):
    """Walk a verified tree and return flat dict goal_id -> bool."""
    sat = {tree["id"]: tree["satisfied"]}
    for l2 in tree.get("children", []):
        sat[l2["id"]] = l2["satisfied"]
        for l3 in l2.get("children", []):
            sat[l3["id"]] = l3["satisfied"]
    return sat


# ---------------------------------------------------------------------------
# Search Node
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    node_id: int
    paragraphs: list
    score: float = 0.0
    is_terminal: bool = False
    is_complete: bool = False
    parent_id: int = None
    depth: int = 0
    operation: str = ""
    n_correct: int = 0
    n_wrong: int = 0
    goal_satisfaction: dict = field(default_factory=dict)
    _sat_cache: dict = field(default_factory=dict)  # goal_id -> bool, pure fn of text
    _vr_done: bool = False  # n_correct/n_wrong already computed

    @property
    def text(self): return "".join(self.paragraphs)
    @property
    def answer(self): return _extract_last_json_obj(self.text)


# ---------------------------------------------------------------------------
# Decompose prompts
# ---------------------------------------------------------------------------

_L2_ORDER_PROMPT = """Puzzle: {quiz}
Answer: {gt_text}

Think about which person's identity is easiest to determine first. Consider:
- Whose statement most directly constrains others?
- Which person's identity can be determined independently?
- What is the logical dependency chain?

After thinking, output your ranking on the LAST line in this exact format:
ORDER: Name1, Name2, Name3"""

_L3_PICK_PROMPT = """Puzzle: {quiz}
Answer: {gt_text}

To prove that {person} is a {role}, here are possible reasoning steps:
{steps_list}

Pick the {k} most important steps and order them by priority.

After thinking, output your picks on the LAST line in this exact format:
PICKS: 1, 3, 5"""


def _parse_l2_order(response, names):
    m = re.search(r'ORDER:\s*(.+)', response, re.IGNORECASE)
    text = m.group(1) if m else response.strip().split('\n')[-1]
    parsed = []
    seen = set()
    for name in names:
        pos = text.lower().find(name.lower())
        if pos >= 0 and name not in seen:
            parsed.append((pos, name)); seen.add(name)
    parsed.sort(key=lambda x: x[0])
    result = [n for _, n in parsed]
    for n in names:
        if n not in seen: result.append(n)
    return result


def _parse_l3_picks(response, n_candidates, k=10):
    target = min(k, n_candidates)
    m = re.search(r'PICKS:\s*(.+)', response, re.IGNORECASE)
    text = m.group(1) if m else response.strip().split('\n')[-1]
    picked = []
    seen = set()
    for num in re.findall(r'\d+', text):
        idx = int(num) - 1
        if 0 <= idx < n_candidates and idx not in seen:
            picked.append(idx); seen.add(idx)
            if len(picked) >= target: break
    if len(picked) < target:
        rem = [i for i in range(n_candidates) if i not in seen]
        random.shuffle(rem)
        for i in rem:
            if len(picked) >= target: break
            picked.append(i)
    return picked


# ---------------------------------------------------------------------------
# Bidirectional Search State
# ---------------------------------------------------------------------------

ANSWER_SUFFIX = (
    'Please think step by step, by considering whether each person is lying '
    'and if that leads to contradiction. At the end, output your final answer '
    'as a JSON object where keys are names and values are 1 for knight or 0 '
    'for knave. For example: {"Alice": 1, "Bob": 0}'
)


class _BidirState:
    def __init__(self, problem, names, statements, solution, gt_json,
                 tokenizer, budget=200, decompose_interval=5):
        self.problem = problem
        self.names = names
        self.statements = statements
        self.solution = solution
        self.gt_json = gt_json
        self.tokenizer = tokenizer
        self.budget = budget
        self.decompose_interval = decompose_interval

        self.expand_params = {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95,
                              "stop": ["\n\n"], "include_stop_str_in_output": True}
        self.finish_params = {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95}

        self.goal_tree = _generate_goal_tree(names, solution)
        self.l3_candidates = {}
        for pidx, l2 in enumerate(self.goal_tree["children"]):
            self.l3_candidates[pidx] = l2["children"]
            l2["children"] = []
        self.l2_order = list(range(len(names)))
        self.l2_expand_idx = 0
        self.decompose_done = False

        self.nodes = []
        self.step = 0
        self.done = False
        self.result_text = ""
        self._nid = 0
        self._exp_rem = 0
        self._exp_pid = None
        self._exp_col = []
        self._fin_q = []

        self._add(_Node(node_id=0, paragraphs=[""], operation="root"))

    def _add(self, n):
        self._nid += 1; n.node_id = self._nid
        self.nodes.append(n); return n

    def _score(self, node):
        tc = json.loads(json.dumps(self.goal_tree))
        _verify_against_tree(node.text, tc, self.gt_json, cache=node._sat_cache)
        node._sat_cache.update(_flatten_satisfaction(tc))
        node.score = _recursive_score(tc)
        node.goal_satisfaction = _flatten_satisfaction(tc)
        if not node._vr_done:
            nc, nw = _verify_response(self.names, self.statements, self.solution, node.text)
            node.n_correct = nc; node.n_wrong = nw
            node._vr_done = True
        if node.n_wrong > 0: node.score = min(node.score, 0.01)
        if node.is_complete and node.answer:
            if _kk_compute_score(node.text, self.gt_json) >= 1.0:
                node.score = 100.0; self.result_text = node.text; self.done = True

    def _prompt(self, paras):
        msgs = [{"role": "user", "content": self.problem + "\n\n" + ANSWER_SUFFIX}]
        p = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return p + "".join(paras)

    def _finish_prompt(self, node):
        msgs = [{"role": "user", "content": self.problem + "\n\n" + ANSWER_SUFFIX}]
        p = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t = node.text
        if not t.rstrip().endswith("### Final Answer"): t += "\n\n### Final Answer\n\n"
        return p + t

    # Selection
    def _select(self, pool, temp):
        if len(pool) == 1: return pool[0]
        cc = {}
        for n in self.nodes:
            if n.parent_id: cc[n.parent_id] = cc.get(n.parent_id, 0) + 1
        scores = [n.score * 10 + (1.0 if cc.get(n.node_id, 0) == 0 else 0.0) for n in pool]
        if temp <= 0: return pool[max(range(len(pool)), key=lambda i: scores[i])]
        sc = [s / max(temp, 0.01) for s in scores]
        c = max(sc)
        w = [math.exp(s - c) for s in sc]
        return random.choices(pool, weights=w, k=1)[0]

    def _boltz(self, nodes, temp):
        if len(nodes) == 1 or temp <= 0: return max(nodes, key=lambda n: n.score)
        sc = [n.score / temp for n in nodes]
        c = max(sc); w = [math.exp(s - c) for s in sc]
        return random.choices(nodes, weights=w, k=1)[0]

    def _joint_recursive_score(self, a, b, goal):
        sat_a = a.goal_satisfaction.get(goal["id"], False)
        sat_b = b.goal_satisfaction.get(goal["id"], False)
        v = max(float(bool(sat_a)), float(bool(sat_b)))
        if v >= 1.0: return 1.0
        children = goal.get("children", [])
        if not children: return v
        return sum(self._joint_recursive_score(a, b, c) for c in children) / len(children)

    def _select_pair_by_coverage(self, pool, temp):
        if len(pool) < 2: return None
        pairs, scores = [], []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                u = self._joint_recursive_score(a, b, self.goal_tree)
                pairs.append((a, b)); scores.append(u * 10)
        if temp <= 0:
            idx = max(range(len(scores)), key=lambda i: scores[i])
            return pairs[idx]
        max_s = max(scores)
        weights = [math.exp((s - max_s) / max(temp, 0.01)) for s in scores]
        total = sum(weights)
        if total <= 0: return random.choice(pairs)
        r = random.random() * total
        cum = 0.0
        for pair, w in zip(pairs, weights):
            cum += w
            if r <= cum: return pair
        return pairs[-1]

    # Mutations
    @staticmethod
    def _pfx(a, b):
        s = 0
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]: s += 1
            else: break
        return s

    @staticmethod
    def _vp(a, b):
        s = _BidirState._pfx(a.paragraphs, b.paragraphs)
        return bool(a.paragraphs[s:]) and bool(b.paragraphs[s:])

    def _merge(self, a, b):
        if not self._vp(a, b): return None
        s = self._pfx(a.paragraphs, b.paragraphs)
        m = a.paragraphs[:s] + a.paragraphs[s:] + b.paragraphs[s:]
        if not m: return None
        f = "".join(m)
        return _Node(0, m, parent_id=a.node_id, is_terminal="### Final Answer" in f,
                     is_complete=_extract_last_json_obj(f) is not None,
                     depth=max(a.depth, b.depth)+1, operation="combine")

    def _delete(self, nd):
        p = nd.paragraphs[:]
        if len(p) <= 2: return None
        del p[random.randint(1, len(p)-2)]
        f = "".join(p)
        return _Node(0, p, parent_id=nd.node_id, is_terminal="### Final Answer" in f,
                     is_complete=_extract_last_json_obj(f) is not None,
                     depth=nd.depth+1, operation="deletion")

    def _translocate(self, a, b):
        if not self._vp(a, b): return None
        s = self._pfx(a.paragraphs, b.paragraphs)
        sa, sb = a.paragraphs[s:], b.paragraphs[s:]
        if not sa or not sb: return None
        ai, bi = random.randint(0, len(sa)-1), random.randint(0, len(sb)-1)
        p = a.paragraphs[:s] + sa[:ai] + [sb[bi]] + sa[ai+1:]
        f = "".join(p)
        return _Node(0, p, parent_id=a.node_id, is_terminal="### Final Answer" in f,
                     is_complete=_extract_last_json_obj(f) is not None,
                     depth=max(a.depth, b.depth)+1, operation="translocation")

    def _crossover(self, a, b):
        if not self._vp(a, b): return None
        s = self._pfx(a.paragraphs, b.paragraphs)
        sa, sb = a.paragraphs[s:], b.paragraphs[s:]
        if not sa or not sb: return None
        m = random.randint(0, len(sa)-1); nn = random.randint(1, len(sb))
        p = a.paragraphs[:s] + sa[:m] + sb[-nn:]
        if not p: return None
        f = "".join(p)
        return _Node(0, p, parent_id=a.node_id, is_terminal="### Final Answer" in f,
                     is_complete=_extract_last_json_obj(f) is not None,
                     depth=max(a.depth, b.depth)+1, operation="crossing_over")

    def _reg(self, child, action):
        child = self._add(child)
        if child.is_terminal and not child.is_complete: self._fin_q.append(child.node_id)
        self._score(child)

    # Request generation
    def next_request(self):
        if self.done: return None

        if self._exp_rem > 0 and self._exp_pid is not None:
            parent = next((n for n in self.nodes if n.node_id == self._exp_pid), None)
            if parent:
                virt = parent.paragraphs + self._exp_col
                return {"type": "expand", "prompt": self._prompt(virt), "params": self.expand_params}
            self._exp_rem = 0; self._exp_pid = None; self._exp_col = []

        if self._fin_q:
            nid = self._fin_q.pop(0)
            nd = next((n for n in self.nodes if n.node_id == nid), None)
            if nd and nd.is_terminal and not nd.is_complete:
                return {"type": "finish", "prompt": self._finish_prompt(nd),
                        "params": self.finish_params, "node_id": nid}

        if not self.decompose_done:
            gt_text = ", ".join(f'{n} is a {"knight" if s else "knave"}'
                                for n, s in zip(self.names, self.solution))
            self.decompose_done = True
            return {"type": "decompose_l2",
                    "content": _L2_ORDER_PROMPT.format(quiz=self.problem, gt_text=gt_text)}

        if self.step > 0 and self.step % self.decompose_interval == 0:
            n = len(self.l2_order)
            found = None
            for tries in range(n):
                cand_idx = (self.l2_expand_idx + tries) % n
                p = self.l2_order[cand_idx]
                picked_ids = {c["id"] for c in self.goal_tree["children"][p]["children"]}
                rem = [c for c in self.l3_candidates[p] if c["id"] not in picked_ids]
                if rem:
                    found = (p, rem, cand_idx); break
            if found is not None:
                pidx, remaining, cand_idx = found
                self.l2_expand_idx = cand_idx + 1
                person = self.names[pidx]
                role = "knight" if self.solution[pidx] else "knave"
                steps = "\n".join(f"  {i+1}. {c['description']}" for i, c in enumerate(remaining))
                gt_text = ", ".join(f'{n} is a {"knight" if s else "knave"}'
                                    for n, s in zip(self.names, self.solution))
                k_pick = min(10, len(remaining))
                return {"type": "decompose_l3",
                        "content": _L3_PICK_PROMPT.format(quiz=self.problem, gt_text=gt_text,
                                                           person=person, role=role,
                                                           steps_list=steps, k=k_pick),
                        "person_idx": pidx,
                        "candidates": remaining}

        if self.step >= self.budget - 1:
            self._finish(); return None

        cands = [n for n in self.nodes if not n.is_terminal]
        if not cands: cands = [self.nodes[0]]

        t = self.step / max(self.budget - 2, 1)
        temp = 2.0 + (1.0 - 2.0) * t
        scored = [n for n in cands if n.paragraphs]
        mp = [n for n in scored if len(n.paragraphs) > 2]

        roll = random.random()
        if roll < 0.10 and len(scored) >= 2:
            pair = self._select_pair_by_coverage(scored, temp)
            if pair is not None:
                a, b = pair
                c = self._merge(a, b)
                if c: self._reg(c, "combine"); return self.next_request()
        elif roll < 0.15 and mp:
            c = self._delete(self._boltz(mp, temp))
            if c: self._reg(c, "deletion"); return self.next_request()
        elif roll < 0.225 and len(mp) >= 2:
            pair = self._select_pair_by_coverage(mp, temp)
            if pair is not None:
                a, b = pair
                c = self._translocate(a, b)
                if c: self._reg(c, "translocation"); return self.next_request()
        elif roll < 0.30 and len(mp) >= 2:
            pair = self._select_pair_by_coverage(mp, temp)
            if pair is not None:
                a, b = pair
                c = self._crossover(a, b)
                if c: self._reg(c, "crossing_over"); return self.next_request()

        sel = self._select(cands, temp)
        k = random.randint(1, 5)
        self._exp_rem = k - 1; self._exp_pid = sel.node_id; self._exp_col = []
        return {"type": "expand", "prompt": self._prompt(sel.paragraphs), "params": self.expand_params}

    def process_result(self, text, req):
        rt = req["type"]
        if rt == "decompose_l2":
            self.l2_order = [self.names.index(n) for n in _parse_l2_order(text, self.names)]
            self.step += 1; return
        if rt == "decompose_l3":
            pidx = req["person_idx"]
            candidates = req["candidates"]
            picks = _parse_l3_picks(text, len(candidates), k=10)
            for i in picks:
                self.goal_tree["children"][pidx]["children"].append(candidates[i])
            if picks:
                for nd in self.nodes:
                    self._score(nd)
            self.step += 1; return
        if rt == "finish":
            t = text.strip()
            if not t: return
            parent = next((n for n in self.nodes if n.node_id == req["node_id"]), None)
            if not parent: return
            p = parent.paragraphs + [t]
            f = "".join(p)
            child = _Node(0, p, parent_id=req["node_id"], is_terminal=True,
                          is_complete=_extract_last_json_obj(f) is not None,
                          depth=parent.depth+1, operation="expand")
            self._reg(child, "expand"); self.step += 1; return

        # expand
        if text and text.strip(): self._exp_col.append(text)
        is_term = "### Final Answer" in (text or "")
        self._exp_rem = max(0, self._exp_rem - 1)
        if self._exp_rem > 0 and not is_term:
            self.step += 1; return
        col = self._exp_col; self._exp_col = []; self._exp_rem = 0
        pid = self._exp_pid; self._exp_pid = None
        if not col: self.step += 1; return
        parent = next((n for n in self.nodes if n.node_id == pid), None)
        if not parent: self.step += 1; return
        p = parent.paragraphs + col; f = "".join(p)
        child = _Node(0, p, parent_id=pid, is_terminal="### Final Answer" in f,
                      is_complete=_extract_last_json_obj(f) is not None,
                      depth=parent.depth+1, operation="expand")
        self._reg(child, "expand"); self.step += 1

    def _finish(self):
        comp = [n for n in self.nodes if n.is_complete]
        best = max(comp, key=lambda n: n.score) if comp else (
            max(self.nodes, key=lambda n: n.score) if self.nodes else None)
        self.result_text = best.text if best else ""
        self.done = True


@register("bes")
class BESAgentLoop(AgentLoopBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        scfg = self.rollout_config.get("search", {})
        self.search_budget = scfg.get("budget", 200)
        self.decompose_interval = scfg.get("decompose_interval", 5)
        self.backward_url = scfg.get("backward_url", "http://localhost:8235/v1")
        self.backward_model = scfg.get("backward_model", "google/gemma-3-1b-it")
        self._backward_client = None

    def _get_backward(self):
        if self._backward_client is None:
            import httpx
            from openai import AsyncOpenAI
            self._backward_client = AsyncOpenAI(
                base_url=self.backward_url, api_key="dummy",
                timeout=httpx.Timeout(timeout=300, connect=10.0))
        return self._backward_client

    async def _backward_gen(self, content):
        client = self._get_backward()
        r = await client.chat.completions.create(
            model=self.backward_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=400, temperature=0.3, top_p=0.95)
        return r.choices[0].message.content

    async def _single_gen(self, prompt_ids, sampling_params):
        """Run one forward-only generation and return AgentLoopOutput."""
        pad_sp = {
            "max_tokens": sampling_params.get("max_tokens", self.response_length),
            "temperature": sampling_params.get("temperature", 0.6),
            "top_p": sampling_params.get("top_p", 0.95),
        }
        out = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=pad_sp)
        resp_ids = out.token_ids[:self.response_length]
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp_ids,
            response_mask=[1] * len(resp_ids),
            response_logprobs=None, routed_experts=None, multi_modal_data={},
            num_turns=1, metrics={}, extra_fields={"turn_scores": [], "tool_rewards": []})

    async def run(self, sampling_params: dict, **kwargs) -> list:
        messages = list(kwargs["raw_prompt"])
        problem_text = ""
        for msg in messages:
            if msg["role"] == "user": problem_text = msg["content"]; break

        reward_info = kwargs.get("reward_model", {})
        gt_json = reward_info.get("ground_truth", "") if isinstance(reward_info, dict) else ""

        extra = kwargs.get("extra_info", {})
        if isinstance(extra, str): extra = json.loads(extra)
        statements_raw = extra.get("statements")
        names = extra.get("names")
        solution = extra.get("solution")

        prompt_ids = await self.apply_chat_template(messages)

        if statements_raw is None or names is None or solution is None:
            logger.warning("Missing statements/names/solution, returning %d single-gen samples", GRPO_N)
            results = []
            for _ in range(GRPO_N):
                results.append(await self._single_gen(prompt_ids, sampling_params))
            return results

        def to_tuple(x):
            if isinstance(x, list): return tuple(to_tuple(i) for i in x)
            return x
        statements = to_tuple(statements_raw)

        state = _BidirState(
            problem=problem_text, names=names, statements=statements,
            solution=solution, gt_json=gt_json, tokenizer=self.tokenizer,
            budget=self.search_budget, decompose_interval=self.decompose_interval)

        while not state.done:
            req = state.next_request()
            if req is None: continue

            if req["type"].startswith("decompose"):
                text = await self._backward_gen(req["content"])
            else:
                sp = dict(sampling_params)
                params = req.get("params", {})
                sp["temperature"] = params.get("temperature", sp.get("temperature", 0.6))
                sp["top_p"] = params.get("top_p", sp.get("top_p", 0.95))
                if params.get("stop"): sp["stop"] = params["stop"]
                if params.get("include_stop_str_in_output"):
                    sp["include_stop_str_in_output"] = True
                if params.get("max_tokens"): sp["max_tokens"] = params["max_tokens"]

                req_prompt_ids = self.tokenizer.encode(req["prompt"])
                try:
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=req_prompt_ids, sampling_params=sp)
                    text = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
                except ValueError as e:
                    logger.warning(
                        "generate skipped type=%s prompt_len=%d: %s",
                        req.get("type"), len(req_prompt_ids), e)
                    text = ""

            state.process_result(text, req)

        # Collect unique complete responses (deduplicated by text)
        comp = [nd for nd in state.nodes if nd.is_complete and nd.text.strip()]
        if not comp:
            comp = [max(state.nodes, key=lambda nd: nd.score)] if state.nodes else []

        unique = []
        seen = set()
        for nd in sorted(comp, key=lambda nd: nd.score, reverse=True):
            if nd.text not in seen:
                seen.add(nd.text)
                unique.append(nd)

        # Truncate: keep 1 correct + (GRPO_N - 1) wrong if > GRPO_N
        if len(unique) > GRPO_N:
            correct = [nd for nd in unique if _kk_compute_score(nd.text, gt_json) >= 1.0]
            wrong = [nd for nd in unique if _kk_compute_score(nd.text, gt_json) < 1.0]
            if correct:
                unique = correct[:1] + wrong[:GRPO_N - 1]
            else:
                unique = wrong[:GRPO_N]

        # Build results from search nodes
        results = []
        for nd in unique:
            resp_ids = self.tokenizer.encode(nd.text, add_special_tokens=False)
            resp_ids = resp_ids[:self.response_length]
            results.append(AgentLoopOutput(
                prompt_ids=prompt_ids, response_ids=resp_ids,
                response_mask=[1] * len(resp_ids),
                response_logprobs=None, routed_experts=None, multi_modal_data={},
                num_turns=2, metrics={},
                extra_fields={"turn_scores": [], "tool_rewards": []}))

        n_pad = GRPO_N - len(results)
        if n_pad > 0:
            logger.debug("Search returned %d unique nodes, padding %d with single-gen", len(unique), n_pad)
            for _ in range(n_pad):
                results.append(await self._single_gen(prompt_ids, sampling_params))

        return results
