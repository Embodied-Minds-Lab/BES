from __future__ import annotations

import math
from typing import List, Optional, Callable
from pydantic import BaseModel
import json
import random
from collections import deque
import torch
from verl import DataProto
import random
import os

DEBUG: bool = os.environ.get('TREE_SEARCH_DEBUG', '').lower() == 'true'

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class TreeNode:
    def __init__(
        self,
        tree_uid: str, # equal to the original question prompt uid
        node_uid: str,
        prompts: Optional[torch.Tensor] = None, # prompts are same as the original input_ids
        input_ids: Optional[torch.Tensor] = None, # input_ids are the left_part input during rollout
        attention_mask: Optional[torch.Tensor] = None, # attention_mask are the left_part input during rollout
        position_ids: Optional[torch.Tensor] = None, # position_ids are the left_part input during rollout
        responses: Optional[torch.Tensor] = None, # responses are the right_part output during rollout
        responses_with_info_mask: Optional[torch.Tensor] = None, # responses_with_info_mask are the right_part output during rollout
        turns_mask: Optional[torch.Tensor] = None, # turn_mask are the right_part output during rollout
        log_prob_node: Optional[float] = 0.0,
        log_prob_list: Optional[list[float]] = [],
        parent_node: Optional['TreeNode'] = None,
        is_root: bool = False,
        is_active: bool = True,
        valid_action_stats: int = 0,
        valid_search_stats: int = 0,
        depth: int = 0,
        is_leaf: bool = False,
        correct_leaf_in_subtree: int = 0,
        reward_mode: str = 'base', 
        tensor_fn = None,
        margin = 0.1,
    ):

        self.tree_uid: int = tree_uid
        self.node_uid: int = node_uid

        self.prompts: torch.Tensor = prompts
        self.input_ids: torch.Tensor = input_ids
        self.attention_mask: torch.Tensor = attention_mask
        self.position_ids: torch.Tensor = position_ids
        self.responses: torch.Tensor = responses
        self.responses_with_info_mask: torch.Tensor = responses_with_info_mask
        self.turns_mask: torch.Tensor = turns_mask

        self.log_prob_node: float = log_prob_node
        self.log_prob_list: list[float] = log_prob_list

        self.parent_node = parent_node
        self._child_node = []

        self.is_root = is_root
        self.is_active = is_active
        self.valid_action_stats = valid_action_stats
        self.valid_search_stats = valid_search_stats
        self.depth = depth
        self.is_leaf = is_leaf

        self.original_score = 0.
        self.final_score = 0.

        self.subtree_leaf_score = 0.

        self.reward_mode = reward_mode
        self.margin = margin

        self.tensor_fn = tensor_fn

    @property
    def child_node(self) -> list['TreeNode']:
        return self._child_node

    @child_node.setter
    def child_node(self, value: list['TreeNode']):
        """
        Debug
        """
        print(f"!!! WARNING: Direct assignment to child_node on node {self.node_uid}. New list has {len(value)} children.")
        
        import traceback
        traceback.print_stack()
        
        # Check
        for child in value:
            if child is self:
                raise ValueError(f"CRITICAL: Attempted to directly assign a list containing the node itself as its own child. Node UID: {self.node_uid}")

        self._child_node = value
        
    def add_child(self, child_node: 'TreeNode'):
        if child_node is self:
            raise ValueError("A node cannot be its own child")
        if child_node.node_uid == self.node_uid:
            raise ValueError("node_uid same!! A node cannot be its own child")
        self._child_node.append(child_node)
    
    def get_subtree_nodes(self):
        """
        Dynamically get all descendant nodes by traversing the tree
        """
        nodes = []
        nodes_to_visit = list(self.child_node) # from child nodes
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node.child_node)
        return nodes

    def get_subtree_leaves_num(self):
        num = 0
        nodes_to_visit = list(self.child_node)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if current_node.is_leaf:
                num += 1
            nodes_to_visit.extend(current_node.child_node)
        return num

    def get_expand_node(self, n: int = 1, mode: str = 'random') -> List['TreeNode']:
        """
        Sample n nodes from the subtree
        """
        candidate_set = [self]
        for node in self.get_subtree_nodes():
            if not node.is_leaf:
                candidate_set.append(node)
        result = random.choices(candidate_set, k=n)
                
        assert len(result) == n, f"get_expand_node error, len(result)={len(result)} != n={n}"
        return result

    def sample_leaf(self, n: int = 1) -> List['TreeNode']:
        """
        Sample n leaves, then prune the tree (drop the unselected nodes)
        """

        candidate_uid_set = []

        for node in self.get_subtree_nodes():
            if node.is_leaf:
                candidate_uid_set.append(node.node_uid)

        if len(candidate_uid_set) < n:
            dprint(f"root={self.node_uid}, candidate_uid_set={candidate_uid_set}")
            subtree_nodes = self.get_subtree_nodes()
            subtree_node_uids = [node.node_uid for node in subtree_nodes]
            dprint(f"all subtree nodes={subtree_node_uids}")
        assert len(candidate_uid_set) >= n, f"root={self.node_uid}, candidate_uid_set len={len(candidate_uid_set)} < n={n}"

        random.shuffle(candidate_uid_set)
        dprint(f'original candidate_uid_set len={len(candidate_uid_set)}')
        candidate_uid_set = candidate_uid_set[:n]
        dprint(f'sampled candidate_uid_set len={len(candidate_uid_set)}')
        self._prune_subtree(candidate_uid_set)

        result = []
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                result.append(node)
        return result

    def set_leaf_original_score(self, score: float):
        """
        Set the original score of the leaf
        """
        self.original_score = score

    @staticmethod
    def dfs_subtree_leaf_score(tmp_node: 'TreeNode') -> float:
        """
        Do dfs and compute the subtree leaf original score
        """
        subtree_leaf_score = tmp_node.original_score
        for node in tmp_node.child_node:
            subtree_leaf_score += TreeNode.dfs_subtree_leaf_score(node)
        tmp_node.subtree_leaf_score = subtree_leaf_score
        return subtree_leaf_score

    def calculate_final_score_from_root(self):
        """
        Calculate the Diff-based final score from root (not for Tree-GRPO. Tree-GRPO uses base mode)
        """

        # First do dfs and compute the subtree leaf original score
        TreeNode.dfs_subtree_leaf_score(self)

        # Then compute the final score for each node
        total_leaf_num = self.get_subtree_leaves_num()
        global_score_mean = self.subtree_leaf_score / total_leaf_num
        for node in self.get_subtree_nodes():
            # TreeRL global score
            curr_leaf_num = 1 if node.is_leaf else node.get_subtree_leaves_num()
            subtree_nodes = node.get_subtree_nodes()
            subtree_node_uids = [x.node_uid for x in subtree_nodes]
            assert curr_leaf_num > 0, f"node_uid={node.node_uid} have no leaves, subtree_node_uids={subtree_node_uids}"
            curr_score_mean = node.subtree_leaf_score / curr_leaf_num    
            global_score = curr_score_mean - global_score_mean
            global_score = 0.
            # TreeRL local score
            parent_leaf_num = node.parent_node.get_subtree_leaves_num()
            parent_score_mean = node.parent_node.subtree_leaf_score / parent_leaf_num
            local_score = curr_score_mean - parent_score_mean
            
            diff_score = global_score + local_score
            diff_score = max(diff_score - self.margin, 0.)

            # final score = diff_score + curr_score_mean
            final_score = diff_score + curr_score_mean
            node.final_score = final_score / math.sqrt(curr_leaf_num)

    def get_token_level_score_from_leaf(self):
        """
        Get the token-level score from the leaf
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)

        # Diff-based Reward
        if self.reward_mode == 'tree_diff':
            valid_response_length_list = []
            scores_list = []
            node = self.parent_node
            while node:
                if node.is_root:
                    break
                valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
                valid_response_length_list.append(valid_response_length)
                scores_list.append(node.final_score)
                node = node.parent_node
            scores_list.append(0.)
            valid_response_length_list.append(0)
            valid_response_length_list.reverse()
            scores_list.reverse()
            for i in range(1, len(valid_response_length_list)):
                score = scores_list[i]
                
                l = valid_response_length_list[i-1]
                r = valid_response_length_list[i] - 1

                if l < r:
                    final_token_level_scores[l:r] = score
        # Tree-GRPO uses base mode
        else:
            valid_response_length = self.tensor_fn.create_attention_mask(self.responses).sum()
            final_token_level_scores[valid_response_length-1] = self.original_score

        return final_token_level_scores

    def _prune_subtree(self, candidate_uid_set: List[int]) -> bool:
        """
        Drop all the leaves not in cdandidate_uid_set
        """
        surviving_children = []
        dprint(f'start, node={self.node_uid}, child_node len={len(self._child_node)}')

        for child in self.child_node:
            dprint(f'node={self.node_uid}, iter for child={child.node_uid}')
            if child._prune_subtree(candidate_uid_set):
                surviving_children.append(child)
        
        # update child node
        self._child_node = surviving_children
                
        # Determine whether the current node should be retained. The conditions for retention are:
        # 1. It is in candidate_uid_set
        # 2. Or, it still has child nodes after pruning
        should_keep_this_node = self.node_uid in candidate_uid_set or (len(self._child_node) > 0)

        dprint(f'end, node={self.node_uid}, is keeped={should_keep_this_node}')

        return should_keep_this_node

    def check_all_nodes_child(self):
        for node in self.get_subtree_nodes():
            node_child_list = node.child_node
            node_child_node_uid_list = []
            for node_child_node in node_child_list:
                node_child_node_uid_list.append(node_child_node.node_uid)
            if node.node_uid in node_child_node_uid_list:
                dprint(f'error!! node.uid in node_child_node_uid_list!!! is_root={node.is_root}, is_leaf={node.is_leaf}, node_uid={node.node_uid}')

    def delete_tree_from_root(self):
        """
        Delete the tree from root
        """
        all_nodes_list = self.get_subtree_nodes()
        all_nodes_list.append(self)

        for node in all_nodes_list:
            node.prompts = None
            node.input_ids = None
            node.attention_mask = None
            node.position_ids = None
            node.responses = None
            node.responses_with_info_mask = None

            node._child_node.clear()
            node.parent_node = None

            node.log_prob_list.clear()

        