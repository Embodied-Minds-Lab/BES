import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import uuid
import random
import numpy as np

from .tree_node import TreeNode, DEBUG, dprint


@dataclass
class GenerationTreeSearchConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    m: int = 4  # number of trees
    n: int = 2  # number of nodes to expand
    l: int = 1  # number of iterations
    k: int = 4 # number of final selected samples per tree
    reward_mode: str = 'tree_diff'
    expand_mode: str = 'random'
    tool_call_mode: bool = False
    tool_call_stub_response: str = "ok"

class LLMGenerationTreeSearchManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        reward_fn,
        config: GenerationTreeSearchConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.reward_fn = reward_fn

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        self.root_list = []

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        def _truncate(resp: str) -> str:
            if '</search>' in resp:
                return resp.split('</search>')[0] + '</search>'
            if '</answer>' in resp:
                return resp.split('</answer>')[0] + '</answer>'
            if self.config.tool_call_mode and '</tool_call>' in resp:
                return resp.split('</tool_call>')[0] + '</tool_call>'
            return resp

        responses_str = [_truncate(resp) for resp in responses_str]

        # if self.config.no_think_rl:
        #     raise ValueError('stop')
        #     # if no_think_rl is enabled, only keep action in the str
        #     actions, _ = self.env.postprocess_predictions(responses_str)
        #     responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
        #     print("RESPONSES:", responses_str)
        
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict(
            tensors={
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:]
            }
        )
        new_rollings.non_tensor_batch = rollings.non_tensor_batch
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                turns_mask: torch.Tensor,
                turns: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]

        # turns mask
        response_content_mask = (response != pad_id).to(dtype=turns_mask.dtype).to(device=turns_mask.device)
        turns_mask_tensors = [turns_mask, response_content_mask * turns.unsqueeze(1)]

        
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            turns_mask_tensors.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        concatenated_turns_mask = torch.cat(turns_mask_tensors, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_turns_mask = concatenated_turns_mask.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, padded_turns_mask
        

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          turns: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          right_truncate: bool = True,
                        ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask, turns_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    right_side['turns_mask'],
                    turns,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, turns_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    right_side['turns_mask'],
                    turns,
                    pad_to_left=False
                )
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        if right_truncate:
            turns_mask = turns_mask[:, :max_len]
            responses = responses[:, :max_len]
            responses_with_info_mask = responses_with_info_mask[:, :max_len]
        else:
            turns_mask = turns_mask[:, -max_len:]
            responses = responses[:, -max_len:]
            responses_with_info_mask = responses_with_info_mask[:, -max_len:]

        return {'responses': responses, 'responses_with_info_mask': responses_with_info_mask, 'turns_mask': turns_mask}

    def _generate_sequences_with_prob(self, ):
        pass


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
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

    def _get_action_log_probs(
            self,
            log_probs: torch.Tensor
        ):
        return torch.mean(log_probs, dim=1)

    def gen_action_chain(
            self,
            gen_batch: DataProto,
            node_list: List[TreeNode],
            turns_stats: torch.Tensor,
        ):
        """
        Generate action chain for each example in gen_batch.

        Args:
            gen_batch (DataProto): gen_batch should include 
                batch keys ['input_ids', 'attention_mask', 'position_ids']
                non_tensor_batch keys ['uid', 'data_source']
            node_list (list[TreeNode]): should have same order and same length as gen_batch
            turns_stats (torch.Tensor): turns_stats is the number of action turns for each example
        """

        # init original_right_side from node
        init_responses_list = []
        init_responses_with_info_mask_list = []
        init_turns_mask_list = []
        for i, node in enumerate(node_list):
            if node.responses is not None:
                responses = node.responses
                responses_with_info_mask = node.responses_with_info_mask
                turns_mask = node.turns_mask
            else:
                responses = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
                responses_with_info_mask = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
                turns_mask = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
            init_responses_list.append(responses)
            init_responses_with_info_mask_list.append(responses_with_info_mask)
            init_turns_mask_list.append(turns_mask)
            
        init_responses = self.tensor_fn.pad_and_stack(init_responses_list, pad_to_left=False)
        init_responses_with_info_mask = self.tensor_fn.pad_and_stack(init_responses_with_info_mask_list, pad_to_left=False)
        init_turns_mask = self.tensor_fn.pad_and_stack(init_turns_mask_list, pad_to_left=False)
        original_right_side = {
            'responses': init_responses, 
            'responses_with_info_mask': init_responses_with_info_mask,
            'turns_mask': init_turns_mask
        }

        tmp_node_list = node_list
        
        initial_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:]
        
        active_mask = torch.ones(initial_input_ids.shape[0], dtype=torch.bool)
        valid_action_stats = torch.zeros(initial_input_ids.shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(initial_input_ids.shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        rollout_token_cnt = 0

        # fix bug. inherit parent node stats
        for i, node in enumerate(tmp_node_list):
            valid_action_stats[i] = node.valid_action_stats
            valid_search_stats[i] = node.valid_search_stats

        # Main generation loop
        for step in range(self.config.max_turns):
            # check if turns is reached
            # get need_act_mask by (turns_stats<config.max_turns)&active_mask
            need_act_mask = turns_stats < self.config.max_turns
            need_act_mask = need_act_mask.to(active_mask.dtype) * active_mask

            if not need_act_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[need_act_mask] for k, v in rollings.batch.items()
                    },
                    meta_info=rollings.meta_info,
                )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            ## choice 1: get vllm log_probs for tree search, log_probs is a 2d tensor
            log_probs = gen_output.batch['infer_log_probs']
            log_probs = self.tensor_fn._example_level_pad_tensor(log_probs, need_act_mask)
            log_probs_step = self._get_action_log_probs(log_probs)

            # meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, need_act_mask)

            # for debug, stats response
            if DEBUG:
                rollout_token_cnt += gen_output.batch['responses'].shape[0] * gen_output.batch['responses'].shape[1]

            # ## choice 2: get log_probs from actor for tree search
            # with torch.no_grad():
            #     log_probs = self.actor_rollout_wg.compute_log_prob()

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, need_act_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            curr_action_mask = need_act_mask
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[need_act_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                turns_stats,
                next_obs_ids,
            )

            # create tree node
            for i in range(gen_batch.batch['input_ids'].shape[0]):
                if not curr_action_mask[i]:
                    continue

                node_uid = str(uuid.uuid4())

                input_ids = rollings.batch['input_ids'][i]
                position_ids = rollings.batch['position_ids'][i]
                attention_mask = rollings.batch['attention_mask'][i]

                responses = original_right_side['responses'][i]
                responses_with_info_mask = original_right_side['responses_with_info_mask'][i]
                turns_mask = original_right_side['turns_mask'][i]

                # prompts not in rollings
                parent_node = tmp_node_list[i]
                prompts = parent_node.prompts
                tree_uid = parent_node.tree_uid

                new_node = TreeNode(
                    tree_uid=tree_uid,
                    node_uid=node_uid,
                    prompts=prompts,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    responses=responses,
                    responses_with_info_mask=responses_with_info_mask,
                    turns_mask=turns_mask,
                    parent_node=parent_node,
                    is_root=False,
                    is_active=bool(curr_active_mask[i].item()),
                    valid_action_stats=int(valid_action_stats[i].item()),
                    valid_search_stats=int(valid_search_stats[i].item()),
                    depth=parent_node.depth+1,
                    is_leaf=bool(dones[i]),
                    reward_mode=self.config.reward_mode,
                    tensor_fn=self.tensor_fn,
                )

                parent_node.add_child(new_node)
                tmp_node_list[i] = new_node
            
                
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    meta_info=rollings.meta_info,
                )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            ## choice 1: get vllm log_probs for tree search, log_probs is a 2d tensor
            log_probs = gen_output.batch['infer_log_probs']
            log_probs_step = self._get_action_log_probs(log_probs)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # for debug, stats response
            if DEBUG:
                rollout_token_cnt += gen_output.batch['responses'].shape[0] * gen_output.batch['responses'].shape[1]

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            curr_action_mask = active_mask.clone()
            turns_stats[active_mask] += 1
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                turns_stats,
            )

            # create tree node
            for i in range(gen_batch.batch['input_ids'].shape[0]):
                if not curr_action_mask[i]:
                    continue

                node_uid = str(uuid.uuid4())

                input_ids = rollings.batch['input_ids'][i]
                position_ids = rollings.batch['position_ids'][i]
                attention_mask = rollings.batch['attention_mask'][i]

                responses = original_right_side['responses'][i]
                responses_with_info_mask = original_right_side['responses_with_info_mask'][i]
                turns_mask = original_right_side['turns_mask'][i]

                # prompts not in rollings
                parent_node = tmp_node_list[i]
                prompts = parent_node.prompts
                tree_uid = parent_node.tree_uid

                new_node = TreeNode(
                    tree_uid=tree_uid,
                    node_uid=node_uid,
                    prompts=prompts,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    responses=responses,
                    responses_with_info_mask=responses_with_info_mask,
                    turns_mask=turns_mask,
                    parent_node=parent_node,
                    is_root=False,
                    is_active=bool(curr_active_mask[i].item()),
                    valid_action_stats=int(valid_action_stats[i].item()),
                    valid_search_stats=int(valid_search_stats[i].item()),
                    depth=parent_node.depth+1,
                    is_leaf=True,
                    reward_mode=self.config.reward_mode,
                    tensor_fn=self.tensor_fn,
                )
                dprint(f"tmp_node_list[{i}] before update:", tmp_node_list[i].node_uid, parent_node.node_uid)
                dprint(f"Updated tmp_node_list[{i}] to new_node:", new_node.node_uid)
                dprint(f"parent={parent_node.node_uid}, child={new_node.node_uid}, child is_leaf={True}")
                parent_node.add_child(new_node)

        dprint(f"ROLLOUT_TOKEN_CNT={rollout_token_cnt}")

        # original_scores = self.reward_fn(final_output)
        # for i in range(len(tmp_node_list)):
        #     tmp_node_list[i].set_leaf_original_score(original_scores[i])

    def run_llm_loop_tree_search(
            self, 
            gen_batch, 
            initial_input_ids: torch.Tensor,
        ) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        expand_gen_batch = gen_batch.repeat(repeat_times=self.config.n, interleave=True)
        final_output = gen_batch.repeat(repeat_times=self.config.k, interleave=True)
        
        ## step 1. generate initial action chain (m,)
        turns_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # First, create the root node for each prompt
        # To start m trees
        for i in range(gen_batch.batch['input_ids'].shape[0]):
            tree_uid = str(uuid.uuid4())
            node_uid = str(uuid.uuid4())
            prompts = gen_batch.batch['prompts'][i]
            input_ids = gen_batch.batch['input_ids'][i]
            attention_mask = gen_batch.batch['attention_mask'][i]
            position_ids = gen_batch.batch['position_ids'][i]
            root_node = TreeNode(
                tree_uid=tree_uid,
                node_uid=node_uid,
                prompts=prompts,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                is_root=True,
                is_active=True,
                valid_action_stats=0,
                valid_search_stats=0,
                depth=0,
                reward_mode=self.config.reward_mode,
                tensor_fn=self.tensor_fn,
            )
            self.root_list.append(root_node)
        # Then, get m initial chains
        self.gen_action_chain(gen_batch, self.root_list.copy(), turns_stats)

        # step 2. Iteration to expand the trees
        for iteration in range(self.config.l):
            expansion_node_list = []

            # get expansion for n nodes
            for root in self.root_list:
                expansion_node_list_i = root.get_expand_node(self.config.n, mode=self.config.expand_mode)
                expansion_node_list.extend(expansion_node_list_i)
                expansion_node_uid_list = [node.node_uid for node in expansion_node_list_i]
                dprint(f'==========get expansion for root={root.node_uid}, uid_list={expansion_node_uid_list}')

            # prompts not need to pad
            prompts = torch.stack([node.prompts for node in expansion_node_list], dim=0)
            
            # need to pad
            input_ids = self.tensor_fn.pad_and_stack([node.input_ids for node in expansion_node_list])
            attention_masks = self.tensor_fn.create_attention_mask(input_ids)
            position_ids = self.tensor_fn.create_position_ids(attention_masks)
            
            turns_stats = torch.tensor([node.depth for node in expansion_node_list], dtype=torch.int)
            expand_gen_batch.batch['input_ids'] = input_ids
            expand_gen_batch.batch['attention_mask'] = attention_masks
            expand_gen_batch.batch['position_ids'] = position_ids

            # do the expansion
            self.gen_action_chain(expand_gen_batch, expansion_node_list, turns_stats)
        
        # step 3. Get final output 
        # The order is BS x M(trees) x K(samples per tree)
        # same as final_output
        final_responses_list = []
        final_responses_with_info_mask_list = []
        final_turns_mask_list = []
        final_tree_uid_list = []
        final_node_list = []
        for root in self.root_list:
            root.check_all_nodes_child()
            # sample trajectory
            final_node_list_tmp = root.sample_leaf(self.config.k)
            final_node_list.extend(final_node_list_tmp)

        for node in final_node_list:
            final_responses_list.append(node.responses)
            final_responses_with_info_mask_list.append(node.responses_with_info_mask)
            final_turns_mask_list.append(node.turns_mask)
            final_tree_uid_list.append(node.tree_uid)

        final_responses = self.tensor_fn.pad_and_stack(final_responses_list, pad_to_left=False)
        final_responses_with_info_mask = self.tensor_fn.pad_and_stack(final_responses_with_info_mask_list, pad_to_left=False)
        final_turns_mask = self.tensor_fn.pad_and_stack(final_turns_mask_list, pad_to_left=False)

        final_output.batch['responses'] = final_responses.long()
        final_output.batch['prompts'] = final_output.batch['prompts'].long()
        final_output.batch['input_ids'] = torch.cat([
            final_output.batch['prompts'],
            final_responses
        ], dim=1)
        final_output.batch['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(final_output.batch['prompts']),
            self.tensor_fn.create_attention_mask(final_responses)
        ], dim=1)
        final_output.batch['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(final_output.batch['prompts']),
            self.tensor_fn.create_attention_mask(final_responses_with_info_mask)
        ], dim=1)
        final_output.batch['position_ids'] = self.tensor_fn.create_position_ids(
            final_output.batch['attention_mask']
        )
        final_output.batch['turns_mask'] = final_turns_mask

        # step 4. Compute scores and return outputs
        # get original score from reward_fn
        dprint('===========start calculate reward')
        original_scores, _ = self.reward_fn(final_output)
        for i, node in enumerate(final_node_list):
            dprint(f'node:{node.node_uid}, original_score={original_scores[i]}')
            node.set_leaf_original_score(original_scores[i])
            # node.set_leaf_original_score(random.random())
        # get final tree-based score for each tree
        for root in self.root_list:
            root.calculate_final_score_from_root()
        # get final token-level score for each leaf
        final_token_level_scores_list = []
        for node in final_node_list:
            final_token_level_scores_list.append(node.get_token_level_score_from_leaf())
        final_token_level_scores = self.tensor_fn.pad_and_stack(final_token_level_scores_list, pad_to_left=False, pad_value=0.0)
        final_output.batch['token_level_scores'] = final_token_level_scores        

        # get meta info
        turns_stats = []
        active_mask = []
        valid_action_stats = []
        valid_search_stats = []
        for node in final_node_list:
            turns_stats.append(node.depth)
            active_mask.append(node.is_active)
            valid_action_stats.append(node.valid_action_stats)
            valid_search_stats.append(node.valid_search_stats)
        
        meta_info = {}
        meta_info['turns_stats'] = turns_stats
        meta_info['active_mask'] = active_mask
        meta_info['valid_action_stats'] = valid_action_stats
        meta_info['valid_search_stats'] = valid_search_stats

        final_output.meta_info = meta_info

        final_output.non_tensor_batch['tree_uid'] = np.array(final_tree_uid_list, dtype=object)

        # Delete the tree
        for root in self.root_list:
            root.delete_tree_from_root()
        self.root_list = []

        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        # Skip the HTTP call when there's nothing to search OR when no retriever is
        # configured (tool_call_mode training has no retriever). An unschema'd URL
        # would otherwise raise MissingSchema inside requests.post.
        has_retriever = bool(self.config.search_url)
        if do_search and len(search_queries) > 0 and has_retriever:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                # Only set to 1 if done in this turn
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    # Only set to 1 if done in this turn
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    # Only set to 1 if done in this turn
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                elif action == 'tool_call':
                    # stub response — the grader scores based on which tools were
                    # called / with what args, not on what the tool returned.
                    next_obs.append(f'\n\n<tool_response>{self.config.tool_call_stub_response}</tool_response>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                else:
                    invalid_hint = 'If I want to search, I should put the query between <search> and </search>.'
                    if self.config.tool_call_mode:
                        invalid_hint = ('If I want to call a tool, I should put the JSON '
                                        '{"name":..., "arguments":{...}} between <tool_call> and </tool_call>.')
                    next_obs.append(f'\nMy previous action is invalid. {invalid_hint} '
                                    f'If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    # Only set to 1 if done in this turn
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        tag_alt = "search|answer|tool_call" if self.config.tool_call_mode else "search|answer"
        pattern = rf'<({tag_alt})>(.*?)</\1>'
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        if retrieval_result is None:
            print("WARNING!!! retrieval_result is None")
        elif len(retrieval_result) > 0 and isinstance(retrieval_result[0], dict):
            for idx, doc_item in enumerate(retrieval_result):
                
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        else:
            for tmp in retrieval_result:
                format_reference += tmp

        return format_reference


###### test
if __name__ == '__main__':
    import ray

    @ray.remote
    def test():

        from verl.utils.fs import copy_local_path_from_hdfs
        from transformers import AutoTokenizer, AutoConfig

        model_path = '/path/to/model'
        copy_local_path_from_hdfs(model_path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(model_path)

        from omegaconf import OmegaConf

        from verl.trainer.main_ppo_format_ts import RewardManager
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                                structure_format_score=0.2, 
                                final_format_score=0.1,
                                retrieval_score=0)

        config = OmegaConf.load('/path/to/ppo_trainer_test.yaml')
        # config = config.actor_rollout_ref

        # from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_ts import vLLMRollout
        # model_hf_config = AutoConfig.from_pretrained(model_path)
        # actor_rollout_wg = vLLMRollout(model_path=model_path, config=config, tokenizer=tokenizer, model_hf_config=model_hf_config)
        
        # from verl.workers.fsdp_workers import ActorRolloutRefWorker
        # actor_rollout_wg = ActorRolloutRefWorker(config=config, role='actor_rollout')
        # actor_rollout_wg.init_model()

        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        from verl.trainer.ppo.ray_trainer_ts import ResourcePoolManager, Role
        from verl.trainer.ppo.ray_trainer_ts import RayPPOTrainer

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=reward_fn,
                                debug=True,
                                )
        trainer.init_workers()
        actor_rollout_wg = trainer.actor_rollout_wg

        dprint('============ init finished =================')

        gen_config = GenerationTreeSearchConfig(
                max_turns=3,
                max_start_length=2048,
                max_prompt_length=4096,
                max_response_length=500,
                max_obs_length=500,
                num_gpus=2,
                no_think_rl=False,
                search_url="http://127.0.0.1:8000/retrieve",
                topk=3,
                m=2,
                n=2,
                l=1,
                k=3,
                reward_mode='base',
                expand_mode='random',
            )
        generation_manager = LLMGenerationTreeSearchManager(
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            reward_fn=reward_fn,
            config=gen_config
        )

        batch_size = 4

        prompt_input_text = []
        prompt_text_list = [
            "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What genre of music was the album The Sabbath Stones?",
            "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Do You Believe Me Now is the second studio album of American country music singer Jimmy Wayne, the album is also Wayne's first album in five years and his debut for Valory Music Group, a subsidiary of which independent American record label specializing in country and pop artists, and  based on Music Row in Nashville, Tennessee, and is distributed by Universal Music Group (UMG)?",
            "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What country of origin does Susan Dalian and Storm have in common?",
            "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Aaj Shahzeb Khanzada Kay Sath is a current affairs show on which private Pakistani news channel?",
        ]
        for prompt_text in prompt_text_list:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            prompt_text_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_input_text.extend([prompt_text_template] * gen_config.m)

        tensor_fn = TensorHelper(TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=4096,
                max_obs_length=500,
                max_start_length=2048
            ))

        tokenized_output = tokenizer(
            prompt_input_text,
            padding=True, # No padding needed for a single example
            truncation=True,
            max_length=4096, # A reasonable max length
            return_tensors='pt' # Return PyTorch tensors
        )

        dprint("\n--- Tensor Shapes ---")
        dprint(f"input_ids shape: {tokenized_output['input_ids'].shape}")
        dprint(f"attention_mask shape: {tokenized_output['attention_mask'].shape}")
        dprint(f"position_ids shape: {tensor_fn.create_position_ids(tokenized_output['attention_mask']).shape}")
        dprint(f"input_ids max: {tokenized_output['input_ids'].max()}, min: {tokenized_output['input_ids'].min()}")
        dprint(f"vocabulary size: {tokenizer.vocab_size}")

        gen_batch = DataProto.from_dict(
            tensors={
                'prompts': tokenized_output['input_ids'],
                'input_ids': tokenized_output['input_ids'],
                'attention_mask': tokenized_output['attention_mask'],
                'position_ids': tensor_fn.create_position_ids(tokenized_output['attention_mask']),
            },
            non_tensors={
                'uid': np.array(['123456'] * batch_size * gen_config.m, dtype=object),
                'data_source': np.array(['hotpotqa'] * batch_size * gen_config.m, dtype=object),
                'reward_model': np.array([{'ground_truth': {'target': ['heavy metal music']}}] * batch_size * gen_config.m, dtype=object)
            },
            # meta_info = {
            #     'eos_token_id': [tokenizer.eos_token_id]*2,
            #     'pad_token_id': [tokenizer.pad_token_id],
            # }
        )
        initial_input_ids = tokenized_output['input_ids']

        generation_manager.run_llm_loop_tree_search(gen_batch, initial_input_ids)

    ray.get(test.remote())

    ## To start the test
    # ray start --head --port=8265
    # ray job submit --address=http://127.0.0.1:8265 --runtime-env=verl/trainer/runtime_env_test.yaml -- python -m search_r1.llm_agent.generation_ts 2>&1 | tee test_generation_ts.log