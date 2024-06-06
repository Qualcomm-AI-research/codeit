# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import traceback

import numpy as np
import torch
from transformers import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from codeit.augment.type_inference import CONSTANT_NAMES
from codeit.dsl.primitives import PRIMITIVE_FUNCTIONS
from codeit.policy.environment import execute_candidate_program
from codeit.task import Task
from codeit.utils import get_grid_size, get_num_pixels, transform_to_function


def abs_grid_distance(grid1, grid2, debug=False):
    size1 = get_grid_size(grid1)
    size2 = get_grid_size(grid2)
    if size1 == size2:
        num_pixels = get_num_pixels(grid1)
        count = np.sum(np.array(grid1) != np.array(grid2))
        if debug:
            print("num pixels", num_pixels)
            print("count", count)
        return count / num_pixels
    else:
        return 1


class BadTokenEnforcedEndLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, eos_token_id) -> None:
        super().__init__()
        self.eos_token_id = eos_token_id
        self.allowed_token_ids = allowed_token_ids
        assert isinstance(allowed_token_ids, torch.cuda.LongTensor)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        last_token_per_sequence = input_ids[:, -1]
        is_allowed = last_token_per_sequence.unsqueeze(1) == self.allowed_token_ids
        is_allowed = is_allowed.any(dim=1)

        scores[~is_allowed] = -1e20
        scores[~is_allowed, self.eos_token_id] = 1e20

        return scores


def get_allowed_tokens(tokenizer):
    terms = CONSTANT_NAMES + PRIMITIVE_FUNCTIONS + ["(", ")", ",", "I", "O", "=", "\n", " "]
    for i in range(0, 100):
        terms.append(f"x{i}")
        terms.append(f"{i}")

    spaced_terms = copy.copy(terms)
    for term in terms:
        if term not in ["\n"]:
            spaced_terms.append(f" {term}")

    spaced_term_tokens = tokenizer(spaced_terms, padding=True, return_tensors="pt")
    spaced_term_unique_token_ids = torch.unique(spaced_term_tokens["input_ids"])
    return spaced_term_unique_token_ids


def evaluate_solution(program, inference_task, training=True, n_examples=100):
    performance = []
    if training:
        examples = inference_task.training_examples[:n_examples]
    else:
        examples = inference_task.test_examples[:n_examples]

    for example in examples:
        output = execute_candidate_program(program, example["input"])
        performance.append(output == example["output"])
    return performance


class Evaluator:
    def __init__(self, tokenizer, text_encoder, valid_grid_func, allowed_tokens=False) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.valid_grid_func = valid_grid_func
        if allowed_tokens:
            self.allowed_tokens = get_allowed_tokens(tokenizer=tokenizer).type(
                torch.cuda.LongTensor
            )
        else:
            self.allowed_tokens = False

    def generate(
        self, model, num_samples, input_ids, attention_mask, temperature=0.95, max_length=512
    ):

        if not isinstance(self.allowed_tokens, bool):
            logits_processor = LogitsProcessorList(
                [
                    BadTokenEnforcedEndLogitsProcessor(
                        allowed_token_ids=self.allowed_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                ]
            )
        else:
            logits_processor = None

        tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            max_length=max_length,
            num_return_sequences=num_samples,
            logits_processor=logits_processor,
            output_scores=False,
            return_dict_in_generate=False,
        )
        return tokens

    def decode_actions(self, tokens):
        action_list_pre = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        action_list = [actions.split("#")[0] for actions in action_list_pre]
        return action_list

    def evaluate_actions(self, action_list, initial_states, goal_states, max_state_size=None):
        rewards = np.empty((len(action_list), len(initial_states)), dtype=object)
        terminal_states = np.empty((len(action_list), len(initial_states)), dtype=object)
        for i, actions in enumerate(action_list):
            for j, initial_state in enumerate(initial_states):
                terminal_state = execute_candidate_program(
                    actions, initial_state, max_state_size=max_state_size
                )
                if self.valid_grid_func(terminal_state):
                    reward = 1 - abs_grid_distance(terminal_state, goal_states[j])
                else:
                    reward = 0
                rewards[i][j] = reward
                terminal_states[i][j] = terminal_state
        return rewards, terminal_states

    def create_tasks_and_solutions(
        self,
        action_list,
        terminal_states,
        initial_states,
        solutions,
        rewards,
        inference_task,
        generated_tasks=None,
        iteration_id=None,
        task_name_extension="__",
        max_grid_size=10_000_000_000,
    ):
        for j, actions in enumerate(action_list):
            valid_training_states = []
            for k, terminal_state in enumerate(terminal_states[j]):
                if self.valid_grid_func(terminal_state):
                    if get_num_pixels(terminal_state) < max_grid_size:
                        valid_training_states.append(
                            {"input": initial_states[k], "output": terminal_state}
                        )
            if len(valid_training_states) > 0:
                task_key = f"{inference_task.task_key}_it{iteration_id}__{j}"
                try:
                    task_demonstration_performance = [0.0]
                    distance = np.mean([1 - reward for reward in rewards])
                    if 1 in rewards[j]:  # we're saving the solution if any examples are solved
                        task_demonstration_performance = evaluate_solution(
                            program=actions, inference_task=inference_task, training=True
                        )
                        test_performance = evaluate_solution(
                            program=actions, inference_task=inference_task, training=False
                        )
                        if inference_task.task_key in solutions.keys():
                            # we can add the same solution!
                            solutions[inference_task.task_key][actions] = {
                                "program": actions,
                                "new_task_key": task_key,
                                "seen_example_performance": rewards[j].tolist(),
                                "test_performance": test_performance,
                                "task_demonstration_performance": task_demonstration_performance,
                            }
                        else:
                            solutions[inference_task.task_key] = {
                                actions: {
                                    "program": actions,
                                    "new_task_key": task_key,
                                    "seen_example_performance": rewards[j].tolist(),
                                    "test_performance": test_performance,
                                    "task_demonstration_performance": task_demonstration_performance,
                                }
                            }
                        pass
                    if generated_tasks is not None:
                        new_task = Task(
                            program=transform_to_function(
                                input_str=actions,
                                function_name=f"solve_{inference_task.task_key}_it{iteration_id}{task_name_extension}{j}",
                            ),
                            training_examples=valid_training_states,
                            test_examples=[],
                            task_key=task_key,
                            extra_info={
                                "rewards": rewards[j],
                                "distance": distance,
                                "task_demonstration_performance": np.mean(
                                    task_demonstration_performance
                                ),
                            },
                            parent_key=inference_task.task_key,
                        )
                        generated_tasks[task_key] = new_task
                except:
                    print(f"could not create task {task_key}")
                    traceback.print_exc()
        return solutions, generated_tasks
