# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import functools
import gc
import json
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from codeit.concept_arc import load_concept_arc
from codeit.dsl.arc_types import is_grid
from codeit.dsl.dsl import palette
from codeit.dsl.primitives import PRIMITIVE_CONSTANTS, PRIMITIVE_FUNCTIONS
from codeit.exit_data_module import collate_fn_seq2seq, get_inference_dataset
from codeit.policy.inference import Evaluator
from codeit.replay_buffer import Buffer
from codeit.task import get_tasks_with_keys


def calculate_performance(solutions, task_key):
    if task_key in solutions:
        programs = {
            program: np.mean(solutions[task_key][program]["task_demonstration_performance"])
            for program in solutions[task_key]
        }

        if programs:
            sorted_data = dict(
                sorted(programs.items(), key=lambda x: (len(x[0]), x[1]), reverse=False)
            )
            sorted_data = dict(sorted(sorted_data.items(), key=lambda x: x[1], reverse=True))
            keys = list(sorted_data.keys())[:3]
            top_three_performance = [
                np.mean(solutions[task_key][program_key]["test_performance"]) == 1
                for program_key in keys
            ]  # all test cases are solved
            max_top_three = max(top_three_performance)

            task_demonstration_performance = max(programs.values()) == 1
            test_performance = max_top_three
        else:
            task_demonstration_performance = False
            test_performance = False
    else:
        task_demonstration_performance = False
        test_performance = False

    return task_demonstration_performance, test_performance


def performance_over_tasks(performance_by_task):
    return np.mean(np.array(performance_by_task) == 1.0)


def valid_grid_fast(value):
    if value in [(), (()), ((),)]:
        return True
    try:
        np_grid = np.array(value, dtype=int)
    except:
        return False

    if np_grid.ndim != 2:
        return False

    return np.all((0 <= np_grid) & (np_grid) <= 9)


def valid_grid_slow(value):
    if not is_grid(value):
        return False
    colours = palette(value)
    for colour in colours:
        if colour not in list(range(0, 10)):
            return False
    return True


class Agent:
    def __init__(self, config):
        self.config = config

        with open(config.data.split_keys_path, "r") as f:
            split_keys = json.load(f)

        test_keys = split_keys["test"][: config.data.n_test]
        train_keys = list(split_keys["train"][: config.data.n_train])
        val_keys = list(split_keys["val"][: config.data.n_val])

        if config.final_experiments:
            if config.concept_arc:
                inference_tasks, _, _ = load_concept_arc(config.data.concept_arc_path)
            else:
                inference_tasks = get_tasks_with_keys(
                    file_path=f"{config.data.evaluation_data_dir}", keys=test_keys
                )
            train_tasks = get_tasks_with_keys(
                file_path=f"{config.data.training_data_dir}", keys=train_keys + val_keys
            )
            self.train_keys = train_keys + val_keys

        else:
            inference_tasks = get_tasks_with_keys(
                file_path=f"{config.data.training_data_dir}", keys=val_keys
            )
            train_tasks = get_tasks_with_keys(
                file_path=f"{config.data.training_data_dir}", keys=train_keys
            )
            self.train_keys = train_keys
            self.train_tasks = train_tasks

        self.replay_buffer = Buffer(config=config, train_tasks=train_tasks)

        self.input_state_max = config.data.dataloader.tokenizer.input_state_max
        self.n_examples = config.data.dataloader.tokenizer.n_examples
        self.max_decoder_tokens = config.model.max_length
        self.max_state_size = config.exit.max_state_size
        self.max_grid_size = config.exit.max_state_size

        self.sparse = config.data.dataloader.tokenizer.sparse
        self.sampling_and_filtering_ablation = config.ablation.sampling_and_filtering

        self.inference_tasks = inference_tasks
        self.tokenizer = self.replay_buffer.tokenizer

        self.n_policy_samples = config.exit.n_policy_samples

        if config.evaluation.valid_grid_function == "fast":
            valid_grid_func = valid_grid_fast
        else:
            valid_grid_func = valid_grid_slow

        self.evaluator = Evaluator(
            tokenizer=self.replay_buffer.tokenizer,
            text_encoder=self.replay_buffer.text_encoder,
            valid_grid_func=valid_grid_func,
            allowed_tokens=config.data.dataloader.tokenizer.allowed_tokens,
        )

        self.inference_dataset = get_inference_dataset(
            tasks=self.inference_tasks,
            tokenizer=self.replay_buffer.tokenizer,
            input_state_max=self.input_state_max,
            n_examples=self.n_examples,
            max_decoder_tokens=self.max_decoder_tokens,
            sparse=self.sparse,
            text_encoder=None,
        )

        self.solutions = {"policy": {"seen_example": {}, "task_demonstration": {}, "test": {}}}

        self.add_policy_samples = config.exit.add_policy_samples

    def add_tasks_to_buffer(self, tasks, iteration_id, mode="policy"):
        for task in tasks.values():
            if (not self.sampling_and_filtering_ablation) or mode != "policy":
                self.replay_buffer.add(task=copy.copy(task), iteration_id=iteration_id, mode=mode)
            else:
                if task.extra_info["task_demonstration_performance"] == 1:
                    self.replay_buffer.add(
                        task=copy.copy(task), iteration_id=iteration_id, mode=mode
                    )

    def add_solution(self, solution, task_key, mode):
        if task_key in self.solutions[mode]["seen_example"]:
            self.solutions[mode]["seen_example"][task_key][solution["program"]] = solution
        else:
            self.solutions[mode]["seen_example"][task_key] = {solution["program"]: solution}
        if np.mean(solution["test_performance"]) == 1:
            if task_key in self.solutions[mode]["test"]:
                self.solutions[mode]["test"][task_key][solution["program"]] = solution
            else:
                self.solutions[mode]["test"][task_key] = {solution["program"]: solution}
        if np.mean(solution["task_demonstration_performance"]) == 1:
            if task_key in self.solutions[mode]["task_demonstration"]:
                self.solutions[mode]["task_demonstration"][task_key][solution["program"]] = solution
            else:
                self.solutions[mode]["task_demonstration"][task_key] = {
                    solution["program"]: solution
                }

    def update_solutions(self, solutions, task, mode, solutions_log):
        if task.task_key in solutions:
            for solution in solutions[task.task_key].values():
                self.add_solution(solution=solution, task_key=task.task_key, mode=mode)
                solutions_log.append(
                    f"task: {task.task_key} program: {solution['program']} seen_example_performance: {np.mean(solution['seen_example_performance'])} test_performance: {np.mean(solution['test_performance'])} task_demonstration_performance: {np.mean(solution['task_demonstration_performance'])}"
                )
        return solutions_log

    def save_solutions(self):
        with open(
            f"{self.config.exit.solutions_dir}/solutions_{self.replay_buffer.current_iteration}.json",
            "w+",
        ) as f:
            json.dump(self.solutions, f)

    def sample_policy_programs(self, iteration_id, model):
        programs = {}
        log = {}
        if self.config.exit.policy_sample_log:
            policy_sample_log_file = f"{self.config.exit.policy_sample_log}/log_{iteration_id}.json"
        else:
            policy_sample_log_file = None

        batch_size = self.config.evaluation.batch_size_sample

        device = model.device
        ds_lengths = [len(x["input_ids"]) for x in self.inference_dataset]
        sorted_indices = np.argsort(ds_lengths)

        sorted_ds = self.inference_dataset.select(sorted_indices)
        collate_fn = functools.partial(
            collate_fn_seq2seq,
            pad_token_id=self.config.models.data.dataloader.tokenizer.pad_token_id,
        )
        dl = DataLoader(
            dataset=sorted_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            for ix, batch in tqdm(
                enumerate(dl),
                desc=f"Policy sampling on {model.device} over tasks of batch size {batch_size}",
            ):
                torch.cuda.empty_cache()
                gc.collect()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                actual_batch_size = len(input_ids)  # needed for reshape later
                original_ds_indices = sorted_indices[
                    ix * batch_size : ix * batch_size + actual_batch_size
                ]
                task_ids = [
                    list(self.tokenizer.batch_decode(self.inference_dataset["task_id"]))[ix]
                    for ix in original_ds_indices
                ]

                tokens = self.evaluator.generate(
                    model=model,
                    num_samples=self.config.exit.n_policy_samples,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=self.config.evaluation.temperature,
                    max_length=self.config.evaluation.max_length,
                )

                tokens = tokens.reshape(actual_batch_size, self.config.exit.n_policy_samples, -1)

                for batch_dim, task_id in enumerate(task_ids):
                    action_list = self.evaluator.decode_actions(tokens[batch_dim, :, :])

                    if task_id in programs:
                        programs[task_id] += action_list
                        log[task_id] += self.replay_buffer.tokenizer.batch_decode(
                            tokens[batch_dim, :, :], skip_special_tokens=False
                        )
                    else:
                        programs[task_id] = action_list
                        log[task_id] = self.replay_buffer.tokenizer.batch_decode(
                            tokens[batch_dim, :, :], skip_special_tokens=False
                        )
        if policy_sample_log_file:
            with open(policy_sample_log_file, "w") as f:
                json.dump(log, f)
        return programs

    def sample_policy_tasks(self, iteration_id, model):
        if self.n_policy_samples > 0:
            programs = self.sample_policy_programs(iteration_id, model)
            generated_tasks = {}
            solutions = {}
            solutions_log = []
            task_demonstration_performance_over_tasks = []
            test_performance_over_tasks = []
            for task in tqdm(self.inference_tasks.values(), desc="Evaluating programs over tasks"):

                (
                    solutions,
                    generated_tasks,
                    task_demonstration_performance,
                    test_performance,
                ) = self.evaluate_programs(
                    programs[task.task_key],
                    task,
                    solutions=solutions,
                    generated_tasks=generated_tasks,
                    iteration_id=iteration_id,
                )
                solutions_log = self.update_solutions(
                    solutions=solutions, task=task, mode="policy", solutions_log=solutions_log
                )
                task_demonstration_performance_over_tasks.append(task_demonstration_performance)
                test_performance_over_tasks.append(test_performance)
            if self.add_policy_samples:
                self.add_tasks_to_buffer(generated_tasks, iteration_id)
            return (
                performance_over_tasks(task_demonstration_performance_over_tasks),
                performance_over_tasks(test_performance_over_tasks),
                solutions_log,
            )
        return 0, 0, []

    def evaluate_programs(
        self, programs, task, solutions=None, generated_tasks=None, iteration_id=None
    ):
        if len(programs) > 0:
            programs = list(set(programs))
            initial_states = [
                training_example["input"]
                for training_example in task.training_examples[: self.n_examples]
            ]
            goal_states = [
                training_example["output"]
                for training_example in task.training_examples[: self.n_examples]
            ]
            rewards, terminal_states = self.evaluator.evaluate_actions(
                programs, initial_states, goal_states, max_state_size=self.max_state_size
            )
            solutions, generated_tasks = self.evaluator.create_tasks_and_solutions(
                action_list=programs,
                terminal_states=terminal_states,
                initial_states=initial_states,
                solutions=solutions,
                rewards=rewards,
                inference_task=task,
                generated_tasks=generated_tasks,
                iteration_id=iteration_id,
                task_name_extension="__",
                max_grid_size=self.max_grid_size,
            )
            task_demonstration_performance, test_performance = calculate_performance(
                solutions, task_key=task.task_key
            )
            return solutions, generated_tasks, task_demonstration_performance, test_performance
        else:
            return {}, {}, 0, 0
