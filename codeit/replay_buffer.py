# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools
import json

import numpy as np

from codeit.policy.tokenize import TextEncoder
from codeit.policy.tokenize import tokenize_task as tokenize_task_seq_2_seq
from codeit.task import Task, from_dict
from codeit.utils import get_num_pixels, get_tokenizer


def normalize(p):
    if sum(p) == 0:
        return [1 / len(p)] * len(p)
    else:
        p = np.array(p)
        p /= p.sum()
    return p


def load_mutated_tasks(file_name, start_id, end_id, id_interval, val_keys=None):
    print("loaded tasks from file", file_name)
    if file_name:
        file_ids = range(start_id, end_id + id_interval, id_interval)
        tasks = {}
        for file_id in file_ids:
            with open(f"{file_name}{file_id}.json", "r") as f:
                task_dict = json.load(f)
            if val_keys:
                filtered_keys = [
                    s
                    for s in task_dict.keys()
                    if not any(s.startswith(prefix) for prefix in val_keys)
                ]
            else:
                filtered_keys = task_dict.keys()
            for task_key in filtered_keys:
                task = task_dict[task_key]
                too_big = False
                for training_example in task["training_examples"]:
                    if get_num_pixels(training_example["output"]) > 1_000:
                        too_big = True
                if not too_big:
                    tasks[task["task_key"]] = task
        return tasks
    else:
        return {}


class Buffer:
    def __init__(self, config, train_tasks={}):
        self.entries = {"policy": {}, "mutated": {}}
        self.capacity = config.replay_buffer.capacity
        self.reduced_size = round(
            config.replay_buffer.capacity * (1 - config.replay_buffer.reduce_percentage)
        )

        self.performance_penalty = config.replay_buffer.performance_penalty
        self.length_penalty = config.replay_buffer.length_penalty
        self.time_penalty = config.replay_buffer.time_penalty
        self.distance_penalty = config.replay_buffer.distance_penalty
        self.length_normalizer = config.replay_buffer.length_normalizer
        self.age_normalizer = config.replay_buffer.age_normalizer

        self.n_examples = config.data.dataloader.tokenizer.n_examples
        self.input_state_max = config.data.dataloader.tokenizer.input_state_max
        self.max_decoder_tokens = config.model.max_length

        self.sparse = config.data.dataloader.tokenizer.sparse

        self.priority_in_mutated = config.replay_buffer.priority_in_mutated

        self.programs = {"policy": set(), "mutated": set()}

        self.tokenizer = get_tokenizer(config)
        self.tokenizer.decoder_start_token_id = config.model.decoder_start_token_id

        self.text_encoder = TextEncoder()

        self.size = {"policy": 0, "mutated": 0}
        self.current_iteration = 0

        self.tokenize_task = functools.partial(
            tokenize_task_seq_2_seq,
            sparse=self.sparse,
            tokenizer=self.tokenizer,
            n_examples=self.n_examples,
            input_state_max=self.input_state_max,
            max_tokens=self.max_decoder_tokens,
            text_encoder=self.text_encoder,
        )

        self.max_mutated_train_tasks = config.replay_buffer.max_mutated_train_tasks

        self.programs = {"policy": set(), "mutated": set()}

        if config.final_experiments:
            val_keys = None
        else:
            with open(config.data.split_keys_path, "r") as f:
                split_keys = json.load(f)
            val_keys = split_keys["val"]

        mutated_train_tasks = load_mutated_tasks(
            file_name=config.replay_buffer.mutated_train_tasks_file,
            start_id=config.replay_buffer.mutated_file_start,
            end_id=config.replay_buffer.mutated_file_end,
            id_interval=config.replay_buffer.mutated_file_interval,
            val_keys=val_keys,
        )

        self.initialise_buffer(train_tasks=train_tasks, mutated_train_tasks=mutated_train_tasks)

    def initialise_buffer(self, train_tasks, mutated_train_tasks):
        for task in train_tasks.values():
            task.parent_key = task.task_key
            task.extra_info["demonstration_performance"] = 1.0
            self.add(task=task, iteration_id=0, mode="mutated")
        print(f"added {len(train_tasks)} train tasks")
        for task in list(mutated_train_tasks.values())[: self.max_mutated_train_tasks]:
            task["extra_info"]["likelihood"] = 0.0
            self.add(task=from_dict(task), iteration_id=0, mode="mutated")
        print(f"added {len(mutated_train_tasks)} mutated train tasks")
        lengths = [
            len(entry["experience"]["input_ids"]) for entry in self.entries["mutated"].values()
        ]
        print(f"max train input length: {max(lengths)}")

    def add(self, task: Task, iteration_id: int, mode: str = "policy"):
        if self.size[mode] >= self.capacity:
            print("*** reducing replay buffer ***")
            self.reduce(mode=mode)
        unique_id = f"{task.parent_key}{task.program_lines}"
        self.programs[mode].add(task.program_lines)
        priority = self.get_priority(task, iteration_id, mode)
        experience = self.tokenize_task(task=task)
        self.entries[mode][unique_id] = {"experience": experience, "priority": priority}
        self.size[mode] += 1

    def reduce(self, mode):
        priorities = [entry["priority"] for entry in self.entries[mode].values()]
        priorities = sorted(priorities, reverse=True)
        threshold = priorities[self.reduced_size]
        new_size = 0
        entries_to_delete = []
        for unique_id in self.entries[mode]:
            if self.entries[mode][unique_id]["priority"] < threshold:
                entries_to_delete.append(unique_id)
            else:
                new_size += 1
        for unique_id in entries_to_delete:
            del self.entries[unique_id]
        self.size[mode] = new_size

    def add_sample(self, samples, sample):
        samples["input_ids"].append(sample["input_ids"])
        samples["attention_mask"].append(sample["attention_mask"])
        samples["labels"].append(sample["labels"])
        samples["task_id"].append(sample["task_id"])
        return samples

    def combine_experiences(self, policy_experiences, mutated_experiences):
        experiences = {}
        experiences["input_ids"] = (
            policy_experiences["input_ids"] + mutated_experiences["input_ids"]
        )
        experiences["attention_mask"] = (
            policy_experiences["attention_mask"] + mutated_experiences["attention_mask"]
        )
        experiences["labels"] = policy_experiences["labels"] + mutated_experiences["labels"]
        experiences["task_id"] = policy_experiences["task_id"] + mutated_experiences["task_id"]
        return experiences

    def sample(self, n_samples, mode):
        samples = {"input_ids": [], "attention_mask": [], "labels": [], "task_id": []}

        priorities = np.array([entry["priority"] for entry in self.entries[mode].values()])
        prob_distribution = normalize(priorities)

        num_entries = len(self.entries[mode])
        indices = np.array([])

        if n_samples < num_entries:
            indices = np.array(
                np.random.choice(num_entries, size=n_samples, p=prob_distribution, replace=False)
            )
        else:
            indices = np.array(range(0, num_entries))

        for index in indices:
            keys = list(self.entries[mode].keys())
            sample = self.entries[mode][keys[int(index)]]["experience"]
            samples = self.add_sample(samples=samples, sample=sample)

        return samples

    def sample_experiences(self, n_policy_experiences, n_mutated_experiences):
        if n_mutated_experiences > 0 and self.size["mutated"] > 0:
            mutated_experiences = self.sample(n_samples=n_mutated_experiences, mode="mutated")
        else:
            raise Exception("Initialise buffer")

        if n_policy_experiences > 0 and self.size["policy"] > 0:
            policy_experiences = self.sample(n_samples=n_policy_experiences, mode="policy")
            return self.combine_experiences(
                policy_experiences=policy_experiences, mutated_experiences=mutated_experiences
            )
        else:
            return mutated_experiences

    def get_priority(self, task, iteration_id, mode):

        total_penalty = (
            self.length_penalty
            + self.time_penalty
            + self.distance_penalty
            + self.performance_penalty
        )

        if total_penalty > 0:
            if mode == "mutated" and (not self.priority_in_mutated):
                return 0
            program_length = 1 - min(
                len(self.tokenizer.encode(task.program_lines)) / self.length_normalizer, 1
            )
            age = min(iteration_id / self.age_normalizer, 1)  # penalise older solutions
            if "distance" in task.extra_info:
                distance = 1 - task.extra_info["distance"]  # penalise large distances
            else:
                distance = 0.0

            if "task_demonstration_performance" in task.extra_info:
                task_demonstration_performance = task.extra_info["task_demonstration_performance"]
            else:
                task_demonstration_performance = 0.0

            assert 0 <= distance <= 1
            assert 0 <= task_demonstration_performance <= 1

            total_weighted_value = (
                program_length * self.length_penalty
                + age * self.time_penalty
                + distance * self.distance_penalty
                + task_demonstration_performance * self.performance_penalty
            )
            return total_weighted_value / total_penalty + 0.0000000000000000001
        else:
            return 0
