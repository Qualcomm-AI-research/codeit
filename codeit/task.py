# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import inspect
import json
import math
import random
from collections import defaultdict
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from codeit.dsl.arc_types import *
from codeit.dsl.solvers import *
from codeit.utils import get_grid_size

# Create a custom colormap with adjusted green shades
colors = [(0, "blue"), (0.35, "blue"), (0.5, "green"), (0.75, "yellow"), (1, "red")]
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = "custom_jet"
# Create the colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def program_to_program_lines(program_string):
    program_lines = []
    for line in program_string.split("\n"):
        if line.count("=") == 1:
            program_lines.append(line.strip())
    return "\n".join(list(program_lines))


def get_data(path):
    data = {}
    for fn in os.listdir(path):
        with open(f"{path}/{fn}") as f:
            data[fn.rstrip(".json")] = json.load(f)
    ast = lambda g: tuple(tuple(r) for r in g)
    return {
        "train": {
            k: [
                {
                    "input": ast(e["input"]),
                    "output": ast(e["output"]),
                }
                for e in v["train"]
            ]
            for k, v in data.items()
        },
        "test": {
            k: [
                {
                    "input": ast(e["input"]),
                    "output": ast(e["output"]),
                }
                for e in v["test"]
            ]
            for k, v in data.items()
        },
    }


def get_tasks_with_keys(file_path, keys):
    task_population = {}
    print("loading tasks from file path: " + file_path)
    for file in tqdm(keys):
        task = Task.from_json(file_path + file + ".json")
        task_population[task.task_key] = task
    return task_population


def from_dict(task_data):
    if "parent_key" in task_data:
        parent_key = task_data["parent_key"]
    else:
        parent_key = None
    test_examples = [
        {"input": list_to_tuple(example["input"]), "output": list_to_tuple(example["output"])}
        for example in task_data["test_examples"]
    ]
    training_examples = [
        {"input": list_to_tuple(example["input"]), "output": list_to_tuple(example["output"])}
        for example in task_data["training_examples"]
    ]
    return Task(
        program=task_data["program"],
        training_examples=training_examples,
        test_examples=test_examples,
        task_key=task_data["task_key"],
        extra_info=task_data["extra_info"],
        parent_key=parent_key,
    )


class Task:
    def __init__(
        self, program, training_examples, test_examples, task_key, parent_key=None, extra_info=""
    ):
        self.program = program
        if program:
            self.program_lines = program_to_program_lines(program)
        else:
            self.program_lines = ""
        self.task_key = task_key
        self.training_examples = training_examples
        self.test_examples = test_examples
        self.training_states = None
        self.training_actions = None
        self.parent_key = parent_key
        if extra_info:
            self.extra_info = extra_info
        else:
            self.extra_info = {}

    def to_dict(self):
        task_data = {
            "program": self.program,
            "training_examples": self.training_examples,
            "test_examples": self.test_examples,
            "task_key": self.task_key,
            "parent_key": self.parent_key,
            "extra_info": self.extra_info if self.extra_info else "",
        }
        return task_data

    def to_json(self, file_path):
        with open(file_path, "w") as json_file:
            json.dump(self.to_dict(), json_file)

    @staticmethod
    def from_json(file_path):
        with open(file_path, "r") as json_file:
            task_data = json.load(json_file)
        return from_dict(task_data)

    def display_task(self):
        num_training_examples = len(self.training_examples)
        num_test_examples = len(self.test_examples)
        fig, axes = plt.subplots(
            2,
            2 * max(num_training_examples, num_test_examples),
            figsize=(4 * max(num_training_examples, num_test_examples), 8),
            gridspec_kw={"wspace": 0.1, "hspace": 0.3},
        )  # add gridspec_kw parameter to adjust spacing
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        for i, example in enumerate(self.training_examples):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])
            axes[0, 2 * i].imshow(input_grid, cmap=cm, vmin=0, vmax=9)
            axes[0, 2 * i].set_title(f"Train {i + 1}: Input")
            axes[0, 2 * i + 1].imshow(output_grid, cmap=cm, vmin=0, vmax=9)
            axes[0, 2 * i + 1].set_title(f"Train {i + 1}: Output")
        for i, example in enumerate(self.test_examples):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])
            axes[1, 2 * i].imshow(input_grid, cmap=cm, vmin=0, vmax=9)
            axes[1, 2 * i].set_title(f"Test {i + 1}: Input")
            axes[1, 2 * i + 1].imshow(output_grid, cmap=cm, vmin=0, vmax=9)
            axes[1, 2 * i + 1].set_title(f"Test {i + 1}: Output")
        fig.suptitle(f"Task {self.task_key}", fontsize=16)
        plt.show()

    def display_example(self, example_type="training", i=0):
        if example_type == "training":
            example = self.training_examples[i]
        elif example_type == "test":
            example = self.test_examples[i]
        else:
            print("Invalid example_type. Please choose either 'training' or 'test'.")
            return
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        ax1.imshow(input_grid, cmap=cm, vmin=0, vmax=9)
        ax1.set_title(f"{example_type.capitalize()} {i + 1}: Input")
        ax2.imshow(output_grid, cmap=cm, vmin=0, vmax=9)
        ax2.set_title(f"{example_type.capitalize()} {i + 1}: Output")
        fig.suptitle(f"Task {self.task_key}", fontsize=16)
        plt.show()
        return plt


def list_to_tuple(listoflists):
    ast = lambda g: tuple(tuple(r) for r in g)
    return ast(listoflists)


def make_tasks_from_dir(raw_dir: str, processed_dir: str, train: bool):
    data = get_data(raw_dir)
    task_keys = list(data["train"].keys())
    tasks = {}
    for task_key in task_keys:
        program = inspect.getsource(eval(f"solve_{task_key}")) if train else None
        training_examples = sort_examples(data["train"][task_key])
        test_examples = sort_examples(data["test"][task_key])
        task = Task(
            program=program,
            training_examples=training_examples,
            test_examples=test_examples,
            task_key=task_key,
        )
        task.to_json(processed_dir + task_key + ".json")
        tasks[task.task_key] = task
    return tasks


def make_tasks(config):
    training_path = config.data.training_data_dir
    evaluation_path = config.data.evaluation_data_dir
    for path in [training_path, evaluation_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    training_tasks = make_tasks_from_dir(
        raw_dir=config.data.raw_training_data_dir,
        processed_dir=config.data.training_data_dir,
        train=True,
    )
    evaluation_tasks = make_tasks_from_dir(
        raw_dir=config.data.raw_evaluation_data_dir,
        processed_dir=config.data.evaluation_data_dir,
        train=False,
    )

    train_keys, val_keys = split_tasks(training_tasks, config)

    split_keys = {"train": train_keys, "val": val_keys, "test": list(evaluation_tasks.keys())}

    if os.path.exists(config.data.split_keys_path):
        print("split keys json already exists")
    else:
        with open(config.data.split_keys_path, "w") as f:
            json.dump(split_keys, f)


def sort_examples(examples):
    return sorted(examples, key=lambda x: get_grid_size(x["input"] + x["output"]))


def split_tasks(tasks, config):
    train_keys = []
    val_keys = []
    split = config.data.train_split
    program_length_task_list = [
        (task.program_lines.count("="), task.task_key) for task in tasks.values()
    ]
    random.shuffle(program_length_task_list)
    program_ordered_length_task_list = sorted(program_length_task_list, key=lambda x: x[0])
    grouped = defaultdict(list)
    for a, b in program_ordered_length_task_list:
        grouped[a].append(b)
    tasks_by_length = dict(grouped)
    for task_length in tasks_by_length.keys():
        if len(grouped[task_length]) > 2:
            n_val = math.ceil(len(grouped[task_length]) * (1 - split))
            n_train = len(grouped[task_length]) - n_val
        else:
            n_train = 2
        train_keys += grouped[task_length][:n_train]
        val_keys += grouped[task_length][n_train:]
    print(f"num train: {len(train_keys)} num val: {len(val_keys)}")
    return train_keys, val_keys
