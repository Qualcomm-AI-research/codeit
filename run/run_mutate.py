# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import os
import random
import traceback
from typing import Any

import hydra
import lightning.pytorch as pl

from codeit.augment.genetic import TaskEvolver
from codeit.augment.mutate_grid import valid_grid
from codeit.augment.program_sampler import ProgramSampler
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *
from codeit.dsl.primitives import PRIMITIVE_CONSTANTS, PRIMITIVE_FUNCTIONS
from codeit.policy.inference import execute_candidate_program
from codeit.task import Task, from_dict, get_tasks_with_keys, program_to_program_lines
from codeit.utils import get_num_pixels


def get_valid_examples(
    program,
    initial_states,
    max_grid_size=10_000_000_000,
    return_terminal_states=False,
    sig_alarm=False,
):
    examples = []
    terminal_states = []
    for j, initial_state in enumerate(initial_states):
        terminal_state = execute_candidate_program(
            program_to_program_lines(program), program_input=initial_state, sig_alarm=sig_alarm
        )
        terminal_states.append(terminal_state)
        if valid_grid(terminal_state):
            if get_num_pixels(terminal_state) < max_grid_size:
                examples.append({"input": initial_state, "output": terminal_state})
    if return_terminal_states:
        return examples, terminal_states
    else:
        return examples


def random_sample(program_sampler, train_tasks, sample_id, iteration_id=0):
    task = train_tasks[random.choice(list(train_tasks.keys()))]
    try:
        program_name = f"solve_{task.task_key}_it{iteration_id}_r_{sample_id}"
        program_sample = program_sampler.sample(
            program_name=program_name, I=task.training_examples[0]["input"]
        )
        generated_task = program_to_task(
            program=program_sample, parent_task=task, program_name=program_name
        )
        return generated_task
    except:
        traceback.print_exc()


def load_tasks(task_dir):
    task_files = [
        file
        for file in os.listdir(task_dir)
        if file.startswith("genetic_") and file.endswith(".json")
    ]
    tasks = {}
    ids = []
    for task_file in task_files:
        with open(task_dir + "/" + task_file, "r") as f:
            tasks_dict = json.load(f)
        for task_dict in tasks_dict.values():
            tasks[task_dict["task_key"]] = from_dict(task_dict)
        ids.append(int(task_file.split("_")[-1].strip(".json")))
    return tasks, max(ids)


def program_to_task(program, parent_task, program_name):
    initial_states = [example["input"] for example in parent_task.training_examples]
    valid_training_states = get_valid_examples(
        program=program, initial_states=initial_states, max_grid_size=1_000
    )
    task_id = program_name.strip("solve_")
    if valid_training_states:
        new_task = Task(
            program=program,
            training_examples=valid_training_states,
            test_examples=[],
            task_key=task_id,
        )
    else:
        new_task = None
    return new_task


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:
    pl.seed_everything(config.seed)
    print(config.baseline.resume)

    # load tasks
    with open(config.data.split_keys_path, "r") as f:
        split_keys = json.load(f)

    val_keys = split_keys["val"][: config.data.n_val]
    train_keys = split_keys["train"][: config.data.n_train]

    # baseline params
    n_samples = config.baseline.n_samples
    log_chunk_size = config.baseline.log_chunk_size  # used for logging

    mutation_baseline = config.baseline.run_mutated

    sig_alarm = config.baseline.sig_alarm

    if config.baseline.validation_set:
        train_tasks = get_tasks_with_keys(
            file_path=f"{config.data.training_data_dir}", keys=train_keys
        )
    elif config.baseline.test_set:
        train_tasks = get_tasks_with_keys(
            file_path=f"{config.data.training_data_dir}", keys=list(train_keys) + list(val_keys)
        )

    if config.baseline.resume:
        tasks, last_id = load_tasks(task_dir=config.run_dir)
        last_id += log_chunk_size
        for task in train_tasks.values():
            tasks[task.task_key] = task
    else:
        tasks = train_tasks
        last_id = 0

    chunks = range(last_id + log_chunk_size, n_samples + log_chunk_size, log_chunk_size)
    print(f"initialising mutation with {len(tasks)} tasks")
    depth_one = config.baseline.depth_one
    task_evolver = TaskEvolver(
        tasks=tasks,
        primitive_functions=PRIMITIVE_FUNCTIONS,
        primitive_constants=PRIMITIVE_CONSTANTS,
        phi_program=1,
        phi_var=config.generator.phi_var,
        phi_func=config.generator.phi_func,
        phi_arg=config.generator.phi_arg,
        data_path=config.data.data_dir,
        iteration_id=str(0),
        select_from_arc=depth_one,
        sig_alarm=sig_alarm,
        load_inferrer=config.baseline.load_inferrer,
    )
    if not mutation_baseline:
        program_sampler = ProgramSampler(data_path=config.data.data_dir)

    for chunk in chunks:
        print(f"evolving chunk: {chunk}")
        generated_tasks_train = {}
        for i in range(chunk, chunk + log_chunk_size):
            # genetic samples
            if mutation_baseline:
                new_task = task_evolver.evolve_with_timeout(sig_alarm=config.generator.sig_alarm)
                if new_task:
                    new_task.parent_key = new_task.task_key.split("_")[0]
                    generated_tasks_train[new_task.task_key] = new_task.to_dict()
                else:
                    pass
            else:
                new_task = None
                while new_task is None:
                    new_task = random_sample(
                        program_sampler=program_sampler,
                        train_tasks=train_tasks,
                        sample_id=i,
                        iteration_id=0,
                    )
                    if new_task:
                        generated_tasks_train[new_task.task_key] = new_task.to_dict()

        # log every chunk
        print(f"saving chunk: {chunk} with {len(generated_tasks_train)} tasks")
        with open(f"{config.run_dir}/mutated_tasks_train_{chunk}.json", "w+") as f:
            json.dump(generated_tasks_train, f)
        with open(f"{config.run_dir}/log_{chunk}.txt", "a+") as f:
            f.write(f"no. mutated tasks: {len(task_evolver.task_population)}\n")


if __name__ == "__main__":
    main()
