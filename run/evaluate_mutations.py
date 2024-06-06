# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import concurrent.futures
import cProfile
import csv
import json
import os
import random
import time
from typing import Any

import hydra
import lightning.pytorch as pl
import numpy as np

from codeit.agent import calculate_performance
from codeit.augment.mutate_grid import valid_grid
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *
from codeit.policy.inference import abs_grid_distance, execute_candidate_program
from codeit.task import get_tasks_with_keys, program_to_program_lines
from codeit.utils import transform_to_function
from run.run_mutate import get_valid_examples


def reduce_tasks(tasks, n):
    reduced_tasks = {}
    for i, task_key in enumerate(list(tasks.keys())[:n]):
        reduced_tasks[task_key] = tasks[task_key]
        if i == 0:
            reduced_tasks[task_key][
                "program"
            ] = "def solve_6150a2bd(I):\n    O = rot180(I)\n    return O\n"
    print(f"returning {len(reduced_tasks)} tasks")
    return reduced_tasks


def get_metrics(terminal_states, goal_states):
    task_demonstration_performance = []
    distance = []
    for i in range(len(goal_states)):
        task_demonstration_performance.append(goal_states[i] == terminal_states[i])
        if valid_grid(terminal_states[i]):
            distance.append(abs_grid_distance(goal_states[i], terminal_states[i]))
        else:
            distance.append(1)
    return task_demonstration_performance, distance


def get_test_performance(test_examples, program, max_state_size, sig_alarm):
    test_performance = []
    for example in test_examples:
        program_input = example["input"]
        program_output = execute_candidate_program(
            program_string=program,
            program_input=program_input,
            max_state_size=max_state_size,
            sig_alarm=sig_alarm,
        )
        test_performance.append(program_output == example["output"])
    return test_performance


def process_task(
    task_dict,
    inference_tasks,
    programs,
    return_test_performance=False,
    max_state_size=1_000,
    only_one_search_task=True,
    sig_alarm=True,
):
    results = {}
    solutions = {}
    program_lines = program_to_program_lines(task_dict["program"])
    if program_lines not in programs:
        print(f"processing program: {program_lines} in task: {task_dict['task_key']}")
        chosen_inference_task = random.randint(0, len(inference_tasks) - 1)
        for j, inference_task in enumerate(inference_tasks.values()):
            new_task_key = task_dict["task_key"] + "_" + str(j)
            new_program = transform_to_function(
                input_str=program_lines, function_name=f"solve_{new_task_key}"
            )
            examples = inference_task["training_examples"]
            initial_states = [example["input"] for example in examples]
            goal_states = [example["output"] for example in examples]
            valid_examples, terminal_states = get_valid_examples(
                program=program_lines,
                initial_states=initial_states,
                return_terminal_states=True,
                max_grid_size=max_state_size,
                sig_alarm=sig_alarm,
            )
            if len(valid_examples) > 0:
                task_demo_performance, distance = get_metrics(terminal_states, goal_states)
                if return_test_performance:
                    test_performance = get_test_performance(
                        inference_task["test_examples"],
                        program_lines,
                        max_state_size,
                        sig_alarm=sig_alarm,
                    )
                else:
                    test_performance = None
                new_task = {
                    "program": new_program,
                    "training_examples": valid_examples,
                    "test_examples": [],
                    "task_key": new_task_key,
                    "parent_key": inference_task["task_key"],
                    "extra_info": {
                        "distance": np.mean(distance),
                        "task_demonstration_performance": np.mean(task_demo_performance),
                        "test_performance": np.mean(test_performance),
                    },
                }
                if only_one_search_task:
                    if j == chosen_inference_task:
                        results[new_task_key] = new_task
                else:
                    results[new_task_key] = new_task
                if np.mean(task_demo_performance) > 0:
                    if inference_task["task_key"] in solutions:
                        solutions[inference_task["task_key"]][program_lines] = {
                            "program": program_lines,
                            "task_demonstration_performance": np.mean(task_demo_performance),
                            "test_performance": np.mean(test_performance),
                        }
                    else:
                        solutions[inference_task["task_key"]] = {
                            program_lines: {
                                "program": program_lines,
                                "task_demonstration_performance": np.mean(task_demo_performance),
                                "test_performance": np.mean(test_performance),
                            }
                        }
    return {"results": results, "solutions": solutions}


def gather_solutions(solutions_chunk, solutions):
    for inference_key in solutions_chunk.keys():
        for solution in solutions_chunk[inference_key].values():
            if inference_key in solutions:
                solutions[inference_key][solution["program"]] = solution
            else:
                solutions[inference_key] = {solution["program"]: solution}
    return solutions


def get_search(
    generated_tasks_train_chunk,
    inference_tasks,
    programs,
    parallel=True,
    n_processes=None,
    return_test_performance=False,
    only_one_search_task=True,
    sig_alarm=True,
    max_state_size=1_000,
):
    if parallel:
        return get_search_parallel(
            generated_tasks_train_chunk,
            inference_tasks,
            n_processes,
            programs,
            return_test_performance=return_test_performance,
            only_one_search_task=only_one_search_task,
            sig_alarm=sig_alarm,
            max_state_size=max_state_size,
        )
    else:
        return get_search_sequential(
            generated_tasks_train_chunk,
            inference_tasks,
            programs,
            return_test_performance=return_test_performance,
            only_one_search_task=only_one_search_task,
            sig_alarm=sig_alarm,
            max_state_size=max_state_size,
        )


def get_search_sequential(
    generated_tasks_train_chunk,
    inference_tasks,
    programs,
    return_test_performance,
    only_one_search_task,
    max_state_size=1000,
    sig_alarm=True,
):
    generated_tasks_search_chunk = {}
    solutions = {}
    for task_dict in generated_tasks_train_chunk.values():
        results = process_task(
            task_dict,
            inference_tasks,
            programs,
            return_test_performance,
            max_state_size,
            only_one_search_task,
            sig_alarm=sig_alarm,
        )
        generated_tasks_search_chunk.update(results["results"])
        solutions = gather_solutions(results["solutions"], solutions)
        print(f"completed task: {task_dict['task_key']}")
    return generated_tasks_search_chunk, solutions


def get_search_parallel(
    generated_tasks_train_chunk,
    inference_tasks,
    n_processes,
    programs,
    return_test_performance=False,
    max_state_size=1000,
    only_one_search_task=True,
    sig_alarm=True,
):
    if n_processes is None:
        n_processes = os.cpu_count() - 6
    generated_tasks_search_chunk = {}
    solutions = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {
            executor.submit(
                process_task,
                task,
                inference_tasks,
                programs,
                return_test_performance,
                max_state_size,
                only_one_search_task,
                sig_alarm,
            ): task["task_key"]
            for task in generated_tasks_train_chunk.values()
        }
        for future in concurrent.futures.as_completed(futures):
            chunk_index = futures[future]
            try:
                result = future.result()
                print(f"completed task: {chunk_index}")
                generated_tasks_search_chunk.update(result["results"])
                solutions = gather_solutions(result["solutions"], solutions)
            except Exception as e:
                print(f"task{chunk_index} raised an exception: {e}")
    return generated_tasks_search_chunk, solutions


def inference_tasks_to_dict(inference_tasks):
    new_dicts = {}
    for task in inference_tasks.values():
        new_dicts[task.task_key] = task.to_dict()
    return new_dicts


def calculate_performance_over_inference_tasks(solutions, inference_keys):
    task_demonstration_performance_list = []
    test_performance_list = []
    for task_key in inference_keys:
        task_demonstration_performance, test_performance = calculate_performance(
            solutions, task_key
        )
        task_demonstration_performance_list.append(task_demonstration_performance)
        test_performance_list.append(test_performance)
    return {
        "task_demonstration_performance": task_demonstration_performance_list,
        "test_performance": test_performance_list,
    }


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:
    pl.seed_everything(config.seed)

    # load tasks
    with open(config.data.split_keys_path, "r") as f:
        split_keys = json.load(f)

    val_keys = split_keys["val"][: config.data.n_val]
    test_keys = split_keys["test"][: config.data.n_test]

    if config.baseline.validation_set:

        inference_tasks = get_tasks_with_keys(
            file_path=f"{config.data.training_data_dir}", keys=val_keys
        )

    elif config.baseline.test_set:

        inference_tasks = get_tasks_with_keys(
            file_path=f"{config.data.evaluation_data_dir}", keys=test_keys
        )

    # baseline params
    n_samples = config.baseline.n_samples
    save_chunk_size = config.baseline.save_chunk_size  # used for saving
    # assert save_chunk_size%len(inference_tasks)==0 # must be in intervals of 89 so we can compare to n_p
    log_chunk_size = config.baseline.log_chunk_size  # used for logging
    # assert log_chunk_size%len(inference_tasks)==0 # must be in intervals of 89 so we can compare to n_p

    log_chunks = range(log_chunk_size, n_samples + log_chunk_size, log_chunk_size)

    max_save_chunk = config.baseline.max_save_chunk
    max_state_size = config.baseline.max_state_size
    n_processes = config.baseline.n_processes
    sig_alarm = config.baseline.sig_alarm

    generated_tasks_train_aggregated_chunk = {}
    generated_tasks_search_aggregated_chunk = {}

    programs = set()
    solutions = {}

    inference_tasks = inference_tasks_to_dict(inference_tasks=inference_tasks)

    for chunk in log_chunks:
        start_time = time.time()
        while not os.path.exists(f"{config.baseline.tasks_file}{chunk}.json"):
            print("sleeping one hour")
            time.sleep(3600)
        if os.path.exists(f"{config.baseline.tasks_file}{chunk}.json"):
            with open(f"{config.baseline.tasks_file}{chunk}.json", "r") as f:
                generated_tasks_train_chunk = json.load(f)
            if config.baseline.max_chunk < len(generated_tasks_train_chunk):
                generated_tasks_train_chunk = reduce_tasks(
                    generated_tasks_train_chunk, config.baseline.max_chunk
                )
        else:
            raise Exception("File does not exist")

        print("creating search tasks")
        generated_tasks_search_chunk, solutions_chunk = get_search(
            generated_tasks_train_chunk,
            inference_tasks,
            n_processes=n_processes,
            parallel=config.baseline.parallel,
            programs=programs,
            return_test_performance=config.baseline.calculate_performance,
            max_state_size=max_state_size,
            sig_alarm=sig_alarm,
            only_one_search_task=config.baseline.only_one_search_task,
        )

        for task_dict in generated_tasks_search_chunk.values():
            programs.add(program_to_program_lines(task_dict["program"]))

        print(f"aggregated tasks over log chunks")
        for task_dict in generated_tasks_train_chunk.values():
            generated_tasks_train_aggregated_chunk[task_dict["task_key"]] = task_dict

        for task_dict in generated_tasks_search_chunk.values():
            generated_tasks_search_aggregated_chunk[task_dict["task_key"]] = task_dict

        print(f"chunk {chunk}: {len(generated_tasks_train_aggregated_chunk)} mutated train tasks")
        print(f"chunk {chunk}: {len(generated_tasks_search_aggregated_chunk)} mutated search tasks")

        if chunk % save_chunk_size == 0 and chunk < max_save_chunk:
            if config.baseline.calculate_performance:
                print("gathering solutions")
                solutions = gather_solutions(solutions_chunk, solutions)
                print("getting metrics over solutions")
                if solutions:
                    performance_by_task = calculate_performance_over_inference_tasks(
                        solutions, inference_tasks.keys()
                    )
                    test_performance = np.mean(performance_by_task["test_performance"])
                    task_demonstration_performance = np.mean(
                        performance_by_task["task_demonstration_performance"]
                    )
                else:
                    test_performance = 0
                    task_demonstration_performance = 0
                print(f"********** n_samples {chunk} % solved: {test_performance}***********")
                with open(f"{config.run_dir}/mutation_performance.csv", "a+", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((chunk, test_performance))
                if chunk in [9600, 19200, 979200, 988800]:
                    with open(
                        f"{config.run_dir}/mutation_solutions_{chunk}.json", "a+", newline=""
                    ) as f:
                        json.dump(solutions, f)

            print("saving tasks")
            if config.baseline.save_train_tasks:
                with open(f"{config.run_dir}/mutated_tasks_train_{chunk}.json", "w+") as f:
                    json.dump(generated_tasks_train_aggregated_chunk, f)
            if config.baseline.save_search_tasks:
                with open(f"{config.run_dir}/mutated_tasks_search_{chunk}.json", "w+") as f:
                    json.dump(generated_tasks_search_aggregated_chunk, f)

            generated_tasks_train_aggregated_chunk = {}
            generated_tasks_search_aggregated_chunk = {}

            print("saved tasks")
        print(f"took {time.time()-start_time} seconds")


if __name__ == "__main__":
    main()
