# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import cProfile
import csv
import gc
import json
import os
import time
from typing import Any

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from codeit.agent import Agent, calculate_performance
from codeit.callbacks import HfModelCheckpoint
from codeit.exit_data_module import ExItDataModule
from codeit.hf_model_module import HFModule
from codeit.task import from_dict
from codeit.utils import get_num_pixels


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


def filter_by_inference_keys(tasks, inference_keys):
    task_keys = [
        s for s in tasks.keys() if not any(s.startswith(prefix) for prefix in inference_keys)
    ]
    filtered_tasks = {}
    for key in task_keys:
        filtered_tasks[key] = tasks[key]
    return filtered_tasks


def initialise_csv_writer(config):
    results_dir = config.run_dir + "/performance.csv"
    with open(results_dir, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "meta_iteration",
                "cumulative_performance",
                "performance",
                "step",
                "num_mutated_tasks",
                "num_policy_tasks",
            ]
        )


def write_performance(
    config,
    meta_iteration,
    cumulative_performance,
    performance,
    step,
    num_mutated_tasks,
    num_policy_tasks,
):
    results_dir = config.run_dir + "/performance.csv"
    with open(results_dir, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                meta_iteration,
                cumulative_performance,
                performance,
                step,
                num_mutated_tasks,
                num_policy_tasks,
            ]
        )


def get_num_programs(agent, mode="policy"):
    return copy.copy(len(agent.replay_buffer.programs[mode]))


def get_num_tasks(agent, mode="policy"):
    return copy.copy(len(agent.replay_buffer.entries[mode]))


def filter_and_load_mutated_tasks(mutated_train_tasks):
    filtered_mutated_tasks = {}
    for task in mutated_train_tasks.values():
        too_big = False
        for training_example in task["training_examples"]:
            if get_num_pixels(training_example["output"]) > 1_000:
                too_big = True
        if not too_big:
            filtered_mutated_tasks[task["task_key"]] = from_dict(task)
    print(f"removed {len(mutated_train_tasks)-len(filtered_mutated_tasks)} tasks from genetic!!!")
    return filtered_mutated_tasks


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:

    print("\n" + "=" * 10, "Configuration", "=" * 10)

    pl.seed_everything(config.seed)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    config.trainer.profiler = None
    solutions_interval = config.exit.solutions_interval

    agent = Agent(config=config)

    initialise_csv_writer(config)

    logger = TensorBoardLogger(
        save_dir=config.run_dir, version=None, name="tensorboard", default_hp_metric=False
    )

    pl_module = HFModule(config)
    checkpoint_callback = HfModelCheckpoint(
        dirpath=f"{config.model.models_dir}/",
        save_top_k=0,
        save_last=config.model.save_last,
        config=config,
    )
    callbacks = [checkpoint_callback]

    print("initialising trainer")
    trainer = pl.Trainer(**config.trainer, logger=logger, callbacks=callbacks)

    # load tasks for ablations
    if config.ablation.used:
        print("preparing ablations")
        assert config.exit.add_policy_samples == False
        print(
            f"sample values from {config.ablation.start_value} to {config.ablation.final_value} at interval {config.ablation.mutation_interval}"
        )
        sample_values = list(
            range(
                config.ablation.start_value,
                config.ablation.final_value + config.ablation.mutation_interval,
                config.ablation.mutation_interval,
            )
        )
        mutated_tasks_files = [
            config.ablation.tasks_file + "_" + str(value) + ".json" for value in sample_values
        ]
        mutated_tasks_list = []
        for file in mutated_tasks_files:
            print(f"loading file {file}")
            with open(file, "r") as f:
                mutated_tasks = json.load(f)
            print(f"loaded {len(mutated_tasks)} tasks")
            print(f"filtering tasks from file {file}")
            mutated_tasks = filter_and_load_mutated_tasks(mutated_tasks)
            if not config.final_experiments:
                print(f"filtering out validation keys from tasks from file {file}")
                mutated_tasks = filter_by_inference_keys(
                    mutated_tasks, agent.inference_tasks.keys()
                )
            print(f"adding {len(mutated_tasks)} tasks to task list")
            mutated_tasks_list.append(mutated_tasks)
        if len(mutated_tasks_list) != config.exit.n_iters:
            raise Exception(
                f"mutated task list length: {len(mutated_tasks_list)} num iters: {config.exit.n_iters}"
            )

    print(
        f'first inference example inputs {agent.replay_buffer.tokenizer.decode(agent.inference_dataset["input_ids"][0])}'
    )
    print(
        f'first inference example labels {agent.replay_buffer.tokenizer.decode(agent.inference_dataset["labels"][0])}'
    )

    for n_iter in range(0, config.exit.n_iters):

        if config.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        print(f"******** iteration {n_iter} ********")

        # add mutated tasks to buffer
        if config.ablation.used:
            agent.add_tasks_to_buffer(
                tasks=mutated_tasks_list[n_iter], iteration_id=n_iter, mode="mutated"
            )

        # sampling replay buffer

        data_module = ExItDataModule(config=config, replay_buffer=agent.replay_buffer)
        data_module.setup()

        trainer.logger.log_metrics(
            {"training set size": len(data_module.train_dataset)}, trainer.global_step
        )

        print(
            f'first training example inputs {agent.replay_buffer.tokenizer.decode(data_module.train_dataset["input_ids"][0])}'
        )
        print(
            f'first training example labels {agent.replay_buffer.tokenizer.decode([token for token in data_module.train_dataset["labels"][0] if token !=-100])}'
        )

        # train
        t = time.time()
        trainer.fit(pl_module, datamodule=data_module)
        trainer.logger.log_metrics({"train_time": time.time() - t}, trainer.global_step)

        # sampling policy
        t = time.time()
        num_programs = get_num_programs(agent, mode="policy")
        num_tasks = get_num_tasks(agent, mode="policy")
        model = trainer.model.transformer.eval().to("cuda:0")
        task_demonstration_performance, test_performance, solutions_log = agent.sample_policy_tasks(
            iteration_id=n_iter, model=model
        )

        torch.cuda.empty_cache()
        gc.collect()

        # log policy metrics
        cumulative_performance = calculate_performance_over_inference_tasks(
            agent.solutions["policy"]["seen_example"], agent.inference_tasks.keys()
        )
        cumulative_test_performance = np.mean(cumulative_performance["test_performance"])

        trainer.logger.log_metrics(
            {
                "sampling_time/policy": time.time() - t,
                "task_demonstration/policy/performance": task_demonstration_performance,
                "test/policy/performance": test_performance,
                "task_demonstration/policy/cumulative_performance": len(
                    agent.solutions["policy"]["task_demonstration"]
                )
                / len(agent.inference_tasks),
                "test/policy/cumulative_performance": cumulative_test_performance,
                "delta_programs/policy": get_num_programs(agent, mode="policy") - num_programs,
                "delta_tasks/policy": get_num_tasks(agent, mode="policy") - num_tasks,
            },
            trainer.global_step,
        )
        log = {}
        log["replay_buffer/num_policy_tasks"] = get_num_tasks(agent, mode="policy")
        log["replay_buffer/num_mutated_tasks"] = get_num_tasks(agent, mode="mutated")
        log["replay_buffer/num_policy_programs"] = get_num_programs(agent, mode="policy")
        log["replay_buffer/num_mutated_programs"] = get_num_programs(agent, mode="mutated")
        trainer.logger.log_metrics(log, trainer.global_step)

        for text_solution in solutions_log:
            trainer.logger.experiment.add_text(
                "policy: " + text_solution[6:].split(" ")[0], text_solution, trainer.global_step
            )

        write_performance(
            config,
            meta_iteration=n_iter,
            cumulative_performance=np.mean(cumulative_test_performance),
            performance=test_performance,
            step=trainer.global_step,
            num_mutated_tasks=len(agent.replay_buffer.entries["mutated"]),
            num_policy_tasks=len(agent.replay_buffer.entries["policy"]),
        )

        agent.replay_buffer.current_iteration += 1
        trainer.fit_loop.max_epochs += config.trainer.max_epochs

        if n_iter % solutions_interval == 0:
            agent.save_solutions()

        if config.profile:
            profiler.disable()
            profiler.dump_stats(f"{config.run_dir}/profile_data_{n_iter}.prof")

    return trainer, data_module, pl_module


if __name__ == "__main__":
    main()
