# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import os

from codeit.task import Task


def load_json(file_path):
    with open(file_path) as f:
        file = json.load(f)
    return file


def list_to_tuple(example_list):
    ast = lambda g: tuple(tuple(r) for r in g)
    return [
        {
            "input": ast(e["input"]),
            "output": ast(e["output"]),
        }
        for e in example_list
    ]


def load_concept_arc(
    data_path="data/concept_arc/",
):
    concept_arc_tasks = {}
    arc_eval_tasks_mapping = {}
    concepts = {}
    for directory in os.listdir(data_path):
        if not directory.startswith(".") and directory != "ProblemsWithErrors":
            for element in os.listdir(data_path + directory):
                if element.endswith(".json") and not element.replace(".json", "").endswith(
                    "Minimal"
                ):
                    task_dict = load_json(data_path + directory + "/" + element)
                    task = Task(
                        program=None,
                        training_examples=list_to_tuple(task_dict["train"]),
                        test_examples=list_to_tuple(task_dict["test"]),
                        task_key=element.replace(".json", ""),
                        parent_key=None,
                        extra_info={"concept": directory},
                    )
                    concept_arc_tasks[element.replace(".json", "")] = task
                    if directory in concepts:
                        concepts[directory].append(element.replace(".json", ""))
                    else:
                        concepts[directory] = [element.replace(".json", "")]
                else:
                    if element.startswith(".") or element.replace(".json", "").endswith("Minimal"):
                        pass
                    elif element != "OriginalARCEvalSet":
                        print(f"what is: {element} in {directory}")
                        raise Exception
                    else:
                        for task_id in os.listdir(data_path + directory + "/" + element):
                            if directory in arc_eval_tasks_mapping:
                                arc_eval_tasks_mapping[directory].append(
                                    task_id.replace(".json", "")
                                )
                            else:
                                arc_eval_tasks_mapping[directory] = [task_id.replace(".json", "")]
    return concept_arc_tasks, arc_eval_tasks_mapping, concepts
