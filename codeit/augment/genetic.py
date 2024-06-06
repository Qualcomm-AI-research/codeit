# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import ast
import copy
import inspect
import os
import pickle
import random
import re
import signal
import traceback
from concurrent.futures import ThreadPoolExecutor

import astunparse

from codeit.augment.mutate_grid import mutate_input_one_step, valid_grid
from codeit.augment.mutate_program import ProgramMutator
from codeit.augment.type_inference import (
    CONSTANT_TO_TYPE_MAPPING,
    TypeInferer,
    contains_non_base_type,
    display_type,
)
from codeit.dsl.arc_types import *
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *
from codeit.task import Task
from codeit.utils import get_grid_size


def filter_population_by_size(task_population, grid_size=(10, 10)):
    filtered_population = {}
    for task in task_population.values():
        new_training_examples = []
        for example in task.training_examples:
            if (
                get_grid_size(example["input"]) == grid_size
                and get_grid_size(example["output"]) == grid_size
            ):
                new_training_examples.append(example)
        if len(new_training_examples) > 0:
            task.training_examples = new_training_examples
            if task.program:
                task.init_training_states()
                task.init_training_actions()
            filtered_population.update({task.task_key: task})
    return filtered_population


def get_function_types(func_def):
    func_ast = ast.parse(func_def).body[0]
    input_types = tuple(arg.annotation.id for arg in func_ast.args.args)
    output_type = func_ast.returns.id
    return input_types, output_type


def get_func_name(func_def):
    func_ast = ast.parse(func_def).body[0]
    return func_ast.name


def change_function_name(program, child_id):
    program_ast = ast.parse(program)
    program_ast.body[0].name = program_ast.body[0].name + f"_{child_id}"
    return astunparse.unparse(program_ast).strip() + "\n"


def build_primitive_function_to_general_type_mapping(type_to_function_mapping):
    primitive_function_to_general_type_mapping = {}
    for output_type in type_to_function_mapping.keys():
        for input_type in type_to_function_mapping[output_type]:
            for primitive_function in type_to_function_mapping[output_type][input_type]:
                primitive_function_to_general_type_mapping[primitive_function] = {
                    "inputs": input_type,
                    "output": output_type,
                }
    return primitive_function_to_general_type_mapping


def build_general_type_to_primitive_function_mapping(primitives):
    mapping = {}
    for func_def in primitives:
        input_types, output_type = get_function_types(func_def)
        if output_type not in mapping:
            mapping[output_type] = {}
        if input_types not in mapping[output_type]:
            mapping[output_type][input_types] = []
        mapping[output_type][input_types].append(get_func_name(func_def))
    return mapping


def get_function_names_with_output_type(output_type, type_to_function_mapping):
    function_names = []
    if output_type in type_to_function_mapping:
        for input_types, func_list in type_to_function_mapping[output_type].items():
            for func_def in func_list:
                func_name = func_def.split("(", 1)[0].split()[-1]
                if func_name != "canvas" and func_name != "cellwise":
                    function_names.append(func_name)
    return function_names


class TimeoutException(Exception):
    pass


def handle_timeout(signum, frame):
    print("timeout occured")


class TaskEvolver:
    def __init__(
        self,
        data_path,
        primitive_functions,
        primitive_constants,
        pop_size_before_parallel=1000,
        train_file_path="data/tasks/raw_train/",
        phi_program=0.5,
        phi_equivalent=1,
        phi_redundant=1,
        phi_var=0.5,
        phi_func=0.2,
        phi_arg=0.2,
        resume=0,
        tasks=None,
        aug_keys=None,
        iteration_id="",
        select_from_arc=False,
        sig_alarm=False,
        load_inferrer=True,
    ):
        self.task_population = tasks
        self.resume = bool(resume)
        self.base_type_to_primitive_function_mapping = None
        self.primitive_function_to_base_type_mapping = None
        self.primitive_constant_to_type_mapping = CONSTANT_TO_TYPE_MAPPING
        self.fitness_threshold = 0.5
        self.population_tree = PopulationTree()
        self.initialise_population(aug_keys, train_file_path, tasks=tasks)
        print("populations initialised")
        self.primitive_functions = primitive_functions
        self.primitive_constants = primitive_constants
        self.general_type_to_primitive_function_mapping = (
            build_general_type_to_primitive_function_mapping(
                [
                    inspect.getsource(eval(primitive_function))
                    for primitive_function in primitive_functions
                ]
            )
        )
        self.primitive_function_to_general_type_mapping = (
            build_primitive_function_to_general_type_mapping(
                self.general_type_to_primitive_function_mapping
            )
        )
        self.primitive_grid_functions = get_function_names_with_output_type(
            "Grid", self.general_type_to_primitive_function_mapping
        )
        self.load_inferrer = load_inferrer
        self.input_mutation_log = ""
        self.program_mutation_log = ""
        self.timeout_log = ""
        self.filter_log = ""
        self.phi_program = phi_program
        self.phi_inputs = 1 - self.phi_program
        self.phi_var = phi_var
        self.phi_func = phi_func
        self.phi_arg = phi_arg
        self.phi_equivalent = phi_equivalent
        self.phi_redundant = phi_redundant
        self.infer_base_types_for_primitive_functions(data_path=data_path)
        self.timed_out = False
        self.pop_size_before_parallel = pop_size_before_parallel
        self.updates = None
        self.update_thread = None
        self.iteration_id = iteration_id
        self.arc_training = copy.copy(self.task_population)
        self.select_from_arc = select_from_arc
        self.sig_alarm = sig_alarm

    def execute_with_timeout(self, func, timeout, *args, **kwargs):
        if self.sig_alarm:
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                result = func(*args, **kwargs)
                signal.setitimer(signal.ITIMER_REAL, 0)
                return result
            except Exception as e:
                traceback.print_exc()
                self.timed_out = True
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                except Exception as e:
                    self.timed_out = True
                    print(f"{func.__name__} took longer than {timeout} seconds")
                    t = traceback.format_exc
                    traceback.print_exc()
                    # raise TimeoutError  # Or any other value you'd like to return in case of a timeout
                else:
                    return result

    def evolve_with_timeout(self, sig_alarm):
        self.timed_out = False
        selected_task = self.select()
        print(selected_task.task_key)
        mutated_task = self.execute_with_timeout(
            self.evolve, timeout=sig_alarm, selected_task=selected_task
        )
        if mutated_task:
            fitness = self.fitness_function(mutated_task, selected_task)
            if fitness > self.fitness_threshold:
                self.task_population[mutated_task.task_key] = mutated_task
                self.population_tree.add_program(mutated_task.task_key, selected_task.task_key)
        else:
            self.timeout_log += f"evolve except: mutating task {selected_task.task_key} timed out\n"
            print(f"evolve except: mutating task {selected_task.task_key} timed out\n")
            print(selected_task.task_key, selected_task.program_lines)
        return mutated_task

    def evolve(self, selected_task):
        mutated_task = self.mutate_task(selected_task)
        return mutated_task

    def initialise_population(self, task_keys, file_path, tasks=None):
        print("initialising training population")
        if self.resume:
            task_keys = [
                name.strip(".json") for name in os.listdir(file_path) if name.endswith(".json")
            ]
        task_population = {}
        if self.task_population or tasks:
            for task_key in self.task_population:
                print(f"Adding task {task_key} to population tree")  # Added logging
                self.population_tree.add_program(task_key)
        else:
            for task_key in task_keys:  # Removed tqdm for now
                print(f"Loading task {task_key}")  # Added logging
                task = Task.from_json(file_path + task_key + ".json")
                task_population[task.task_key] = task
                print(f"Adding task {task_key} to population tree")  # Added logging
                self.population_tree.add_program(task.task_key)
            self.task_population = task_population
        print("done")

    def select(self, max_grid_size=None):
        """selects a task from the population to be mutated"""
        if self.select_from_arc:
            return self.arc_training[random.choice(list(self.arc_training.keys()))]
        else:
            if max_grid_size:
                filtered_task_population = filter_tasks_by_input_dimensions(
                    self.task_population, max_rows=max_grid_size[0], max_columns=max_grid_size[1]
                )
                return self.task_population[random.choice(list(filtered_task_population.keys()))]
            else:
                return self.task_population[random.choice(list(self.task_population.keys()))]

    def mutate_task(self, task):
        """mutates a task by either mutating the program or the input"""
        mutation_choice = random.choices(
            ["input", "program"], weights=[self.phi_inputs, self.phi_program]
        )[0]
        try:
            mutation = ["inputs"]
            mutated_training_examples = []
            mutated_test_examples = []
            if len(self.iteration_id) > 0:
                new_child_id = f"it{self.iteration_id}_{self.get_new_child_id(task.task_key)}"
            else:
                new_child_id = self.get_new_child_id(task.task_key)

            if mutation_choice == "program":
                mutation_results = self.mutate_program(task=task)
                program = change_function_name(mutation_results["output"], new_child_id)
                mutation = mutation_results["mutation"]
                self.program_mutation_log += mutation_results["log"]
                equivalent = True
                exec(program, globals(), locals())
                for example_type in ["training_examples", "test_examples"]:
                    if self.timed_out:
                        raise ValueError(f"timed out on program execution in mutate task")
                    example_list = (
                        task.training_examples
                        if example_type == "training_examples"
                        else task.test_examples
                    )
                    mutated_examples = []
                    for example in example_list:
                        I = example["input"]
                        output = eval(f"solve_{task.task_key}_{new_child_id}(I)")
                        mutated_examples.append({"input": I, "output": output})
                        if output != example["output"]:
                            equivalent = False
                    if example_type == "training_examples":
                        mutated_training_examples = mutated_examples
                    else:
                        mutated_test_examples = mutated_examples

            else:
                program = task.program
                program = change_function_name(program, new_child_id)
                for example_type in ["training_examples", "test_examples"]:
                    example_list = (
                        task.training_examples
                        if example_type == "training_examples"
                        else task.test_examples
                    )
                    mutated_examples = []
                    for i, example in enumerate(example_list):
                        mutation_results = self.mutate_input(example=example, program=program)
                        input_grid = mutation_results["output"]["input"]
                        output_grid = mutation_results["output"]["output"]
                        mutation.append(mutation_results["mutation"])
                        self.input_mutation_log += mutation_results["log"]
                        mutated_examples.append({"input": input_grid, "output": output_grid})
                    if example_type == "training_examples":
                        mutated_training_examples = mutated_examples
                    else:
                        mutated_test_examples = mutated_examples
                equivalent = False
            new_key = f"{task.task_key}_{new_child_id}"
            return Task(
                program=program,
                training_examples=mutated_training_examples,
                test_examples=mutated_test_examples,
                task_key=new_key,
                extra_info={"mutation": mutation, "equivalent": equivalent},
            )
        except:
            print("error in mutate task not timeout")
            t = traceback.format_exc()
            traceback.print_exc()
            if mutation_choice == "program":
                print("error mutating program:", self.program_mutation_log.split("Traceback")[-1])
                self.program_mutation_log += "mutate task except:" + traceback.format_exc()
            else:
                print("error mutating input:", self.input_mutation_log.split("Traceback")[-1])
                self.input_mutation_log += "mutate task except:" + traceback.format_exc()

    def fitness_function(self, mutated_task, selected_task):
        """function to evaluate the fitness of a mutated task"""
        return 1

    def get_new_child_id(self, task_key):
        """returns the next child id for a given task key"""
        children = self.population_tree.get_children(task_key)
        if children:
            return max([int(child.split("_")[-1]) for child in children]) + 1
        else:
            return 0

    def save_population(self, population, file_path):
        for task_key, task in population.items():
            task.to_json(f"{file_path}{task_key}.json")

    def filter_population_for_equivalence_and_reduce(self):
        filtered_population = {}
        for task in self.task_population.values():
            processed_task = copy.copy(task)
            if random.random() <= self.phi_redundant:
                task.program = remove_redundant_lines(processed_task.program)
            if "equivalent" in task.extra_info.keys():
                if random.random() <= self.phi_equivalent and task.extra_info["equivalent"]:
                    pass
                else:
                    filtered_population.update({processed_task.task_key: processed_task})
            else:
                filtered_population.update({processed_task.task_key: processed_task})
        self.filter_log += f"removed {len(self.task_population) - len(filtered_population)} tasks from population by filtering for equivalence and redundancy\n"
        self.task_population = filtered_population

    def filter_population_by_size(self, grid_size=(10, 10)):
        filtered_population = filter_population_by_size(self.task_population, grid_size=(10, 10))
        self.filter_log += "removed {len(self.task_population) - len(filtered_population)} tasks from population by filtering on size {grid_size}"
        self.task_population = filtered_population

    def infer_base_types_for_primitive_functions(self, data_path, save=False):
        if (
            os.path.exists(f"{data_path}primitive_function_to_base_type_mapping.pkl")
            and self.load_inferrer
        ):
            self.primitive_function_to_base_type_mapping = pickle.load(
                open(f"{data_path}primitive_function_to_base_type_mapping.pkl", "rb")
            )
            self.base_type_to_primitive_function_mapping = pickle.load(
                open(f"{data_path}base_type_to_primitive_function_mapping.pkl", "rb")
            )
        else:
            type_dict = {}
            for task in self.task_population.values():
                type_inferer = TypeInferer(task.training_examples[0]["input"])
                program_ast = ast.parse(task.program)
                type_inferer.infer_type_from_ast(program_ast)
                for key in type_inferer.type_dict.keys():
                    if (
                        (not key.startswith("x"))
                        and (not key == "I")
                        and (not key == "O")
                        and (not key.isupper())
                    ):
                        if key in type_dict:
                            for term_type in type_inferer.type_dict[key]:
                                if (term_type not in type_dict[key]) and (
                                    not contains_non_base_type(term_type)
                                ):
                                    type_dict[key].append(term_type)
                        else:
                            for term_type in type_inferer.type_dict[key]:
                                if not contains_non_base_type(term_type):
                                    if key in type_dict:
                                        type_dict[key].append(term_type)
                                    else:
                                        type_dict[key] = [term_type]
            type_to_primitives = {}
            prims = []
            for prim in type_dict:
                for prim_type in type_dict[prim]:
                    hashable_prim_type = display_type(prim_type)
                    prims.append(prim)
                    if hashable_prim_type in type_to_primitives:
                        type_to_primitives[hashable_prim_type].append(prim)
                    else:
                        type_to_primitives[hashable_prim_type] = [prim]
            type_dict_hashed = {}
            for prim_type in type_to_primitives.keys():
                for prim in type_to_primitives[prim_type]:
                    if prim in type_dict_hashed:
                        if prim_type not in type_dict_hashed[prim]:
                            type_dict_hashed[prim].append(prim_type)
                    else:
                        type_dict_hashed[prim] = [prim_type]
            self.primitive_function_to_base_type_mapping = type_dict
            self.base_type_to_primitive_function_mapping = type_to_primitives
            if save:
                pickle.dump(
                    type_dict, open(f"{data_path}primitive_function_to_base_type_mapping.pkl", "wb")
                )
                pickle.dump(
                    type_to_primitives,
                    open(f"{data_path}base_type_to_primitive_function_mapping.pkl", "wb"),
                )

    def mutate_input(self, example, program):
        mutated_input_log = ""
        self.timed_out = False
        while not self.timed_out:
            try:
                mutated_input, mutated_output, grid_function = mutate_input_one_step(
                    example,
                    program,
                    self.primitive_grid_functions,
                    self.primitive_constants,
                    self.primitive_function_to_base_type_mapping,
                    self.base_type_to_primitive_function_mapping,
                )
            except:
                mutated_input = None
                mutated_output = None
                mutated_input_log += f"mutate input except:{traceback.format_exc()}\n"
            if (
                valid_grid(mutated_input)
                and example["input"] != mutated_input
                and valid_grid(mutated_output)
            ):
                return {
                    "output": {"input": mutated_input, "output": mutated_output},
                    "log": mutated_input_log,
                    "mutation": grid_function,
                }
            else:
                mutated_input_log += "invalid grid\n"

    def mutate_program(self, task):
        program = task.program
        I = task.training_examples[0]["input"]
        type_inferer = TypeInferer(I)
        mutated_program_log = ""
        self.timed_out = False
        while not self.timed_out:
            try:
                program_mutator = ProgramMutator(
                    program=program,
                    type_inferer=type_inferer,
                    phi_var=self.phi_var,
                    phi_func=self.phi_func,
                    phi_arg=self.phi_arg,
                    primitive_function_to_general_type_mapping=self.primitive_function_to_general_type_mapping,
                    primitive_function_to_base_type_mapping=self.primitive_function_to_base_type_mapping,
                    primitive_constant_to_type_mapping=self.primitive_constant_to_type_mapping,
                    general_type_to_primitive_function_mapping=self.general_type_to_primitive_function_mapping,
                    base_type_to_primitive_function_mapping=self.base_type_to_primitive_function_mapping,
                    type_to_primitive_constant_mapping=self.primitive_constants,
                )
                mutation = program_mutator.mutate()
                exec(ast.unparse(program_mutator.program_ast), globals(), locals())
                not_identity = False
                for i, example in enumerate(task.training_examples + task.test_examples):
                    if self.timed_out:
                        raise ValueError(f"timed out on program execution in mutate program")
                    I = example["input"]
                    O = eval(f"solve_{task.task_key}(I)")
                    if not valid_grid(O):
                        raise ValueError(
                            f"invalid grid generated by mutated program on input example {i}"
                        )
                    if I != O:
                        not_identity = True
                if not not_identity:
                    raise ValueError(f"all output examples are the identity")
                return {
                    "output": ast.unparse(program_mutator.program_ast),
                    "log": mutated_program_log,
                    "mutation": mutation,
                }
            except:
                mutated_program_log += f"mutate program except:{traceback.format_exc()}\n"
                pass


class PopulationTree:
    def __init__(self):
        self.tree = {}

    def add_program(self, program_name, parent_name=None):
        if parent_name is not None and parent_name not in self.tree:
            raise ValueError(f"Parent '{parent_name}' not found in the tree.")
        self.tree[program_name] = parent_name

    def get_parent(self, program_name):
        return self.tree.get(program_name)

    def get_children(self, parent_name):
        return [prog for prog, parent in self.tree.items() if parent == parent_name]

    def __repr__(self):
        return str(self.tree)


def filter_tasks_by_input_dimensions(tasks, max_rows, max_columns):
    filtered_tasks = {}
    for task_key, task in tasks.items():
        add = True
        for example in task.training_examples + task.test_examples:
            input_grid_size = get_grid_size(example["input"])
            output_grid_size = get_grid_size(example["output"])
            if (
                input_grid_size[0] > max_rows
                or input_grid_size[1] > max_columns
                or output_grid_size[0] > max_rows
                or output_grid_size[1] > max_columns
            ):
                add = False
        if add:
            filtered_tasks[task_key] = tasks[task_key]
    return filtered_tasks


def remove_redundant_lines(program_str):
    used_vars = set([])
    lines = [l for l in program_str.split("\n") if l]
    for line in lines[1:-1]:
        used_vars.update(re.findall(r"x[0-9]+", line.strip().split("=")[1].strip()))
    lines_to_keep = [
        line
        for line in lines
        if line.strip().split("=")[0].strip() in used_vars
        or line.strip().split("=")[0].strip() == "O"
    ]
    new_program = [lines[0]]
    for i, line in enumerate(lines_to_keep):
        new_program.append(f"x{i + 1} = {line.strip().split('=')[1].strip()}")
    new_program.append(lines[-1].strip())
    return "\n    ".join(new_program)
