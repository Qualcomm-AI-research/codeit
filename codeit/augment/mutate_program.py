# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import ast
import copy
import random
import traceback

from codeit.augment.mutate_grid import valid_grid
from codeit.augment.type_inference import (
    CONSTANT_TO_TYPE_MAPPING,
    TypeInferer,
    display_type,
)
from codeit.dsl.arc_types import *
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *


class ProgramMutator:
    def __init__(
        self,
        program,
        type_inferer,
        phi_var,
        phi_func,
        phi_arg,
        primitive_function_to_general_type_mapping,
        primitive_function_to_base_type_mapping,
        primitive_constant_to_type_mapping,
        general_type_to_primitive_function_mapping,
        base_type_to_primitive_function_mapping,
        type_to_primitive_constant_mapping,
    ):
        self.program = program
        self.program_ast = ast.parse(program)
        self.type_inferer = type_inferer
        self.phi_var = phi_var
        self.phi_func = phi_func
        self.phi_arg = phi_arg
        self.primitive_function_to_general_type_mapping = primitive_function_to_general_type_mapping
        self.primitive_function_to_base_type_mapping = primitive_function_to_base_type_mapping
        self.primitive_constant_to_type_mapping = primitive_constant_to_type_mapping
        self.general_type_to_primitive_function_mapping = general_type_to_primitive_function_mapping
        self.base_type_to_primitive_function_mapping = base_type_to_primitive_function_mapping
        self.type_to_primitive_constant_mapping = type_to_primitive_constant_mapping
        self.memory_index = None
        self.type_inferer.infer_type_from_ast(self.program_ast)

    def mutate(self):
        mutation_choice = random.choices(
            ["replace_argument", "replace_function", "replace_variable"],
            weights=[self.phi_arg, self.phi_func, 1 - (self.phi_func + self.phi_arg)],
        )[0]
        assignments = [node for node in ast.walk(self.program_ast) if isinstance(node, ast.Assign)]
        node_to_mutate = random.choice(assignments)
        self.memory_index = assignments.index(node_to_mutate) + 1
        if mutation_choice == "replace_argument":
            arg_to_replace_id = random.choice(range(len(node_to_mutate.value.args)))
            arg_to_replace = node_to_mutate.value.args[arg_to_replace_id].id
            new_arg, new_variable_mutation = self.replace_argument(arg_to_replace=arg_to_replace)
            assignments[self.memory_index - 1].value.args[arg_to_replace_id].id = new_arg
            if new_variable_mutation:
                mutation = ("arg", arg_to_replace, new_arg, new_variable_mutation)
            else:
                mutation = ("arg", arg_to_replace, new_arg)
        elif mutation_choice == "replace_function":
            function_to_replace = node_to_mutate.value.func.id
            new_function = self.replace_function(function_to_replace=function_to_replace)
            assignments[self.memory_index - 1].value.func.id = new_function
            mutation = ("func", function_to_replace, new_function)
        else:
            variable_to_replace = node_to_mutate.targets[0].id
            new_variable_value = self.replace_variable(variable_to_replace=variable_to_replace)
            mutation = (
                "var_def",
                f"{node_to_mutate.value.func.id}({', '.join([arg.id for arg in node_to_mutate.value.args])})",
                new_variable_value,
            )
            new_node = ast.parse(new_variable_value).body[0].value
            assignments[self.memory_index - 1].value = new_node
        return mutation

    def sample_term_with_type(self, term_type, terms_to_exclude):
        filtered_type_dict = self.filter_type_dict_by_index()
        candidate_terms = []
        # add primitive functions
        if isinstance(term_type, dict):
            candidate_terms += self.general_type_to_primitive_function_mapping[term_type["output"]][
                term_type["inputs"]
            ]
        elif isinstance(term_type, Arrow):
            candidate_terms += self.base_type_to_primitive_function_mapping[display_type(Arrow)]
        # add primitive constants
        else:
            if term_type in self.type_to_primitive_constant_mapping.keys():
                candidate_terms += self.type_to_primitive_constant_mapping[term_type]
        # add variables in memory
        for var_name, var_type in filtered_type_dict.items():
            if var_type[0] == term_type:
                candidate_terms.append(var_name)
        # filter out excluded terms
        candidate_terms = [term for term in candidate_terms if term not in terms_to_exclude]
        if not candidate_terms:
            raise ValueError(f"No candidate terms of type {display_type(term_type)} found")
        return random.choice(candidate_terms)

    def sample_function_with_output_type(self, output_type):
        filtered_type_dict = self.filter_type_dict_by_index()
        candidate_terms = []
        # add primitive functions
        for input_type in self.general_type_to_primitive_function_mapping[output_type]:
            candidate_terms += self.general_type_to_primitive_function_mapping[output_type][
                input_type
            ]
        # add variables in memory
        for var_name, var_type in filtered_type_dict.items():
            if var_type[0].startswith("Arrow") and var_type[0].endswith(f", {output_type})"):
                candidate_terms.append(var_name)
        # filter out functions that are do not have base type hints
        candidate_terms = [
            candidate_term
            for candidate_term in candidate_terms
            if candidate_term in self.primitive_function_to_base_type_mapping.keys()
        ]
        if not candidate_terms:
            raise ValueError(
                f"No candidate functions with output type {display_type(output_type)} found"
            )
        return random.choice(candidate_terms)

    def filter_memory_by_index(self):
        filtered_memory = {"I": self.type_inferer.memory["I"]}
        for var_index in range(1, self.memory_index):
            var_name = f"x{var_index}"
            filtered_memory[var_name] = self.type_inferer.memory[var_name]
        return filtered_memory

    def filter_type_dict_by_index(self):
        filtered_type_dict = {"I": self.type_inferer.type_dict["I"]}
        for var_index in range(1, self.memory_index):
            var_name = f"x{var_index}"
            filtered_type_dict[var_name] = self.type_inferer.type_dict[var_name]
        return filtered_type_dict

    def bump_up_variable_names(self, start_idx):
        # bump up variable names in program ast
        for node in ast.walk(self.program_ast):
            if isinstance(node, ast.Name) and node.id.startswith("x"):
                var_num = int(node.id[1:])
                if var_num >= start_idx:
                    node.id = f"x{var_num + 1}"
        # bump up variable names in type inferer and memory
        for var_name in copy.copy(list(self.type_inferer.type_dict.keys())):
            if var_name.startswith("x") and int(var_name[1:]) >= start_idx:
                self.type_inferer.type_dict[
                    f"x{int(var_name[1:])+1}r"
                ] = self.type_inferer.type_dict.pop(var_name)
                self.type_inferer.memory[
                    f"x{int(var_name[1:]) + 1}r"
                ] = self.type_inferer.memory.pop(var_name)
        for var_name in copy.copy(list(self.type_inferer.type_dict.keys())):
            if var_name.startswith("x") and var_name.endswith("r"):
                self.type_inferer.type_dict[var_name[:-1]] = self.type_inferer.type_dict.pop(
                    var_name
                )
                self.type_inferer.memory[var_name[:-1]] = self.type_inferer.memory.pop(var_name)

    def add_variable_to_ast(self, index_to_insert, new_variable_value):
        self.bump_up_variable_names(index_to_insert + 1)
        new_assignment = ast.parse(f"x{index_to_insert+1} = {new_variable_value}").body[0]
        for node in ast.walk(self.program_ast):
            if isinstance(node, ast.FunctionDef):
                node.body.insert(index_to_insert, new_assignment)
                break

    def add_variable_to_memory(self, index_to_insert, new_variable_value):
        local_memory = self.filter_memory_by_index()
        for primitive in list(self.primitive_function_to_general_type_mapping.keys()) + list(
            self.primitive_constant_to_type_mapping.keys()
        ):
            local_memory[primitive] = eval(primitive)
        new_variable = eval(new_variable_value, {}, local_memory)
        self.type_inferer.memory[f"x{index_to_insert}"] = new_variable

    def add_variable_type_dict(self, index_to_insert, new_variable_type):
        self.type_inferer.type_dict[f"x{index_to_insert}"] = [new_variable_type]

    def replace_argument(self, arg_to_replace):
        """replace an argument with an argument of the same type, add new variable with probability phi_var and replace argument with this new variable"""
        if arg_to_replace.startswith("x") or arg_to_replace == "I":
            arg_type = self.type_inferer.type_dict[arg_to_replace][0]
            if isinstance(arg_type, Arrow):
                raise ValueError(
                    f"argument type {display_type(arg_type)} replacement not supported"
                )
        elif arg_to_replace in self.primitive_constant_to_type_mapping.keys():
            arg_type = CONSTANT_TO_TYPE_MAPPING[arg_to_replace]
        elif arg_to_replace in self.primitive_function_to_general_type_mapping.keys():
            arg_type = self.primitive_function_to_general_type_mapping[arg_to_replace]
        else:
            raise ValueError(f"argument {arg_to_replace} not found")
        if random.random() < self.phi_var:
            new_variable_type = arg_type
            new_variable_value = self.create_variable(new_variable_type)
            self.add_variable_to_ast(
                index_to_insert=self.memory_index - 1, new_variable_value=new_variable_value
            )
            self.add_variable_to_memory(
                index_to_insert=self.memory_index - 1, new_variable_value=new_variable_value
            )
            self.add_variable_type_dict(
                index_to_insert=self.memory_index - 1, new_variable_type=new_variable_type
            )
            new_arg = f"x{copy.copy(self.memory_index)}"
            # self.memory_index += 1
            mutation = new_variable_value
        else:
            new_arg = self.sample_term_with_type(
                term_type=arg_type, terms_to_exclude=[arg_to_replace]
            )
            mutation = None
        return new_arg, mutation

    def replace_function(self, function_to_replace):
        """replace a function with a function of the same type"""
        if function_to_replace.startswith("x"):
            function_type = self.type_inferer.type_dict[function_to_replace]
            if "Callable" in display_type(function_type):
                raise ValueError(
                    f"function type {display_type(function_type)} replacement not supported"
                )
            else:
                new_function = self.sample_term_with_type(
                    term_type=function_type, terms_to_exclude=[function_to_replace]
                )
        else:
            function_type = self.primitive_function_to_general_type_mapping[function_to_replace]
            new_function = self.sample_term_with_type(
                term_type=function_type, terms_to_exclude=[function_to_replace]
            )
        return new_function

    def create_variable(self, variable_type):
        args = []
        if isinstance(variable_type, Arrow):
            raise ValueError(
                f"variable type {display_type(variable_type)} replacement not supported"
            )
        else:
            new_func = self.sample_function_with_output_type(output_type=variable_type)
            if new_func.startswith("x"):
                new_func_type = self.type_inferer.type_dict[new_func][0]
            else:
                new_func_type = random.choice(
                    self.primitive_function_to_base_type_mapping[new_func]
                )
            for arg_type in new_func_type.inputs:
                arg = self.sample_term_with_type(term_type=arg_type, terms_to_exclude=[])
                args.append(arg)
        new_variable_value = f"{new_func}({','.join(args)})"
        return new_variable_value

    def replace_variable(self, variable_to_replace):
        """replace a variable with a variable of the same type"""
        variable_type = self.type_inferer.type_dict[variable_to_replace][0]
        new_variable_value = self.create_variable(variable_type=variable_type)
        return new_variable_value
