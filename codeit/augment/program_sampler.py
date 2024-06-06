# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import ast
import copy
import inspect
import os
import pickle
import random
import traceback

import numpy as np
from arrow import Arrow

from codeit.augment.genetic import (
    build_general_type_to_primitive_function_mapping,
    build_primitive_function_to_general_type_mapping,
)
from codeit.augment.type_inference import (
    CONSTANT_TO_TYPE_MAPPING,
    TypeInferer,
    display_type,
)
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *
from codeit.dsl.primitives import PRIMITIVE_CONSTANTS, PRIMITIVE_FUNCTIONS


def add_variable_definition_to_ast(program_ast, program_name, variable_definition):
    if program_ast is None:
        base_code = f"def {program_name}(I):\n   return O"
        program_ast = ast.parse(base_code)
    func_def = program_ast.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    assert func_def.name == program_name
    new_node = ast.parse(variable_definition).body[0]
    func_def.body.insert(-1, new_node)
    ast.fix_missing_locations(new_node)
    return program_ast


class ProgramSampler:
    def __init__(self, data_path) -> None:
        self.general_type_to_primitive_function_mapping = (
            build_general_type_to_primitive_function_mapping(
                [
                    inspect.getsource(eval(primitive_function))
                    for primitive_function in PRIMITIVE_FUNCTIONS
                ]
            )
        )
        self.primitive_function_to_general_type_mapping = (
            build_primitive_function_to_general_type_mapping(
                self.general_type_to_primitive_function_mapping
            )
        )
        self.primitive_constant_to_type_mapping = CONSTANT_TO_TYPE_MAPPING
        self.infer_base_types_for_primitive_functions
        self.type_to_primitive_constant_mapping = PRIMITIVE_CONSTANTS
        self.infer_base_types_for_primitive_functions(data_path)

    def infer_base_types_for_primitive_functions(self, data_path):
        if os.path.exists(f"{data_path}primitive_function_to_base_type_mapping.pkl"):
            self.primitive_function_to_base_type_mapping = pickle.load(
                open(f"{data_path}primitive_function_to_base_type_mapping.pkl", "rb")
            )
            self.base_type_to_primitive_function_mapping = pickle.load(
                open(f"{data_path}base_type_to_primitive_function_mapping.pkl", "rb")
            )
        else:
            raise NotImplementedError

    def sample(self, program_name, I):
        program_sample = ProgramSample(
            program_name=program_name,
            I=I,
            primitive_function_to_general_type_mapping=self.primitive_function_to_general_type_mapping,
            primitive_constant_to_type_mapping=self.primitive_constant_to_type_mapping,
            general_type_to_primitive_function_mapping=self.general_type_to_primitive_function_mapping,
            base_type_to_primitive_function_mapping=self.base_type_to_primitive_function_mapping,
            type_to_primitive_constant_mapping=self.type_to_primitive_constant_mapping,
            primitive_function_to_base_type_mapping=self.primitive_function_to_base_type_mapping,
        )
        return program_sample.sample()


class ProgramSample:
    def __init__(
        self,
        program_name,
        I,
        primitive_function_to_general_type_mapping,
        primitive_function_to_base_type_mapping,
        primitive_constant_to_type_mapping,
        general_type_to_primitive_function_mapping,
        base_type_to_primitive_function_mapping,
        type_to_primitive_constant_mapping,
    ):
        self.program_name = program_name
        self.program_ast = ast.parse(f"def {program_name}(I):\n   return O")
        self.program = ast.unparse(self.program_ast)
        self.type_inferer = TypeInferer(I)
        self.primitive_function_to_general_type_mapping = primitive_function_to_general_type_mapping
        self.primitive_function_to_base_type_mapping = primitive_function_to_base_type_mapping
        self.primitive_constant_to_type_mapping = primitive_constant_to_type_mapping
        self.general_type_to_primitive_function_mapping = general_type_to_primitive_function_mapping
        self.base_type_to_primitive_function_mapping = base_type_to_primitive_function_mapping
        self.type_to_primitive_constant_mapping = type_to_primitive_constant_mapping
        self.memory_index = 1
        self.type_inferer.infer_type_from_ast(self.program_ast)
        self.end_prob = 0.8
        self.type_choices = list(general_type_to_primitive_function_mapping.keys())

    def sample(self):
        while "O =" not in self.program:
            self.sample_line()
            self.memory_index += 1
        return self.program

    def sample_line(self):
        while True:
            try:
                variable_type = np.random.choice(self.type_choices)
                assert isinstance(variable_type, str)

                if variable_type == "Grid":
                    end_program = np.random.rand(1)[0] < self.end_prob
                else:
                    end_program = False

                new_variable_value = self.create_variable(variable_type)
                break
            except:
                pass

        if end_program:
            new_variable = "O = " + new_variable_value
        else:
            new_variable = f"x{self.memory_index} = {new_variable_value}"

        self.program_ast = add_variable_definition_to_ast(
            self.program_ast, self.program_name, variable_definition=new_variable
        )
        self.add_variable_to_memory(
            index_to_insert=self.memory_index, new_variable_value=new_variable_value
        )
        self.add_variable_type_dict(
            index_to_insert=self.memory_index, new_variable_type=variable_type
        )
        self.program = ast.unparse(self.program_ast)

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

    def sample_function(self):
        filtered_type_dict = self.filter_type_dict_by_index()
        candidate_terms = []
        # add primitive functions
        candidate_terms += list(self.primitive_function_to_base_type_mapping.keys())
        # add variables in memory
        for var_name, var_type in filtered_type_dict.items():
            if var_type[0].startswith("Arrow"):
                candidate_terms.append(var_name)
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

    def create_variable(self, variable_type):
        args = []
        if isinstance(variable_type, Arrow):
            raise ValueError(
                f"variable type {display_type(variable_type)} replacement not supported"
            )
        else:
            new_func = self.sample_function()
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
