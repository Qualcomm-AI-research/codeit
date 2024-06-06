# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import ast
import inspect
import random
import re

from codeit.dsl.arc_types import *
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *

BASE_TYPES = [
    "Boolean",
    "Integer",
    "IntegerTuple",
    "IntegerSet",
    "Grid",
    "Cell",
    "Object",
    "Objects",
    "Indices",
    "IndicesSet",
]
NON_BASE_TYPES = [
    "Numerical",
    "Patch",
    "Element",
    "Piece",
    "TupleTuple",
    "ContainerContainer",
    "Tuple",
    "Any",
    "Container",
    "Callable",
    "FrozenSet",
]

CONSTANT_TO_TYPE_MAPPING = {}
for key, values in PRIMITIVE_CONSTANTS.items():
    for value in values:
        CONSTANT_TO_TYPE_MAPPING[value] = key

CONSTANT_NAMES = list(CONSTANT_TO_TYPE_MAPPING.keys())


def get_primitive_function_type(primitive_function_name):
    func_def = inspect.getsource(eval(primitive_function_name))
    func_ast = ast.parse(func_def).body[0]
    input_types = [arg.annotation.id for arg in func_ast.args.args]
    output_type = func_ast.returns.id
    return Arrow(input_types, output_type)


def get_primitive_constant_type(primitive_constant_name):
    return CONSTANT_TO_TYPE_MAPPING[primitive_constant_name]


def find_first_x(input_string: str) -> str:
    # Find the first occurrence of 'x' followed by one or more digits
    match = re.search(r"x[1-9]\d*\b", input_string)
    if match:
        return match.group()
    else:
        return None


def replace_x(x_old: str, x_new: str, input_string: str) -> str:
    x_old = r"\b" + x_old + r"\b"  # Add word boundaries to the pattern
    return re.sub(x_old, x_new, input_string, count=1)


def display_type(term_type):
    try:
        return f"Arrow(( {', '.join([display_type(arrow_input) for arrow_input in term_type.inputs])} ), {display_type(term_type.output)})"
    except:
        return term_type


def contains_non_base_type(term_type):
    if not isinstance(term_type, str):
        term_type = display_type(term_type)
    for non_base_type in NON_BASE_TYPES:
        if non_base_type in term_type:
            return True
    return False


class TypeInferer:
    def __init__(self, input_grid):
        self.type_dict = {"I": ["Grid"]}
        self.memory = {"I": input_grid}

    def add(self, term, term_type, value=None):
        if term in self.type_dict.keys():
            if term_type not in self.type_dict[term] and term_type:
                self.type_dict[term].append(term_type)
        else:
            if term_type:
                self.type_dict[term] = [term_type]
        if value is not None:
            self.memory[term] = value

    def infer_type_from_ast(self, ast_node):
        for child in ast.iter_child_nodes(ast_node):
            self.infer_type_from_ast(child)
        self.process_node(ast_node)

    def parse_node(self, node):
        var = None
        f = None
        args = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                var = target.id
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                f = node.value.func.id
            for arg in node.value.args:
                if isinstance(arg, ast.Name):
                    args.append(arg.id)
        return var, f, args

    def process_node(self, node):
        if isinstance(node, ast.Assign):
            var, f, args = self.parse_node(node)
            function_type, arg_types = self.infer_types(f, args)
            var_type = function_type if isinstance(function_type, str) else function_type.output
            self.add(term=var, term_type=var_type, value=f"{f}({', '.join(args)})")
            self.add(term=f, term_type=function_type)
            for i, arg in enumerate(args):
                self.add(term=arg, term_type=arg_types[i])

    def reduce_to_primitives(self, code_string):
        while find_first_x(code_string):
            var = find_first_x(code_string)
            code_string = replace_x(var, self.memory[var], code_string)
        return code_string

    def infer_types(self, f, args):
        arg_types = []
        # argument type inference
        for arg in args:
            if arg.startswith("x") or arg in ["I", "O"]:
                arg_types.append(self.type_dict[arg][0])
            elif arg not in PRIMITIVE_FUNCTIONS:
                arg_types.append(get_primitive_constant_type(arg))
            else:
                arg_type = get_primitive_function_type(arg)
                if arg_type in BASE_TYPES:
                    arg_types.append(arg_type)
                else:
                    arg_types.append("Callable")
        # function type inference
        if f.startswith("x") or f in ["I", "O"]:
            function_type = self.type_dict[f][0]
        else:
            function_type = get_primitive_function_type(f)
            # check args
            for i, arg_type in enumerate(function_type.inputs):
                if arg_type not in BASE_TYPES:
                    if args[i] in CONSTANT_TO_TYPE_MAPPING.keys():
                        function_type.inputs[i] = CONSTANT_TO_TYPE_MAPPING[args[i]]
                    elif args[i] in PRIMITIVE_FUNCTIONS:
                        function_type.inputs[i] = get_primitive_function_type(args[i])
                    else:
                        function_type.inputs[i] = self.type_dict[args[i]][0]
            # check output
            if function_type.output not in BASE_TYPES:
                I = self.memory["I"]
                O = eval(self.reduce_to_primitives(f"{f}({', '.join(args)})"))
                function_type.output = get_custom_type(O)
                if not function_type.output:
                    if isinstance(O, Callable):
                        function_type.output = "Callable"
                    elif isinstance(O, Container):
                        function_type.output = "Container"
        return function_type, arg_types
