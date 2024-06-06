# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import re
import signal
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from codeit.augment.mutate_grid import valid_grid
from codeit.augment.type_inference import CONSTANT_TO_TYPE_MAPPING
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *


def execute_candidate_program(program_string, program_input, max_state_size=1_000, sig_alarm=False):
    program_string = program_string.rstrip("\n")
    valid_syntax = check_syntax(program_string)
    if valid_syntax != "Valid Syntax":
        return valid_syntax
    if not valid_grid(program_input):
        return "Invalid Input"
    lines = program_string.split("\n")
    memory = {"I": program_input}
    for line in lines:
        var_name = None
        try:
            var_name, var_def = line.split("=")
            environment = {**globals(), **memory}
            state = execute_with_timeout(
                func=eval_code,
                timeout=0.25,
                var_def=var_def.strip(),
                environment=environment,
                sig_alarm=sig_alarm,
            )
            if max_state_size:
                if hasattr(state, "__len__"):
                    if len(state) < max_state_size:
                        memory[var_name.strip()] = state
                else:
                    memory[var_name.strip()] = state
            else:
                memory[var_name.strip()] = state
        except:
            return f"Error evaluating {var_name}: {traceback.format_exc()}"
    if "O" not in memory:
        return "No output variable defined"
    return memory["O"]


class TimeoutException(Exception):
    pass


def handle_timeout(signum, frame):
    raise TimeoutException


def execute_with_timeout(
    func,
    timeout,
    sig_alarm=False,
    *args,
    **kwargs,
):
    if sig_alarm:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            result = func(*args, **kwargs)
            signal.setitimer(signal.ITIMER_REAL, 0)
            return result
        except TimeoutException:
            print(f"{func.__name__} took longer than {timeout} seconds")
            raise TimeoutError
        except Exception as e:
            raise e
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
            except TimeoutError:
                print(f"{func.__name__} took longer than {timeout} seconds")
                raise TimeoutError  # Or any other value you'd like to return in case of a timeout
            else:
                return result


def check_syntax(program_string):
    variables = set()
    calls = set()
    primitive_functions = PRIMITIVE_FUNCTIONS
    primitive_constants = CONSTANT_TO_TYPE_MAPPING.keys()
    lines = program_string.split("\n")
    for i, line in enumerate(lines):
        line_strip = line.lstrip().split(" = ")
        if len(line_strip) != 2:
            return f"Line {line} is not a variable assignment"
        variable, call = line_strip
        call_strip = call.rstrip().split("(")
        if len(call_strip) != 2:
            return f"Line {line} does not have exactly one open parenthesis"
        function, args = call_strip
        if i == len(lines) - 1:
            if variable != "O":
                return f"Line {line} does not have variable name of O"
        if variable in primitive_functions:
            return f"Line {line} has a variable name that is a primitive function"
        if variable in variables:
            return f"Line {line} has a variable name that has already been used"
        if call in calls:
            return f"Line {line} has a function call that has already been used"
        variables.add(variable)
        calls.add(call)
        if (function not in primitive_functions) and (function not in variables):
            return f"Line {line} has a function call that is not a primitive function or a variable"
        if args:
            if args[-1] != ")":
                return f"Line {line} does not have a closing parenthesis"
        else:
            return f"Line {line} has a function call with no arguments"
        args = [args[:-1]] if "," not in args else args[:-1].split(", ")
        for arg in args:
            if not any(
                [
                    arg in variables,
                    arg in primitive_functions,
                    arg in primitive_constants,
                    arg == "I",
                ]
            ):
                return f"Line {line} has an argument {arg} that is not a variable, primitive function, primitive constant, or input"
    return "Valid Syntax"


def eval_code(var_def, environment):
    return eval(var_def, environment)
