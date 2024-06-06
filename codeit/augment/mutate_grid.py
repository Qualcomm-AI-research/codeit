# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import traceback

import numpy as np

from codeit.augment.type_inference import *
from codeit.dsl.arc_types import *
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *


def valid_grid(value):
    if isinstance(value, bool):
        return False
    if value in [(), (()), ((),)]:
        return True
    try:
        np_grid = np.array(value, dtype=int)
    except:
        return False

    if np_grid.ndim != 2:
        return False

    return np.all((0 <= np_grid) & (np_grid) <= 9)


def mutate_input_one_step(
    example,
    program,
    primitive_grid_functions,
    primitive_constants,
    primitive_function_to_types,
    type_to_primitive_functions,
):
    program_name = ast.parse(program).body[0].name
    exec(program, globals(), locals())
    grid_function_types = []
    I = example["input"]
    OBJECTS = objects(
        I,
        random.choice(primitive_constants["Boolean"]),
        random.choice(primitive_constants["Boolean"]),
        random.choice(primitive_constants["Boolean"]),
    )
    while not grid_function_types:
        grid_function = random.choice(
            [
                function
                for function in primitive_grid_functions
                if function in primitive_function_to_types
            ]
        )
        grid_function_types = [
            function_type
            for function_type in primitive_function_to_types[grid_function]
            if "Grid" in function_type.inputs
        ]
    grid_function_type = random.choice(grid_function_types)
    args = []
    for arg_type in grid_function_type.inputs:
        if arg_type == "Grid":
            args.append(I)
        elif arg_type == "Object":
            args.append(object_sampler(I, OBJECTS))
        elif arg_type == "Objects":
            args.append(sample_objects(I, OBJECTS))
        elif arg_type == "Boolean":
            args.append(eval(random.choice(primitive_constants["Boolean"])))
        elif arg_type == "Integer":
            args.append(eval(random.choice(primitive_constants["Integer"])))
        elif arg_type == "IntegerTuple":
            args.append(eval(random.choice(primitive_constants["IntegerTuple"])))
        elif arg_type == "Cell":
            args.append(cell_sampler(I, OBJECTS))
        elif arg_type == "Indices":
            args.append(sample_indices(I, OBJECTS))
        elif arg_type == "IndicesSet":
            args.append(sample_indices_set(I, OBJECTS))
        elif display_type(arg_type) in type_to_primitive_functions:
            types = type_to_primitive_functions[display_type(arg_type)]
            args.append(random.choice(types))
        elif arg_type == "IntegerSet":
            args.append(sample_integer_set(primitive_constants))
        else:
            raise ValueError(f"Unknown argument type {arg_type}")
    transformation = f"{grid_function}(*args)"
    mutated_input_grid = eval(transformation)
    mutated_output_grid = eval(f"{program_name}(mutated_input_grid)")
    return mutated_input_grid, mutated_output_grid, grid_function


def sample_integer_set(primitive_constants):
    integer_set = []
    while True:
        integer_choice = eval(random.choice(primitive_constants["Integer"]))
        integer_set.append(integer_choice)
        if random.random() < 0.5:
            break
    return frozenset(integer_set)


def sample_indices(I, OBJECTS=None):
    OBJECT = object_sampler(I, OBJECTS)
    INDICES = toindices(OBJECT)
    return INDICES


def sample_indices_set(I, OBJECTS=None):
    OBJECTS = sample_objects(I, OBJECTS)
    indices_set = []
    for object in list(OBJECTS):
        indices_set.append(toindices(object))
    return FrozenSet(indices_set)


def sample_objects(I, OBJECTS=None):
    if OBJECTS:
        return OBJECTS
    else:
        OBJECTS = []
        while True:
            OBJECTS.append(object_sampler(I, None))
            if random.random() < 0.5:
                break
        return FrozenSet(OBJECTS)


def cell_sampler(I, OBJECTS=None):
    return random.choice(list(object_sampler(I, OBJECTS)))


def object_sampler(I, OBJECTS=None):
    if OBJECTS:
        return random.choice(list(OBJECTS))
    else:
        return sample_object_from_patch(patch_sampler(I))


def patch_sampler(I, grow_probability=0.75):
    patch = {random.choice(list(asindices(I)))}
    while len(patch) < round(0.9 * (len(asindices(I)))):
        grow_start = random.choice(list(patch))
        new_cell = random.choice(list(normalise_cell(neighbors(grow_start), I)))
        patch.add(new_cell)
        if random.random() > grow_probability:
            break
    return frozenset(patch)


def normalise_cell(cell, I):
    normalised_cell = set()
    for (x, y) in cell:
        if x < 0:
            x += len(I)
        if y < 0:
            y += len(I[0])
        normalised_cell.add((x, y))
    return frozenset(normalised_cell)


def sample_object_from_patch(patch):
    colour = random.choice(range(10))
    object = {(colour, cell) for cell in patch}
    return frozenset(object)
