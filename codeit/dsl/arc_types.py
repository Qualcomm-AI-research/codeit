# Code adapted from https://github.com/michaelhodel/arc-dsl
# Copyright (c) 2023 Michael Hodel, licensed under MIT

from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable, Optional
)

Boolean = bool
Integer = int
IntegerTuple = Tuple[Integer, Integer]
Numerical = Union[Integer, IntegerTuple]
IntegerSet = FrozenSet[Integer]
Grid = Tuple[Tuple[Integer]]
Cell = Tuple[Integer, IntegerTuple]
Object = FrozenSet[Cell]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerTuple]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]
GridTuple = Tuple[Grid, ...]

CUSTOM_TYPE_NAMES = {
    Boolean: "Boolean",
    Integer: "Integer",
    IntegerTuple: "IntegerTuple",
    IntegerSet: "IntegerSet",
    Grid: "Grid",
    Cell: "Cell",
    Object: "Object",
    Objects: "Objects",
    Indices: "Indices",
    IndicesSet: "IndicesSet",
    # added ones
    GridTuple: "GridTuple",
}


def is_grid(value):
    if not isinstance(value, Tuple):
        return False
    if len(value) == 0:
        return True
    if not isinstance(value[0], Tuple):
        return False
    first_length = len(value[0]) if value else None
    return all(
        isinstance(inner, Tuple)
        and len(inner) == first_length
        and all(isinstance(item, int) for item in inner)
        for inner in value
    )


def check_custom_type(value: Any, custom_type: Any) -> bool:
    if custom_type is Boolean:
        return isinstance(value, bool)
    elif custom_type is Integer:
        return isinstance(value, int)
    elif custom_type is IntegerTuple:
        return (
            isinstance(value, Tuple) and len(value) == 2 and all(isinstance(x, int) for x in value)
        )
    elif custom_type is IntegerSet:
        return isinstance(value, FrozenSet) and all(isinstance(x, int) for x in value)
    elif custom_type is Grid:
        return isinstance(value, Tuple) and all(
            isinstance(x, Tuple) and all(isinstance(y, int) for y in x) for x in value
        )
    elif custom_type is Cell:
        return (
            isinstance(value, Tuple)
            and len(value) == 2
            and isinstance(value[0], int)
            and isinstance(value[1], Tuple)
            and len(value[1]) == 2
            and all(isinstance(x, int) for x in value[1])
        )
    elif custom_type is Object:
        return isinstance(value, FrozenSet) and all(
            isinstance(x, Tuple)
            and len(x) == 2
            and isinstance(x[0], int)
            and isinstance(x[1], Tuple)
            and len(x[1]) == 2
            and all(isinstance(y, int) for y in x[1])
            for x in value
        )
    elif custom_type is Objects:
        return isinstance(value, FrozenSet) and all(
            isinstance(x, FrozenSet)
            and all(
                isinstance(y, Tuple)
                and len(y) == 2
                and isinstance(y[0], int)
                and isinstance(y[1], Tuple)
                and len(y[1]) == 2
                and all(isinstance(z, int) for z in y[1])
                for y in x
            )
            for x in value
        )
    elif custom_type is Indices:
        return isinstance(value, FrozenSet) and all(
            isinstance(x, Tuple) and len(x) == 2 and all(isinstance(y, int) for y in x)
            for x in value
        )
    elif custom_type is IndicesSet:
        return isinstance(value, FrozenSet) and all(
            isinstance(x, FrozenSet)
            and all(
                isinstance(y, Tuple) and len(y) == 2 and all(isinstance(z, int) for z in y)
                for y in x
            )
            for x in value
        )
    elif custom_type is GridTuple:
        return isinstance(value, Tuple) and all(check_custom_type(x, Grid) for x in value)
    else:
        return False


def get_custom_type(value: Any) -> Optional[Any]:
    outer_type = type(value)
    possible_custom_types = []

    if outer_type is tuple:
        possible_custom_types = [IntegerTuple, Grid, Cell, GridTuple]
    elif outer_type is frozenset:
        possible_custom_types = [IntegerSet, Object, Objects, Indices, IndicesSet]

    for custom_type in possible_custom_types:
        if check_custom_type(value, custom_type):
            return CUSTOM_TYPE_NAMES[custom_type]
    else:
        possible_custom_types = [Boolean, Integer]
        for custom_type in possible_custom_types:
            if check_custom_type(value, custom_type):
                return CUSTOM_TYPE_NAMES[custom_type]
    return None


class Arrow(object):
    """Helper class for indicating function type: input -> output and written func(args)."""

    def __init__(self, inputs: list, output):
        self.inputs = list(inputs)
        self.output = output
        self.n_inputs = len(inputs)

    def __eq__(self, other):
        if isinstance(other, str):
            return False
        return self.inputs == other.inputs and self.output == other.output

    def __hash__(self):
        return hash((self.inputs, self.output))
