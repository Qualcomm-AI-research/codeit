# Code adapted from https://github.com/michaelhodel/arc-dsl
# Copyright (c) 2023 Michael Hodel, licensed under MIT

import os
import re

from codeit import PROJECT_FOLDER_PATH


def find_function_names(filename):
    with open(filename, "r") as file:
        contents = file.read()
    pattern = r"def\s+(\w+)\("
    function_names = re.findall(pattern, contents)
    return function_names


F = False
T = True

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
TEN = 10

NEG_ONE = -1
NEG_TWO = -2

DOWN = (1, 0)
RIGHT = (0, 1)
UP = (-1, 0)
LEFT = (0, -1)

ORIGIN = (0, 0)
UNITY = (1, 1)
NEG_UNITY = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)

ZERO_BY_TWO = (0, 2)
TWO_BY_ZERO = (2, 0)
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)

PRIMITIVE_CONSTANTS = {
    "Boolean": ["F", "T"],
    "Integer": [
        "ZERO",
        "ONE",
        "TWO",
        "THREE",
        "FOUR",
        "FIVE",
        "SIX",
        "SEVEN",
        "EIGHT",
        "NINE",
        "TEN",
        "NEG_ONE",
        "NEG_TWO",
    ],
    "IntegerTuple": [
        "DOWN",
        "RIGHT",
        "UP",
        "LEFT",
        "ORIGIN",
        "UNITY",
        "NEG_UNITY",
        "UP_RIGHT",
        "DOWN_LEFT",
        "ZERO_BY_TWO",
        "TWO_BY_ZERO",
        "TWO_BY_TWO",
        "THREE_BY_THREE",
    ],
}
PRIMITIVE_FUNCTIONS = find_function_names(
    filename=os.path.join(PROJECT_FOLDER_PATH, "codeit/dsl/dsl.py")
)
