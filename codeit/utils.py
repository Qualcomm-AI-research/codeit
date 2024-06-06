# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from importlib import import_module
from typing import Any, Dict, Iterable, List, Union

from matplotlib import pyplot as plt


def transform_to_function(input_str: str, function_name: str) -> str:
    header = f"def {function_name}(I):\n"
    indented_content = "    " + input_str.strip().replace("\n", "\n    ")
    footer = "\n    return O"
    return header + indented_content + footer


def get_tokenizer(config):
    tokenizer_class = get_class(config.data.dataloader.tokenizer.cls)
    tokenizer = tokenizer_class.from_pretrained(
        config.data.dataloader.tokenizer.name, cache_dir=config.model.cache_dir
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = config.data.dataloader.tokenizer.pad_token_id
    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = config.data.dataloader.tokenizer.eos_token_id
    if (
        tokenizer.pad_token_id != config.data.dataloader.tokenizer.pad_token_id
        or tokenizer.eos_token_id != config.data.dataloader.tokenizer.eos_token_id
    ):
        raise Exception("Mismatch in tokenizer")
    return tokenizer


def get_class(class_str: str) -> Any:
    """
    A little util to import a class from any package, withot the need for an explicit import
    statement in the header. Useful when your classes can come from many different modules (like HF models).
    Can actually work also with modules which are not classes, like functions.
    :param class_str: The name of the class, inclusive of namespace from the root of the
     module is comes from (e.g. torch.nn.Linear, NOT just Linear
    :return: The class itself.
    """
    package = import_module(".".join(class_str.split(".")[:-1]))
    return getattr(package, class_str.split(".")[-1])


def get_grid_size(grid):
    num_rows = len(grid)
    num_columns = len(grid[0]) if num_rows > 0 else 0
    return (num_rows, num_columns)


def get_num_pixels(grid):
    num_rows = len(grid)
    num_columns = len(grid[0]) if num_rows > 0 else 0
    return num_rows if num_columns == 0 else num_rows * num_columns
