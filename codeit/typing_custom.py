# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import torch
from numpy import typing as npt

# Represents a generic tensor type.
# This could be an np.ndarray, or a torch.Tensor.
TensorType = Union[npt.ArrayLike, torch.Tensor]
# Either a plain tensor, or a dict or tuple of tensors (or StructTensors).
TensorStructType = Union[TensorType, dict, tuple]

# A shape of a tensor.
TensorShape = Union[Tuple[int], List[int]]

# A boolean TensorType like object (can't enforce this)
BoolTensorType = TensorType  # This should only contain boolean values.


class GenerationConfig(TypedDict):
    method: str
    kwargs: Dict[Any, Any]
