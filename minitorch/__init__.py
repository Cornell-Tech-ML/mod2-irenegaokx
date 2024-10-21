"""The package aggregates and re-exports various modules for tensor operations,
automatic differentiation, datasets, optimizers, and testing utilities.

Modules included:
- tensor_data: Contains utilities for tensor storage, indexing, and broadcasting.
- tensor: Provides the core tensor class and related operations.
- tensor_ops: Implements operations on tensors.
- tensor_functions: Contains differentiable functions for tensors.
- datasets: Includes datasets and data utilities.
- optim: Contains optimizers for training models.
- testing: Provides testing utilities for validating implementations.
- module: Provides a base class for creating models and modules.
- autodiff: Implements automatic differentiation.
- scalar: Defines scalar operations.
- scalar_functions: Contains differentiable scalar functions.

This package structure helps modularize tensor manipulation, differentiation,
and machine learning-related operations for reusability and clarity.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
