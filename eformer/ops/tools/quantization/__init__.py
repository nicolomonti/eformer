# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantization tools for memory-efficient neural network operations.

This module provides various quantized array implementations and operators
designed to reduce memory usage and improve computational efficiency while
maintaining acceptable precision for neural network computations.

Quantized Array Types:
    Array1B: 1-bit quantized arrays for extreme compression
    Array8B: 8-bit quantized arrays for balanced precision/efficiency
    ArrayNF4: NF4 (Normal Float 4-bit) quantized arrays for fine-grained control

Specialized Operators:
    RSROperatorBinary: Binary reduced precision operations
    RSROperatorTernary: Ternary reduced precision operations

These implementations enable:
    - Significant memory reduction for large models
    - Faster computation through reduced precision arithmetic
    - Configurable precision-efficiency trade-offs
    - Integration with the eformer.ops optimization framework

Example Usage:
    >>> from eformer.ops.tools.quantization import Array8B, ArrayNF4
    >>> # Create quantized arrays for memory-efficient storage
    >>> quantized_weights = Array8B.from_array(large_weight_matrix)
    >>> nf4_activations = ArrayNF4.from_array(activation_tensor)

Note:
    Quantization involves trade-offs between memory efficiency and numerical
    precision. Choose the appropriate quantization level based on your
    model's accuracy requirements and computational constraints.
"""

from .implicit_array_1bit import Array1B
from .implicit_array_8bit import Array8B
from .implicit_array_nf4 import ArrayNF4
from .implicit_array_rsr import RSROperatorBinary, RSROperatorTernary

__all__ = (
    "Array1B",
    "Array8B",
    "ArrayNF4",
    "RSROperatorBinary",
    "RSROperatorTernary",
)
