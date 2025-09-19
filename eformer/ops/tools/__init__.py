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

"""Tools and utilities for eformer.ops operations.

This module provides specialized tools and utilities for advanced operation
implementations, including quantization support and other optimization tools.

Submodules:
    quantization: Quantized array implementations and operators
        - Array1B: 1-bit quantized arrays
        - Array8B: 8-bit quantized arrays
        - ArrayNF4: NF4 quantized arrays
        - RSR operators: Reduced precision operators

The tools module serves as a container for specialized functionality that
extends the core operations framework with domain-specific optimizations
and alternative implementations.

Example Usage:
    >>> from eformer.ops.tools import quantization
    >>> # Use quantization tools for memory-efficient operations
"""

from . import quantization

__all__ = ("quantization",)
