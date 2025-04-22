# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp


class _Empty: ...


Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = tp.Sequence[int]

Mesh = jax.sharding.Mesh

AxisNames = tuple[str, ...]
AxisIdxes = tuple[int, ...]
AxisType = tp.Optional[tp.Union[tp.Tuple[str, ...], str, tp.Any]]

BATCH = "__BATCH__"
LENGTH = "__LENGTH__"
KV_LENGTH = "__KV_LENGTH__"
QUERY_LENGTH = "__QUERY_LENGTH__"
EMBED = "__EMBED__"
HEAD = "__HEAD__"
MLP_INTERMEDIATE = "__MLP_INTERMEDIATE__"
VOCAB = "__VOCAB__"
EXPERT = "__EXPERT__"
EXPERT_GATE = "__EXPERT_GATE__"
HEAD_DIM = "__HEAD_DIM__"
BIAS_HEAD_SEQ = "__BIAS_HEAD_SEQ__"
BIAS_KV_SEQ = "__BIAS_KV_SEQ__"
MODE_DECODE = "__autoregressive__"
MODE_PREFILL = "__prefill__"
MODE_TRAIN = "__train__"
MODE_INSERT = "__insert__"

GENERATION_MODES = {
	MODE_DECODE,
	MODE_INSERT,
}

RUNTIME_MODES_TYPES = tp.Literal[
	MODE_DECODE,
	MODE_PREFILL,
	MODE_TRAIN,
	MODE_INSERT,
]


class DynamicShardingAxes(tp.NamedTuple):
	axes: tp.Sequence[tp.Optional[str]]
	mode: RUNTIME_MODES_TYPES | int  # type:ignore


class HiddenStateSharding(DynamicShardingAxes):
	axes = [BATCH, QUERY_LENGTH, EMBED]
	mode = 1


class AttnQSharding(DynamicShardingAxes):
	axes = [BATCH, QUERY_LENGTH, HEAD, HEAD_DIM]
	mode = 1


class AttnKVSharding(DynamicShardingAxes):
	axes = [BATCH, KV_LENGTH, HEAD, HEAD_DIM]
	mode = 1


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NOT_GIVEN = _Empty()
