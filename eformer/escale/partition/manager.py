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
import contextvars
import dataclasses
import typing as tp

import jax
from jax.sharding import PartitionSpec

from eformer.common_types import (
	BATCH,
	BIAS_HEAD_SEQ,
	BIAS_KV_SEQ,
	EMBED,
	EXPERT,
	EXPERT_GATE,
	GENERATION_MODES,
	HEAD,
	HEAD_DIM,
	KV_LENGTH,
	LENGTH,
	MLP_INTERMEDIATE,
	MODE_DECODE,
	MODE_TRAIN,
	NOT_GIVEN,
	QUERY_LENGTH,
	RUNTIME_MODES_TYPES,
	VOCAB,
	AxisType,
	DynamicShardingAxes,
)
from eformer.pytree import xTree

from .constraints import get_corrected_named_sharding, with_sharding_constraint


class PartitionAxis(xTree):
	"""
	Configuration for partitioning model axes across a device mesh.

	Defines the mesh dimension names for standard parallelism strategies and maps
	logical model axes to these dimensions. Allows overriding defaults.

	Mesh Dimensions:
	    data_parallel_axis: Name for data parallel mesh dim. Default: "dp".
	    fully_sharded_data_parallel_axis: Name for FSDP mesh dim. Default: "fsdp".
	    tensor_parallel_axis: Name for tensor parallel mesh dim. Default: "tp".
	    sequence_parallel_axis: Name for sequence parallel mesh dim. Default: "sp".
	    expert_parallel_axis: Name for expert parallel mesh dim (MoE). Default: "ep".

	Logical Model Axes:
	    Maps logical tensor axes (like batch, sequence, hidden) to one or more
	    mesh dimension names defined above, or None if not partitioned.
	    Defaults are derived from the standard mesh dimension names but can be
	    overridden during instantiation. For example, `head_axis` defaults to
	    the value of `tensor_parallel_axis` ('tp').

	"""

	data_parallel_axis: str = "dp"
	fully_sharded_data_parallel_axis: str = "fsdp"
	tensor_parallel_axis: str = "tp"
	sequence_parallel_axis: str = "sp"
	expert_parallel_axis: str = "ep"

	batch_axis: AxisType = NOT_GIVEN
	sequence_axis: AxisType = NOT_GIVEN
	query_sequence_axis: AxisType = NOT_GIVEN
	head_axis: AxisType = NOT_GIVEN
	key_sequence_axis: AxisType = NOT_GIVEN
	hidden_state_axis: AxisType = NOT_GIVEN
	mlp_intermediate_axis: AxisType = NOT_GIVEN
	vocab_axis: AxisType = NOT_GIVEN
	expert_axis: AxisType = NOT_GIVEN
	expert_gate_axis: AxisType = None

	attention_dim_axis: AxisType = None
	bias_head_sequence_axis: AxisType = None
	bias_key_sequence_axis: AxisType = None

	decode_batch_axis: AxisType = NOT_GIVEN
	decode_query_sequence_axis: AxisType = None
	decode_head_axis: AxisType = NOT_GIVEN
	decode_key_sequence_axis: AxisType = NOT_GIVEN
	decode_attention_dim_axis: AxisType = None

	_SEMANTIC_MAP: tp.ClassVar[tp.Dict[str, str]] = {
		BATCH: "batch_axis",
		LENGTH: "sequence_axis",
		QUERY_LENGTH: "query_sequence_axis",
		KV_LENGTH: "key_sequence_axis",
		EMBED: "hidden_state_axis",
		HEAD: "head_axis",
		MLP_INTERMEDIATE: "mlp_intermediate_axis",
		VOCAB: "vocab_axis",
		EXPERT: "expert_axis",
		EXPERT_GATE: "expert_gate_axis",
		HEAD_DIM: "attention_dim_axis",
		BIAS_HEAD_SEQ: "bias_head_sequence_axis",
		BIAS_KV_SEQ: "bias_key_sequence_axis",
		"_": None,
	}

	_STANDARD_TO_GENERATION_ATTR_MAP: tp.ClassVar[tp.Dict[str, str]] = {
		"batch_axis": "decode_batch_axis",
		"query_sequence_axis": "decode_query_sequence_axis",
		"key_sequence_axis": "decode_key_sequence_axis",
		"head_axis": "decode_head_axis",
		"attention_dim_axis": "decode_attention_dim_axis",
	}

	def __post_init__(self):
		resolved_values = {}

		def resolve_field(name, default_logic):
			current_value = getattr(self, name)
			if current_value is NOT_GIVEN:
				resolved_values[name] = default_logic()
			elif name not in resolved_values:
				resolved_values[name] = current_value

		def get_resolved(name):
			return resolved_values.get(name, getattr(self, name))

		resolve_field(
			"batch_axis",
			lambda: (self.fully_sharded_data_parallel_axis, self.data_parallel_axis),
		)
		resolve_field("sequence_axis", lambda: self.sequence_parallel_axis)
		resolve_field("query_sequence_axis", lambda: self.sequence_parallel_axis)
		# Default qS = S rule
		resolve_field("head_axis", lambda: self.tensor_parallel_axis)
		resolve_field("key_sequence_axis", lambda: self.sequence_parallel_axis)
		# Default kS = S rule
		resolve_field("hidden_state_axis", lambda: self.tensor_parallel_axis)
		resolve_field("mlp_intermediate_axis", lambda: self.tensor_parallel_axis)
		resolve_field("vocab_axis", lambda: self.tensor_parallel_axis)
		resolve_field("expert_axis", lambda: self.expert_parallel_axis)

		resolve_field("decode_batch_axis", lambda: get_resolved("batch_axis"))
		resolve_field("decode_head_axis", lambda: get_resolved("head_axis"))
		resolve_field("decode_key_sequence_axis", lambda: get_resolved("key_sequence_axis"))

		for fld in dataclasses.fields(self):
			if fld.name not in resolved_values and fld.name not in [
				"_SEMANTIC_MAP",
				"_STANDARD_TO_GENERATION_ATTR_MAP",
			]:
				resolved_values[fld.name] = getattr(self, fld.name)

		for name, value in resolved_values.items():
			object.__setattr__(self, name, value)

		self._safety_check()

	def _safety_check(self):
		for fld in dataclasses.fields(self):
			if fld.name not in ["_SEMANTIC_MAP", "_STANDARD_TO_GENERATION_ATTR_MAP"]:
				val = getattr(self, fld.name)
				if val == NOT_GIVEN:
					raise ValueError(f"Partitioning rule `{fld.name}` was not resolved.")

	def resolve_spec(
		self,
		axes: tp.Sequence[tp.Optional[str]],
		mode: RUNTIME_MODES_TYPES,  # type:ignore
	) -> PartitionSpec:
		"""
		Generates a PartitionSpec from a sequence of semantic axis names and a mode.

		Args:
				axes: A sequence of semantic axis name strings (e.g., [BATCH, LENGTH, HEAD])
							or None (or "_") for axes that shouldn't be sharded.
				mode: The current operational mode (e.g., MODE_TRAIN,
							MODE_DECODE) which determines if generation-specific
							rules should be applied.

		Returns:
				A jax.sharding.PartitionSpec instance.

		Raises:
				ValueError: If an unknown semantic axis name is encountered.
				LookupError: If an internal attribute name isn't found (shouldn't happen).
		"""
		resolved_rules: list[AxisType] = []

		for axis_name in axes:
			if axis_name is None or axis_name == "_":
				resolved_rules.append(None)
				continue

			standard_attr_name = self._SEMANTIC_MAP.get(axis_name)
			if standard_attr_name is None:
				raise ValueError(f"Unknown semantic axis name: '{axis_name}'")

			target_attr_name = standard_attr_name

			if mode in GENERATION_MODES:
				gen_attr_name = self._STANDARD_TO_GENERATION_ATTR_MAP.get(standard_attr_name)
				if gen_attr_name:
					if hasattr(self, gen_attr_name):
						gen_val = getattr(self, gen_attr_name)
						if gen_val is not None and gen_val is not NOT_GIVEN:
							target_attr_name = gen_attr_name
			try:
				mesh_axis_rule: AxisType = getattr(self, target_attr_name)
			except AttributeError as e:
				raise LookupError(
					f"Internal error: Attribute '{target_attr_name}' not found in PartitionAxis instance."
				) from e

			if mesh_axis_rule is NOT_GIVEN:
				raise ValueError(
					f"Resolved axis rule for '{axis_name}' ('{target_attr_name}') is still NOT_GIVEN."
				)

			resolved_rules.append(mesh_axis_rule)

		return PartitionSpec(*resolved_rules)


_CURRENT_PARTITION_MANAGER: contextvars.ContextVar[tp.Optional["PartitionManager"]] = (
	contextvars.ContextVar("current_partition_manager", default=None)
)


_LAST_PARTITION_MANAGER: contextvars.ContextVar[tp.Optional["PartitionManager"]] = (
	contextvars.ContextVar("last_partition_manager", default=None)
)


class PartitionManager:
	"""
	Context manager for applying sharding constraints using PartitionAxis.

	Sets a context-local variable to make the current manager implicitly
	available via `get_current_partition_manager()` or `shard()`.

	Args:
	    paxis: The PartitionAxis instance defining the sharding strategy.
	"""

	def __init__(self, paxis: PartitionAxis):
		if not isinstance(paxis, PartitionAxis):
			raise TypeError(f"Expected PartitionAxis, got {type(paxis)}")
		self.paxis = paxis
		self._reset_token: tp.Optional[contextvars.Token] = None
		_LAST_PARTITION_MANAGER.set(self)

	def __enter__(self):
		"""Sets this manager as the active one in the current context."""
		self._reset_token = _CURRENT_PARTITION_MANAGER.set(self)
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		"""Resets the context variable to its previous state."""
		if self._reset_token is None:
			raise RuntimeError(
				"PartitionManager context exited without being properly entered."
			)

		_CURRENT_PARTITION_MANAGER.reset(self._reset_token)
		self._reset_token = None

	@staticmethod
	def shard(
		x: jax.Array,
		axes: tp.Sequence[tp.Optional[str]] = NOT_GIVEN,
		mode: RUNTIME_MODES_TYPES | int = NOT_GIVEN,  # type:ignore
		dynamic_axes: tp.Optional[DynamicShardingAxes] = NOT_GIVEN,
		auto_correct: bool = True,
	) -> jax.Array:
		"""
		Applies sharding constraint to an array based on the active PartitionManager context.

		Retrieves the current PartitionManager implicitly and uses it to resolve
		the shorthand and apply the constraint.

		Returns:
				The array `x` with the sharding constraint applied.

		Raises:
				LookupError: If called outside of an active PartitionManager context.
		"""

		if axes is NOT_GIVEN or mode is NOT_GIVEN:
			assert dynamic_axes is not NOT_GIVEN, (
				"if axes or mode is empty you should provide dynamic axes"
			)
			axes = dynamic_axes.axes
			mode = dynamic_axes.mode

		if isinstance(mode, int):
			mode = MODE_DECODE if x.shape[mode] == 1 else MODE_TRAIN
		manager = get_current_partition_manager()
		spec = manager.paxis.resolve_spec(axes, mode)
		if auto_correct:
			spec = get_corrected_named_sharding(x.shape, spec).spec
		return with_sharding_constraint(x, spec)

	def __str__(self):
		return "PartitionManager(...)"

	def __repr__(self):
		return "PartitionManager(...)"


def get_current_partition_manager() -> PartitionManager:
	"""
	Retrieves the currently active PartitionManager from the context.

	Raises:
	    LookupError: If called outside of an active PartitionManager context.
	"""
	manager = _CURRENT_PARTITION_MANAGER.get()
	if manager is None:
		raise LookupError(
			"Cannot get partition manager: Not currently within a "
			"`with PartitionManager(...)` context."
		)
	return manager


def get_partition_manager() -> PartitionManager:
	"""
	Retrieves the last active PartitionManager from the context.

	Raises:
	    LookupError: If called outside of an active PartitionManager context.
	"""
	manager = _LAST_PARTITION_MANAGER.get()
	if manager is None:
		raise LookupError("Cannot get partition manager: you havent created one yet!")
	return manager
