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
import contextlib
import dataclasses
import os
import re
import typing as tp
import warnings
from dataclasses import dataclass
from functools import partial

import chex
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import tree_util as tu
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..tree_util import named_tree_map

MIN_SHARDING_SIZE = int(os.environ.get("MIN_SHARDING_SIZE", "16384"))
LOG_SHARDING_MOVE = os.environ.get("LOG_SHARDING_MOVE", "false") in [
	"true",
	"yes",
	"1",
	"on",
]

AxisType = tp.Optional[tp.Union[tp.Tuple[str, ...], str, tp.Any]]


def names_in_current_mesh(*names: str) -> bool:
	"""
	Check if the given names are present in the current JAX mesh.

	Args:
	    *names: Variable number of axis names to check.

	Returns:
	    True if all given names are present in the current mesh, False otherwise.
	"""
	mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
	return set(names) <= set(mesh_axis_names)


def make_shard_and_gather_fns(
	partition_specs: tp.Dict[str, PartitionSpec],
	mesh: tp.Optional[Mesh] = None,
) -> tp.Tuple[tp.Dict[str, tp.Callable], tp.Dict[str, tp.Callable]]:
	"""
	Create shard and gather functions based on given partition specs and mesh.

	This function generates dictionaries of shard and gather functions that can be used
	to distribute and collect arrays across a JAX mesh. The functions are specifically
	designed for use with Flax's `tu.tree_map`.

	Args:
	        partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.
	        mesh: The JAX mesh to use for sharding. If None, the current mesh is used.

	Returns:
	        A tuple containing two dictionaries:
	                - `shard_fns`: A dictionary mapping parameter names to their corresponding shard functions.
	                - `gather_fns`: A dictionary mapping parameter names to their corresponding gather functions.
	"""
	if mesh is None:
		mesh = get_incontext_mesh()

	named_shardings = tu.tree_map(
		lambda p: NamedSharding(mesh=mesh, spec=p),
		partition_specs,
	)

	def make_shard_fn(sharding: NamedSharding) -> tp.Callable:
		"""
		Create a shard function for a specific partition spec.
		"""
		if jax.process_count() > 1:

			@partial(jax.jit, out_shardings=sharding)
			def _self_shard(tensor):
				return jnp.asarray(tensor)

			def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
				with mesh:
					tensor = jax.block_until_ready(_self_shard(tensor))
					assert tensor.sharding == sharding, "sharding Failed!."
				return tensor

			return shard_fn
		else:

			def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
				with mesh:
					tensor = with_sharding_constraint(tensor, sharding=sharding)
				return tensor

			return shard_fn

	def make_gather_fn(sharding: NamedSharding) -> tp.Callable:
		"""
		Create a gather function for a specific partition spec.
		"""

		@partial(jax.jit, out_shardings=NamedSharding(mesh=mesh, spec=PartitionSpec()))
		def _self_gather(tensor):
			return jnp.asarray(tensor)

		def gather_fn(tensor: jnp.ndarray) -> jnp.ndarray:
			return jax.device_get(jax.block_until_ready(_self_gather(tensor)))

		return gather_fn

	shard_fns = tu.tree_map(make_shard_fn, named_shardings)
	gather_fns = tu.tree_map(make_gather_fn, named_shardings)
	return shard_fns, gather_fns


def get_names_from_partition_spec(
	partition_specs: tp.Dict[str, PartitionSpec],
) -> tp.List[str]:
	"""
	Extract axis names from a partition specification.

	This function recursively iterates through the provided `partition_specs`
	dictionary and extracts all unique axis names used in the sharding specifications.

	Args:
	        partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.

	Returns:
	        A list of unique axis names used in the partition specs.
	"""
	names = set()
	if isinstance(partition_specs, dict):
		partition_specs = partition_specs.values()
	for item in partition_specs:
		if item is None:
			continue
		elif isinstance(item, str):
			names.add(item)
		else:
			names.update(get_names_from_partition_spec(item))
	return list(names)


def with_sharding_constraint(
	arr: jnp.ndarray,
	sharding: tp.Dict[str, tp.Union[PartitionSpec, NamedSharding]],
) -> jnp.ndarray:
	"""
	Apply sharding constraints if axis names are present in the current mesh.

	This is a smarter version of `jax.lax.with_sharding_constraint`. It only applies the
	sharding constraint if all the axis names specified in the `partition_specs` are
	present in the current JAX mesh.

	Args:
	        arr: The JAX array to apply sharding constraints to.
	        sharding: A dictionary mapping parameter names to their respective `PartitionSpec`.

	Returns:
	        The JAX array with sharding constraints applied (if applicable).
	"""
	if isinstance(arr, (jax.Array, jnp.ndarray)):
		if isinstance(sharding, NamedSharding):
			mesh = sharding.mesh
			sharding = sharding.spec
		else:
			mesh = None
		if mesh is None:
			mesh = get_incontext_mesh()
		axis_names = get_names_from_partition_spec(sharding)
		if names_in_current_mesh(*axis_names):
			with mesh or contextlib.nullcontext():
				arr = _with_sharding_constraint(arr, sharding)
	return arr


def match_partition_rules(
	rules: tp.List[tp.Tuple[str, PartitionSpec]],
	tree: tp.Dict,
) -> tp.Dict:
	"""
	Match partition rules to parameters based on their names.

	This function takes a list of partition rules (regular expressions and
	corresponding `PartitionSpec`) and applies them to a dictionary of parameters
	based on their names. It's useful for automatically defining sharding strategies.

	Args:
	        rules: A list of tuples, where each tuple contains:
	                         - A regular expression to match parameter names.
	                         - A `PartitionSpec` to apply if the name matches.
	        tree: A dictionary of parameters, where keys are parameter names.

	Returns:
	        A dictionary with the same keys as `tree`, but values are replaced
	        with the corresponding `PartitionSpec` based on matching rules.
	"""

	def get_partition_spec(name: str, leaf: jnp.ndarray) -> PartitionSpec:
		"""
		Determine the partition spec for a parameter based on its name.
		"""

		if not hasattr(leaf, "shape"):
			return PartitionSpec()
		size = np.prod(leaf.shape)
		if len(leaf.shape) == 0:
			""" Don't partition scalar values. """
			return PartitionSpec()

		for rule, ps in rules:
			if re.search(rule, name) is not None:
				if size < MIN_SHARDING_SIZE:
					if LOG_SHARDING_MOVE:
						warnings.warn(
							f"PartitionSpec Related to {name} was safer and faster being local array.",
							stacklevel=1,
						)
					return PartitionSpec()
				if len(ps) > leaf.ndim:
					ps = PartitionSpec(*tuple(ps[: leaf.ndim]))
					if LOG_SHARDING_MOVE:
						warnings.warn(
							f"PartitionSpec Related to {name} went out of range (will be auto trimed to {ps}).",
							stacklevel=1,
						)
				return ps
		raise ValueError(f"Partition rule not found for param: {name}")

	return named_tree_map(get_partition_spec, tree, sep="/")


def analyze_sharding_strategy(
	pytree: tp.Any,
	partition_specs: tp.Dict[str, PartitionSpec],
	mesh: tp.Optional[Mesh] = None,
) -> tp.Dict:
	"""
	Analyzes the effectiveness of a sharding strategy.

	Returns metrics like:
	- Memory usage per device
	- Load balance
	- Communication costs
	"""
	if mesh is None:
		mesh = get_incontext_mesh()

	analysis = {
		"total_parameters": 0,
		"sharded_parameters": 0,
		"memory_per_device": {},
		"balance_score": 0.0,
		"partition_stats": {},
	}

	def analyze_leaf(path: str, array: np.ndarray, spec: PartitionSpec):
		total_size = np.prod(array.shape) * array.dtype.itemsize
		analysis["total_parameters"] += np.prod(array.shape)

		if spec != PartitionSpec():
			analysis["sharded_parameters"] += np.prod(array.shape)

		# Calculate per-device memory
		sharded_size = total_size
		for axis, name in enumerate(spec):
			if name is not None:
				sharded_size //= mesh.shape[name]

		return sharded_size

	# Traverse the pytree and collect statistics
	tu.tree_map_with_path(analyze_leaf, pytree, partition_specs)

	return analysis


def create_pattern_based_partition_spec(
	pattern: str,
	mesh: tp.Optional[Mesh] = None,
	default_spec: tp.Optional[PartitionSpec] = None,
) -> tp.Callable[[str, chex.Array], PartitionSpec]:
	"""
	Creates a function that returns PartitionSpec based on parameter name patterns.

	Example:
	        pattern_fn = create_pattern_based_partition_spec(
	                "attention|mlp->data,hidden->model"
	        )
	"""
	if default_spec is None:
		default_spec = PartitionSpec()
	if mesh is None:
		mesh = get_incontext_mesh()

	rules = []
	for rule in pattern.split(","):
		if "->" in rule:
			patterns, spec = rule.split("->")
			patterns = patterns.split("|")
			spec = PartitionSpec(*spec.split("."))
			rules.extend((pattern, spec) for pattern in patterns)

	def get_partition_spec(name: str, array: chex.Array) -> PartitionSpec:
		for pattern, spec in rules:
			if re.search(pattern, name):
				return spec
		return default_spec

	return get_partition_spec


def extract_sharding_structure(pytree: tp.Any) -> tp.Any:
	"""
	Extract a PyTree of NamedShardings matching the input structure.
	Returns None for leaves without shardings.
	"""
	leaves, treedef = jax.tree_util.tree_flatten(pytree)

	sharding_leaves = []
	for leaf in leaves:
		if isinstance(leaf, jax.Array) and (shard := leaf.sharding) is not None:
			sharding_leaves.append(shard if isinstance(shard, NamedSharding) else None)
		else:
			sharding_leaves.append(None)

	return jax.tree_util.tree_unflatten(treedef, sharding_leaves)


def get_shardings_with_structure(pytree: tp.Any) -> tp.Any:
	"""
	Returns a PyTree matching the input structure containing either:
	- NamedSharding objects where present
	- None for leaves without NamedShardings
	"""
	return extract_sharding_structure(pytree)


def get_incontext_mesh() -> Mesh:
	"""Retrieves the mesh object active in the current execution context.

	This function accesses the physical mesh defined within the thread's
	resource environment (pxla.thread_resources.env.physical_mesh).

	Returns:
	    MeshType: The active mesh object for the current context.

	Raises:
	    AssertionError: If no mesh is found in the current context
	                    (i.e., mesh.empty() is True).
	"""
	mesh = pxla.thread_resources.env.physical_mesh
	if mesh.empty() if callable(mesh.empty) else mesh.empty:
		raise AssertionError("No mesh found under this context manager.")
	# It might be better practice to raise a more specific exception type
	# e.g., class NoActiveMeshError(RuntimeError): pass
	# raise NoActiveMeshError("No mesh found under this context manager.")
	return mesh


def get_axes_size_in_mesh(axis_names: AxisType, mesh: tp.Optional[Mesh] = None) -> int:
	"""
	Calculates the total size of the specified mesh axes.

	If a single axis name (string) is provided, it returns the size of that
	dimension in the mesh. If a sequence (list or tuple) of axis names is
	provided, it returns the product of the sizes of all specified axes.

	If no mesh is explicitly provided, it uses the mesh active in the
	current context obtained via `get_current_mesh()`.

	Args:
	    axis_names: The name of a single mesh axis (str) or a sequence
	                (list/tuple) of axis names whose sizes should be multiplied.
	    mesh: The mesh object to query. If None, the current context's mesh
	          is used. Defaults to None.

	Returns:
	    int: The size of the single specified axis, or the product of the sizes
	         of the sequence of specified axes.

	Raises:
	    KeyError: If any of the specified `axis_names` are not found in the
	              mesh's dimensions.
	    AssertionError: If `mesh` is None and no mesh is found in the current
	                   context (raised by `get_current_mesh()`).
	"""
	if mesh is None:
		mesh = get_incontext_mesh()

	# Assuming mesh.shape behaves like a dictionary {axis_name: size}
	mesh_shape: tp.Dict[str, int] = mesh.shape

	if isinstance(axis_names, str):
		# Raises KeyError if axis_names is not a valid key
		return mesh_shape[axis_names]
	elif isinstance(axis_names, (list, tuple)):
		product = 1
		# Iterate in the provided order, though order doesn't matter for product
		for axis in axis_names:
			# Raises KeyError if axis is not a valid key
			product *= mesh_shape[axis]
		return product
	else:
		# Handle unexpected type for axis_names
		raise TypeError(f"axis_names must be str or Sequence[str], got {type(axis_names)}")


def get_mesh_axis_names(mesh: tp.Optional[Mesh] = None) -> tp.List[str]:
	"""Retrieves the names of all axes defined in the mesh.

	These names typically correspond to the dimensions used for sharding or
	parallelism.

	If no mesh is explicitly provided, it uses the mesh active in the
	current context obtained via `get_current_mesh()`.

	Args:
	    mesh: The mesh object to query. If None, the current context's mesh
	          is used. Defaults to None.

	Returns:
	    List[str]: A list containing the names of all axes in the mesh.

	Raises:
	    AssertionError: If `mesh` is None and no mesh is found in the current
	                   context (raised by `get_current_mesh()`).
	"""
	if mesh is None:
		mesh = get_incontext_mesh()

	mesh_shape: tp.Dict[str, int] = mesh.shape
	return list(mesh_shape.keys())


def get_mesh_axis_size(axis_names: AxisType) -> int:
	"""Calculates the total number of devices along the specified mesh axis or axes.

	Args:
	    axis_names: The name of a single mesh axis (str) or a sequence (list/tuple)
	                of mesh axis names. The order in the sequence does not affect
	                the result (product is commutative).

	Returns:
	    The total number of devices (size) in the submesh defined by the axis/axes.
	    Returns 1 if axis_names is an empty sequence.

	Raises:
	    TypeError: If axis_names is not a str or a sequence of str.
	"""
	if isinstance(axis_names, str):
		# Size along a single axis dimension
		return lax.psum(1, axis_name=axis_names)
	elif isinstance(axis_names, (list, tuple)):
		if not axis_names:
			return 1  # The size of a submesh with zero dimensions is 1

		# Calculate the product of sizes along each specified axis
		product = 1
		for axis in axis_names:
			product *= lax.psum(1, axis_name=axis)
		return product
		# Alternative using math.prod (Python 3.8+)
		# return math.prod(lax.psum(1, axis_name=ax) for ax in axis_names)
	else:
		raise TypeError(
			f"Input 'axis_names' must be a string or sequence (list/tuple), "
			f"but got type {type(axis_names)}"
		)


def get_submesh_device_index(axis_names: AxisType) -> int:
	"""
	Calculates the linear index of the current device within the specified mesh axes.

	This effectively flattens the multi-dimensional coordinates of the device
	within the submesh defined by `axis_names` into a single integer index.

	IMPORTANT: It assumes the input `axis_names` sequence is ordered from
	most major to most minor dimension. The calculation performs a
	row-major-like flattening based on this order.

	Args:
	    axis_names: The name of a single mesh axis (str) or a sequence (list/tuple)
	                of mesh axis names, ordered from major to minor.

	Returns:
	    The 0-based linear index of the current device within the submesh.
	    Returns 0 if axis_names is an empty sequence.

	Raises:
	    TypeError: If axis_names is not a str or a sequence of str.
	"""
	if isinstance(axis_names, str):
		# Index along a single axis dimension
		return lax.axis_index(axis_name=axis_names)
	elif isinstance(axis_names, (list, tuple)):
		if not axis_names:
			return 0  # Index within a zero-dimensional submesh is 0

		linear_index = 0
		stride = 1
		# Iterate from the minor axis to the major axis (reverse of the input order)
		# This implements the formula: idx = sum(local_idx[dim] * stride[dim])
		# where stride[dim] = product(size[k] for k > dim)
		for axis in reversed(axis_names):
			index_on_axis = lax.axis_index(axis_name=axis)
			linear_index += index_on_axis * stride

			# Update stride for the next (more major) dimension
			axis_size = lax.psum(1, axis_name=axis)  # Use lax.psum, not the other func
			stride *= axis_size
		return linear_index
	else:
		raise TypeError(
			f"Input 'axis_names' must be a string or sequence (list/tuple), "
			f"but got type {type(axis_names)}"
		)


@dataclass(frozen=True)
class PartitionAxis:
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

	# --- Mesh Dimension Names ---
	data_parallel_axis: str = "dp"
	fully_sharded_data_parallel_axis: str = "fsdp"
	tensor_parallel_axis: str = "tp"
	sequence_parallel_axis: str = "sp"
	expert_parallel_axis: str = "ep"

	batch_axis: AxisType = ...
	sequence_axis: AxisType = ...
	query_sequence_axis: AxisType = ...
	head_axis: AxisType = ...
	key_sequence_axis: AxisType = ...
	hidden_state_axis: AxisType = ...
	mlp_intermediate_axis: AxisType = ...
	vocab_axis: AxisType = ...
	expert_axis: AxisType = ...
	expert_gate_axis: AxisType = None

	attention_dim_axis: AxisType = None  # Usually not partitioned
	bias_head_sequence_axis: AxisType = None
	bias_key_sequence_axis: AxisType = None

	# --- Generation Specific ---
	generation_batch_axis: AxisType = None
	generation_query_sequence_axis: AxisType = None  # Often length 1, not sharded
	generation_head_axis: AxisType = ...
	generation_key_sequence_axis: AxisType = ...
	generation_attention_dim_axis: AxisType = None

	def __post_init__(self):
		"""
		Resolve default partitioning strategies after initialization.

		Since the dataclass is frozen, we need to use object.__setattr__ to modify fields.
		"""

		# Helper to set attribute on frozen dataclass
		def set_attr(obj, name, value):
			object.__setattr__(obj, name, value)

		def _operate(val):
			return val is Ellipsis

		# Resolve fields that need defaults
		if _operate(self.batch_axis):
			# Default batch sharding uses both FSDP and DP dimensions

			_shardin = (self.fully_sharded_data_parallel_axis, self.data_parallel_axis)
			set_attr(self, "batch_axis", _shardin)

		if _operate(self.sequence_axis):
			set_attr(self, "sequence_axis", self.sequence_parallel_axis)

		if _operate(self.query_sequence_axis):
			set_attr(self, "query_sequence_axis", self.sequence_parallel_axis)

		if _operate(self.head_axis):
			set_attr(self, "head_axis", self.tensor_parallel_axis)

		if _operate(self.key_sequence_axis):
			set_attr(self, "key_sequence_axis", self.sequence_parallel_axis)

		if _operate(self.hidden_state_axis):
			set_attr(self, "hidden_state_axis", self.tensor_parallel_axis)

		if _operate(self.mlp_intermediate_axis):
			set_attr(self, "mlp_intermediate_axis", self.tensor_parallel_axis)

		if _operate(self.vocab_axis):
			set_attr(self, "vocab_axis", self.tensor_parallel_axis)

		if _operate(self.expert_axis):
			set_attr(self, "expert_axis", self.expert_parallel_axis)

		if _operate(self.generation_head_axis):
			set_attr(self, "generation_head_axis", self.tensor_parallel_axis)

		if _operate(self.generation_key_sequence_axis):
			set_attr(self, "generation_key_sequence_axis", self.sequence_parallel_axis)

		self._safety_check()

	def _safety_check(self):
		"""Ensures no essential attributes are left uninitialized (as Ellipsis)."""
		for field in dataclasses.fields(self):
			val = getattr(self, field.name)
			if val == Ellipsis:
				raise ValueError(f"`{field.name}` shouldn't be ellipsis")
