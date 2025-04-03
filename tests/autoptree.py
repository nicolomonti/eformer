import unittest
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np

from eformer.pytree import FrozenPyTree, PyTree, auto_pytree, field


class TestPytreeStyles(unittest.TestCase):
	"""Test both decorator and inheritance styles for pytree nodes."""

	def test_decorator_style_basic(self):
		"""Test basic functionality with decorator style."""

		@auto_pytree
		class Vector:
			x: float
			y: float

		vec = Vector(1.0, 2.0)
		self.assertEqual(vec.x, 1.0)
		self.assertEqual(vec.y, 2.0)

		# Test mutability
		vec.x = 3.0
		self.assertEqual(vec.x, 3.0)

	def test_decorator_style_frozen(self):
		"""Test frozen functionality with decorator style."""

		@auto_pytree(frozen=True)
		class FrozenVector:
			x: float
			y: float

		vec = FrozenVector(1.0, 2.0)

		# Test immutability
		with self.assertRaises(FrozenInstanceError):
			vec.x = 3.0

		# Test replace
		new_vec = vec.replace(x=3.0)
		self.assertEqual(new_vec.x, 3.0)
		self.assertEqual(new_vec.y, 2.0)

	def test_inheritance_style_basic(self):
		"""Test basic functionality with inheritance style."""

		class Vector(PyTree):
			x: float
			y: float

		vec = Vector(1.0, 2.0)
		self.assertEqual(vec.x, 1.0)
		self.assertEqual(vec.y, 2.0)

		# Test mutability
		vec.x = 3.0
		self.assertEqual(vec.x, 3.0)

	def test_inheritance_style_frozen(self):
		"""Test frozen functionality with inheritance style."""

		class FrozenVector(PyTree, frozen=True):
			x: float
			y: float

		vec = FrozenVector(1.0, 2.0)

		# Test immutability
		with self.assertRaises(FrozenInstanceError):
			vec.x = 3.0

		# Test replace
		new_vec = vec.replace(x=3.0)
		self.assertEqual(new_vec.x, 3.0)
		self.assertEqual(new_vec.y, 2.0)

	def test_frozen_pytree_node(self):
		"""Test the FrozenPyTree convenience class."""

		class Vector(FrozenPyTree):
			x: float
			y: float

		vec = Vector(1.0, 2.0)

		# Test immutability
		with self.assertRaises(FrozenInstanceError):
			vec.x = 3.0

		# Test replace
		new_vec = vec.replace(x=3.0)
		self.assertEqual(new_vec.x, 3.0)
		self.assertEqual(new_vec.y, 2.0)

	def test_field_metadata_decorator(self):
		"""Test field metadata with decorator style."""

		@auto_pytree
		class Model:
			weights: jnp.ndarray
			bias: jnp.ndarray
			name: str = field(pytree_node=False)  # Explicitly mark as metadata

		weights = jnp.array([1.0, 2.0])
		bias = jnp.array(0.5)
		model = Model(weights, bias, "test_model")

		# Check that name is in meta_fields
		self.assertIn("name", model.__pytree_meta__["meta_fields"])
		self.assertNotIn("name", model.__pytree_meta__["data_fields"])

	def test_field_metadata_inheritance(self):
		"""Test field metadata with inheritance style."""

		class Model(PyTree):
			weights: jnp.ndarray
			bias: jnp.ndarray
			name: str = field(pytree_node=False)  # Explicitly mark as metadata

		weights = jnp.array([1.0, 2.0])
		bias = jnp.array(0.5)
		model = Model(weights, bias, "test_model")

		# Check that name is in meta_fields
		self.assertIn("name", model.__pytree_meta__["meta_fields"])
		self.assertNotIn("name", model.__pytree_meta__["data_fields"])

	def test_jax_transformations_decorator(self):
		"""Test JAX transformations with decorator style."""

		@auto_pytree(frozen=True)
		class Model:
			weights: jnp.ndarray
			bias: jnp.ndarray

		weights = jnp.array([1.0, 2.0])
		bias = jnp.array(0.5)
		model = Model(weights, bias)

		# Test jit
		@jax.jit
		def apply(model, x):
			return jnp.dot(x, model.weights) + model.bias

		x = jnp.array([1.0, 2.0])
		result = apply(model, x)
		expected = jnp.array(5.5)  # 1*1 + 2*2 + 0.5
		np.testing.assert_allclose(result, expected)

	def test_jax_transformations_inheritance(self):
		"""Test JAX transformations with inheritance style."""

		class Model(FrozenPyTree):
			weights: jnp.ndarray
			bias: jnp.ndarray

		weights = jnp.array([1.0, 2.0])
		bias = jnp.array(0.5)
		model = Model(weights, bias)

		# Test jit
		@jax.jit
		def apply(model, x):
			return jnp.dot(x, model.weights) + model.bias

		x = jnp.array([1.0, 2.0])
		result = apply(model, x)
		expected = jnp.array(5.5)  # 1*1 + 2*2 + 0.5
		np.testing.assert_allclose(result, expected)

	def test_json_serialization_decorator(self):
		"""Test JSON serialization with decorator style."""

		@auto_pytree(json_serializable=True)
		class Config:
			name: str
			values: tuple

		config = Config("test", (1, 2, 3))
		json_str = config.to_json()
		new_config = Config.from_json(json_str)

		self.assertEqual(new_config.name, "test")
		# The tuple becomes a list during JSON serialization
		self.assertListEqual(list(new_config.values), [1, 2, 3])

	def test_json_serialization_inheritance(self):
		"""Test JSON serialization with inheritance style."""

		class Config(PyTree):
			name: str
			values: tuple

		config = Config("test", (1, 2, 3))
		json_str = config.to_json()
		new_config = Config.from_json(json_str)

		self.assertEqual(new_config.name, "test")
		# The tuple becomes a list during JSON serialization
		self.assertListEqual(list(new_config.values), [1, 2, 3])

	def test_disable_json_serialization(self):
		"""Test disabling JSON serialization."""

		class Config(PyTree, json_serializable=False):
			name: str
			values: tuple

		config = Config("test", (1, 2, 3))
		self.assertFalse(hasattr(config, "to_json"))
		self.assertFalse(hasattr(config, "from_json"))

	def test_nested_structures(self):
		"""Test nested PyTree structures."""

		class Point(FrozenPyTree):
			x: float
			y: float

		class Line(FrozenPyTree):
			start: Point
			end: Point

		start = Point(0.0, 0.0)
		end = Point(1.0, 1.0)
		line = Line(start, end)

		# Test nested immutability
		with self.assertRaises(FrozenInstanceError):
			line.start = Point(2.0, 2.0)

		with self.assertRaises(FrozenInstanceError):
			line.start.x = 2.0

		# Test replace with nested structure
		new_start = start.replace(x=2.0)
		new_line = line.replace(start=new_start)

		self.assertEqual(new_line.start.x, 2.0)
		self.assertEqual(line.start.x, 0.0)  # Original unchanged

	def test_explicit_meta_fields(self):
		"""Test explicit meta_fields specification."""

		class Vector(PyTree, meta_fields=("z",)):
			x: float
			y: float
			z: float  # Should be meta despite being float

		vec = Vector(1.0, 2.0, 3.0)

		# Check that z is in meta_fields
		self.assertIn("z", vec.__pytree_meta__["meta_fields"])
		self.assertNotIn("z", vec.__pytree_meta__["data_fields"])

	def test_auto_detection(self):
		"""Test auto-detection of non-JAX types."""

		class Model(PyTree):
			weights: jnp.ndarray
			bias: jnp.ndarray
			name: str  # Should be auto-detected as meta

			# Define the callable as a method
			def process_fn(self, x):
				return x + 1

		weights = jnp.array([1.0, 2.0])
		bias = jnp.array(0.5)
		model = Model(weights, bias, "model")

		# Check meta fields
		meta_fields = model.__pytree_meta__["meta_fields"]
		self.assertIn("name", meta_fields)

	def test_override_auto_detection(self):
		"""Test overriding auto-detection with field."""

		class Model(PyTree):
			weights: jnp.ndarray = field()  # Explicitly a data field
			bias: jnp.ndarray = field(pytree_node=False)  # Override to meta field
			name: str = field(pytree_node=True)  # Override to data field

		weights = jnp.array([1.0, 2.0])
		bias = jnp.array(0.5)
		model = Model(weights, bias, "model")

		# Check fields
		data_fields = model.__pytree_meta__["data_fields"]
		meta_fields = model.__pytree_meta__["meta_fields"]

		self.assertIn("weights", data_fields)
		self.assertIn("bias", meta_fields)
		self.assertIn("name", data_fields)  # Overridden to be a data field


if __name__ == "__main__":
	unittest.main()
