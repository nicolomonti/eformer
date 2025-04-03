import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

from eformer.pytree import FrozenPyTree, PyTree


def benchmark_frozen_vs_nonfrozen():
	class FrozenModel(FrozenPyTree):
		weights: jnp.ndarray
		biases: jnp.ndarray

	class MutableModel(PyTree):
		weights: jnp.ndarray
		biases: jnp.ndarray

	# Sizes to benchmark
	sizes = [32, 64, 128, 256, 512, 1024, 2048]

	# Results storage
	jit_times_frozen = []
	jit_times_mutable = []
	grad_times_frozen = []
	grad_times_mutable = []
	vmap_times_frozen = []
	vmap_times_mutable = []

	# Number of repetitions for each measurement
	n_repeats = 10

	for size in sizes:
		print(f"Benchmarking size: {size}")

		# Create random data
		key = jax.random.PRNGKey(42)
		key, subkey = jax.random.split(key)
		weights = jax.random.normal(key, (size, size))
		key, subkey = jax.random.split(key)
		biases = jax.random.normal(subkey, (size,))
		key, subkey = jax.random.split(key)
		x = jax.random.normal(subkey, (size,))
		key, subkey = jax.random.split(key)
		target = jax.random.normal(subkey, (size,))

		# Create models
		frozen_model = FrozenModel(weights=weights, biases=biases)
		mutable_model = MutableModel(weights=weights, biases=biases)

		# Define model function
		def model_fn(model, x):
			return jnp.dot(x, model.weights) + model.biases

		# Define loss function
		def loss_fn(model, x, target):
			pred = model_fn(model, x)
			return jnp.mean((pred - target) ** 2)

		# JIT compilation benchmark
		jitted_fn_frozen = jax.jit(partial(model_fn, frozen_model))
		jitted_fn_mutable = jax.jit(partial(model_fn, mutable_model))

		# Warmup
		_ = jitted_fn_frozen(x).block_until_ready()
		_ = jitted_fn_mutable(x).block_until_ready()

		# Measure JIT performance
		start = time.time()
		for _ in range(n_repeats):
			_ = jitted_fn_frozen(x).block_until_ready()
		jit_time_frozen = (time.time() - start) / n_repeats

		start = time.time()
		for _ in range(n_repeats):
			_ = jitted_fn_mutable(x).block_until_ready()
		jit_time_mutable = (time.time() - start) / n_repeats

		jit_times_frozen.append(jit_time_frozen)
		jit_times_mutable.append(jit_time_mutable)

		# Gradient benchmark
		grad_fn_frozen = jax.grad(partial(loss_fn, frozen_model), argnums=1)
		grad_fn_mutable = jax.grad(partial(loss_fn, mutable_model), argnums=1)

		# Warmup
		_ = grad_fn_frozen(x, target).block_until_ready()
		_ = grad_fn_mutable(x, target).block_until_ready()

		# Measure gradient performance
		start = time.time()
		for _ in range(n_repeats):
			_ = grad_fn_frozen(x, target).block_until_ready()
		grad_time_frozen = (time.time() - start) / n_repeats

		start = time.time()
		for _ in range(n_repeats):
			_ = grad_fn_mutable(x, target).block_until_ready()
		grad_time_mutable = (time.time() - start) / n_repeats

		grad_times_frozen.append(grad_time_frozen)
		grad_times_mutable.append(grad_time_mutable)

		# VMAP benchmark
		# Create batch data
		batch_size = 16
		key, subkey = jax.random.split(key)
		batch_x = jax.random.normal(subkey, (batch_size, size))

		vmap_fn_frozen = jax.vmap(partial(model_fn, frozen_model))
		vmap_fn_mutable = jax.vmap(partial(model_fn, mutable_model))

		# Warmup
		_ = vmap_fn_frozen(batch_x).block_until_ready()
		_ = vmap_fn_mutable(batch_x).block_until_ready()

		# Measure vmap performance
		start = time.time()
		for _ in range(n_repeats):
			_ = vmap_fn_frozen(batch_x).block_until_ready()
		vmap_time_frozen = (time.time() - start) / n_repeats

		start = time.time()
		for _ in range(n_repeats):
			_ = vmap_fn_mutable(batch_x).block_until_ready()
		vmap_time_mutable = (time.time() - start) / n_repeats

		vmap_times_frozen.append(vmap_time_frozen)
		vmap_times_mutable.append(vmap_time_mutable)

	# Plot results
	plt.figure(figsize=(15, 10))

	# JIT times
	plt.subplot(3, 1, 1)
	plt.plot(sizes, jit_times_frozen, "b-", label="Frozen")
	plt.plot(sizes, jit_times_mutable, "r-", label="Mutable")
	plt.title("JIT Execution Time")
	plt.xlabel("Size")
	plt.ylabel("Time (s)")
	plt.legend()
	plt.grid(True)

	# Gradient times
	plt.subplot(3, 1, 2)
	plt.plot(sizes, grad_times_frozen, "b-", label="Frozen")
	plt.plot(sizes, grad_times_mutable, "r-", label="Mutable")
	plt.title("Gradient Computation Time")
	plt.xlabel("Size")
	plt.ylabel("Time (s)")
	plt.legend()
	plt.grid(True)

	# VMAP times
	plt.subplot(3, 1, 3)
	plt.plot(sizes, vmap_times_frozen, "b-", label="Frozen")
	plt.plot(sizes, vmap_times_mutable, "r-", label="Mutable")
	plt.title("VMAP Execution Time")
	plt.xlabel("Size")
	plt.ylabel("Time (s)")
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.savefig("frozen_vs_mutable_benchmark.png")
	plt.show()

	# Print speedup ratios
	print("\nSpeedup Ratios (Frozen/Mutable):")
	print("Size\tJIT\tGrad\tVMAP")
	for i, size in enumerate(sizes):
		jit_ratio = jit_times_frozen[i] / jit_times_mutable[i]
		grad_ratio = grad_times_frozen[i] / grad_times_mutable[i]
		vmap_ratio = vmap_times_frozen[i] / vmap_times_mutable[i]
		print(f"{size}\t{jit_ratio:.3f}\t{grad_ratio:.3f}\t{vmap_ratio:.3f}")

	# Calculate average speedups
	avg_jit = sum(jit_times_frozen) / sum(jit_times_mutable)
	avg_grad = sum(grad_times_frozen) / sum(grad_times_mutable)
	avg_vmap = sum(vmap_times_frozen) / sum(vmap_times_mutable)

	print("\nAverage Speedup Ratios (Frozen/Mutable):")
	print(f"JIT: {avg_jit:.3f}")
	print(f"Grad: {avg_grad:.3f}")
	print(f"VMAP: {avg_vmap:.3f}")

	# Return the data for further analysis
	return {
		"sizes": sizes,
		"jit_times_frozen": jit_times_frozen,
		"jit_times_mutable": jit_times_mutable,
		"grad_times_frozen": grad_times_frozen,
		"grad_times_mutable": grad_times_mutable,
		"vmap_times_frozen": vmap_times_frozen,
		"vmap_times_mutable": vmap_times_mutable,
	}


if __name__ == "__main__":
	benchmark_frozen_vs_nonfrozen()
