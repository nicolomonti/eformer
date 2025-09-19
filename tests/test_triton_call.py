import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("triton", reason="Triton is required")
import triton
import triton.language as tl

if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip("GPU backend required for Triton tests", allow_module_level=True)

from eformer.callib import triton_call


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    length,
    output_ptr,
    block_size: tl.constexpr,
):
    """Adds two vectors elementwise."""
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < length
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    out = x + y
    tl.store(output_ptr + offsets, out, mask=mask)


def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    assert x.shape == y.shape and x.dtype == y.dtype, "x and y must match shape/dtype"
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = 256
    grid = (triton.cdiv(int(x.size), block_size),)
    return triton_call(
        x,
        y,
        int(x.size),
        kernel=add_kernel,
        out_shape=out_shape,
        grid=grid,
        block_size=block_size,
    )


@triton.jit
def sigmoid_kernel(
    x_ptr,
    length,
    output_ptr,
    block_size: tl.constexpr,
):
    """Elementwise sigmoid with fp32 math, cast back to input dtype."""
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offs = block_start + tl.arange(0, block_size)
    mask = offs < length

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x32))
    tl.store(output_ptr + offs, y.to(x.dtype), mask=mask)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    assert jnp.issubdtype(x.dtype, jnp.floating), "sigmoid expects floating dtype"
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    block_size = 256
    grid = (triton.cdiv(int(x.size), block_size),)
    return triton_call(
        x,
        int(x.size),
        kernel=sigmoid_kernel,
        out_shape=out_shape,
        grid=grid,
        block_size=block_size,
    )


@triton.jit
def softmax_lastdim_kernel(
    x_ptr,
    n_rows,
    n_cols,
    output_ptr,
    block_size: tl.constexpr,
):
    """
    Row-wise softmax over last dimension.
    One program per row, processes up to block_size columns.
    Assumes block_size >= n_cols (wrapper chooses accordingly).
    """
    row_id = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = cols < n_cols
    row_start = row_id * n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf"))
    x32 = x.to(tl.float32)

    x_max = tl.max(x32, axis=0)
    x_shift = x32 - x_max
    num = tl.exp(x_shift)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(output_ptr + row_start + cols, y.to(x.dtype), mask=mask)


def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    assert jnp.issubdtype(x.dtype, jnp.floating), "softmax expects floating dtype"
    if axis != -1:
        x = jnp.moveaxis(x, axis, -1)
        undo_axis = True
    else:
        undo_axis = False

    assert x.ndim >= 1
    n_rows = int(x.size // x.shape[-1])
    n_cols = int(x.shape[-1])

    # Choose a power-of-two block large enough, capped at 1024
    block_size = 1 << (n_cols - 1).bit_length()
    block_size = min(block_size, 1024)
    if block_size < n_cols:
        # For very large n_cols you'd need tiling; tests avoid this case.
        raise ValueError("n_cols exceeds supported block size; add tiling if needed.")

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    y = triton_call(
        x,
        n_rows,
        n_cols,
        kernel=softmax_lastdim_kernel,
        out_shape=out_shape,
        grid=(n_rows,),
        block_size=block_size,
    )
    if undo_axis:
        y = jnp.moveaxis(y, -1, axis)
    return y


def _make_inputs(shape, dtype):
    size = int(jnp.prod(jnp.array(shape)))
    if jnp.issubdtype(dtype, jnp.integer):
        x = jnp.arange(size, dtype=dtype).reshape(shape)
        y = jnp.arange(size, dtype=dtype).reshape(shape) * jnp.array(2, dtype=dtype)
    else:
        base = jnp.arange(size, dtype=jnp.float32).reshape(shape)
        x = (base * 0.1 - 3.0).astype(dtype)
        y = (base * 0.2 + 1.0).astype(dtype)
    return x, y


def _tol(dtype):
    if dtype == jnp.float16:
        return dict(rtol=1e-2, atol=1e-2)
    return dict(rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.int32])
@pytest.mark.parametrize("shape", [(1,), (7,), (8,), (9,), (4, 5), (2, 3, 7)])
def test_add_matches_jnp_add(shape, dtype):
    x, y = _make_inputs(shape, dtype)
    out = add(x, y)
    expected = x + y

    if jnp.issubdtype(dtype, jnp.integer):
        assert jnp.array_equal(out, expected)
    else:
        assert jnp.allclose(out, expected, **_tol(dtype))

    assert out.shape == x.shape and out.dtype == x.dtype


def test_add_works_under_jit():
    x, y = _make_inputs((17,), jnp.float32)  # include tail coverage
    eager = add(x, y)
    compiled = jax.jit(add)(x, y)
    assert jnp.allclose(eager, compiled, rtol=1e-6, atol=1e-6)


def test_add_grad_if_supported():
    # If your triton_call registers JVP/VJP rules, this will pass; else we skip.
    x, y = _make_inputs((32,), jnp.float32)
    try:
        gx, gy = jax.grad(lambda a, b: jnp.sum(add(a, b)), argnums=(0, 1))(x, y)
    except Exception as e:
        pytest.skip(f"Autodiff not defined for triton_call/add: {e}")
    assert jnp.allclose(gx, jnp.ones_like(x), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(gy, jnp.ones_like(y), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(1,), (8,), (3, 5), (2, 2, 7), (4, 3, 2, 5)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_sigmoid_matches_jax(shape, dtype):
    x = (jnp.linspace(-6, 6, num=int(jnp.prod(jnp.array(shape))), dtype=jnp.float32).reshape(shape)).astype(dtype)
    y = sigmoid(x)
    y_ref = jax.nn.sigmoid(x)
    assert jnp.allclose(y, y_ref, **_tol(dtype))
    assert y.shape == x.shape and y.dtype == x.dtype


def test_sigmoid_jit_and_grad():
    x = jnp.linspace(-5, 5, 1024, dtype=jnp.float32)
    y_eager = sigmoid(x)
    y_jit = jax.jit(sigmoid)(x)
    assert jnp.allclose(y_eager, y_jit, rtol=1e-6, atol=1e-6)

    try:
        g = jax.grad(lambda t: jnp.sum(sigmoid(t)))(x)
    except Exception as e:
        pytest.skip(f"Autodiff not defined for triton_call/sigmoid: {e}")
    g_ref = jax.grad(lambda t: jnp.sum(jax.nn.sigmoid(t)))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(2, 8), (4, 33), (1, 257), (2, 3, 129)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_softmax_matches_jax(shape, dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, dtype=dtype) * 2.5
    y = softmax(x, axis=-1)
    y_ref = jax.nn.softmax(x, axis=-1)
    assert jnp.allclose(y, y_ref, **_tol(dtype))
    assert y.shape == x.shape and y.dtype == x.dtype


def test_softmax_shift_invariance():
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 97), dtype=jnp.float32)
    c = jnp.linspace(-100.0, 100.0, 4, dtype=jnp.float32).reshape(-1, 1)  # per-row shift
    y1 = softmax(x, axis=-1)
    y2 = softmax(x + c, axis=-1)
    assert jnp.allclose(y1, y2, rtol=1e-6, atol=1e-6)


def test_softmax_rows_sum_to_one():
    x = jax.random.normal(jax.random.PRNGKey(2), (7, 123), dtype=jnp.float32)
    y = softmax(x, axis=-1)
    s = jnp.sum(y, axis=-1)
    assert jnp.allclose(s, jnp.ones_like(s), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
