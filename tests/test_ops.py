from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp

from eformer.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    PersistentCache,
    benchmark,
)

# Types for optional modifiers
ScoreMod = Callable[[jax.Array], jax.Array]  # logits -> logits
MaskMod = Callable[[tuple[int, ...]], jax.Array]  # shape -> bool mask


def _coerce_precision(precision):
    if precision is None or not isinstance(precision, tuple):
        precision = (precision, precision)
    qk_prec, pv_prec = precision
    return qk_prec, pv_prec


@dataclass(frozen=True)
class NativeAttnCfg:
    impl: Literal["stable"] = "stable"


class NativeAttention(Kernel[NativeAttnCfg, jax.Array]):
    op_id: str = "attention.Native"
    version: str = "0"

    def prepare(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        *,
        precision: jax.lax.PrecisionLike | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike] | None = None,
        score_mod: ScoreMod | None = None,
        mask_mod: MaskMod | None = None,
        dropout_mask: jax.Array | None = None,
        dropout_rate: float = 0.0,
        normalize_output: bool = True,
        return_residuals: bool = False,
        q_sharding: Any | None = None,
        k_sharding: Any | None = None,
    ):
        if q_sharding is not None or k_sharding is not None:
            raise NotImplementedError("Sharding not supported in NativeAttention.")
        if (dropout_rate != 0.0) and (dropout_mask is None):
            raise ValueError("dropout_mask must be provided if dropout_rate != 0.0")

        if k.shape[-2] not in (1, q.shape[-2]):
            if q.shape[-2] % k.shape[-2] != 0:
                raise ValueError("num_heads_q must be a multiple of num_heads_kv")
            repeats = q.shape[-2] // k.shape[-2]
            k = jnp.repeat(k, repeats, axis=-2)
            v = jnp.repeat(v, repeats, axis=-2)

        kwargs = dict(
            precision=_coerce_precision(precision),
            score_mod=score_mod,
            mask_mod=mask_mod,
            dropout_mask=dropout_mask,
            dropout_rate=dropout_rate,
            normalize_output=normalize_output,
            return_residuals=return_residuals,  # ignored in run()
        )
        return (q, k, v), kwargs

    def run(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        *,
        cfg: NativeAttnCfg,
        precision: tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike],
        score_mod: ScoreMod | None,
        mask_mod: MaskMod | None,
        dropout_mask: jax.Array | None,
        dropout_rate: float,
        normalize_output: bool,
        return_residuals: bool,  # not used; kept for API compatibility
    ) -> jax.Array:
        qk_prec, pv_prec = precision

        logits = jnp.einsum("...qhd,...khd->...hqk", q, k, precision=qk_prec)

        if score_mod is not None:
            logits = score_mod(logits)

        if mask_mod is not None:
            mask = mask_mod(logits.shape)
            mask_val = float(jnp.finfo(logits.dtype).min)
            logits = jnp.where(mask, logits, mask_val)

        redtype = jnp.promote_types(logits.dtype, jnp.float32)
        x_max = jnp.max(logits.astype(redtype), axis=-1, keepdims=True)
        probs_unnorm = jnp.exp(logits - x_max)

        probs = jax.lax.cond(
            normalize_output,
            lambda _: probs_unnorm / (jnp.sum(probs_unnorm, axis=-1, keepdims=True) + jnp.finfo(redtype).tiny),
            lambda _: probs_unnorm,
            operand=None,
        )

        if dropout_mask is not None:
            probs = probs * dropout_mask.astype(probs.dtype) / (1.0 - dropout_rate)

        out = jnp.einsum("...hqk,...khd->...qhd", probs.astype(v.dtype), v, precision=pv_prec)
        return out.astype(q.dtype)

    def heuristic_cfg(self, inv: Invocation[NativeAttnCfg, Any]) -> NativeAttnCfg:
        return NativeAttnCfg()

    def candidate_cfgs(self, inv: Invocation[NativeAttnCfg, Any]) -> Iterable[NativeAttnCfg]:
        return [self.heuristic_cfg(inv)]


def attention(
    executor: Executor,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    precision=None,
    score_mod: ScoreMod | None = None,
    mask_mod: MaskMod | None = None,
    dropout_mask: jax.Array | None = None,
    dropout_rate: float = 0.0,
    normalize_output: bool = True,
    return_residuals: bool = False,
    q_sharding=None,
    k_sharding=None,
):
    kernel = NativeAttention()
    return executor(
        kernel,
        q,
        k,
        v,
        precision=precision,
        score_mod=score_mod,
        mask_mod=mask_mod,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
        q_sharding=q_sharding,
        k_sharding=k_sharding,
    )


selector = ConfigSelectorChain(
    cache=ConfigCache(),
    policy=AutotunePolicy(cache_miss_fallback="heuristics"),
    persistent=PersistentCache("ops-cache.json"),
)
execute = Executor(selector)

q = jnp.ones((1, 128, 32, 64), jnp.bfloat16)  # (B, Tq, Hq, D)
k = jnp.ones((1, 256, 1, 64), jnp.bfloat16)  # (B, Tk, Hkv, D)
v = jnp.ones((1, 256, 1, 64), jnp.bfloat16)


def call_fn(*args, **kwargs):
    return attention(execute, *args, **kwargs)


tooks = benchmark(
    call_fn,
    q,
    k,
    v,
    score_mod=lambda s: s / s.shape[-1] ** 0.5,
    mask_mod=lambda shape: jnp.tril(jnp.ones(shape, dtype=bool)),
    iters=50,
)
print(tooks)
