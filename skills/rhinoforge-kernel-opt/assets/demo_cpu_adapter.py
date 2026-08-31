"""Small deterministic CPU adapter for ``scripts/bench.py``.

This demonstrates the adapter protocol and exercises fused-style output
contracts.  ``rmsnorm_quant_mxfp8`` uses an int8 payload plus a per-row scale
to keep the example portable; it is a protocol fixture, not an MXFP8 device
implementation or performance reference.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


_WORKLOADS = {
    "gemm",
    "gemm_silu_mul",
    "gemm_add",
    "gemm_rope",
    "rmsnorm_quant_mxfp8",
    "bad_candidate",
}
_GRAPH = {
    "workload": None,
    "build_count": 0,
    "replay_count": 0,
    "cache_size": 0,
    "invariant_ok": True,
}
_BARE_GRAPH = {
    "workload": None,
    "build_count": 0,
    "replay_count": 0,
    "cache_size": 0,
    "invariant_ok": True,
}


def _reset_graph(workload: str) -> None:
    _GRAPH.update(
        {
            "workload": workload,
            "build_count": 0,
            "replay_count": 0,
            "cache_size": 0,
            "invariant_ok": True,
        }
    )
    _BARE_GRAPH.update(
        {
            "workload": workload,
            "build_count": 0,
            "replay_count": 0,
            "cache_size": 0,
            "invariant_ok": True,
        }
    )


def _advance_graph(state: dict[str, Any]) -> None:
    if state["build_count"] == 0:
        state["build_count"] = 1
        state["cache_size"] = 1
    else:
        state["replay_count"] += 1


def make_case(workload: str, seed: int, device: str) -> dict[str, Any]:
    if workload not in _WORKLOADS:
        raise ValueError(f"unknown demo workload {workload!r}")
    if _GRAPH["workload"] != workload:
        _reset_graph(workload)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows, inner, columns = 8, 16, 12
    case: dict[str, Any] = {
        "workload": workload,
        "x": torch.randn(rows, inner, generator=generator, dtype=torch.float32),
    }
    if workload != "rmsnorm_quant_mxfp8":
        case["weight"] = torch.randn(
            inner, columns, generator=generator, dtype=torch.float32
        ) / math.sqrt(inner)
    if workload == "gemm_silu_mul":
        case["weight_up"] = torch.randn(
            inner, columns, generator=generator, dtype=torch.float32
        ) / math.sqrt(inner)
    elif workload == "gemm_add":
        case["residual"] = torch.randn(
            rows, columns, generator=generator, dtype=torch.float32
        )
    elif workload == "gemm_rope":
        case["weight_k"] = torch.randn(
            inner, columns, generator=generator, dtype=torch.float32
        ) / math.sqrt(inner)
        case["position"] = torch.arange(rows, dtype=torch.float32)
        case["inv_frequency"] = torch.linspace(
            1.0, 0.01, columns // 2, dtype=torch.float32
        )
    elif workload == "rmsnorm_quant_mxfp8":
        case["norm_weight"] = torch.randn(
            inner, generator=generator, dtype=torch.float32
        ) * 0.05 + 1.0
        case["epsilon"] = 1.0e-6
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }


def _rope(value: torch.Tensor, position: torch.Tensor, inv_frequency: torch.Tensor) -> torch.Tensor:
    angle = position[:, None] * inv_frequency[None, :]
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    result = torch.empty_like(value)
    result[..., 0::2] = even * cosine - odd * sine
    result[..., 1::2] = even * sine + odd * cosine
    return result


def _rmsnorm(case: dict[str, Any]) -> torch.Tensor:
    value = case["x"]
    variance = value.float().square().mean(dim=-1, keepdim=True)
    normalized = value.float() * torch.rsqrt(variance + float(case["epsilon"]))
    return normalized * case["norm_weight"].float()


def _rmsnorm_quant(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = _rmsnorm(case)
    scale = normalized.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
    payload = torch.round(normalized / scale).clamp(-127, 127).to(torch.int8)
    # The nested payload+scale contract is deliberate: the benchmark must not
    # silently flatten or ignore either output.
    return payload, scale


def _compute(case: dict[str, Any]) -> Any:
    workload = case["workload"]
    if workload == "rmsnorm_quant_mxfp8":
        return _rmsnorm_quant(case)
    gemm = torch.matmul(case["x"], case["weight"])
    if workload in {"gemm", "bad_candidate"}:
        return gemm
    if workload == "gemm_silu_mul":
        up = torch.matmul(case["x"], case["weight_up"])
        return F.silu(gemm) * up
    if workload == "gemm_add":
        return gemm + case["residual"]
    if workload == "gemm_rope":
        key = torch.matmul(case["x"], case["weight_k"])
        return (
            _rope(gemm, case["position"], case["inv_frequency"]),
            _rope(key, case["position"], case["inv_frequency"]),
        )
    raise AssertionError(f"unhandled workload {workload!r}")


def reference(case: dict[str, Any]) -> Any:
    return _compute(case)


def candidate(case: dict[str, Any]) -> Any:
    _advance_graph(_GRAPH)
    result = _compute(case)
    if case["workload"] == "bad_candidate":
        return result + 0.25
    return result


def synchronize() -> None:
    # CPU operations used here are synchronous.  A real RPU adapter must call
    # the backend synchronization primitive at this boundary.
    return None


def fp32_anchor(case: dict[str, Any]) -> Any:
    # The quantized profile returns the unencoded FP32 target; bench.py compares
    # it to dequantize(candidate_output, case). Other profiles return their
    # direct FP32 output tree.
    if case["workload"] == "rmsnorm_quant_mxfp8":
        return _rmsnorm(case)
    return _compute(case)


def dequantize(output: Any, case: dict[str, Any]) -> torch.Tensor:
    if case["workload"] != "rmsnorm_quant_mxfp8":
        raise NotImplementedError("dequantize is defined only for quantized output")
    payload, scale = output
    return payload.float() * scale.float()


def quantized_output_roles() -> dict[str, str]:
    """Describe every encoded leaf without inferring semantics from dtype."""

    return {"$[0]": "payload", "$[1]": "scale"}


def bare_operation(case: dict[str, Any]) -> Any:
    """Return the profile's frozen bare compute baseline."""

    _advance_graph(_BARE_GRAPH)
    if case["workload"] == "rmsnorm_quant_mxfp8":
        return _rmsnorm(case)
    if case["workload"] == "gemm_rope":
        return (
            torch.matmul(case["x"], case["weight"]),
            torch.matmul(case["x"], case["weight_k"]),
        )
    if case["workload"] == "gemm_silu_mul":
        return (
            torch.matmul(case["x"], case["weight"]),
            torch.matmul(case["x"], case["weight_up"]),
        )
    return torch.matmul(case["x"], case["weight"])


def bare_reference(case: dict[str, Any]) -> Any:
    """Independent semantic oracle for the profile's bare baseline."""

    if case["workload"] == "rmsnorm_quant_mxfp8":
        value = case["x"].float()
        variance = value.square().mean(dim=-1, keepdim=True)
        return (
            value * torch.rsqrt(variance + float(case["epsilon"]))
        ) * case["norm_weight"].float()
    if case["workload"] == "gemm_rope":
        return (
            torch.einsum("mk,kn->mn", case["x"], case["weight"]),
            torch.einsum("mk,kn->mn", case["x"], case["weight_k"]),
        )
    if case["workload"] == "gemm_silu_mul":
        return (
            torch.einsum("mk,kn->mn", case["x"], case["weight"]),
            torch.einsum("mk,kn->mn", case["x"], case["weight_up"]),
        )
    return torch.einsum("mk,kn->mn", case["x"], case["weight"])


def graph_state() -> dict[str, Any]:
    return {
        "build_count": int(_GRAPH["build_count"]),
        "replay_count": int(_GRAPH["replay_count"]),
        "cache_size": int(_GRAPH["cache_size"]),
        "invariant_ok": bool(_GRAPH["invariant_ok"]),
    }


def arm_graph_state(arm: str) -> dict[str, Any]:
    """Return independent BUILD/REPLAY state for a timed comparison arm."""

    if arm == "candidate":
        state = _GRAPH
    elif arm == "bare_operation":
        state = _BARE_GRAPH
    else:
        raise ValueError(f"unknown timing arm {arm!r}")
    return {
        "build_count": int(state["build_count"]),
        "replay_count": int(state["replay_count"]),
        "cache_size": int(state["cache_size"]),
        "invariant_ok": bool(state["invariant_ok"]),
    }


def mutate_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    replacement = torch.randn(
        tuple(case["x"].shape), generator=generator, dtype=torch.float32
    ).to(case["x"].device)
    # copy_ changes semantic bytes without replacing the input allocation.
    case["x"].copy_(replacement)
    return case
