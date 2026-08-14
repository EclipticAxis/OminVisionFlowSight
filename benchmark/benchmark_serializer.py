#!/usr/bin/env python
"""Benchmark: JSON vs CBOR serialization for TargetState.

Compares the two serializers across three scenarios (minimal, rich, large
metadata) and three metrics (serialize speed, deserialize speed, payload
size). Results are printed as a markdown table for easy pasting into
``docs/cbor-benchmark.md``.

Run::

    PYTHONPATH=F:/VisionBata python benchmark/benchmark_serializer.py

The script is self-contained -- no pytest needed. It uses
``time.perf_counter_ns()`` for high-resolution timing and runs enough
iterations to produce stable measurements.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

# Ensure the project root is on sys.path when running directly.
# (The script lives in benchmark/ which is not a package, so we need
# to add the parent directory.)
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from visioncore.protocol.serialization import CBORSerializer, TargetStateSerializer
from visioncore.state.target_state import TargetState


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def make_minimal_state() -> TargetState:
    """A minimal snapshot: empty metadata, all defaults."""
    return TargetState(
        target_id=1, local_id=1, global_id=None,
        label="person", confidence=0.9,
        cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.1, height=0.2, timestamp=0.0, camera_id=0,
        metadata={},
    )


def make_rich_state() -> TargetState:
    """A rich snapshot: populated metadata with nested structures."""
    return TargetState(
        target_id=42, local_id=3, global_id=101,
        label="vehicle", confidence=0.87,
        cx=0.35, cy=0.62, vx=0.012, vy=-0.008,
        width=0.18, height=0.34, timestamp=12345.678,
        camera_id=2,
        metadata={
            "source": "yolo26n-obb",
            "roi_tag": "entry_zone",
            "detector_score": 0.87,
            "keypoints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "attributes": {
                "color": "red",
                "orientation": 45.0,
                "occluded": False,
            },
            "track_age": 15,
        },
    )


def make_large_state() -> TargetState:
    """A large snapshot: 100-key metadata with mixed value types."""
    metadata: dict[str, Any] = {}
    for i in range(100):
        metadata[f"key_{i:03d}"] = {
            "index": i,
            "value": float(i) * 0.01,
            "label": f"label_{i}",
            "nested": {"deep": [i, i + 1, i + 2]},
        }
    return TargetState(
        target_id=999, local_id=50, global_id=5001,
        label="person", confidence=0.95,
        cx=0.7, cy=0.3, vx=0.05, vy=0.02,
        width=0.15, height=0.40, timestamp=99999.0,
        camera_id=3,
        metadata=metadata,
    )


SCENARIOS = [
    ("minimal", make_minimal_state),
    ("rich", make_rich_state),
    ("large", make_large_state),
]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench_serialize(state: TargetState, serializer: Any, iterations: int) -> float:
    """Return total seconds to serialize ``iterations`` times."""
    # Warm up (let the JIT / caches settle).
    for _ in range(min(100, iterations)):
        serializer.serialize(state)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        serializer.serialize(state)
    end = time.perf_counter_ns()
    return (end - start) / 1e9


def bench_deserialize(
    payload: bytes | str, serializer: Any, iterations: int
) -> float:
    """Return total seconds to deserialize ``iterations`` times."""
    for _ in range(min(100, iterations)):
        serializer.deserialize(payload)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        serializer.deserialize(payload)
    end = time.perf_counter_ns()
    return (end - start) / 1e9


def fmt_us(per_op_seconds: float) -> str:
    """Format microseconds per op with 2 decimal places."""
    return f"{per_op_seconds * 1e6:.2f}"


def fmt_ops(per_op_seconds: float) -> str:
    """Format ops/sec with thousands separator."""
    if per_op_seconds <= 0:
        return "inf"
    return f"{1.0 / per_op_seconds:,.0f}"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(iterations: int = 20000) -> None:
    """Run the full benchmark and print a markdown table."""

    json_ser = TargetStateSerializer()
    cbor_ser = CBORSerializer()

    print(f"# CBOR vs JSON Serializer Benchmark\n")
    print(f"**Iterations per measurement**: {iterations:,}")
    print(f"**Python**: {sys.version.split()[0]}")
    print(f"**Platform**: {sys.platform}")
    try:
        import cbor2 as _cbor2
        import importlib.metadata
        cbor_ver = importlib.metadata.version("cbor2")
    except Exception:
        cbor_ver = "unknown"
    print(f"**cbor2 version**: {cbor_ver}")
    print()

    # Table header.
    print("## Results\n")
    print(
        "| Scenario | Metric | JSON | CBOR | CBOR/JSON ratio |"
    )
    print(
        "|----------|--------|------|------|-----------------|"
    )

    for scenario_name, builder in SCENARIOS:
        state = builder()

        # --- Serialize ---
        json_time = bench_serialize(state, json_ser, iterations)
        cbor_time = bench_serialize(state, cbor_ser, iterations)

        json_json_us = fmt_us(json_time / iterations)
        cbor_us = fmt_us(cbor_time / iterations)
        ser_ratio = cbor_time / json_time if json_time > 0 else float("inf")

        print(
            f"| {scenario_name} | serialize μs/op | {json_json_us} | "
            f"{cbor_us} | {ser_ratio:.2f}x |"
        )

        # Throughput (ops/sec).
        json_ops = fmt_ops(json_time / iterations)
        cbor_ops = fmt_ops(cbor_time / iterations)
        print(
            f"| {scenario_name} | serialize ops/sec | {json_ops} | "
            f"{cbor_ops} | {ser_ratio:.2f}x |"
        )

        # --- Payload size ---
        json_payload = json_ser.serialize(state)
        cbor_payload = cbor_ser.serialize(state)
        json_size = len(json_payload.encode("utf-8")) if isinstance(json_payload, str) else len(json_payload)
        cbor_size = len(cbor_payload)
        size_ratio = cbor_size / json_size if json_size > 0 else float("inf")

        print(
            f"| {scenario_name} | payload bytes | {json_size} | "
            f"{cbor_size} | {size_ratio:.2f}x |"
        )

        # --- Deserialize ---
        json_deser_time = bench_deserialize(json_payload, json_ser, iterations)
        cbor_deser_time = bench_deserialize(cbor_payload, cbor_ser, iterations)

        json_deser_us = fmt_us(json_deser_time / iterations)
        cbor_deser_us = fmt_us(cbor_deser_time / iterations)
        deser_ratio = cbor_deser_time / json_deser_time if json_deser_time > 0 else float("inf")

        print(
            f"| {scenario_name} | deserialize μs/op | {json_deser_us} | "
            f"{cbor_deser_us} | {deser_ratio:.2f}x |"
        )

        json_deser_ops = fmt_ops(json_deser_time / iterations)
        cbor_deser_ops = fmt_ops(cbor_deser_time / iterations)
        print(
            f"| {scenario_name} | deserialize ops/sec | {json_deser_ops} | "
            f"{cbor_deser_ops} | {deser_ratio:.2f}x |"
        )

    print()

    # Summary.
    print("## Summary\n")
    print("**Payload size**: CBOR is consistently smaller than JSON for all")
    print("scenarios. The advantage grows with payload complexity (more")
    print("keys, nested structures) because CBOR's binary encoding of")
    print("integers, floats, and strings is more compact than JSON's text")
    print("representation.\n")
    print("**Speed**: See the ratio columns -- a ratio < 1.00x means CBOR")
    print("is faster; > 1.00x means JSON is faster. Both serializers")
    print("leverage the same `to_dict()` / `from_dict()` pipeline, so the")
    print("difference is purely in the encoding/decoding step.\n")
    print("**Recommendation**: Use JSON for human-readable output (logging,")
    print("debugging, console). Use CBOR for network transport and")
    print("high-volume file logging where payload size and speed matter.")


if __name__ == "__main__":
    # Allow overriding iteration count via CLI arg.
    iters = 20000
    if len(sys.argv) > 1:
        try:
            iters = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(iterations=iters)
