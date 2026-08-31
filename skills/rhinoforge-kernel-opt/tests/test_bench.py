from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

try:
    import torch
except ImportError:  # The help/schema tests remain discoverable without PyTorch.
    torch = None

SKILL_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = SKILL_ROOT / "scripts" / "bench.py"
ADAPTER_PATH = SKILL_ROOT / "assets" / "demo_cpu_adapter.py"
COMMIT = "a" * 40
PARENT = "b" * 40
CONTRACT = "c" * 64


def _load_bench():
    spec = importlib.util.spec_from_file_location("test_rhinoforge_kernel_bench", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_receipt(result, workload_ids):
    common_path = SKILL_ROOT / "scripts" / "campaign_common.py"
    spec = importlib.util.spec_from_file_location("test_campaign_common", common_path)
    assert spec is not None and spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    contract = {
        "measurement": {"metric": "latency_us", "direction": "min", "samples": 4},
        "workloads": [{"id": workload_id} for workload_id in workload_ids],
        "gates": {
            "max_abs": 1.0e-5,
            "max_rel": 1.0e-5,
            "anchor_max_abs": 1.0e-5,
            "anchor_max_rel": 1.0e-5,
            "require_same_dtype": True,
            "require_fp32_anchor": True,
            "require_graph_replay": True,
            "require_independent_outputs": True,
        },
    }
    return common.validate_result(result, contract, CONTRACT)


def _arguments(output: Path, *workloads: str) -> list[str]:
    result = ["--adapter", str(ADAPTER_PATH)]
    for workload in workloads:
        result.extend(("--workload", workload))
    result.extend(
        (
            "--device",
            "cpu",
            "--seed",
            "17",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--candidate-id",
            "cpu-demo",
            "--candidate-commit",
            COMMIT,
            "--parent-commit",
            PARENT,
            "--contract-sha256",
            CONTRACT,
            "--verdict",
            "signal",
            "--output",
            str(output),
        )
    )
    return result


class BenchTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "PyTorch is required for the CPU adapter")
    def test_cpu_campaign_covers_nested_mutation_and_raw_samples(self):
        bench = _load_bench()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        temporary_path = Path(temporary.name)
        output = temporary_path / "result.json"

        self.assertEqual(
            bench.main(
                _arguments(output, "gemm_silu_mul", "gemm_rope")
            ),
            0,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        json.dumps(result, allow_nan=False)
        _validate_receipt(result, ["gemm_silu_mul", "gemm_rope"])

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["candidate_id"], "cpu-demo")
        self.assertEqual(result["candidate_commit"], COMMIT)
        self.assertEqual(result["parent_commit"], PARENT)
        self.assertEqual(result["contract_sha256"], CONTRACT)
        self.assertEqual(result["verdict"], "signal")
        self.assertEqual(result["metric"]["name"], "latency_us")
        self.assertEqual(result["metric"]["direction"], "min")
        self.assertTrue(all(result["hard_gates"].values()))

        workloads = {item["id"]: item for item in result["workloads"]}
        self.assertEqual(set(workloads), {"gemm_silu_mul", "gemm_rope"})
        for workload in workloads.values():
            self.assertTrue(workload["eligible"])
            self.assertTrue(workload["hard_gates"])
            self.assertTrue(all(workload["hard_gates"].values()))
            self.assertEqual(len(workload["raw_samples"]["candidate_ns"]), 4)
            self.assertEqual(len(workload["raw_samples"]["reference_ns"]), 4)
            self.assertTrue(
                all(
                    isinstance(sample, int) and sample > 0
                    for samples in workload["raw_samples"].values()
                    for sample in samples
                )
            )
            summary = workload["summary"]
            for implementation in ("candidate", "reference"):
                self.assertGreater(summary[f"{implementation}_p50_ns"], 0)
                self.assertGreaterEqual(
                    summary[f"{implementation}_p95_ns"],
                    summary[f"{implementation}_p50_ns"],
                )
                self.assertGreaterEqual(summary[f"{implementation}_mad_ns"], 0)
                self.assertTrue(math.isfinite(summary[f"{implementation}_cv"]))
                self.assertEqual(summary[f"{implementation}_sample_count"], 4)
            self.assertTrue(workload["hard_gates"]["same_pointer_changed_value"])
            self.assertTrue(workload["hard_gates"]["different_pointer_same_value"])
            self.assertTrue(workload["hard_gates"]["retained_output"])
            self.assertTrue(workload["correctness"]["same_dtype"])
            self.assertTrue(workload["correctness"]["fp32_anchor"])
            self.assertTrue(workload["lifecycle"]["passed"])
            self.assertEqual(workload["lifecycle"]["graph_build_count"], 1)
            self.assertGreater(workload["lifecycle"]["graph_replay_count"], 0)
            self.assertTrue(workload["lifecycle"]["cache_size_stable"])
            self.assertTrue(workload["lifecycle"]["cache_invariant_ok"])
            self.assertTrue(workload["lifecycle"]["independent_outputs"])

        fused = workloads["gemm_silu_mul"]
        self.assertNotIn("bare_operation_ns", fused["raw_samples"])
        self.assertTrue(
            all("/" not in block for block in fused["summary"]["timing_order_blocks"])
        )

        rope = workloads["gemm_rope"]
        baseline = rope["correctness"]["probes"]["baseline"]["same_dtype"]
        self.assertEqual(baseline["leaf_count"], 2)
        self.assertEqual(
            {leaf["candidate_dtype"] for leaf in baseline["leaves"]},
            {"torch.float32"},
        )
        self.assertNotIn("bare_operation_ns", rope["raw_samples"])

        expected_us = statistics.median(
            workload["summary"]["candidate_p50_ns"]
            for workload in workloads.values()
        ) / 1_000.0
        self.assertEqual(result["metric"]["value"], expected_us)
        self.assertFalse(list(temporary_path.glob(".result.json.*.tmp")))

    @unittest.skipUnless(torch is not None, "PyTorch is required for the CPU adapter")
    def test_failed_candidate_is_ineligible_and_has_no_performance_metric(self):
        bench = _load_bench()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "failed.json"

        self.assertEqual(bench.main(_arguments(output, "bad_candidate")), 2)
        result = json.loads(output.read_text(encoding="utf-8"))
        _validate_receipt(result, ["bad_candidate"])
        workload = result["workloads"][0]

        self.assertFalse(workload["eligible"])
        self.assertFalse(workload["correctness"]["passed"])
        self.assertFalse(workload["hard_gates"]["same_dtype_parity"])
        self.assertFalse(workload["hard_gates"]["timing_samples"])
        self.assertEqual(workload["raw_samples"]["candidate_ns"], [])
        self.assertEqual(workload["raw_samples"]["reference_ns"], [])
        self.assertIsNone(result["metric"]["value"])
        self.assertFalse(result["hard_gates"]["all_workloads_eligible"])
        self.assertNotIn("champion", json.dumps(result).lower())

    @unittest.skipUnless(torch is not None, "PyTorch is required for the CPU adapter")
    def test_candidate_input_mutation_is_a_hard_failure(self):
        bench = _load_bench()
        adapter = bench._load_adapter(ADAPTER_PATH)
        original_make_case = adapter.make_case
        original_candidate = adapter.candidate

        def make_case(workload, seed, device):
            case = original_make_case(workload, seed, device)
            case["immutability_canary"] = torch.zeros(4, device=device)
            return case

        def mutating_candidate(case):
            output = original_candidate(case)
            case["immutability_canary"].add_(1)
            return output

        adapter.make_case = make_case
        adapter.candidate = mutating_candidate
        result = bench.run_campaign(
            adapter,
            ["gemm"],
            candidate_id="mutating-candidate",
            candidate_commit=COMMIT,
            parent_commit=PARENT,
            contract_sha256=CONTRACT,
            verdict="signal",
            device="cpu",
            warmup=1,
            repeats=2,
        )
        workload = result["workloads"][0]

        self.assertTrue(
            workload["correctness"]["probes"]["baseline"]["same_dtype"]["passed"]
        )
        self.assertFalse(workload["correctness"]["passed"])
        self.assertFalse(workload["correctness"]["input_immutable"])
        self.assertFalse(workload["hard_gates"]["input_immutable"])
        self.assertFalse(workload["eligible"])
        self.assertEqual(workload["raw_samples"]["candidate_ns"], [])
        self.assertIsNone(result["metric"]["value"])

    @unittest.skipUnless(torch is not None, "PyTorch is required for the CPU adapter")
    def test_same_value_input_storage_replacement_is_a_hard_failure(self):
        bench = _load_bench()
        adapter = bench._load_adapter(ADAPTER_PATH)
        original_candidate = adapter.candidate

        def replacing_candidate(case):
            case["x"] = case["x"].clone()
            return original_candidate(case)

        adapter.candidate = replacing_candidate
        result = bench.run_campaign(
            adapter,
            ["gemm"],
            candidate_id="replaced-input-storage",
            candidate_commit=COMMIT,
            parent_commit=PARENT,
            contract_sha256=CONTRACT,
            verdict="signal",
            profile="gemm",
            device="cpu",
            warmup=1,
            repeats=2,
        )
        workload = result["workloads"][0]
        self.assertFalse(workload["correctness"]["input_immutable"])
        self.assertFalse(workload["hard_gates"]["input_immutable"])
        self.assertFalse(workload["eligible"])

    def test_help_states_the_real_security_boundary(self):
        bench = _load_bench()
        help_text = " ".join(bench.build_parser().format_help().lower().split())
        self.assertIn("not a malicious-code sandbox", help_text)
        self.assertIn("controlled read-only harness", help_text)
        self.assertIn("clean process", help_text)


if __name__ == "__main__":
    unittest.main()
