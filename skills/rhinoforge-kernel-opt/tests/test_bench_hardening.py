from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

try:
    import torch
except ImportError:  # pragma: no cover - minimal host discovery
    torch = None


SKILL_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = SKILL_ROOT / "scripts" / "bench.py"
ADAPTER_PATH = SKILL_ROOT / "assets" / "demo_cpu_adapter.py"
COMMIT = "a" * 40
PARENT = "b" * 40
CONTRACT = "c" * 64
PREFLIGHT = "d" * 64
TRACE = "e" * 64


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "test_rhinoforge_kernel_bench_hardening", BENCH_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_common():
    path = SKILL_ROOT / "scripts" / "campaign_common.py"
    spec = importlib.util.spec_from_file_location(
        "test_rhinoforge_kernel_campaign_common", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_arguments(bench, adapter, workload: str = "gemm_silu_mul"):
    return bench.run_campaign(
        adapter,
        [workload],
        candidate_id="full-fusion",
        candidate_commit=COMMIT,
        parent_commit=PARENT,
        contract_sha256=CONTRACT,
        preflight_sha256=PREFLIGHT,
        verdict="full",
        profile="gemm+silu_mul",
        device="cpu",
        seed=19,
        warmup=1,
        repeats=2,
        max_epilogue_tax_pct=1.0e9,
        fusion_confidence=0.95,
        fusion_bootstrap_trials=1000,
        fusion_bootstrap_seed=20260831,
        trusted_execution={
            workload: {
                "single_device_program": True,
                "launch_count": 1,
                "kernel_names": ["release_test_gemm_silu_mul"],
                "trace_summary_sha256": TRACE,
                "preflight_sha256": PREFLIGHT,
            }
        },
    )


@unittest.skipUnless(torch is not None, "PyTorch is required for the CPU fixture")
class BenchHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bench = _load_bench()
        self.adapter = self.bench._load_adapter(ADAPTER_PATH)

    def test_signal_records_unattested_real_process_identity(self) -> None:
        result = self.bench.run_campaign(
            self.adapter,
            ["gemm"],
            candidate_id="signal-process",
            candidate_commit=COMMIT,
            parent_commit=PARENT,
            contract_sha256=CONTRACT,
            verdict="signal",
            device="cpu",
            warmup=1,
            repeats=2,
        )

        process = result["process"]
        self.assertEqual(process["pid"], os.getpid())
        uuid.UUID(process["boot_id"])
        uuid.UUID(process["run_uuid"])
        self.assertGreater(process["start_ticks"], 0)
        self.assertIs(process["attested"], False)
        second = self.bench._read_process_identity()
        self.assertEqual(second["boot_id"], process["boot_id"])
        self.assertEqual(second["pid"], process["pid"])
        self.assertEqual(second["start_ticks"], process["start_ticks"])
        self.assertNotEqual(second["run_uuid"], process["run_uuid"])
        self.assertEqual(result["preflight_sha256"], "0" * 64)
        self.assertIsNone(result["workloads"][0]["execution"])

    def test_full_requires_exact_profile_and_nonzero_preflight(self) -> None:
        with self.assertRaisesRegex(self.bench.BenchError, "exact --profile"):
            self.bench.run_campaign(
                self.adapter,
                ["gemm"],
                candidate_id="missing-profile",
                candidate_commit=COMMIT,
                parent_commit=PARENT,
                contract_sha256=CONTRACT,
                preflight_sha256=PREFLIGHT,
                verdict="full",
                device="cpu",
            )

    def test_exact_profile_missing_anchor_or_graph_is_a_failed_gate(self) -> None:
        for hook_name, gate_name in (
            ("fp32_anchor", "fp32_anchor"),
            ("graph_state", "graph_lifecycle"),
        ):
            with self.subTest(hook=hook_name):
                adapter = self.bench._load_adapter(ADAPTER_PATH)
                setattr(adapter, hook_name, None)
                result = self.bench.run_campaign(
                    adapter,
                    ["gemm"],
                    candidate_id=f"missing-{hook_name}",
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
                self.assertFalse(workload["hard_gates"][gate_name])
                self.assertFalse(workload["eligible"])
                self.assertEqual(workload["raw_samples"]["candidate_ns"], [])

    def test_exact_profile_refuses_timing_before_steady_replay(self) -> None:
        self.adapter.graph_state = lambda: {
            "build_count": 1,
            "replay_count": 0,
            "cache_size": 1,
            "invariant_ok": True,
        }
        result = self.bench.run_campaign(
            self.adapter,
            ["gemm"],
            candidate_id="cold-graph",
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
        self.assertFalse(workload["hard_gates"]["graph_lifecycle"])
        self.assertFalse(workload["eligible"])
        self.assertEqual(workload["raw_samples"]["candidate_ns"], [])
        self.assertIn("already-warm Graph state", workload["summary"]["timing_error"])

        with self.assertRaisesRegex(self.bench.BenchError, "non-zero preflight"):
            self.bench.run_campaign(
                self.adapter,
                ["gemm"],
                candidate_id="missing-preflight",
                candidate_commit=COMMIT,
                parent_commit=PARENT,
                contract_sha256=CONTRACT,
                verdict="full",
                profile="gemm",
                device="cpu",
            )

    def test_full_fusion_binds_execution_bare_samples_and_paired_ci(self) -> None:
        result = _full_arguments(self.bench, self.adapter)
        workload = result["workloads"][0]
        execution = workload["execution"]
        summary = workload["summary"]

        self.assertEqual(result["preflight_sha256"], PREFLIGHT)
        self.assertEqual(execution["preflight_sha256"], PREFLIGHT)
        self.assertTrue(execution["single_device_program"])
        self.assertEqual(execution["launch_count"], 1)
        self.assertEqual(execution["trace_summary_sha256"], TRACE)
        self.assertEqual(len(workload["raw_samples"]["bare_operation_ns"]), 4)
        recomputed = self.bench._paired_epilogue_tax_interval(
            workload["raw_samples"]["candidate_ns"],
            workload["raw_samples"]["bare_operation_ns"],
            confidence=0.95,
            bootstrap_trials=1000,
            bootstrap_seed=20260831,
        )
        for key in (
            "epilogue_tax_p50_pct",
            "epilogue_tax_ci_lower_pct",
            "epilogue_tax_ci_upper_pct",
            "epilogue_tax_confidence",
            "epilogue_tax_bootstrap_trials",
            "epilogue_tax_bootstrap_seed",
        ):
            self.assertEqual(summary[key], recomputed[key])
        self.assertTrue(workload["hard_gates"]["epilogue_tax"])
        self.assertTrue(workload["hard_gates"]["single_device_program"])

    def test_full_fusion_cannot_skip_missing_or_notimplemented_bare(self) -> None:
        self.adapter.bare_operation = None
        missing = _full_arguments(self.bench, self.adapter)
        missing_workload = missing["workloads"][0]
        self.assertFalse(missing_workload["hard_gates"]["bare_operation"])
        self.assertEqual(missing_workload["raw_samples"]["bare_operation_ns"], [])
        self.assertIsNone(missing["metric"]["value"])

        adapter = self.bench._load_adapter(ADAPTER_PATH)
        adapter.bare_operation = lambda _case: (
            _ for _ in ()
        ).throw(NotImplementedError())
        unavailable = _full_arguments(self.bench, adapter)
        unavailable_workload = unavailable["workloads"][0]
        self.assertFalse(unavailable_workload["hard_gates"]["bare_operation"])
        self.assertEqual(
            unavailable_workload["raw_samples"]["bare_operation_ns"], []
        )
        self.assertIsNone(unavailable["metric"]["value"])

    def test_full_missing_verifier_execution_evidence_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            self.bench.BenchError, "verifier-owned execution evidence"
        ):
            self.bench.run_campaign(
                self.adapter,
                ["gemm"],
                candidate_id="missing-evidence",
                candidate_commit=COMMIT,
                parent_commit=PARENT,
                contract_sha256=CONTRACT,
                preflight_sha256=PREFLIGHT,
                verdict="full",
                profile="gemm",
                device="cpu",
                warmup=1,
                repeats=2,
            )

    def test_signal_rejects_trusted_execution_evidence(self) -> None:
        with self.assertRaisesRegex(self.bench.BenchError, "must not accept"):
            self.bench.run_campaign(
                self.adapter,
                ["gemm"],
                candidate_id="signal-evidence",
                candidate_commit=COMMIT,
                parent_commit=PARENT,
                contract_sha256=CONTRACT,
                verdict="signal",
                device="cpu",
                warmup=1,
                repeats=2,
                trusted_execution={"gemm": {}},
            )

    def test_value_equal_wrong_stride_output_is_ineligible(self) -> None:
        original_candidate = self.adapter.candidate

        def wrong_layout(case):
            output = original_candidate(case)
            return output.transpose(0, 1).contiguous().transpose(0, 1)

        self.adapter.candidate = wrong_layout
        result = self.bench.run_campaign(
            self.adapter,
            ["gemm"],
            candidate_id="wrong-layout",
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
        self.assertFalse(workload["hard_gates"]["layout"])
        self.assertFalse(workload["eligible"])
        self.assertFalse(
            workload["correctness"]["probes"]["baseline"]["same_dtype"][
                "layout_match"
            ]
        )

    def test_quantized_profile_uses_bare_rmsnorm_and_exact_encoded_gates(self) -> None:
        evidence = {
            "rmsnorm_quant_mxfp8": {
                "single_device_program": True,
                "launch_count": 1,
                "kernel_names": ["release_rms_mxfp8_fusion"],
                "trace_summary_sha256": TRACE,
                "preflight_sha256": PREFLIGHT,
            }
        }
        result = self.bench.run_campaign(
            self.adapter,
            ["rmsnorm_quant_mxfp8"],
            candidate_id="full-quant",
            candidate_commit=COMMIT,
            parent_commit=PARENT,
            contract_sha256=CONTRACT,
            preflight_sha256=PREFLIGHT,
            verdict="full",
            profile="rmsnorm+quant_mxfp8",
            device="cpu",
            seed=29,
            warmup=1,
            repeats=2,
            anchor_atol=0.02,
            anchor_rtol=1.0,
            max_cv=1.0e9,
            max_epilogue_tax_pct=1.0e9,
            fusion_confidence=0.95,
            fusion_bootstrap_trials=1000,
            fusion_bootstrap_seed=20260831,
            trusted_execution=evidence,
        )
        workload = result["workloads"][0]
        self.assertEqual(len(workload["raw_samples"]["bare_operation_ns"]), 4)
        self.assertTrue(workload["hard_gates"]["bare_operation"])
        self.assertTrue(workload["hard_gates"]["quantized_output_structure"])
        self.assertTrue(workload["hard_gates"]["quantized_output_bit_exact"])
        self.assertTrue(workload["hard_gates"]["quantized_output_layout"])
        self.assertTrue(workload["hard_gates"]["paired_steady_replay"])

        common = _load_common()
        result["process"]["attested"] = True
        contract = {
            "campaign": {"op": "rmsnorm+quant_mxfp8"},
            "runtime": {
                "required_kernels": ["release_rms_mxfp8_fusion"],
                "fused_kernel": "release_rms_mxfp8_fusion",
            },
            "measurement": {
                "metric": "latency_us",
                "direction": "min",
                "samples": 4,
            },
            "gates": {
                "max_abs": 1.0e-5,
                "max_rel": 1.0e-5,
                "anchor_max_abs": 0.02,
                "anchor_max_rel": 1.0,
                "max_cv": 1.0e9,
                "require_same_dtype": True,
                "require_fp32_anchor": True,
                "require_graph_replay": True,
                "require_independent_outputs": True,
                "max_epilogue_tax_pct": 1.0e9,
                "fusion_confidence": 0.95,
                "fusion_bootstrap_trials": 1000,
                "fusion_bootstrap_seed": 20260831,
            },
            "workloads": [{"id": "rmsnorm_quant_mxfp8"}],
        }
        common.validate_result(result, contract, CONTRACT)

        forged_quant = copy.deepcopy(result)
        forged_quant["workloads"][0]["correctness"]["probes"]["baseline"][
            "same_dtype"
        ]["leaves"][0]["byte_exact"] = False
        with self.assertRaisesRegex(
            common.ValidationError, "quantized_output_bit_exact"
        ):
            common.validate_result(forged_quant, contract, CONTRACT)

        forged_bare = copy.deepcopy(result)
        forged_bare["workloads"][0]["correctness"]["bare_operation"][
            "comparison"
        ] = {"passed": True}
        with self.assertRaisesRegex(common.ValidationError, "missing keys"):
            common.validate_result(forged_bare, contract, CONTRACT)

        forged_arm_state = copy.deepcopy(result)
        arm_states = forged_arm_state["workloads"][0]["lifecycle"]["arm_states"]
        arm_states["final"]["bare_operation"]["replay_count"] = arm_states[
            "stable"
        ]["bare_operation"]["replay_count"]
        with self.assertRaisesRegex(common.ValidationError, "paired_steady_replay"):
            common.validate_result(forged_arm_state, contract, CONTRACT)

    def test_quantized_exact_comparison_distinguishes_signed_zero_bits(self) -> None:
        candidate = (torch.tensor([1], dtype=torch.int8), torch.tensor([-0.0]))
        reference = (torch.tensor([1], dtype=torch.int8), torch.tensor([0.0]))
        tolerant = self.bench._compare_trees(
            candidate,
            reference,
            atol=0.0,
            rtol=0.0,
            require_dtype=True,
        )
        exact = self.bench._compare_trees(
            candidate,
            reference,
            atol=0.0,
            rtol=0.0,
            require_dtype=True,
            require_bit_exact=True,
            require_layout=True,
        )
        self.assertTrue(tolerant["passed"])
        self.assertFalse(exact["passed"])
        self.assertFalse(exact["byte_exact"])

    def test_trusted_execution_file_is_identity_bound_and_strict(self) -> None:
        envelope = {
            "schema_version": 1,
            "kind": "rhinoforge-local-trusted-execution",
            "contract_sha256": CONTRACT,
            "preflight_sha256": PREFLIGHT,
            "candidate_commit": COMMIT,
            "parent_commit": PARENT,
            "workloads": {
                "gemm": {
                    "single_device_program": True,
                    "launch_count": 1,
                    "kernel_names": ["release_gemm"],
                    "trace_summary_sha256": TRACE,
                    "preflight_sha256": PREFLIGHT,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            loaded = self.bench._load_trusted_execution_file(
                path,
                workloads=["gemm"],
                contract_sha256=CONTRACT,
                preflight_sha256=PREFLIGHT,
                candidate_commit=COMMIT,
                parent_commit=PARENT,
            )
            self.assertEqual(loaded, envelope["workloads"])
            envelope["candidate_commit"] = "f" * 40
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(self.bench.BenchError, "candidate_commit"):
                self.bench._load_trusted_execution_file(
                    path,
                    workloads=["gemm"],
                    contract_sha256=CONTRACT,
                    preflight_sha256=PREFLIGHT,
                    candidate_commit=COMMIT,
                    parent_commit=PARENT,
                )

    def test_full_cli_consumes_verifier_file_without_adapter_claim_hook(self) -> None:
        envelope = {
            "schema_version": 1,
            "kind": "rhinoforge-local-trusted-execution",
            "contract_sha256": CONTRACT,
            "preflight_sha256": PREFLIGHT,
            "candidate_commit": COMMIT,
            "parent_commit": PARENT,
            "workloads": {
                "gemm_silu_mul": {
                    "single_device_program": True,
                    "launch_count": 1,
                    "kernel_names": ["release_test_gemm_silu_mul"],
                    "trace_summary_sha256": TRACE,
                    "preflight_sha256": PREFLIGHT,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trusted = root / "trusted.json"
            output = root / "result.json"
            trusted.write_text(json.dumps(envelope), encoding="utf-8")
            code = self.bench.main(
                [
                    "--adapter",
                    str(ADAPTER_PATH),
                    "--workload",
                    "gemm_silu_mul",
                    "--profile",
                    "gemm+silu_mul",
                    "--device",
                    "cpu",
                    "--warmup",
                    "1",
                    "--repeats",
                    "2",
                    "--candidate-id",
                    "full-cli",
                    "--candidate-commit",
                    COMMIT,
                    "--parent-commit",
                    PARENT,
                    "--contract-sha256",
                    CONTRACT,
                    "--preflight-sha256",
                    PREFLIGHT,
                    "--trusted-execution",
                    str(trusted),
                    "--max-epilogue-tax-pct",
                    "1000000000",
                    "--fusion-confidence",
                    "0.95",
                    "--fusion-bootstrap-trials",
                    "1000",
                    "--fusion-bootstrap-seed",
                    "20260831",
                    "--verdict",
                    "full",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                result["workloads"][0]["execution"],
                envelope["workloads"]["gemm_silu_mul"],
            )


if __name__ == "__main__":
    unittest.main()
