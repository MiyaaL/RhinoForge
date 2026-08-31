from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"full_verdict_test_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_script("run_full_verdict")


class TraceAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fused = "release_gemm_rope_fusion"
        self.baseline = "llama_rope"
        self.producer_sha256 = "a" * 64
        self.contract = {
            "campaign": {"op": "gemm+rope"},
            "commands": {"benchmark_adapter_sha256": self.producer_sha256},
            "runtime": {
                "trace_schema": "rhinoforge-rpu-chrome-v1",
                "trace_event_category": runner.RPU_DEVICE_EVENT_CATEGORY,
                "required_kernels": [self.baseline, self.fused],
                "fused_kernel": self.fused,
            },
        }

    def _raw_trace(self, names: list[str]) -> Path:
        path = self.root / f"raw-{len(list(self.root.iterdir()))}.json"
        path.write_text(
            json.dumps(
                {
                    "rhinoforgeTrace": {
                        "schema": "rhinoforge-rpu-chrome-v1",
                        "device_event_category": runner.RPU_DEVICE_EVENT_CATEGORY,
                        "device_program_events_exhaustive": True,
                        "producer_sha256": self.producer_sha256,
                    },
                    "traceEvents": [
                        {
                            "ph": "X",
                            "name": name,
                            "cat": runner.RPU_DEVICE_EVENT_CATEGORY,
                            "dur": 1.0,
                        }
                        for name in names
                    ]
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def test_exact_single_fused_manifest_launch_is_admitted(self) -> None:
        raw_trace = self._raw_trace([self.fused])
        summary = self.root / "summary.json"
        evidence = runner._trace_execution(
            raw_trace,
            summary,
            contract=self.contract,
            manifest_names={self.fused, self.baseline},
            preflight_hash="b" * 64,
        )
        self.assertEqual(evidence["kernel_names"], [self.fused])
        self.assertEqual(evidence["launch_count"], 1)
        self.assertEqual(
            evidence["trace_summary_sha256"],
            hashlib.sha256(summary.read_bytes()).hexdigest(),
        )

    def test_graph_with_two_manifest_kernels_cannot_masquerade_as_fusion(self) -> None:
        raw_trace = self._raw_trace([self.fused, self.baseline])
        summary = self.root / "summary.json"
        with self.assertRaisesRegex(
            runner.ValidationError, "exactly one complete admitted device-program"
        ):
            runner._trace_execution(
                raw_trace,
                summary,
                contract=self.contract,
                manifest_names={self.fused, self.baseline},
                preflight_hash="b" * 64,
            )

    def test_repeated_fused_launch_is_not_one_program_invocation(self) -> None:
        raw_trace = self._raw_trace([self.fused, self.fused])
        summary = self.root / "summary.json"
        with self.assertRaises(runner.ValidationError):
            runner._trace_execution(
                raw_trace,
                summary,
                contract=self.contract,
                manifest_names={self.fused, self.baseline},
                preflight_hash="b" * 64,
            )

    def test_unknown_complete_node_cannot_hide_behind_one_manifest_launch(self) -> None:
        raw_trace = self._raw_trace([self.fused, "private_graph_epilogue_node"])
        summary = self.root / "summary.json"
        with self.assertRaisesRegex(runner.ValidationError, "exactly one complete"):
            runner._trace_execution(
                raw_trace,
                summary,
                contract=self.contract,
                manifest_names={self.fused, self.baseline},
                preflight_hash="b" * 64,
            )

    def test_host_complete_events_are_excluded_by_exact_device_category(self) -> None:
        raw_trace = self.root / "host-and-device.json"
        raw_trace.write_text(
            json.dumps(
                {
                    "rhinoforgeTrace": {
                        "schema": "rhinoforge-rpu-chrome-v1",
                        "device_event_category": runner.RPU_DEVICE_EVENT_CATEGORY,
                        "device_program_events_exhaustive": True,
                        "producer_sha256": self.producer_sha256,
                    },
                    "traceEvents": [
                        {"ph": "X", "name": "python_host_scope", "cat": "cpu", "dur": 9},
                        {
                            "ph": "X",
                            "name": self.fused,
                            "cat": runner.RPU_DEVICE_EVENT_CATEGORY,
                            "dur": 1,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        evidence = runner._trace_execution(
            raw_trace,
            self.root / "host-and-device-summary.json",
            contract=self.contract,
            manifest_names={self.fused, self.baseline},
            preflight_hash="b" * 64,
        )
        self.assertEqual(evidence["kernel_names"], [self.fused])

    def test_trace_requires_exact_exhaustive_producer_identity(self) -> None:
        raw_trace = self._raw_trace([self.fused])
        payload = json.loads(raw_trace.read_text(encoding="utf-8"))
        payload["rhinoforgeTrace"]["device_program_events_exhaustive"] = False
        raw_trace.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(runner.ValidationError, "exhaustive"):
            runner._trace_execution(
                raw_trace,
                self.root / "untrusted-producer-summary.json",
                contract=self.contract,
                manifest_names={self.fused, self.baseline},
                preflight_hash="b" * 64,
            )

    def test_child_timeout_terminates_before_reporting_failure(self) -> None:
        child = mock.Mock()
        child.communicate.side_effect = [
            runner.subprocess.TimeoutExpired("child", 1.0),
            ("", "last error"),
        ]
        with self.assertRaisesRegex(runner.ValidationError, "timed out"):
            runner._communicate_with_timeout(child, 1.0, "unit child")
        child.terminate.assert_called_once_with()

    def test_full_verdict_rejects_non_rpu_before_filesystem_work(self) -> None:
        with self.assertRaisesRegex(runner.ValidationError, "exactly 'rpu'"):
            runner.run_full_verdict(
                self.root / "missing-contract.toml",
                self.root / "missing-adapter.py",
                self.root / "missing-preflight.json",
                self.root / "output",
                rhinoforge_root=self.root,
                device_compiler=None,
                candidate_id_prefix="unit",
                device="cpu",
                seed=1,
            )

    def test_lpaddr_substring_cannot_impersonate_explicit_gemm_add_fusion(self) -> None:
        bare = "gemm_fp16_spm_lpaddr_peak"
        fused = "release_gemm_residual_fusion"
        contract = {
            "campaign": {"op": "gemm+add"},
            "commands": {"benchmark_adapter_sha256": self.producer_sha256},
            "runtime": {
                "trace_schema": "rhinoforge-rpu-chrome-v1",
                "trace_event_category": runner.RPU_DEVICE_EVENT_CATEGORY,
                "required_kernels": [bare, fused],
                "fused_kernel": fused,
            },
        }
        raw_trace = self._raw_trace([bare])
        with self.assertRaisesRegex(runner.ValidationError, "exactly one"):
            runner._trace_execution(
                raw_trace,
                self.root / "lpaddr-summary.json",
                contract=contract,
                manifest_names={bare, fused},
                preflight_hash="b" * 64,
            )

    def test_profile_command_creates_fresh_trace_inside_launcher(self) -> None:
        profile_runner = self.root / "profile_runner.py"
        profile_runner.write_text(
            "import json,os,pathlib\n"
            f"event={{'ph':'X','name':{self.fused!r},'cat':"
            f"{runner.RPU_DEVICE_EVENT_CATEGORY!r},'dur':1.0}}\n"
            f"meta={{'schema':'rhinoforge-rpu-chrome-v1','device_event_category':"
            f"{runner.RPU_DEVICE_EVENT_CATEGORY!r},'device_program_events_exhaustive':True,"
            "'producer_sha256':os.environ['RHINOFORGE_PROFILE_PRODUCER_SHA256']}\n"
            "pathlib.Path(os.environ['RHINOFORGE_PROFILE_TRACE_OUTPUT']).write_text("
            "json.dumps({'rhinoforgeTrace':meta,'traceEvents':[event]}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        contract_path = self.root / "contract.toml"
        contract_path.write_text("fixture\n", encoding="utf-8")
        contract = {
            **self.contract,
            "commands": {
                "profile_one_call": [sys.executable, str(profile_runner)],
                "benchmark_adapter_sha256": self.producer_sha256,
            },
        }
        raw_trace = self.root / "fresh-raw.json"
        summary = self.root / "fresh-summary.json"
        lease = SimpleNamespace(check_integrity=lambda: None)
        evidence, metadata = runner._collect_one_call_trace(
            contract_path=contract_path,
            contract=contract,
            contract_hash="a" * 64,
            workload="m1-n1-k1",
            raw_trace_path=raw_trace,
            summary_path=summary,
            manifest_names={self.fused, self.baseline},
            preflight_hash="b" * 64,
            candidate_commit="c" * 40,
            parent_commit="d" * 40,
            device="rpu",
            board_lease=lease,
        )
        self.assertEqual(evidence["kernel_names"], [self.fused])
        self.assertEqual(metadata["complete_event_count"], 1)
        self.assertTrue(raw_trace.is_file())
        with self.assertRaisesRegex(runner.ValidationError, "must be new"):
            runner._collect_one_call_trace(
                contract_path=contract_path,
                contract=contract,
                contract_hash="a" * 64,
                workload="m1-n1-k1",
                raw_trace_path=raw_trace,
                summary_path=summary,
                manifest_names={self.fused, self.baseline},
                preflight_hash="b" * 64,
                candidate_commit="c" * 40,
                parent_commit="d" * 40,
                device="rpu",
                board_lease=lease,
            )

    def test_trusted_execution_envelope_binds_session_identities(self) -> None:
        evidence = {
            "m1-n1-k1": {
                "single_device_program": True,
                "launch_count": 1,
                "kernel_names": [self.fused],
                "trace_summary_sha256": "e" * 64,
                "preflight_sha256": "b" * 64,
            }
        }
        envelope = runner._trusted_execution_envelope(
            contract_hash="a" * 64,
            preflight_hash="b" * 64,
            candidate_commit="c" * 40,
            parent_commit="d" * 40,
            workloads=evidence,
        )
        self.assertEqual(envelope["workloads"], evidence)
        self.assertEqual(envelope["candidate_commit"], "c" * 40)
        self.assertEqual(envelope["preflight_sha256"], "b" * 64)

    def test_cli_returns_nonzero_when_collection_succeeds_but_gates_fail(self) -> None:
        with mock.patch.object(
            runner,
            "run_full_verdict",
            return_value={"all_gates_passed": False, "collection_ok": True},
        ):
            code = runner.main(
                [
                    str(self.root / "contract.toml"),
                    "--adapter",
                    str(self.root / "adapter.py"),
                    "--preflight",
                    str(self.root / "preflight.json"),
                    "--rhinoforge-root",
                    str(self.root),
                    "--output-dir",
                    str(self.root / "output"),
                    "--candidate-id-prefix",
                    "candidate",
                ]
            )
            self.assertEqual(code, 2)

    def test_cli_reports_collection_failure_with_session_schema(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            runner,
            "run_full_verdict",
            side_effect=runner.ValidationError("identity changed"),
        ), redirect_stdout(output):
            code = runner.main(
                [
                    str(self.root / "contract.toml"),
                    "--adapter",
                    str(self.root / "adapter.py"),
                    "--preflight",
                    str(self.root / "preflight.json"),
                    "--rhinoforge-root",
                    str(self.root),
                    "--output-dir",
                    str(self.root / "output"),
                    "--candidate-id-prefix",
                    "unit",
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertFalse(report["collection_ok"])
        self.assertFalse(report["all_gates_passed"])
        self.assertFalse(report["release_eligible"])

    def test_cli_serializes_board_lease_integrity_failure(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            runner,
            "run_full_verdict",
            side_effect=ValueError("board lease journal integrity failure"),
        ), redirect_stdout(output):
            code = runner.main(
                [
                    str(self.root / "contract.toml"),
                    "--adapter",
                    str(self.root / "adapter.py"),
                    "--preflight",
                    str(self.root / "preflight.json"),
                    "--rhinoforge-root",
                    str(self.root),
                    "--output-dir",
                    str(self.root / "output"),
                    "--candidate-id-prefix",
                    "lease-failure",
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertFalse(report["collection_ok"])
        self.assertFalse(report["all_gates_passed"])
        self.assertIn("board lease", report["error"])


if __name__ == "__main__":
    unittest.main()
