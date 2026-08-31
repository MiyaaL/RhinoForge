from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import uuid
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_signal():
    path = SCRIPTS / "run_signal.py"
    spec = importlib.util.spec_from_file_location("kernel_signal_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


signal = _load_signal()
COMMIT = "a" * 40
PARENT = "b" * 40
CONTRACT_HASH = "c" * 64


def _contract(adapter: Path, *, operation: str = "gemm") -> dict:
    gates = {
        "max_abs": 0.125,
        "max_rel": 0.25,
        "anchor_max_abs": 0.03125,
        "anchor_max_rel": 0.0625,
        "max_cv": 0.05,
        "require_same_dtype": True,
        "require_fp32_anchor": True,
        "require_graph_replay": True,
        "require_independent_outputs": True,
    }
    if operation != "gemm":
        gates.update(
            {
                "max_epilogue_tax_pct": 1.5,
                "fusion_confidence": 0.975,
                "fusion_bootstrap_trials": 2345,
                "fusion_bootstrap_seed": 91,
            }
        )
    return {
        "campaign": {
            "id": "unit-signal",
            "op": operation,
            "candidate_root": "candidate",
            "reference_root": "reference",
            "reference_tree_sha256": signal.content_tree_sha256(
                adapter.parent / "reference"
            ),
        },
        "commands": {
            "benchmark_adapter": str(adapter),
            "benchmark_adapter_sha256": hashlib.sha256(
                adapter.read_bytes()
            ).hexdigest(),
        },
        "measurement": {
            "metric": "latency_us",
            "direction": "min",
            "warmup": 7,
            "samples": 5,
            "verdict_processes": 2,
        },
        "gates": gates,
        "workloads": [
            {"id": "shape-a"},
            {"id": "shape-b"},
        ],
    }


class FakeLease:
    def __init__(self, timeout_s, command):
        self.timeout_s = timeout_s
        self.command = command
        self.finished = False
        self.exit_code = None
        self.lease_id = str(uuid.uuid4())
        self.start_sequence = 1
        self.finish_sequence = 0
        self.previous_head_hash = "0" * 64
        self.start_record_sha256 = "1" * 64
        self.finish_record_sha256 = "0" * 64
        self.head_hash = self.start_record_sha256
        self.started_monotonic_ns = 10
        self.finished_monotonic_ns = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def check_integrity(self):
        return None

    def finish(self, exit_code):
        self.finished = True
        self.exit_code = exit_code
        self.finish_sequence = 2
        self.finish_record_sha256 = "2" * 64
        self.head_hash = self.finish_record_sha256
        self.finished_monotonic_ns = 20


class SignalLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter.py"
        self.adapter.write_text("# frozen unit adapter\n", encoding="utf-8")
        self.reference = self.root / "reference"
        self.reference.mkdir()
        (self.reference / "oracle.py").write_text(
            "# frozen reference fixture\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_command_derives_all_fusion_gates_and_sampling_from_contract(self) -> None:
        contract = _contract(self.adapter, operation="gemm+add")
        result = self.root / "result.json"
        command = signal._build_bench_command(
            contract=contract,
            adapter=self.adapter,
            candidate_id="candidate-7",
            candidate_commit=COMMIT,
            parent_commit=PARENT,
            contract_hash=CONTRACT_HASH,
            result_path=result,
            device="rpu",
            seed=17,
        )

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(value("--adapter"), str(self.adapter))
        self.assertEqual(value("--profile"), "gemm+add")
        self.assertEqual(value("--warmup"), "7")
        self.assertEqual(value("--repeats"), "3")
        self.assertEqual(value("--atol"), "0.125")
        self.assertEqual(value("--rtol"), "0.25")
        self.assertEqual(value("--anchor-atol"), "0.03125")
        self.assertEqual(value("--anchor-rtol"), "0.0625")
        self.assertEqual(value("--max-cv"), "0.05")
        self.assertEqual(value("--max-epilogue-tax-pct"), "1.5")
        self.assertEqual(value("--fusion-confidence"), "0.975")
        self.assertEqual(value("--fusion-bootstrap-trials"), "2345")
        self.assertEqual(value("--fusion-bootstrap-seed"), "91")
        self.assertEqual(
            [
                command[index + 1]
                for index, item in enumerate(command)
                if item == "--workload"
            ],
            ["shape-a", "shape-b"],
        )
        self.assertNotIn("--trusted-execution", command)
        self.assertEqual(value("--verdict"), "signal")

    def test_adapter_hash_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(signal.ValidationError, "SHA-256 mismatches"):
            signal._adapter_identity(self.adapter, "0" * 64)

    def test_non_rpu_device_is_rejected_before_contract_or_board_work(self) -> None:
        with mock.patch.object(signal, "load_contract") as load, mock.patch.object(
            signal, "BoardLease"
        ) as lease:
            with self.assertRaisesRegex(signal.ValidationError, "exactly 'rpu'"):
                signal.collect_signal(
                    self.root / "missing-contract.toml",
                    candidate_id="candidate-1",
                    output_dir=self.root / "journal",
                    device="cpu",
                )
        load.assert_not_called()
        lease.assert_not_called()

    def test_process_observation_failure_terminates_child(self) -> None:
        child = mock.Mock()
        child.pid = 42
        child.poll.return_value = None
        child.wait.return_value = 0
        with mock.patch.object(
            signal.subprocess, "Popen", return_value=child
        ), mock.patch.object(
            signal,
            "_proc_identity",
            side_effect=signal.ValidationError("cannot observe"),
        ):
            with self.assertRaisesRegex(signal.ValidationError, "cannot observe"):
                signal._run_bench_child(
                    ["bench"],
                    cwd=self.root,
                    board_lease=FakeLease(0, ["bench"]),
                    timeout_s=1,
                )
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=2)

    def test_failed_gates_are_observed_and_recorded_but_report_failure(self) -> None:
        contract_path = self.root / "contract.toml"
        contract_path.write_text("approved fixture\n", encoding="utf-8")
        candidate_root = self.root / "candidate"
        candidate_root.mkdir()
        output_dir = self.root / "journal"
        contract = _contract(self.adapter)
        process = {
            "boot_id": str(uuid.uuid4()),
            "pid": 4321,
            "start_ticks": 9876,
            "run_uuid": str(uuid.uuid4()),
            "attested": False,
        }
        canonical = {
            "verdict": "signal",
            "preflight_sha256": "0" * 64,
            "candidate_commit": COMMIT,
            "parent_commit": PARENT,
            "process": process,
            "metric": {"value": None},
            "workloads": [
                {
                    "id": workload["id"],
                    "eligible": False,
                    "hard_gates": {"finite": False},
                }
                for workload in contract["workloads"]
            ],
            "hard_gates": {
                "workloads_present": True,
                "all_workloads_eligible": False,
                "metric_finite": False,
            },
        }
        observed = {
            "boot_id": process["boot_id"],
            "pid": process["pid"],
            "start_ticks": process["start_ticks"],
        }

        def run_child(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            output.write_text("{}\n", encoding="utf-8")
            return 2, dict(observed), "gate failure"

        receipt = {"ok": True, "candidate_id": "candidate-7"}
        patches = (
            mock.patch.object(signal, "load_contract", return_value=contract),
            mock.patch.object(
                signal, "contract_sha256", return_value=CONTRACT_HASH
            ),
            mock.patch.object(
                signal,
                "_candidate_identity",
                return_value=(candidate_root, COMMIT, PARENT),
            ),
            mock.patch.object(signal, "BoardLease", FakeLease),
            mock.patch.object(signal, "_run_bench_child", side_effect=run_child),
            mock.patch.object(
                signal, "load_json", return_value={"process": process}
            ),
            mock.patch.object(signal, "validate_result", return_value=canonical),
            mock.patch.object(signal, "record_iteration", return_value=receipt),
        )
        with ExitStack() as stack:
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])
            identity = stack.enter_context(patches[2])
            stack.enter_context(patches[3])
            stack.enter_context(patches[4])
            stack.enter_context(patches[5])
            stack.enter_context(patches[6])
            record = stack.enter_context(patches[7])
            report = signal.collect_signal(
                contract_path,
                candidate_id="candidate-7",
                output_dir=output_dir,
            )

        self.assertTrue(report["collection_ok"])
        self.assertFalse(report["signal_gates_passed"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["bench_exit_code"], 2)
        self.assertEqual(report["board_lease"]["finish_sequence"], 2)
        self.assertEqual(report["recorded_receipt"], receipt)
        self.assertEqual(identity.call_count, 2)
        call = record.call_args
        self.assertEqual(call.args[:3], (contract_path, call.args[1], output_dir))
        self.assertEqual(
            call.kwargs["observed_process"],
            {**observed, "run_uuid": process["run_uuid"]},
        )

    def test_main_returns_nonzero_for_recorded_gate_failure(self) -> None:
        report = {
            "schema_version": 1,
            "ok": False,
            "collection_ok": True,
            "signal_gates_passed": False,
        }
        output = io.StringIO()
        with mock.patch.object(
            signal, "collect_signal", return_value=report
        ), redirect_stdout(output):
            status = signal.main(
                [
                    str(self.root / "contract.toml"),
                    "--candidate-id",
                    "failed-1",
                    "--output-dir",
                    str(self.root / "journal"),
                ]
            )
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_main_reports_collection_failure_separately(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            signal,
            "collect_signal",
            side_effect=signal.ValidationError("adapter changed"),
        ), redirect_stdout(output):
            status = signal.main(
                [
                    str(self.root / "contract.toml"),
                    "--candidate-id",
                    "broken-1",
                    "--output-dir",
                    str(self.root / "journal"),
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertFalse(report["collection_ok"])
        self.assertFalse(report["signal_gates_passed"])
        self.assertEqual(report["error"], "adapter changed")


if __name__ == "__main__":
    unittest.main()
