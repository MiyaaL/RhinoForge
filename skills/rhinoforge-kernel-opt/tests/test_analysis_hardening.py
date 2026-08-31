from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = SKILL_ROOT / "scripts" / f"{name}.py"
    module_name = f"rpu_skill_hardening_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class EfficiencyEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = load_script("analyze_efficiency")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source_file = Path(self.temporary.name) / "reviewed-source.json"
        self.source_file.write_bytes(b'{"source":"reviewed calibration v1"}\n')
        self.source_sha256 = hashlib.sha256(self.source_file.read_bytes()).hexdigest()

    def _arguments(self, peak_kind: str = "authoritative") -> list[str]:
        return [
            "--required-ops",
            "2000",
            "--mandatory-bytes",
            "1000",
            "--peak-ops-per-s",
            "1000000000",
            "--bandwidth-bytes-per-s",
            "1000000000",
            "--launch-floor-ns",
            "500",
            "--latency-ns",
            "4000",
            "--peak-kind",
            peak_kind,
            "--source-id",
            "reviewed-peak-v1",
            "--source-sha256",
            self.source_sha256,
            "--source-file",
            str(self.source_file),
        ]

    def test_both_peak_kinds_bind_reviewed_source_identity(self) -> None:
        expected_kinds = {
            "authoritative": "authoritative_specification",
            "empirical": "empirical_calibration",
        }
        for peak_kind, expected_source_kind in expected_kinds.items():
            with self.subTest(peak_kind=peak_kind):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(self.analysis.main(self._arguments(peak_kind)), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report["roofline"]["source"]["id"], "reviewed-peak-v1")
                self.assertEqual(
                    report["roofline"]["source"]["sha256"], self.source_sha256
                )
                self.assertEqual(
                    report["roofline"]["source"]["kind"], expected_source_kind
                )

    def test_missing_or_malformed_source_identity_is_a_command_failure(self) -> None:
        missing = self._arguments()
        source_id_index = missing.index("--source-id")
        del missing[source_id_index : source_id_index + 2]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.analysis.main(missing)
        self.assertEqual(raised.exception.code, 2)

        malformed = self._arguments()
        sha_index = malformed.index("--source-sha256") + 1
        malformed[sha_index] = "not-a-sha256"
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.analysis.main(malformed)
        self.assertEqual(raised.exception.code, 2)

    def test_source_file_must_exist_and_match_the_declared_hash(self) -> None:
        forged = self._arguments()
        sha_index = forged.index("--source-sha256") + 1
        forged[sha_index] = "0" * 64
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.analysis.main(forged)
        self.assertEqual(raised.exception.code, 2)

        missing = self._arguments()
        file_index = missing.index("--source-file") + 1
        missing[file_index] = str(Path(self.temporary.name) / "missing-source")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.analysis.main(missing)
        self.assertEqual(raised.exception.code, 2)

    def test_efficiency_above_one_and_nonphysical_derived_values_fail(self) -> None:
        above_floor = self._arguments()
        latency_index = above_floor.index("--latency-ns") + 1
        above_floor[latency_index] = "1000"
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.analysis.main(above_floor)
        self.assertEqual(raised.exception.code, 2)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.analysis.roofline(
                required_ops=1e308,
                mandatory_bytes=1,
                peak_ops_per_s=1e-308,
                bandwidth_bytes_per_s=1,
                launch_floor_ns=1,
                latency_ns=1,
                peak_kind="empirical",
                source_id="calibration-v1",
                source_sha256=self.source_sha256,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.analysis.roofline(
                required_ops=float("nan"),
                mandatory_bytes=1,
                peak_ops_per_s=1,
                bandwidth_bytes_per_s=1,
                launch_floor_ns=1,
                latency_ns=1,
                peak_kind="empirical",
                source_id="calibration-v1",
                source_sha256=self.source_sha256,
            )


class TraceSummaryTests(unittest.TestCase):
    def test_summary_binds_trace_hash_without_leaking_sensitive_fields(self) -> None:
        summarizer = load_script("summarize_hwperf")
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "sensitive-trace.json"
            raw = json.dumps(
                {
                    "displayTimeUnit": "ns",
                    "traceEvents": [
                        {
                            "ph": "X",
                            "name": (
                                "token=SUPERSECRET ptrA0x7fffdeadbeef "
                                "/secret/API Keys/topsecret.txt"
                            ),
                            "cat": r"\\server\private\credential.bin",
                            "dur": 4.0,
                            "pid": 918273,
                            "tid": 817263,
                            "ts": 716253,
                            "args": {
                                "bytes": 400,
                                "address": "0xcafebabe",
                                "source_path": "/secret/source.cpp",
                            },
                        }
                    ],
                },
                sort_keys=True,
            ).encode("utf-8")
            trace.write_bytes(raw)

            summary = summarizer.summarize_trace(trace)

        self.assertEqual(summary["trace_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(summary["events"][0]["bytes"]["total"], 400.0)
        encoded = json.dumps(summary, sort_keys=True)
        for forbidden in (
            "deadbeef",
            "cafebabe",
            "/secret",
            "SUPERSECRET",
            "topsecret",
            "credential",
            "server",
            "source_path",
            '"pid"',
            '"tid"',
            '"ts"',
            "918273",
            "817263",
            "716253",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_only_exact_allowlisted_kernel_name_may_remain_visible(self) -> None:
        summarizer = load_script("summarize_hwperf")
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "kernel-names.json"
            trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "name": "gemm_fp16",
                                "cat": "secret-category",
                                "dur": 1.0,
                            },
                            {
                                "ph": "X",
                                "name": "gemm_fp16 secret-suffix",
                                "cat": "secret-category",
                                "dur": 2.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = summarizer.summarize_trace(trace, ["gemm_fp16"])

        names = {event["name"] for event in summary["events"]}
        self.assertIn("gemm_fp16", names)
        self.assertIn("<redacted>", names)
        encoded = json.dumps(summary, sort_keys=True)
        self.assertNotIn("secret-suffix", encoded)
        self.assertNotIn("secret-category", encoded)


class CanonicalBoardLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lease = load_script("board_lease")

    def test_default_lock_identity_is_process_global_and_canonical(self) -> None:
        self.assertEqual(
            self.lease.CANONICAL_LOCK_PATH,
            Path("/tmp/rhinoforge-rpu-board-v3.lock"),
        )

    def test_canonical_journal_chains_complete_lease_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            canonical = Path(temp) / "rhinoforge-rpu-board.lock"
            with mock.patch.object(self.lease, "CANONICAL_LOCK_PATH", canonical):
                for _ in range(2):
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = self.lease.run_with_lease(
                            canonical,
                            0.0,
                            [sys.executable, "-c", "raise SystemExit(0)"],
                        )
                    self.assertEqual(code, 0)

            records = [
                json.loads(line) for line in canonical.read_text().splitlines()
            ]

        self.assertEqual(
            [record["event"] for record in records],
            ["lease_started", "lease_finished", "lease_started", "lease_finished"],
        )
        self.assertEqual(records[0]["lease_id"], records[1]["lease_id"])
        self.assertEqual(records[2]["lease_id"], records[3]["lease_id"])
        self.assertNotEqual(records[0]["lease_id"], records[2]["lease_id"])
        self.assertIn("command_argv_sha256", records[0])
        self.assertNotIn("command_argv", records[0])
        previous = "0" * 64
        for record in records:
            self.assertEqual(record["prev_record_sha256"], previous)
            previous = record["record_sha256"]

    def test_alternate_lock_path_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "rhinoforge-rpu-board.lock"
            alternate = root / "alternate.lock"
            with mock.patch.object(self.lease, "CANONICAL_LOCK_PATH", canonical):
                with self.assertRaisesRegex(ValueError, "alternate locks are forbidden"):
                    self.lease.run_with_lease(
                        alternate,
                        0.0,
                        [sys.executable, "-c", "raise SystemExit(0)"],
                    )

                target = root / "target.lock"
                target.touch()
                canonical.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    self.lease.run_with_lease(
                        canonical,
                        0.0,
                        [sys.executable, "-c", "raise SystemExit(0)"],
                    )

    def test_process_global_guard_refuses_a_second_lock_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            canonical = Path(temp) / "rhinoforge-rpu-board.lock"
            with mock.patch.object(self.lease, "CANONICAL_LOCK_PATH", canonical):
                with self.lease.BoardLease(0.0, ["first"]) as first:
                    self.assertTrue(first.lease_id)
                    with self.assertRaises(TimeoutError):
                        with self.lease.BoardLease(0.0, ["second"]):
                            pass

    def test_child_cannot_replace_pathname_to_create_an_unnoticed_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            canonical = Path(temp) / "rhinoforge-rpu-board.lock"
            program = (
                "import os,time; "
                f"p={str(canonical)!r}; os.unlink(p); open(p,'w').close(); time.sleep(.2)"
            )
            with mock.patch.object(self.lease, "CANONICAL_LOCK_PATH", canonical):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        self.lease.LeaseIntegrityError, "replaced"
                    ):
                        self.lease.run_with_lease(
                            canonical, 0.0, [sys.executable, "-c", program]
                        )

    def test_tampered_hash_chain_is_rejected_before_next_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            canonical = Path(temp) / "rhinoforge-rpu-board.lock"
            with mock.patch.object(self.lease, "CANONICAL_LOCK_PATH", canonical):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        self.lease.run_with_lease(
                            canonical,
                            0.0,
                            [sys.executable, "-c", "raise SystemExit(0)"],
                        ),
                        0,
                    )
                records = canonical.read_text(encoding="utf-8").splitlines()
                first = json.loads(records[0])
                first["pid"] = int(first["pid"]) + 1
                records[0] = json.dumps(first, sort_keys=True)
                canonical.write_text("\n".join(records) + "\n", encoding="utf-8")
                os.chmod(canonical, 0o600)
                with self.assertRaisesRegex(
                    self.lease.LeaseIntegrityError, "hash"
                ):
                    self.lease.run_with_lease(
                        canonical,
                        0.0,
                        [sys.executable, "-c", "raise SystemExit(0)"],
                    )

    def test_truncation_to_unpaired_valid_prefix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            canonical = Path(temp) / "rhinoforge-rpu-board.lock"
            with mock.patch.object(self.lease, "CANONICAL_LOCK_PATH", canonical):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        self.lease.run_with_lease(
                            canonical,
                            0.0,
                            [sys.executable, "-c", "raise SystemExit(0)"],
                        ),
                        0,
                    )
                first = canonical.read_text(encoding="utf-8").splitlines()[0]
                canonical.write_text(first + "\n", encoding="utf-8")
                os.chmod(canonical, 0o600)
                with self.assertRaisesRegex(
                    self.lease.LeaseIntegrityError, "incomplete lease"
                ):
                    self.lease.run_with_lease(
                        canonical,
                        0.0,
                        [sys.executable, "-c", "raise SystemExit(0)"],
                    )

if __name__ == "__main__":
    unittest.main()
