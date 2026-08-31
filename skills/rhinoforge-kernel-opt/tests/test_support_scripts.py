from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = SKILL_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rpu_skill_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_is_non_overwriting(self) -> None:
        bootstrap = load_script("bootstrap_campaign")
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "campaign"
            bootstrap.bootstrap(workspace, "gemm", False)
            self.assertTrue((workspace / "contract.toml").is_file())
            self.assertEqual((workspace / "results.jsonl").read_text(), "")
            self.assertTrue((workspace / "solution" / ".gitkeep").is_file())
            with self.assertRaises(RuntimeError):
                bootstrap.bootstrap(workspace, "gemm", False)

    def test_git_init_isolates_candidate_from_evaluator(self) -> None:
        bootstrap = load_script("bootstrap_campaign")
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "campaign"
            bootstrap.bootstrap(workspace, "gemm", True)
            top = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(workspace / "solution"),
                    "rev-parse",
                    "--show-toplevel",
                ],
                text=True,
            ).strip()
            self.assertEqual(Path(top).resolve(), (workspace / "solution").resolve())
            self.assertFalse((workspace / ".git").exists())


class LeaseTests(unittest.TestCase):
    def test_lease_records_sanitized_command_identity(self) -> None:
        lease = load_script("board_lease")
        lock = lease.CANONICAL_LOCK_PATH
        code = lease.run_with_lease(
            lock,
            0.0,
            [sys.executable, "-c", "raise SystemExit(0)"],
        )
        self.assertEqual(code, 0)
        records = [json.loads(line) for line in lock.read_text().splitlines()]
        started, finished = records[-2:]
        self.assertEqual(finished["exit_code"], 0)
        self.assertIn("command_argv_sha256", started)
        self.assertNotIn("command_argv", started)


class EfficiencyTests(unittest.TestCase):
    def test_roofline_and_paired_interval(self) -> None:
        analysis = load_script("analyze_efficiency")
        report = analysis.roofline(
            required_ops=2000,
            mandatory_bytes=1000,
            peak_ops_per_s=1e9,
            bandwidth_bytes_per_s=1e9,
            launch_floor_ns=500,
            latency_ns=4000,
            peak_kind="empirical",
            source_id="unit-calibration",
            source_sha256="a" * 64,
        )
        self.assertEqual(report["roofline_label"], "empirical roofline")
        self.assertAlmostEqual(report["lower_bound_ns"], 2000)
        self.assertAlmostEqual(report["efficiency"], 0.5)

        fusion = analysis.fusion_tax(
            [101, 102, 100, 103, 101],
            [100, 100, 100, 100, 100],
            confidence=0.95,
            bootstrap_trials=1000,
            seed=7,
            max_tax_pct=5.0,
        )
        self.assertLessEqual(fusion["epilogue_tax_ci_upper_pct"], 5.0)
        self.assertEqual(
            fusion["statistic"], "median_of_paired_epilogue_tax_pct"
        )
        common = load_script("campaign_common")
        lower, upper = common.paired_median_bootstrap_ci(
            [101, 102, 100, 103, 101],
            [100, 100, 100, 100, 100],
            confidence=0.95,
            trials=1000,
            seed=7,
        )
        self.assertEqual(fusion["epilogue_tax_ci_lower_pct"], lower)
        self.assertEqual(fusion["epilogue_tax_ci_upper_pct"], upper)
        self.assertTrue(fusion["indistinguishable_at_registered_resolution"])

    def test_nonfinite_input_fails_closed(self) -> None:
        analysis = load_script("analyze_efficiency")
        with self.assertRaises(ValueError):
            analysis.roofline(
                required_ops=float("nan"),
                mandatory_bytes=1,
                peak_ops_per_s=1,
                bandwidth_bytes_per_s=1,
                launch_floor_ns=1,
                latency_ns=1,
                peak_kind="empirical",
                source_id="unit-calibration",
                source_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
