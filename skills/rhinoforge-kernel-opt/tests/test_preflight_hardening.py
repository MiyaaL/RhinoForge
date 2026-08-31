from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
RHINOFORGE_ROOT = SKILL_ROOT.parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_preflight():
    path = SCRIPTS / "preflight.py"
    spec = importlib.util.spec_from_file_location(
        "kernel_preflight_hardening_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight()

COMMIT = "a" * 40


class PreflightFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "candidate").mkdir()
        (self.root / "reference").mkdir()
        (self.root / "reference" / "oracle.py").write_text(
            "# frozen reference fixture\n", encoding="utf-8"
        )
        self.runner = self.root / "runner.py"
        self.runner.write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.asset = self.root / "operator.ref"
        self.asset.write_bytes(b"opaque-test-asset")
        self.manifest = Path(str(self.asset) + ".kernels")
        self.manifest.write_text(
            "rhinoforge-kernels-v1\n"
            f"asset-size={self.asset.stat().st_size}\n"
            "gemm_fp16_test\n",
            encoding="utf-8",
        )
        self.launch = self.root / "librhino_launch.so"
        self.launch.write_bytes(b"launch-test-library")
        self.contract_path = self.root / "contract.toml"
        self.contract_path.write_text("schema_version = 1\n", encoding="utf-8")
        self.contract = {
            "campaign": {
                "id": "preflight-hardening",
                "op": "gemm",
                "candidate_root": str(self.root / "candidate"),
                "reference_root": str(self.root / "reference"),
                "reference_tree_sha256": preflight.content_tree_sha256(
                    self.root / "reference"
                ),
            },
            "commands": {
                "correctness": [sys.executable, str(self.runner)],
                "benchmark": [sys.executable, str(self.runner)],
                "profile_one_call": [sys.executable, str(self.runner)],
                "benchmark_adapter": str(self.runner),
                "benchmark_adapter_sha256": self._sha(self.runner),
            },
            "attestation": {
                "mode": "external",
                "scheme": "ed25519",
                "namespace": "rhinoforge-full-verdict-v1",
                "signer_id": "unit-release-authority",
                "public_key_base64": "11qYAYKxCrfVS/7TyWQHOg7hcvPapiMlrwIaaPcHURo=",
                "policy_sha256": hashlib.sha256(b"unit-policy").hexdigest(),
                "signer_build_sha256": hashlib.sha256(b"unit-signer-build").hexdigest(),
                "verifier_executable": "/usr/bin/openssl",
                "verifier_sha256": hashlib.sha256(
                    Path("/usr/bin/openssl").read_bytes()
                ).hexdigest(),
            },
            "runtime": {
                "trace_schema": "rhinoforge-rpu-chrome-v1",
                "trace_event_category": "rpu_device_program",
                "rhinoforge_commit": COMMIT,
                "runtime_set": "test-runtime",
                "torch_version": "test-torch",
                "launch_library": str(self.launch),
                "launch_library_sha256": self._sha(self.launch),
                "operator_asset": str(self.asset),
                "operator_asset_sha256": self._sha(self.asset),
                "kernel_manifest": str(self.manifest),
                "kernel_manifest_sha256": self._sha(self.manifest),
                "required_kernels": ["gemm_fp16_test"],
                "requires_device_compiler": False,
                "device_compiler_authorized": False,
                "device_compiler_id": "not-required",
            },
            "workloads": [{"id": "m1-n1-k1"}],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _run(self, **kwargs):
        with mock.patch.object(
            preflight, "load_contract", return_value=self.contract
        ), mock.patch.object(
            preflight, "_is_rhinoforge_root", return_value=True
        ), mock.patch.object(
            preflight, "_git_head", return_value=COMMIT
        ), mock.patch.object(
            preflight, "_git_clean", return_value=True
        ), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(
            preflight, "_device_capable", return_value=True
        ), mock.patch.dict(os.environ, {}, clear=True):
            return preflight.run_preflight(
                self.contract_path,
                rhinoforge_root=RHINOFORGE_ROOT,
                **kwargs,
            )


class RootAndReceiptTests(PreflightFixture):
    def test_reference_tree_mutation_fails_frozen_identity(self) -> None:
        (self.root / "reference" / "oracle.py").write_text(
            "# mutated reference fixture\n", encoding="utf-8"
        )
        report = self._run()
        statuses = {item["name"]: item["status"] for item in report["checks"]}
        self.assertEqual(statuses["campaign.reference_tree_identity"], "fail")
        self.assertFalse(report["ok"])

    def test_unavailable_attestation_allows_local_preflight_but_skips_release(self) -> None:
        self.contract["attestation"] = {
            "mode": "unavailable",
            "scheme": "none",
            "namespace": "rhinoforge-full-verdict-v1",
            "signer_id": "not-available",
            "public_key_base64": "not-available",
            "policy_sha256": "0" * 64,
            "signer_build_sha256": "0" * 64,
            "verifier_executable": "not-available",
            "verifier_sha256": "0" * 64,
        }
        report = self._run()
        verifier = next(
            check
            for check in report["checks"]
            if check["name"] == "attestation.release_verifier"
        )
        self.assertEqual(verifier["status"], "skip")
        self.assertTrue(report["ok"], report)

    def test_real_checkout_is_recognized_and_unrelated_directory_is_not(self) -> None:
        self.assertTrue(preflight._is_rhinoforge_root(RHINOFORGE_ROOT))
        unrelated = self.root / "unrelated"
        (unrelated / ".git").mkdir(parents=True)
        (unrelated / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        self.assertFalse(preflight._is_rhinoforge_root(unrelated))

    def test_explicit_arbitrary_clean_git_root_fails_identity_gate(self) -> None:
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        with mock.patch.object(
            preflight, "load_contract", return_value=self.contract
        ), mock.patch.object(
            preflight, "_git_head", return_value=COMMIT
        ), mock.patch.object(
            preflight, "_git_clean", return_value=True
        ), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True):
            report = preflight.run_preflight(
                self.contract_path, rhinoforge_root=unrelated
            )
        failed = {
            item["name"] for item in report["checks"] if item["status"] == "fail"
        }
        self.assertIn("runtime.rhinoforge_root_identity", failed)
        self.assertFalse(report["ok"])

    def test_receipt_contains_exact_contract_sha256(self) -> None:
        report = self._run()
        self.assertEqual(
            report["contract_sha256"],
            hashlib.sha256(self.contract_path.read_bytes()).hexdigest(),
        )

    def test_arbitrary_character_devices_cannot_impersonate_rpu_or_mem(self) -> None:
        with mock.patch.object(
            preflight, "load_contract", return_value=self.contract
        ), mock.patch.object(
            preflight, "_is_rhinoforge_root", return_value=True
        ), mock.patch.object(
            preflight, "_git_head", return_value=COMMIT
        ), mock.patch.object(
            preflight, "_git_clean", return_value=True
        ), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/null"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True):
            report = preflight.run_preflight(
                self.contract_path,
                rhinoforge_root=RHINOFORGE_ROOT,
                device_glob="/dev/null",
                dev_mem=Path("/dev/null"),
            )
        failed = {
            item["name"] for item in report["checks"] if item["status"] == "fail"
        }
        self.assertIn("device.rpu", failed)
        self.assertIn("device.mem", failed)

    def test_cli_always_requests_torch_rpu_roundtrip_without_flag(self) -> None:
        report = {"schema_version": 1, "ok": True, "contract_sha256": "f" * 64}
        with mock.patch.object(
            preflight, "run_preflight", return_value=report
        ) as run, mock.patch("builtins.print"):
            status = preflight.main([str(self.contract_path)])
        self.assertEqual(status, 0)
        self.assertTrue(run.call_args.kwargs["probe_torch_rpu"])

    def test_cli_writes_strict_json_receipt_atomically(self) -> None:
        report = {"schema_version": 1, "ok": False, "contract_sha256": "e" * 64}
        output = self.root / "preflight.json"
        with mock.patch.object(
            preflight, "run_preflight", return_value=report
        ), mock.patch("builtins.print"):
            status = preflight.main(
                [str(self.contract_path), "--output", str(output)]
            )
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        self.assertEqual(list(self.root.glob(".preflight.json.*.tmp")), [])

    def test_atomic_receipt_rejects_symbolic_link_destination(self) -> None:
        target = self.root / "must-not-change.json"
        target.write_text('{"sentinel":true}\n', encoding="utf-8")
        output = self.root / "preflight.json"
        output.symlink_to(target)
        with self.assertRaisesRegex(preflight.ValidationError, "symbolic link"):
            preflight._atomic_write_json(
                output,
                {"schema_version": 1, "ok": True, "contract_sha256": "d" * 64},
            )
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8")), {"sentinel": True}
        )


class CompilerAttestationTests(PreflightFixture):
    def _enable_compiler(self) -> Path:
        compiler = self.root / "hxcc"
        compiler.write_text(
            "#!/bin/sh\nprintf '%s\\n' 'hxcc version test-build'\n",
            encoding="utf-8",
        )
        compiler.chmod(0o700)
        authorization = self.root / "compiler-authorization.receipt"
        authorization.write_bytes(b"release-owner-approved-compiler")
        self.contract["runtime"].update(
            {
                "requires_device_compiler": True,
                "device_compiler_authorized": True,
                "device_compiler_id": "hxcc version test-build",
                "device_compiler_sha256": self._sha(compiler),
                "compiler_authorization_receipt": str(authorization),
                "compiler_authorization_receipt_sha256": self._sha(authorization),
            }
        )
        return compiler

    def test_compiler_binary_and_authorization_receipt_are_hash_pinned(self) -> None:
        compiler = self._enable_compiler()
        report = self._run(device_compiler=compiler)
        statuses = {item["name"]: item["status"] for item in report["checks"]}
        self.assertEqual(statuses["runtime.device_compiler_sha256"], "pass")
        self.assertEqual(
            statuses["runtime.device_compiler_authorization_receipt"], "pass"
        )
        self.assertEqual(statuses["runtime.device_compiler_identity"], "pass")
        self.assertTrue(report["ok"], report)

    def test_missing_authorization_receipt_fields_fail_closed_clearly(self) -> None:
        compiler = self._enable_compiler()
        del self.contract["runtime"]["compiler_authorization_receipt"]
        del self.contract["runtime"]["compiler_authorization_receipt_sha256"]
        report = self._run(device_compiler=compiler)
        check = next(
            item
            for item in report["checks"]
            if item["name"] == "runtime.device_compiler_authorization_receipt"
        )
        self.assertEqual(check["status"], "fail")
        self.assertIn("missing", check["detail"])
        self.assertFalse(report["ok"])

    def test_compiler_sha256_substitution_fails_before_version_probe(self) -> None:
        compiler = self._enable_compiler()
        self.contract["runtime"]["device_compiler_sha256"] = "1" * 64
        with mock.patch.object(preflight.subprocess, "run") as version_probe:
            report = self._run(device_compiler=compiler)
        statuses = {item["name"]: item["status"] for item in report["checks"]}
        self.assertEqual(statuses["runtime.device_compiler_sha256"], "fail")
        self.assertEqual(statuses["runtime.device_compiler_identity"], "fail")
        version_probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
