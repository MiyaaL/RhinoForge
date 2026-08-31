from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    module_name = f"kernel_campaign_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


import campaign_common as common


preflight = load_script("preflight")
recorder = load_script("record_iteration")
selector = load_script("select_best")
summarizer = load_script("summarize_hwperf")
import attestation


COMMIT = "a" * 40
PARENT = "b" * 40
KERNEL = "gemm_fp16_test"


class CampaignFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate"
        self.reference = self.root / "reference"
        self.candidate.mkdir()
        self.reference.mkdir()
        (self.reference / "oracle.py").write_text(
            "# frozen reference fixture\n", encoding="utf-8"
        )
        self.runner = self.root / "runner.py"
        self.runner.write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.openssl = Path("/usr/bin/openssl")
        self.signing_key = self.root / "unit-release-key.pem"
        subprocess.run(
            [str(self.openssl), "genpkey", "-algorithm", "ED25519", "-out", str(self.signing_key)],
            check=True,
            capture_output=True,
        )
        public_der = subprocess.check_output(
            [
                str(self.openssl),
                "pkey",
                "-in",
                str(self.signing_key),
                "-pubout",
                "-outform",
                "DER",
            ]
        )
        self.public_key = public_der[-32:]
        self.public_key_base64 = base64.b64encode(self.public_key).decode("ascii")
        self.asset = self.root / "operator.ref"
        self.asset.write_bytes(b"opaque-do-not-parse")
        self.manifest = Path(str(self.asset) + ".kernels")
        self.manifest.write_text(
            "rhinoforge-kernels-v1\n"
            f"asset-size={self.asset.stat().st_size}\n"
            f"{KERNEL}\n",
            encoding="utf-8",
        )
        self.launch_library = self.root / "librhino_launch.so.1.0.0"
        self.launch_library.write_bytes(b"public-launch-test-fixture")
        self.contract = self.root / "contract.toml"
        self.preflight_receipt = self.root / "preflight.json"
        self.process_counter = 0
        self.clean_patch = mock.patch.object(preflight, "_git_clean", return_value=True)
        self.clean_patch.start()
        self.root_patch = mock.patch.object(
            preflight, "_is_rhinoforge_root", return_value=True
        )
        self.root_patch.start()
        self.record_git_patch = mock.patch.object(
            recorder, "verify_candidate_git", return_value=self.candidate
        )
        self.record_git_patch.start()
        self.select_git_patch = mock.patch.object(
            selector, "verify_candidate_git", return_value=self.candidate
        )
        self.select_git_patch.start()
        self.source_tree_patch = mock.patch.object(
            selector,
            "_source_tree_identity",
            return_value=("d" * 40, "e" * 64),
        )
        self.source_tree_patch.start()
        self.write_contract()

    def tearDown(self) -> None:
        self.source_tree_patch.stop()
        self.select_git_patch.stop()
        self.record_git_patch.stop()
        self.root_patch.stop()
        self.clean_patch.stop()
        self.temporary.cleanup()

    def write_contract(
        self,
        *,
        state: str = "approved",
        required_kernel: str = KERNEL,
        roofline: bool = False,
        requires_compiler: bool = False,
        compiler_authorized: bool = False,
        release_owner: str = "UNRESOLVED",
    ) -> None:
        roofline_text = ""
        if roofline:
            roofline_source = self.root / "roofline-source.json"
            roofline_source.write_bytes(b'{"calibration":"unit-v1"}\n')
            roofline_text = (
                "\n[roofline]\n"
                "peak_ops_per_s = 1000000000\n"
                "bandwidth_bytes_per_s = 2000000000\n"
                "launch_floor_s = 0.000001\n"
                "peak_kind = \"empirical\"\n"
                "source_id = \"unit-calibration-v1\"\n"
                f"source_file = \"{roofline_source}\"\n"
                f"source_sha256 = \"{hashlib.sha256(roofline_source.read_bytes()).hexdigest()}\"\n"
            )
        self.contract.write_text(
            f'''schema_version = 1
state = "{state}"

[campaign]
id = "unit-gemm"
op = "gemm"
candidate_root = "{self.candidate}"
reference_root = "{self.reference}"
reference_tree_sha256 = "{common.content_tree_sha256(self.reference)}"

[commands]
correctness = ["{sys.executable}", "runner.py", "--mode", "correctness"]
benchmark = ["{sys.executable}", "runner.py", "--mode", "benchmark"]
profile_one_call = ["{sys.executable}", "runner.py", "--mode", "profile-one-call"]
benchmark_adapter = "runner.py"
benchmark_adapter_sha256 = "{hashlib.sha256(self.runner.read_bytes()).hexdigest()}"

[attestation]
mode = "external"
scheme = "ed25519"
namespace = "rhinoforge-full-verdict-v1"
signer_id = "unit-release-authority"
public_key_base64 = "{self.public_key_base64}"
policy_sha256 = "{hashlib.sha256(b'unit-policy').hexdigest()}"
signer_build_sha256 = "{hashlib.sha256(b'unit-signer-build').hexdigest()}"
verifier_executable = "{self.openssl}"
verifier_sha256 = "{hashlib.sha256(self.openssl.read_bytes()).hexdigest()}"

[runtime]
trace_schema = "rhinoforge-rpu-chrome-v1"
trace_event_category = "rpu_device_program"
rhinoforge_commit = "{COMMIT}"
runtime_set = "unit-runtime-v1"
torch_version = "test-torch"
launch_version = "test-launch"
launch_library = "{self.launch_library}"
launch_library_sha256 = "{hashlib.sha256(self.launch_library.read_bytes()).hexdigest()}"
operator_version = "test-operator"
operator_asset = "{self.asset}"
operator_asset_sha256 = "{hashlib.sha256(self.asset.read_bytes()).hexdigest()}"
kernel_manifest = "{self.manifest}"
kernel_manifest_sha256 = "{hashlib.sha256(self.manifest.read_bytes()).hexdigest()}"
required_kernels = ["{required_kernel}"]
requires_device_compiler = {str(requires_compiler).lower()}
device_compiler_authorized = {str(compiler_authorized).lower()}
device_compiler_id = "{('hxcc 1.0+0a6ba7e4' if requires_compiler else 'not-required')}"
device_compiler_sha256 = "{hashlib.sha256(b'test-compiler').hexdigest() if requires_compiler else '0' * 64}"
compiler_authorization_receipt = "{str(self.root / 'compiler-authorization.json') if requires_compiler else 'not-required'}"
compiler_authorization_receipt_sha256 = "{hashlib.sha256(b'test-authorization').hexdigest() if requires_compiler else '0' * 64}"
asset_release_owner = "{release_owner}"

[measurement]
metric = "latency_us"
direction = "min"
warmup = 2
samples = 3
verdict_processes = 2

[gates]
max_abs = 0.1
max_rel = 0.2
anchor_max_abs = 0.1
anchor_max_rel = 0.2
max_cv = 1.0
require_same_dtype = true
require_fp32_anchor = true
require_graph_replay = true
require_independent_outputs = true

[[workloads]]
id = "m16-n32-k64"
m = 16
n = 32
k = 64
dtype = "float16"
layout = "x-row-major,w-row-major-transposed-math"
semantic_inputs = ["alpha=1", "bias=none", "output=materialized"]
{roofline_text}''',
            encoding="utf-8",
        )

    def result(
        self,
        candidate_id: str = "candidate-1",
        *,
        commit: str = COMMIT,
        verdict: str = "full",
        samples: list[float] | None = None,
        nonfinite_count: int = 0,
        ) -> dict:
        self.process_counter += 1
        contract_hash = hashlib.sha256(self.contract.read_bytes()).hexdigest()
        preflight_value = {
            "schema_version": 1,
            "ok": True,
            "contract_sha256": contract_hash,
            "checks": [{"name": "probe.torch_rpu", "status": "pass"}],
        }
        self.preflight_receipt.write_text(
            json.dumps(preflight_value, sort_keys=True), encoding="utf-8"
        )
        preflight_hash = hashlib.sha256(self.preflight_receipt.read_bytes()).hexdigest()
        candidate_samples = samples or [1000.0, 1100.0, 1200.0]
        reference_samples = [
            1300.0 + 100.0 * index for index in range(len(candidate_samples))
        ]

        def p95(values: list[float]) -> float:
            ordered = sorted(values)
            position = (len(ordered) - 1) * 0.95
            lower = math.floor(position)
            upper = math.ceil(position)
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight

        candidate_p50 = float(statistics.median(candidate_samples))
        reference_p50 = float(statistics.median(reference_samples))
        finite = nonfinite_count == 0
        per_gates = {
            "candidate_execution": True,
            "finite": finite,
            "same_dtype_parity": finite,
            "layout": finite,
            "residency": finite,
            "fp32_anchor": finite,
            "graph_lifecycle": finite,
            "timing_samples": True,
            "stability": finite,
            "single_device_program": finite,
        }
        eligible = all(per_gates.values())
        top_gates = {
            "workloads_present": True,
            "all_workloads_eligible": eligible,
            "metric_finite": True,
        }
        result_value = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "candidate_commit": commit,
            "parent_commit": PARENT,
            "contract_sha256": contract_hash,
            "preflight_sha256": preflight_hash,
            "process": {
                "boot_id": "12345678-1234-4123-8123-123456789abc",
                "pid": 1000 + self.process_counter,
                "start_ticks": 100000 + self.process_counter,
                "run_uuid": str(uuid.uuid4()),
                "attested": False,
            },
            "verdict": verdict,
            "metric": {
                "name": "latency_us",
                "direction": "min",
                "value": candidate_p50 / 1000.0,
                "aggregation": "median_workload_candidate_p50",
            },
            "workloads": [
                {
                    "id": "m16-n32-k64",
                    "raw_samples": {
                        "candidate_ns": candidate_samples,
                        "reference_ns": reference_samples,
                    },
                    "summary": {
                        "candidate_p50_ns": candidate_p50,
                        "reference_p50_ns": reference_p50,
                        "candidate_p95_ns": p95(candidate_samples),
                        "reference_p95_ns": p95(reference_samples),
                        "candidate_mad_ns": float(
                            statistics.median(
                                abs(value - candidate_p50)
                                for value in candidate_samples
                            )
                        ),
                        "reference_mad_ns": float(
                            statistics.median(
                                abs(value - reference_p50)
                                for value in reference_samples
                            )
                        ),
                        "candidate_cv": statistics.pstdev(candidate_samples)
                        / statistics.fmean(candidate_samples),
                        "reference_cv": statistics.pstdev(reference_samples)
                        / statistics.fmean(reference_samples),
                        "candidate_sample_count": len(candidate_samples),
                        "reference_sample_count": len(reference_samples),
                        "timing_order_blocks": ["reference/candidate/ABBA"],
                    },
                    "correctness": {
                        "passed": finite,
                        "nonfinite_count": nonfinite_count,
                        "max_abs": 0.01,
                        "max_rel": 0.02,
                        "anchor_max_abs": 0.01,
                        "anchor_max_rel": 0.02,
                        "same_dtype": finite,
                        "fp32_anchor": finite,
                        "fp32_anchor_present": True,
                        "probes": {
                            "baseline": {
                                "same_dtype": {
                                    "passed": finite,
                                    "layout_match": finite,
                                    "device_match": finite,
                                    "leaves": [
                                        {
                                            "path": "$",
                                            "candidate_strides": [32, 1],
                                            "reference_strides": [32, 1],
                                            "layout_match": finite,
                                            "candidate_device": "rpu:0",
                                            "reference_device": "rpu:0",
                                            "device_match": finite,
                                        }
                                    ],
                                }
                            }
                        },
                    },
                    "lifecycle": {
                        "passed": finite,
                        "graph_build_count": 1,
                        "graph_replay_count": 3,
                        "cache_size_stable": True,
                        "cache_invariant_ok": True,
                        "independent_outputs": finite,
                        "graph_state_present": True,
                        "stable_state": {
                            "build_count": 1,
                            "replay_count": 1,
                            "cache_size": 1,
                            "invariant_ok": True,
                        },
                        "final_state": {
                            "build_count": 1,
                            "replay_count": 3,
                            "cache_size": 1,
                            "invariant_ok": True,
                        },
                    },
                    "execution": {
                        "kernel_names": [KERNEL],
                        "launch_count": 1,
                        "single_device_program": True,
                        "trace_summary_sha256": "c" * 64,
                        "preflight_sha256": preflight_hash,
                    },
                    "hard_gates": per_gates,
                    "eligible": eligible,
                }
            ],
            "hard_gates": top_gates,
        }
        if not finite:
            workload = result_value["workloads"][0]
            workload["raw_samples"] = {"candidate_ns": [], "reference_ns": []}
            workload["summary"] = {
                "candidate_p50_ns": None,
                "reference_p50_ns": None,
            }
            result_value["metric"]["value"] = None
            result_value["hard_gates"]["metric_finite"] = False
        return result_value

    def record_result(self, source: Path, output: Path) -> dict:
        raw = json.loads(source.read_text(encoding="utf-8"))
        observed = dict(raw["process"])
        observed.pop("attested", None)
        return recorder.record_iteration(
            self.contract,
            source,
            output,
            observed_process=observed,
            preflight_receipt=self.preflight_receipt,
        )

    def release_proof(self, output: Path, selected_commit: str) -> tuple[Path, str]:
        journal_path = output / "results.jsonl"
        raw_journal = journal_path.read_bytes()
        records = [json.loads(line) for line in raw_journal.decode("utf-8").splitlines()]
        record_hashes = [
            hashlib.sha256(common.canonical_json(record).encode("utf-8")).hexdigest()
            for record in records
        ]
        selected_records = [
            (record, record_hash)
            for record, record_hash in zip(records, record_hashes)
            if record["candidate_commit"] == selected_commit
        ]
        self.assertTrue(selected_records)
        selected = selected_records[0][0]
        command_hash = lambda name: hashlib.sha256(
            common.canonical_json(common.load_contract(self.contract)["commands"][name]).encode(
                "utf-8"
            )
        ).hexdigest()
        processes = [
            {
                "process_id": "preflight",
                "role": "preflight",
                "workload_id": None,
                "receipt_sha256": None,
                "boot_id": "12345678-1234-4123-8123-123456789abc",
                "pid": 899,
                "start_ticks": 89900,
                "run_uuid": None,
                "executable_sha256": "8" * 64,
                "argv_sha256": "9" * 64,
                "environment_sha256": "a" * 64,
                "started_monotonic_ns": 101,
                "finished_monotonic_ns": 109,
                "exit_code": 0,
            },
            {
                "process_id": "profile-m16-n32-k64",
                "role": "profile",
                "workload_id": "m16-n32-k64",
                "receipt_sha256": None,
                "boot_id": "12345678-1234-4123-8123-123456789abc",
                "pid": 900,
                "start_ticks": 90000,
                "run_uuid": None,
                "executable_sha256": "1" * 64,
                "argv_sha256": command_hash("profile_one_call"),
                "environment_sha256": "2" * 64,
                "started_monotonic_ns": 110,
                "finished_monotonic_ns": 120,
                "exit_code": 0,
            }
        ]
        for index, (record, record_hash) in enumerate(selected_records):
            process = record["process"]
            processes.append(
                {
                    "process_id": f"benchmark-{index}",
                    "role": "benchmark",
                    "workload_id": None,
                    "receipt_sha256": record_hash,
                    "boot_id": process["boot_id"],
                    "pid": process["pid"],
                    "start_ticks": process["start_ticks"],
                    "run_uuid": process["run_uuid"],
                    "executable_sha256": "3" * 64,
                    "argv_sha256": command_hash("benchmark"),
                    "environment_sha256": "4" * 64,
                    "started_monotonic_ns": 130 + index * 2,
                    "finished_monotonic_ns": 131 + index * 2,
                    "exit_code": 0,
                }
            )
        challenge = "unit-release-challenge"
        statement = {
            "schema_version": 1,
            "kind": "rhinoforge-kernel-release-proof",
            "proof_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "signer_id": "unit-release-authority",
            "challenge": challenge,
            "sequence": 1,
            "previous_proof_sha256": "0" * 64,
            "issued_at": "2026-08-31T00:00:00Z",
            "issuer": {
                "policy_sha256": hashlib.sha256(b"unit-policy").hexdigest(),
                "signer_build_sha256": hashlib.sha256(b"unit-signer-build").hexdigest(),
                "collection_mode": "authority-controlled-board-session",
            },
            "inputs": {
                "campaign_id": "unit-gemm",
                "contract_sha256": hashlib.sha256(self.contract.read_bytes()).hexdigest(),
                "preflight_sha256": selected["preflight_sha256"],
                "adapter_sha256": hashlib.sha256(self.runner.read_bytes()).hexdigest(),
                "benchmark_argv_sha256": command_hash("benchmark"),
                "profile_argv_sha256": command_hash("profile_one_call"),
            },
            "source": {
                "repository_id": "unit-candidate",
                "candidate_commit": selected_commit,
                "parent_commit": selected["parent_commit"],
                "git_tree_oid": "d" * 40,
                "source_root_sha256": "e" * 64,
            },
            "execution": {
                "board_id": "unit-board",
                "lease": {
                    "lease_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "start_sequence": 1,
                    "finish_sequence": 2,
                    "previous_journal_head_sha256": "0" * 64,
                    "start_record_sha256": "5" * 64,
                    "finish_record_sha256": "6" * 64,
                    "final_journal_head_sha256": "6" * 64,
                    "started_monotonic_ns": 100,
                    "finished_monotonic_ns": 200,
                },
                "processes": processes,
                "traces": [
                    {
                        "workload_id": "m16-n32-k64",
                        "capture_id": "unit-capture",
                        "collector_process_id": "profile-m16-n32-k64",
                        "raw_sha256": "7" * 64,
                        "raw_size": 128,
                        "summary_sha256": "c" * 64,
                        "summary_size": 128,
                        "complete_event_count": 1,
                        "manifest_launch_count": 1,
                        "unknown_complete_event_count": 0,
                        "admitted_kernel_names": [KERNEL],
                    }
                ],
            },
            "decision": {
                "record_sha256s": record_hashes,
                "results_snapshot": {
                    "sha256": hashlib.sha256(raw_journal).hexdigest(),
                    "size": len(raw_journal),
                    "line_count": len(records),
                },
                "selected_candidate_commit": selected_commit,
            },
        }
        payload = common.canonical_json(statement).encode("utf-8")
        pae_path = output / "unit-proof-pae.bin"
        signature_path = output / "unit-proof-signature.bin"
        pae_path.write_bytes(
            attestation._pae(attestation.PAYLOAD_TYPE.encode("utf-8"), payload)
        )
        subprocess.run(
            [
                str(self.openssl),
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(self.signing_key),
                "-in",
                str(pae_path),
                "-out",
                str(signature_path),
            ],
            check=True,
            capture_output=True,
        )
        signature = signature_path.read_bytes()
        envelope = {
            "payloadType": attestation.PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode("ascii"),
            "signatures": [
                {
                    "keyid": attestation.key_id(self.public_key),
                    "sig": base64.b64encode(signature).decode("ascii"),
                }
            ],
        }
        path = output / "release-proof.dsse.json"
        path.write_text(common.canonical_json(envelope) + "\n", encoding="utf-8")
        return path, challenge


class ContractAndPreflightTests(CampaignFixture):
    def test_schema_v1_and_python310_subset_parser(self) -> None:
        text = self.contract.read_text(encoding="utf-8")
        parsed = common._load_toml_subset(text)
        validated = common.validate_contract(parsed)
        self.assertEqual(validated["campaign"]["op"], "gemm")
        self.assertEqual(validated["workloads"][0]["semantic_inputs"][0], "alpha=1")

    def test_draft_state_fails_closed_clearly(self) -> None:
        self.write_contract(state="draft")
        report = preflight.run_preflight(self.contract)
        self.assertFalse(report["ok"])
        self.assertIn("state must be exactly 'approved'", report["checks"][0]["detail"])

    def test_preflight_hashes_opaque_asset_without_parsing_or_output(self) -> None:
        original_hash = preflight._stream_sha256
        hashed_paths: list[Path] = []

        def observed_hash(path: Path) -> str:
            hashed_paths.append(path)
            return original_hash(path)

        with mock.patch.object(preflight, "_stream_sha256", side_effect=observed_hash), mock.patch.object(
            preflight, "_git_head", return_value=COMMIT
        ), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(
            preflight, "_device_capable", return_value=True
        ):
            report = preflight.run_preflight(
                self.contract, rhinoforge_root=self.root
            )
        self.assertTrue(report["ok"], report)
        self.assertIn(self.asset, hashed_paths)
        encoded = json.dumps(report)
        self.assertNotIn("opaque-do-not-parse", encoded)
        self.assertNotIn(
            hashlib.sha256(self.asset.read_bytes()).hexdigest(), encoded
        )
        self.assertTrue(report["empirical_floor_required"])
        self.assertEqual(report["required_kernel_count"], 1)

    def test_same_size_asset_substitution_fails_sha256_identity(self) -> None:
        original_size = self.asset.stat().st_size
        self.asset.write_bytes(b"X" * original_size)
        self.assertEqual(self.asset.stat().st_size, original_size)
        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True):
            report = preflight.run_preflight(self.contract, rhinoforge_root=self.root)
        self.assertFalse(report["ok"])
        failed = {item["name"] for item in report["checks"] if item["status"] == "fail"}
        self.assertIn("runtime.operator_asset_identity", failed)
        self.assertIn("runtime.asset_manifest", failed)

    def test_compiler_requires_explicit_authorization_and_owner(self) -> None:
        self.write_contract(requires_compiler=True)
        with self.assertRaisesRegex(common.ValidationError, "authorized=true"):
            common.load_contract(self.contract)
        self.write_contract(
            requires_compiler=True,
            compiler_authorized=True,
            release_owner="UNRESOLVED",
        )
        with self.assertRaisesRegex(common.ValidationError, "resolved asset_release_owner"):
            common.load_contract(self.contract)

    def test_unresolved_semantics_and_nonfusion_names_fail_closed(self) -> None:
        text = self.contract.read_text(encoding="utf-8")
        self.contract.write_text(
            text.replace('"alpha=1"', '"rounding=must-be-frozen"'),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(common.ValidationError, "unresolved sentinel"):
            common.load_contract(self.contract)

        self.write_contract(required_kernel="llama_rms_norm")
        parsed = common.load_toml(self.contract)
        parsed["campaign"]["op"] = "rmsnorm+quant_mxfp8"
        parsed["runtime"]["required_kernels"] = [
            "llama_rms_norm",
            "parallel_linear_wNVFP4a16",
        ]
        with self.assertRaisesRegex(common.ValidationError, "requires runtime.fused_kernel"):
            common.validate_contract(parsed)

    def test_missing_devices_and_required_kernel_fail_closed(self) -> None:
        self.write_contract(required_kernel="missing_kernel")
        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight.glob, "glob", return_value=[]
        ):
            report = preflight.run_preflight(
                self.contract, rhinoforge_root=self.root
            )
        self.assertFalse(report["ok"])
        failed = {item["name"] for item in report["checks"] if item["status"] == "fail"}
        self.assertIn("runtime.required_kernels", failed)
        self.assertIn("device.rpu", failed)
        self.assertIn("device.mem", failed)

    def test_python_command_requires_existing_runner_argument(self) -> None:
        self.runner.unlink()
        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True):
            report = preflight.run_preflight(self.contract, rhinoforge_root=self.root)
        failed = {item["name"] for item in report["checks"] if item["status"] == "fail"}
        self.assertIn("command.correctness", failed)
        self.assertIn("command.benchmark", failed)

    def test_dirty_rhinoforge_tree_fails_closed(self) -> None:
        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight, "_git_clean", return_value=False
        ), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True):
            report = preflight.run_preflight(
                self.contract, rhinoforge_root=self.root
            )
        failed = {item["name"] for item in report["checks"] if item["status"] == "fail"}
        self.assertIn("runtime.rhinoforge_clean_tree", failed)

    def test_optional_torch_probe_requires_fp16_roundtrip(self) -> None:
        dtype = object()

        class FakeTensor:
            def __init__(self, values):
                self.values = list(values)
                self.dtype = dtype

            def to(self, _device):
                return FakeTensor(self.values)

            def tolist(self):
                return list(self.values)

        synchronize_count = [0]
        namespace = SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: synchronize_count.__setitem__(
                0, synchronize_count[0] + 1
            ),
        )
        fake_torch = SimpleNamespace(
            __version__="test-torch",
            rpu=namespace,
            float16=dtype,
            tensor=lambda values, dtype: FakeTensor(values),
        )

        def fake_import(name: str):
            return fake_torch if name == "torch" else object()

        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True), mock.patch.object(
            preflight.importlib, "import_module", side_effect=fake_import
        ):
            report = preflight.run_preflight(
                self.contract,
                rhinoforge_root=self.root,
                probe_torch_rpu=True,
            )
        self.assertTrue(report["ok"], report)
        self.assertEqual(synchronize_count[0], 2)
        probe = next(item for item in report["checks"] if item["name"] == "probe.torch_rpu")
        self.assertEqual(probe["status"], "pass")

    def test_compiler_identity_uses_authorized_version_probe(self) -> None:
        compiler = self.root / "hxcc"
        compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        compiler.chmod(0o755)
        self.write_contract(
            requires_compiler=True,
            compiler_authorized=True,
            release_owner="release-owner",
        )
        (self.root / "compiler-authorization.json").write_bytes(
            b"test-authorization"
        )
        contract_text = self.contract.read_text(encoding="utf-8")
        contract_text = contract_text.replace(
            hashlib.sha256(b"test-compiler").hexdigest(),
            hashlib.sha256(compiler.read_bytes()).hexdigest(),
        )
        self.contract.write_text(contract_text, encoding="utf-8")
        completed = SimpleNamespace(
            returncode=0,
            stdout="hxcc 1.0+0a6ba7e4\nrestricted detail ignored\n",
            stderr="",
        )
        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True), mock.patch.object(
            preflight.subprocess, "run", return_value=completed
        ) as run:
            report = preflight.run_preflight(
                self.contract,
                rhinoforge_root=self.root,
                device_compiler=compiler,
            )
        self.assertTrue(report["ok"], report)
        run.assert_called_once_with(
            [str(compiler.resolve()), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_complete_roofline_does_not_claim_empirical_floor_needed(self) -> None:
        self.write_contract(roofline=True)
        with mock.patch.object(preflight, "_git_head", return_value=COMMIT), mock.patch.object(
            preflight.glob, "glob", return_value=["/dev/rpu0"]
        ), mock.patch.object(preflight, "_device_capable", return_value=True):
            report = preflight.run_preflight(self.contract, rhinoforge_root=self.root)
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["empirical_floor_required"])


class RecordTests(CampaignFixture):
    def _write_result(self, value: dict, name: str = "result.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
        return path

    def test_atomic_append_markdown_and_duplicate_rejection(self) -> None:
        output = self.root / "journal"
        source = self._write_result(self.result())
        report = self.record_result(source, output)
        self.assertTrue(report["local_gate_eligible"])
        self.assertFalse(report["release_eligible"])
        lines = (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["candidate_id"], "candidate-1")
        markdown = output / "iterations" / "candidate-1.md"
        before = markdown.read_bytes()
        with self.assertRaises(common.ValidationError):
            self.record_result(source, output)
        self.assertEqual((output / "results.jsonl").read_text().count("\n"), 1)
        self.assertEqual(markdown.read_bytes(), before)

    def test_nonfinite_raw_sample_is_rejected(self) -> None:
        value = self.result()
        value["workloads"][0]["raw_samples"]["candidate_ns"][1] = float("nan")
        source = self._write_result(value)
        with self.assertRaises(common.ValidationError):
            self.record_result(source, self.root / "journal")

    def test_nonfinite_output_is_recorded_ineligible(self) -> None:
        value = self.result(nonfinite_count=1)
        value["workloads"][0]["correctness"]["fp32_anchor"] = None
        value["workloads"][0]["correctness"]["error"] = "non-finite output"
        source = self._write_result(value)
        report = self.record_result(source, self.root / "journal")
        self.assertFalse(report["local_gate_eligible"])
        stored = json.loads(
            (self.root / "journal" / "results.jsonl").read_text().strip()
        )
        self.assertFalse(stored["workloads"][0]["eligible"])
        self.assertFalse(stored["workloads"][0]["hard_gates"]["finite"])
        self.assertIsNone(stored["metric"]["value"])

    def test_failed_timing_records_partial_graph_evidence(self) -> None:
        value = self.result(nonfinite_count=1)
        workload = value["workloads"][0]
        workload["correctness"]["fp32_anchor"] = None
        workload["correctness"]["error"] = "timing failed after warmup"
        workload["lifecycle"].update(
            {
                "final_state": None,
                "graph_build_count": 0,
                "graph_replay_count": 0,
                "cache_size_stable": False,
                "cache_invariant_ok": False,
                "error": "timing failed after warmup",
            }
        )
        source = self._write_result(value)
        report = self.record_result(source, self.root / "journal")
        self.assertFalse(report["local_gate_eligible"])

    def test_timed_receipt_requires_complete_graph_even_if_gate_is_disabled(
        self,
    ) -> None:
        self.contract.write_text(
            self.contract.read_text(encoding="utf-8").replace(
                "require_graph_replay = true",
                "require_graph_replay = false",
            ),
            encoding="utf-8",
        )
        value = self.result()
        value["workloads"][0]["lifecycle"]["final_state"] = None
        source = self._write_result(value)
        with self.assertRaisesRegex(
            common.ValidationError,
            "timed workload requires stable_state and final_state",
        ):
            self.record_result(source, self.root / "journal")

    def test_failed_receipt_cannot_claim_latency(self) -> None:
        value = self.result(nonfinite_count=1)
        value["metric"]["value"] = 0.001
        source = self._write_result(value)
        with self.assertRaisesRegex(common.ValidationError, "must use null metric"):
            self.record_result(source, self.root / "journal")

    def test_inconsistent_summary_fails_closed(self) -> None:
        value = self.result()
        value["workloads"][0]["summary"]["candidate_p50_ns"] += 1
        source = self._write_result(value)
        with self.assertRaisesRegex(common.ValidationError, "does not match raw"):
            self.record_result(source, self.root / "journal")

    def test_graph_transition_is_recomputed_from_exact_states(self) -> None:
        value = self.result()
        value["workloads"][0]["lifecycle"]["stable_state"]["replay_count"] = 0
        source = self._write_result(value)
        with self.assertRaisesRegex(
            common.ValidationError, "lifecycle.passed contradicts"
        ):
            self.record_result(source, self.root / "journal")

    def test_graph_aggregate_must_match_final_state(self) -> None:
        value = self.result()
        value["workloads"][0]["lifecycle"]["graph_replay_count"] = 4
        source = self._write_result(value)
        with self.assertRaisesRegex(
            common.ValidationError, "graph_replay_count contradicts final_state"
        ):
            self.record_result(source, self.root / "journal")

    def test_signal_uses_its_smaller_sample_floor_but_full_uses_contract(self) -> None:
        self.contract.write_text(
            self.contract.read_text(encoding="utf-8").replace(
                "samples = 3", "samples = 5"
            ),
            encoding="utf-8",
        )
        short_samples = [100.0, 101.0, 102.0]
        signal = self.result(verdict="signal", samples=short_samples)
        signal_source = self._write_result(signal, "signal.json")
        report = self.record_result(signal_source, self.root / "signal-journal")
        self.assertFalse(report["local_gate_eligible"])

        full = self.result(candidate_id="full-short", samples=short_samples)
        full_source = self._write_result(full, "full.json")
        with self.assertRaisesRegex(common.ValidationError, "at least 5 samples"):
            self.record_result(full_source, self.root / "full-journal")

    def test_signal_rejects_fewer_than_three_samples_per_arm(self) -> None:
        value = self.result(verdict="signal", samples=[100.0, 101.0])
        source = self._write_result(value, "short-signal.json")
        with self.assertRaisesRegex(common.ValidationError, "at least 3 samples"):
            self.record_result(source, self.root / "journal")

    def test_self_attested_full_receipt_is_rejected(self) -> None:
        value = self.result()
        value["process"]["attested"] = True
        source = self._write_result(value)
        with self.assertRaisesRegex(common.ValidationError, "self-attested"):
            recorder.record_iteration(
                self.contract,
                source,
                self.root / "journal",
                preflight_receipt=self.preflight_receipt,
            )

    def test_candidate_git_commit_parent_and_clean_tree_are_verified(self) -> None:
        actual_recorder = load_script("record_iteration")
        subprocess.run(
            ["git", "init", "--initial-branch", "main", str(self.candidate)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(self.candidate), "config", "user.name", "Unit Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.candidate), "config", "user.email", "unit@example.invalid"],
            check=True,
        )
        source = self.candidate / "candidate.txt"
        source.write_text("parent\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.candidate), "add", "candidate.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(self.candidate), "commit", "-m", "parent"],
            check=True,
            capture_output=True,
        )
        parent = subprocess.check_output(
            ["git", "-C", str(self.candidate), "rev-parse", "HEAD"], text=True
        ).strip()
        source.write_text("candidate\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.candidate), "add", "candidate.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(self.candidate), "commit", "-m", "candidate"],
            check=True,
            capture_output=True,
        )
        head = subprocess.check_output(
            ["git", "-C", str(self.candidate), "rev-parse", "HEAD"], text=True
        ).strip()
        contract = common.load_contract(self.contract)
        actual_recorder.verify_candidate_git(
            self.contract,
            contract,
            {"candidate_commit": head, "parent_commit": parent},
        )
        source.write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(common.ValidationError, "clean"):
            actual_recorder.verify_candidate_git(
                self.contract,
                contract,
                {"candidate_commit": head, "parent_commit": parent},
            )

    def test_candidate_root_must_be_isolated_git_top_level(self) -> None:
        actual_recorder = load_script("record_iteration")
        contract = common.load_contract(self.contract)
        with mock.patch.object(
            actual_recorder, "_git", return_value=str(self.root)
        ):
            with self.assertRaisesRegex(common.ValidationError, "isolated Git"):
                actual_recorder.verify_candidate_git(
                    self.contract,
                    contract,
                    {"candidate_commit": COMMIT, "parent_commit": PARENT},
                )


class PromotionTests(CampaignFixture):
    def _append(self, output: Path, result: dict, index: int) -> None:
        source = self.root / f"result-{index}.json"
        source.write_text(json.dumps(result), encoding="utf-8")
        self.record_result(source, output)

    def test_only_full_all_gate_candidate_can_win_and_commit_is_exact(self) -> None:
        output = self.root / "journal"
        signal = self.result(
            "signal-fast", commit="1" * 40, verdict="signal", samples=[100, 101, 102]
        )
        failed = self.result(
            "full-nonfinite",
            commit="2" * 40,
            verdict="full",
            samples=[200, 201, 202],
            nonfinite_count=1,
        )
        lonely_fast = self.result(
            "full-lonely-fast", commit="5" * 40, samples=[300, 400, 500]
        )
        slower_a = self.result(
            "full-slower-a", commit="3" * 40, samples=[900, 1000, 1100]
        )
        slower_b = self.result(
            "full-slower-b", commit="3" * 40, samples=[950, 1050, 1150]
        )
        winner_a = self.result(
            "full-winner-a", commit="4" * 40, samples=[700, 800, 900]
        )
        winner_b = self.result(
            "full-winner-b", commit="4" * 40, samples=[750, 850, 950]
        )
        for index, value in enumerate(
            (signal, failed, lonely_fast, slower_a, slower_b, winner_a, winner_b)
        ):
            self._append(output, value, index)

        best_path = output / "BEST.json"
        proof, challenge = self.release_proof(output, "4" * 40)
        best = selector.select_best(
            self.contract,
            output / "results.jsonl",
            best_path,
            release_proof=proof,
            expected_challenge=challenge,
        )
        self.assertEqual(best["candidate_id"], "full-winner-a")
        self.assertEqual(best["candidate_commit"], "4" * 40)
        self.assertEqual(best["process_count"], 2)
        self.assertEqual(
            best["evidence_candidate_ids"], ["full-winner-a", "full-winner-b"]
        )
        self.assertNotIn("signal-fast", best_path.read_text())
        self.assertEqual(json.loads(best_path.read_text())["candidate_commit"], "4" * 40)

    def test_one_full_receipt_cannot_satisfy_two_process_contract(self) -> None:
        output = self.root / "journal"
        self._append(output, self.result("only-full"), 0)
        proof, challenge = self.release_proof(output, COMMIT)
        with self.assertRaisesRegex(common.ValidationError, "enough all-gate"):
            selector.select_best(
                self.contract,
                output / "results.jsonl",
                output / "BEST.json",
                release_proof=proof,
                expected_challenge=challenge,
            )
        self.assertFalse((output / "BEST.json").exists())

    def test_same_observed_process_with_two_ids_cannot_promote(self) -> None:
        output = self.root / "journal"
        first = self.result("same-process-a")
        second = copy.deepcopy(first)
        second["candidate_id"] = "same-process-b"
        second["process"]["run_uuid"] = str(uuid.uuid4())
        self._append(output, first, 0)
        self._append(output, second, 1)
        proof, challenge = self.release_proof(output, COMMIT)
        with self.assertRaisesRegex(common.ValidationError, "enough all-gate"):
            selector.select_best(
                self.contract,
                output / "results.jsonl",
                output / "BEST.json",
                release_proof=proof,
                expected_challenge=challenge,
            )
        self.assertFalse((output / "BEST.json").exists())

    def test_no_eligible_candidate_does_not_overwrite_existing_best(self) -> None:
        output = self.root / "journal"
        self._append(output, self.result("only-signal", verdict="signal"), 0)
        best_path = output / "BEST.json"
        best_path.write_text('{"sentinel":true}\n', encoding="utf-8")
        proof, challenge = self.release_proof(output, COMMIT)
        with self.assertRaisesRegex(common.ValidationError, "enough all-gate"):
            selector.select_best(
                self.contract,
                output / "results.jsonl",
                best_path,
                release_proof=proof,
                expected_challenge=challenge,
            )
        self.assertEqual(json.loads(best_path.read_text()), {"sentinel": True})

    def test_handwritten_journal_cannot_promote_without_external_proof(self) -> None:
        output = self.root / "journal"
        self._append(output, self.result("fake-a"), 0)
        self._append(output, self.result("fake-b"), 1)
        with self.assertRaisesRegex(common.ValidationError, "external-authority"):
            selector.select_best(
                self.contract, output / "results.jsonl", output / "BEST.json"
            )
        self.assertFalse((output / "BEST.json").exists())

    def test_unavailable_attestation_mode_blocks_selection_immediately(self) -> None:
        contract = common.load_contract(self.contract)
        contract["attestation"]["mode"] = "unavailable"
        with mock.patch.object(selector, "load_contract", return_value=contract), mock.patch.object(
            selector, "contract_sha256", return_value="f" * 64
        ):
            with self.assertRaisesRegex(common.ValidationError, "mode is unavailable"):
                selector.select_best(
                    self.contract,
                    self.root / "missing-results.jsonl",
                    self.root / "BEST.json",
                )

    def test_release_proof_rejects_stale_challenge_and_journal_rewrite(self) -> None:
        output = self.root / "journal"
        self._append(output, self.result("signed-a"), 0)
        self._append(output, self.result("signed-b"), 1)
        proof, challenge = self.release_proof(output, COMMIT)
        with self.assertRaisesRegex(common.ValidationError, "challenge"):
            selector.select_best(
                self.contract,
                output / "results.jsonl",
                output / "BEST.json",
                release_proof=proof,
                expected_challenge=challenge + "-stale",
            )
        lines = (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
        (output / "results.jsonl").write_text(lines[-1] + "\n", encoding="utf-8")
        with self.assertRaises(common.ValidationError):
            selector.select_best(
                self.contract,
                output / "results.jsonl",
                output / "BEST.json",
                release_proof=proof,
                expected_challenge=challenge,
            )
        self.assertFalse((output / "BEST.json").exists())

    def test_release_proof_rejects_payload_tampering(self) -> None:
        output = self.root / "journal"
        self._append(output, self.result("signed-a"), 0)
        self._append(output, self.result("signed-b"), 1)
        proof, challenge = self.release_proof(output, COMMIT)
        envelope = json.loads(proof.read_text(encoding="utf-8"))
        payload = json.loads(base64.b64decode(envelope["payload"]))
        payload["source"]["source_root_sha256"] = "f" * 64
        envelope["payload"] = base64.b64encode(
            common.canonical_json(payload).encode("utf-8")
        ).decode("ascii")
        proof.write_text(common.canonical_json(envelope) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(common.ValidationError, "signature"):
            selector.select_best(
                self.contract,
                output / "results.jsonl",
                output / "BEST.json",
                release_proof=proof,
                expected_challenge=challenge,
            )


class HardwareProfileSummaryTests(CampaignFixture):
    def test_addresses_paths_and_unknown_profile_fields_are_redacted(self) -> None:
        trace = self.root / "trace.json"
        secret = "do-not-leak-source-name"
        trace.write_text(
            json.dumps(
                {
                    "displayTimeUnit": "ns",
                    "secret": secret,
                    "traceEvents": [
                        {
                            "ph": "X",
                            "name": "gemm 0xdeadbeef /private/kernel.cpp",
                            "cat": "/private/category",
                            "dur": 4.0,
                            "ts": 999,
                            "pid": 7,
                            "tid": 8,
                            "args": {
                                "bytes": 400,
                                "address": "0xcafebabe",
                                "source_path": f"/private/{secret}",
                            },
                        },
                        {"ph": "M", "name": secret, "args": {"addr": "0x1234"}},
                    ],
                }
            ),
            encoding="utf-8",
        )
        summary = summarizer.summarize_trace(trace)
        encoded = json.dumps(summary, sort_keys=True)
        for forbidden in (
            "deadbeef",
            "cafebabe",
            "/private",
            secret,
            '"address"',
            '"source_path"',
            '"pid"',
            '"tid"',
            '"ts"',
        ):
            self.assertNotIn(forbidden, encoded)
        event = summary["events"][0]
        self.assertEqual(event["duration_us"]["total"], 4.0)
        self.assertEqual(event["bytes"]["total"], 400.0)
        self.assertEqual(event["bandwidth_bytes_per_s"]["mean"], 100_000_000.0)
        self.assertNotIn("redacted_field_count", summary)

    def test_redacted_labels_do_not_leak_equality_or_group_count(self) -> None:
        summaries = []
        for index, names in enumerate((("secret-a", "secret-a"), ("secret-a", "secret-b"))):
            trace = self.root / f"secret-groups-{index}.json"
            trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "X", "name": name, "cat": f"cat-{name}", "dur": 1.0}
                            for name in names
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = summarizer.summarize_trace(trace)
            summaries.append(summary["events"])
        self.assertEqual(len(summaries[0]), 1)
        self.assertEqual(len(summaries[1]), 1)
        self.assertEqual(summaries[0], summaries[1])

    def test_nonfinite_trace_duration_fails_closed(self) -> None:
        trace = self.root / "trace.json"
        trace.write_text(
            '{"traceEvents":[{"ph":"X","name":"gemm","dur":NaN}]}',
            encoding="utf-8",
        )
        with self.assertRaises(common.ValidationError):
            summarizer.summarize_trace(trace)


if __name__ == "__main__":
    unittest.main()
