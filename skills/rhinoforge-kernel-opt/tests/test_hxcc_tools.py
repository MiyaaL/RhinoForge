from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = SKILL_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rpu_hxcc_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hxcc = load_script("hxcc_preflight")
asm = load_script("inspect_rpu_asm")
hwperf = load_script("normalize_hwperf")


class HxccIdentityTests(unittest.TestCase):
    def test_preflight_rejects_nonfinite_and_unrepresentable_timeouts(self) -> None:
        # Validate before executable discovery so malformed values cannot reach
        # subprocess.run (which otherwise raises inconsistent ValueError or
        # OverflowError variants).
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1, 10**400):
            with self.subTest(timeout=repr(timeout)):
                with self.assertRaises(hxcc.ValidationError):
                    hxcc.run_preflight(
                        "/definitely/missing/hxcc",
                        identity_only=True,
                        timeout=timeout,
                    )

    def test_preflight_cli_reports_nonfinite_timeout(self) -> None:
        status = hxcc.main(
            [
                "--compiler",
                "/definitely/missing/hxcc",
                "--identity-only",
                "--timeout",
                "nan",
            ]
        )
        self.assertEqual(status, 2)

    def test_driver_requires_target_o2_assembler_and_r1(self) -> None:
        dry = (
            '"/pkg/clang-17" "-cc1" "-triple" "rpu-rhino-rpuhsa" "-O2"\n'
            '"/tmp/rpuas" "foo.s"\n'
            '"/tmp/rhino_gen_oplib" "-f" "foo.s" "-m" "r1"\n'
        )
        report = hxcc._parse_driver(dry, hxcc.DEFAULT_TARGET)
        self.assertTrue(report["chain_ok"])
        self.assertEqual(report["target_triple"], hxcc.DEFAULT_TARGET)

        bad = hxcc._parse_driver(dry.replace("-O2", "-O0"), hxcc.DEFAULT_TARGET)
        self.assertFalse(bad["chain_ok"])
        self.assertTrue(bad["o0_present"])

    def test_architecture_and_resource_resolution(self) -> None:
        self.assertEqual(hxcc._normalize_arch("arm64"), "aarch64")
        self.assertEqual(hxcc._normalize_arch("AMD64"), "x86_64")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource = root / "data" / "lib" / "clang" / "17"
            resource.mkdir(parents=True)
            for name in ("clang-17", "rpuas", "rhino_gen_oplib"):
                tool = root / "data" / "bin" / name
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.write_bytes(b"#!/bin/sh\n")
                tool.chmod(0o700)
            found = hxcc._discover_package_tools(resource)
            self.assertEqual(found["clang"], root / "data" / "bin" / "clang-17")

    def test_safe_source_rejects_symlink_components_and_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "private"
            destination.mkdir()
            real = root / "real"
            real.mkdir()
            source = real / "kernel.rc"
            payload = "__rprog__ void k() {}\n"
            source.write_text(payload, encoding="utf-8")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            for spelling in (
                alias / source.name,
                alias / ".." / "real" / source.name,
            ):
                with self.subTest(spelling=str(spelling)):
                    with self.assertRaises(hxcc.ValidationError):
                        hxcc._safe_source(spelling, destination)
            final_alias = root / "kernel-link.rc"
            final_alias.symlink_to(source)
            with self.assertRaises(hxcc.ValidationError):
                hxcc._safe_source(final_alias, destination)

            raced_payload = "__rprog__ void q() {}\n"
            original_read = hxcc.os.read
            raced = False

            def racing_read(fd: int, size: int) -> bytes:
                nonlocal raced
                block = original_read(fd, size)
                if block and not raced:
                    raced = True
                    before_ns = source.stat().st_mtime_ns
                    source.write_text(raced_payload, encoding="utf-8")
                    os.utime(source, ns=(before_ns + 1_000_000, before_ns + 1_000_000))
                return block

            hxcc.os.read = racing_read
            try:
                with self.assertRaises(hxcc.ValidationError):
                    hxcc._safe_source(source, destination)
            finally:
                hxcc.os.read = original_read
            self.assertFalse((destination / source.name).exists())


class SourceAndAssemblyTests(unittest.TestCase):
    def test_source_lint_catches_manual_forbidden_features(self) -> None:
        source = """
        #pragma rpu_hwloop_xx
        int helper() { return 1; }
        __rprog__ int bad() { new int(1); return bad(); }
        """
        report = asm.lint_source(source)
        kinds = {item["kind"] for item in report["violations"]}
        self.assertIn("dynamic_allocation", kinds)
        self.assertIn("nonvoid_rprog_entry", kinds)
        self.assertIn("recursive_entry_call", kinds)
        self.assertIn("hwloop_pragma_not_immediately_before_for", kinds)

    def test_assembly_counts_and_strict_wjump_gate(self) -> None:
        text = """
        loop 0, 4, 1, 31
        vld_e.ra_dir.ra_lpaddr.a_s32 0, wr0
        vmat_e.ra_dir.ra_lpaddr.outon 0, wr1
        fence.vld 0
        vst_r.ra_dir.a_s32 0, wr0
        """
        report = asm.inspect_assembly(
            text, strict=True, require_vmat=True, require_lpaddr=True
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts"]["hardware_loop"], 1)
        self.assertEqual(report["counts"]["repeat"], 1)
        failed = asm.inspect_assembly(text + "\nwjump -1\n", strict=True)
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["counts"]["software_wjump"], 1)

    def test_async_requires_fence_when_source_is_supplied(self) -> None:
        source = "__rprog__ void k() { vldExecute_async(); }"
        report = asm.inspect_assembly(
            "exit\n", source_text=source, require_async_fence=True
        )
        self.assertFalse(report["ok"])
        self.assertEqual(
            report["violations"][0]["kind"],
            "async_unit_without_matching_fence",
        )

    def test_async_fence_is_matched_per_unit_and_requires_execute(self) -> None:
        source = "__rprog__ void k() { vldExecute_async(); vmatExecute_async(); }"
        missing = asm.inspect_assembly(
            "vld_r 0\nfence.vld 0\n", source_text=source, require_async_fence=True
        )
        self.assertFalse(missing["ok"])
        self.assertTrue(any(item["unit"] == "vmat" for item in missing["violations"]))
        setup_only = asm.inspect_assembly(
            "vld_m 0\nfence.vld 0\n",
            source_text="__rprog__ void k() { vldExecute_async(); }",
            require_async_fence=True,
        )
        self.assertFalse(setup_only["ok"])
        good = asm.inspect_assembly(
            "vld_r 0\nfence.vld 0\nvmat_r 0\nfence.vmat 0\n",
            source_text=source,
            require_async_fence=True,
        )
        self.assertTrue(good["ok"])

    def test_empty_assembly_is_not_evidence(self) -> None:
        self.assertFalse(asm.inspect_assembly("")["ok"])

    def test_assembly_numeric_gate_arguments_are_strict(self) -> None:
        with self.assertRaises(asm.ValidationError):
            asm.inspect_assembly("loop 0", max_loop_depth=True)
        with self.assertRaises(asm.ValidationError):
            asm.inspect_assembly("wjump 0", max_wjump=float("nan"))

    def test_atomic_inspection_output_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "kernel.s"
            source.write_text("exit\n", encoding="utf-8")
            output = root / "receipt.json"
            status = asm.main([str(source), "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["ok"])
            # A final symlink must never be followed/overwritten.
            target = root / "sentinel"
            target.write_text("keep\n", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(asm.main([str(source), "--output", str(link)]), 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")

    def test_assembly_input_is_snapshot_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            source = real / "kernel.s"
            payload = "vmat_r 0\n"
            source.write_text(payload, encoding="utf-8")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(asm.ValidationError):
                asm._read_regular(alias / source.name, "assembly")
            final_alias = root / "kernel-link.s"
            final_alias.symlink_to(source)
            with self.assertRaises(asm.ValidationError):
                asm._read_regular(final_alias, "assembly")

            original_read = asm.os.read
            raced = False

            def racing_read(fd: int, size: int) -> bytes:
                nonlocal raced
                block = original_read(fd, size)
                if block and not raced:
                    raced = True
                    before_ns = source.stat().st_mtime_ns
                    source.write_text("vmat_x 0\n", encoding="utf-8")
                    os.utime(source, ns=(before_ns + 1_000_000, before_ns + 1_000_000))
                return block

            asm.os.read = racing_read
            try:
                with self.assertRaises(asm.ValidationError):
                    asm._read_regular(source, "assembly")
            finally:
                asm.os.read = original_read


class HwPerfNormalizationTests(unittest.TestCase):
    def test_other_data_requires_bounded_decimal_integers(self) -> None:
        invalid_values = (True, False, 1.0, -1, "-1", "1.0", "", "9" * 21)
        for key in ("frequency_hz", "base_cycle", "batch_events", "records_captured"):
            for value in invalid_values:
                with self.subTest(key=key, value=repr(value)):
                    with self.assertRaises(hwperf.ValidationError):
                        hwperf.normalize_trace(
                            {
                                "otherData": {key: value},
                                "traceEvents": [],
                            },
                            producer_sha256="a" * 64,
                        )
        with self.assertRaisesRegex(hwperf.ValidationError, "frequency_hz"):
            hwperf.normalize_trace(
                {
                    "otherData": {"frequency_hz": "0"},
                    "traceEvents": [],
                },
                producer_sha256="a" * 64,
            )

    def test_balances_native_be_and_keeps_only_public_args(self) -> None:
        trace = {
            "traceEvents": [
                {
                    "name": "kernel",
                    "cat": "Compute",
                    "ph": "B",
                    "ts": 10,
                    "pid": 1,
                    "tid": 2,
                    "id": "k",
                    "args": {"bytes": 64, "secret": "/dev/mem"},
                },
                {"ph": "E", "ts": 13, "pid": 1, "tid": 2, "id": "k"},
            ]
        }
        normalized, report = hwperf.normalize_trace(
            trace,
            producer_sha256="a" * 64,
            assert_exhaustive=True,
        )
        self.assertTrue(report["exhaustive_asserted"])
        self.assertEqual(normalized["traceEvents"][0]["ph"], "X")
        self.assertEqual(normalized["traceEvents"][0]["dur"], 3.0)
        self.assertEqual(normalized["traceEvents"][0]["cat"], "rpu_device_program")
        self.assertEqual(normalized["traceEvents"][0]["args"]["bytes"], 64)
        self.assertIn("bandwidth_bytes_per_s", normalized["traceEvents"][0]["args"])
        self.assertEqual(report["native_category_counts"], {"Compute": 1})

    def test_exhaustiveness_is_opt_in_and_unpaired_end_fails(self) -> None:
        normalized, report = hwperf.normalize_trace(
            [
                {"name": "k", "cat": "Compute", "ph": "B", "ts": 0},
                {"name": "k", "ph": "E", "ts": 1},
            ],
            producer_sha256="b" * 64,
        )
        self.assertFalse(report["exhaustive_asserted"])
        self.assertFalse(
            normalized["rhinoforgeTrace"]["device_program_events_exhaustive"]
        )
        with self.assertRaises(hwperf.ValidationError):
            hwperf.normalize_trace(
                [{"name": "k", "ph": "E", "ts": 1}],
                producer_sha256="b" * 64,
            )

    def test_official_compute_dma_sync_mapping_and_other_data(self) -> None:
        trace = {
            "otherData": {
                "source": "rhino-launch-kernel HW perf trace",
                "frequency_hz": "800000000",
                "base_cycle": "12",
                "batch_events": "3",
                "records_captured": "6",
            },
            "traceEvents": [
                {
                    "name": "gemm_fp16",
                    "cat": "Compute",
                    "ph": "B",
                    "ts": 0.0,
                    "pid": 0,
                    "tid": "stream_0",
                    "args": {"op_type": "matmul"},
                },
                {
                    "name": "gemm_fp16",
                    "cat": "Compute",
                    "ph": "E",
                    "ts": 2.0,
                    "pid": 0,
                    "tid": "stream_0",
                    "args": {"duration_us": 2.0, "bandwidth_GBps": 1.0},
                },
                {
                    "name": "dma_1_ch0",
                    "cat": "DMA",
                    "ph": "B",
                    "ts": 2.1,
                    "pid": 0,
                    "tid": "stream_0",
                    "args": {"size_bytes": 128},
                },
                {
                    "name": "dma_1_ch0",
                    "cat": "DMA",
                    "ph": "E",
                    "ts": 3.1,
                    "pid": 0,
                    "tid": "stream_0",
                },
                {
                    "name": "barrier_0",
                    "cat": "SYNC",
                    "ph": "s",
                    "pid": 0,
                    "tid": "stream_0",
                    "id": "0",
                    "ts": 2.0,
                },
                {
                    "name": "barrier_0",
                    "cat": "SYNC",
                    "ph": "f",
                    "pid": 0,
                    "tid": "stream_2",
                    "id": "0",
                    "ts": 2.1,
                },
            ],
        }
        normalized, report = hwperf.normalize_trace(
            trace,
            producer_sha256="c" * 64,
            assert_exhaustive=True,
            require_frequency=True,
        )
        self.assertEqual(report["other_data"]["frequency_hz"], 800000000)
        self.assertTrue(report["barrier_flow_complete"])
        self.assertEqual(report["native_category_counts"], {"Compute": 1, "DMA": 1})
        categories = [event.get("cat") for event in normalized["traceEvents"] if event.get("ph") == "X"]
        self.assertEqual(categories, ["rpu_device_program", "rpu_dma"])

    def test_malformed_trace_and_unknown_phase_fail_cleanly(self) -> None:
        with self.assertRaises(hwperf.ValidationError):
            hwperf.normalize_trace(
                [{"ph": ["B"], "name": "bad"}], producer_sha256="d" * 64
            )
        with self.assertRaises(hwperf.ValidationError):
            hwperf.normalize_trace(
                [{"ph": "Z", "name": "hidden"}], producer_sha256="d" * 64
            )

    def test_metadata_and_begin_end_timing_are_checked(self) -> None:
        trace = {
            "otherData": {
                "frequency_hz": "800000000",
                "batch_events": "1",
                "records_captured": "2",
            },
            "traceEvents": [
                {
                    "name": "k",
                    "cat": "Compute",
                    "ph": "B",
                    "ts": 10.0,
                    "tid": 1,
                    "args": {
                        "duration_us": 2.0,
                        "duration_cycles": 1600,
                    },
                },
                {
                    "name": "k",
                    "cat": "Compute",
                    "ph": "E",
                    "ts": 12.0,
                    "tid": 1,
                    "args": {
                        "duration_us": 2.0,
                        "duration_cycles": 1600,
                    },
                },
            ],
        }
        _, report = hwperf.normalize_trace(
            trace,
            producer_sha256="1" * 64,
            assert_exhaustive=True,
            require_frequency=True,
        )
        self.assertEqual(report["other_data"]["frequency_hz"], 800000000)

        fractional_frequency = json.loads(json.dumps(trace))
        fractional_frequency["otherData"]["frequency_hz"] = 800000000.5
        with self.assertRaises(hwperf.ValidationError):
            hwperf.normalize_trace(
                fractional_frequency,
                producer_sha256="1" * 64,
                require_frequency=True,
            )

        contradictory_begin = json.loads(json.dumps(trace))
        contradictory_begin["traceEvents"][0]["args"]["duration_us"] = 9.0
        with self.assertRaises(hwperf.ValidationError):
            hwperf.normalize_trace(
                contradictory_begin,
                producer_sha256="1" * 64,
            )

        contradictory_bandwidth = json.loads(json.dumps(trace))
        contradictory_bandwidth["traceEvents"][0]["args"]["bandwidth_GBps"] = 1.0
        contradictory_bandwidth["traceEvents"][0]["args"]["bytes"] = 2000
        contradictory_bandwidth["traceEvents"][1]["args"]["bandwidth_GBps"] = 2.0
        contradictory_bandwidth["traceEvents"][1]["args"]["bytes"] = 2000
        with self.assertRaises(hwperf.ValidationError):
            hwperf.normalize_trace(
                contradictory_bandwidth,
                producer_sha256="1" * 64,
            )

    def test_normalizer_rejects_output_symlink_and_input_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "native.json"
            source.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"name": "k", "cat": "Compute", "ph": "B", "ts": 0},
                            {"name": "k", "cat": "Compute", "ph": "E", "ts": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            target = root / "sentinel"
            target.write_text("keep\n", encoding="utf-8")
            link = root / "out.json"
            link.symlink_to(target)
            with self.assertRaises(hwperf.ValidationError):
                hwperf.convert_file(source, link, producer_sha256="e" * 64)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")
            with self.assertRaises(hwperf.ValidationError):
                hwperf.convert_file(source, source, producer_sha256="e" * 64)

    def test_normalizer_rejects_input_symlink_components_before_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            source = real / "native.json"
            source.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"name": "k", "cat": "Compute", "ph": "B", "ts": 0},
                            {"name": "k", "cat": "Compute", "ph": "E", "ts": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            output = root / "normalized.json"
            for spelling in (alias / "native.json", alias / ".." / "real" / "native.json"):
                with self.subTest(spelling=str(spelling)):
                    with self.assertRaises(hwperf.ValidationError):
                        hwperf.convert_file(spelling, output, producer_sha256="e" * 64)
            final_alias = root / "native-link.json"
            final_alias.symlink_to(source)
            with self.assertRaises(hwperf.ValidationError):
                hwperf.convert_file(final_alias, output, producer_sha256="e" * 64)

    def test_normalizer_rejects_source_mutation_during_read(self) -> None:
        # Force a same-size rewrite after the first read.  Descriptor identity
        # and size alone would miss this; the fstat mtime/ctime snapshot must
        # reject it before any normalized output is committed.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "native.json"
            payload = json.dumps(
                {
                    "traceEvents": [
                        {"name": "k", "cat": "Compute", "ph": "B", "ts": 0},
                        {"name": "k", "cat": "Compute", "ph": "E", "ts": 1},
                    ]
                }
            )
            raced_payload = payload.replace('"k"', '"q"', 1)
            source.write_text(payload, encoding="utf-8")
            output = root / "normalized.json"
            original_read = hwperf.os.read
            raced = False

            def racing_read(fd: int, size: int) -> bytes:
                nonlocal raced
                block = original_read(fd, size)
                if block and not raced:
                    raced = True
                    before_ns = source.stat().st_mtime_ns
                    source.write_text(raced_payload, encoding="utf-8")
                    # Some temporary filesystems coalesce immediate writes;
                    # make the synthetic race's metadata transition explicit.
                    os.utime(source, ns=(before_ns + 1_000_000, before_ns + 1_000_000))
                return block

            original = hwperf.os.read
            hwperf.os.read = racing_read
            try:
                with self.assertRaises(hwperf.ValidationError):
                    hwperf.convert_file(source, output, producer_sha256="e" * 64)
            finally:
                hwperf.os.read = original
            self.assertFalse(output.exists())

    def test_many_unique_native_events_use_indexed_pairing(self) -> None:
        # Each END has a unique name/id.  This exercises the normal fast path
        # of the inverted index; a full scan of all open stacks regresses to
        # quadratic work and makes a moderately sized trace impractical.
        count = 8_000
        events = [
            {
                "name": f"kernel_{index}",
                "cat": "Compute",
                "ph": "B",
                "ts": float(index),
                "pid": 0,
                "tid": "stream_0",
                "id": index,
            }
            for index in range(count)
        ]
        events.extend(
            {
                "name": f"kernel_{index}",
                "cat": "Compute",
                "ph": "E",
                "ts": float(index) + 1.0,
                "pid": 0,
                "tid": "stream_0",
                "id": index,
            }
            for index in range(count)
        )
        started = time.perf_counter()
        normalized, report = hwperf.normalize_trace(
            events, producer_sha256="f" * 64, assert_exhaustive=True
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(report["complete_event_count"], count)
        self.assertEqual(
            sum(event.get("cat") == "rpu_device_program" for event in normalized["traceEvents"]),
            count,
        )
        # Keep a wide margin for slower CI hosts while catching an accidental
        # O(n^2) implementation (which is several seconds for this fixture).
        self.assertLess(elapsed, 4.0)


if __name__ == "__main__":
    unittest.main()
