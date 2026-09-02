from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    module_name = f"rhinoforge_io_hardening_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    os.sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


import campaign_common as common


summarizer = _load_script("summarize_hwperf")


class CampaignIoHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_read_limited_bytes_rejects_final_and_intermediate_symlinks(self) -> None:
        real = self.root / "real"
        real.mkdir()
        source = real / "trace.json"
        source.write_bytes(b'{"ok":true}')

        final_link = self.root / "final-link.json"
        final_link.symlink_to(source)
        with self.assertRaisesRegex(common.ValidationError, "symbolic|regular"):
            common.read_limited_bytes(final_link, 1024, "trace")

        parent_link = self.root / "parent-link"
        parent_link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(common.ValidationError, "symbolic"):
            common.read_limited_bytes(parent_link / "trace.json", 1024, "trace")

        # Do not normalize away a symlink hidden by ``..`` before checking it.
        with self.assertRaisesRegex(common.ValidationError, "symbolic"):
            common.read_limited_bytes(
                parent_link / ".." / self.root.name / "real" / "trace.json",
                1024,
                "trace",
            )

    def test_read_limited_bytes_detects_same_size_mutation(self) -> None:
        source = self.root / "trace.json"
        source.write_bytes(b"a" * 4096)
        original_read = common.os.read
        mutated = False

        def racing_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            block = original_read(descriptor, count)
            if block and not mutated:
                mutated = True
                before = source.stat()
                source.write_bytes(b"b" * 4096)
                # Ensure filesystems with coarse timestamp resolution expose
                # the mutation to the descriptor identity check.
                os.utime(
                    source,
                    ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
                )
            return block

        with mock.patch.object(common.os, "read", side_effect=racing_read):
            with self.assertRaisesRegex(common.ValidationError, "changed"):
                common.read_limited_bytes(source, 8192, "trace")

    def test_read_limited_bytes_detects_path_replacement_after_open(self) -> None:
        source = self.root / "trace.json"
        replacement = self.root / "replacement.json"
        source.write_bytes(b"old")
        replacement.write_bytes(b"new")
        original_read = common.os.read
        swapped = False

        def racing_read(descriptor: int, count: int) -> bytes:
            nonlocal swapped
            block = original_read(descriptor, count)
            if block and not swapped:
                swapped = True
                source.unlink()
                replacement.rename(source)
            return block

        with mock.patch.object(common.os, "read", side_effect=racing_read):
            with self.assertRaisesRegex(common.ValidationError, "changed"):
                common.read_limited_bytes(source, 1024, "trace")

    def test_safe_destination_creates_only_regular_parent_components(self) -> None:
        output = self.root / "new" / "nested" / "summary.json"
        self.assertEqual(common.safe_destination(output), output)
        self.assertTrue(output.parent.is_dir())

        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(common.ValidationError, "symbolic"):
            common.safe_destination(alias / "summary.json")

    def test_content_tree_rejects_intermediate_symlink_root(self) -> None:
        real = self.root / "reference"
        real.mkdir()
        (real / "oracle.py").write_text("frozen\n", encoding="utf-8")
        alias = self.root / "reference-alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(common.ValidationError, "symbolic"):
            common.content_tree_sha256(alias)

    def _trace(self) -> Path:
        trace = self.root / "trace.json"
        trace.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"ph": "X", "name": "k", "cat": "c", "dur": 1.0}
                    ]
                }
            ),
            encoding="utf-8",
        )
        return trace

    def test_summarizer_atomic_output_rejects_symlink_parent_and_input_collision(self) -> None:
        trace = self._trace()
        outside = self.root / "outside"
        outside.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        output = alias / "summary.json"
        with self.assertRaisesRegex(common.ValidationError, "symbolic"):
            summarizer._atomic_write(output, "{}\n")
        self.assertFalse((outside / "summary.json").exists())

        with self.assertRaisesRegex(common.ValidationError, "differ from input"):
            summarizer._atomic_write(trace, "{}\n", forbidden=(trace,))

    def test_summarizer_atomic_output_is_valid_and_overwrites_regular_file(self) -> None:
        trace = self._trace()
        output = self.root / "nested" / "summary.json"
        summarizer._atomic_write(output, '{"ok":true}\n', forbidden=(trace,))
        self.assertEqual(output.read_text(encoding="utf-8"), '{"ok":true}\n')
        summarizer._atomic_write(output, '{"ok":false}\n', forbidden=(trace,))
        self.assertEqual(output.read_text(encoding="utf-8"), '{"ok":false}\n')
        self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))

    def test_numeric_overflow_is_rejected_as_validation_error(self) -> None:
        with self.assertRaises(common.ValidationError):
            common._number(10**10000, "value")
        with self.assertRaises(common.ValidationError):
            summarizer._number(10**10000, "trace duration")

    def test_summarizer_does_not_silently_accept_native_duration_pairs(self) -> None:
        trace = self.root / "native.json"
        trace.write_text(
            json.dumps(
                {
                    "otherData": {
                        "source": "rhino-launch-kernel HW perf trace",
                        "frequency_hz": "800000000",
                    },
                    "traceEvents": [
                        {"ph": "B", "cat": "Compute", "name": "k", "ts": 0},
                        {"ph": "E", "cat": "Compute", "name": "k", "ts": 1},
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(common.ValidationError, "normalized"):
            summarizer.summarize_trace(trace)


if __name__ == "__main__":
    unittest.main()
