"""Board-free selection tests for the optional source operator SDK."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compile_and_run(tmp_path, source, sdk=False):
    compiler = shutil.which("c++")
    if not compiler: pytest.skip("C++ compiler unavailable")
    args = [compiler, "-std=c++17", "-I", str(ROOT / "src/core")]
    if sdk:
        include = Path(os.environ.get("RPU_OPS_INCLUDE_DIR", ROOT.parent / "rpu_ops/include"))
        if not (include / "rpu_ops/kernel_abi.hpp").is_file():
            pytest.skip("rpu_ops headers unavailable; set RPU_OPS_INCLUDE_DIR")
        args += ["-DRPU_HAS_SOURCE_OPS=1", "-I", str(include)]
    cpp = tmp_path / "test.cpp"; cpp.write_text(source)
    exe = tmp_path / "test"
    subprocess.run([*args, str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    env = dict(os.environ); env.pop("RPU_SOURCE_OPS", None)
    subprocess.run([str(exe)], check=True, env=env, capture_output=True, text=True)


def test_source_configuration_and_process_freeze(tmp_path):
    compile_and_run(tmp_path, r'''
#include "rpu_source_ops.h"
#include <cassert>
int main() {
  using namespace rpu_source_ops;
  assert(!parse(nullptr).gelu && !parse("").layernorm);
  assert(parse("gelu").gelu && !parse("gelu").layernorm);
  assert(parse("layernorm,gelu").gelu && parse("layernorm,gelu").layernorm);
  assert(parse("rmsnorm").rmsnorm && !parse("rmsnorm").gelu);
  assert(!parse(nullptr).rmsnorm);
  for (auto value : {"all", "rope", "GELU", "gelu,", ",gelu", "gelu,unknown"}) {
    bool rejected = false;
    try { parse(value); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  setenv("RPU_SOURCE_OPS", "gelu", 1);
  assert(configuration().gelu && !configuration().layernorm);
  setenv("RPU_SOURCE_OPS", "layernorm", 1);
  assert(configuration().gelu && !configuration().layernorm);
}
''')


def test_source_plan_and_reference_fallback(tmp_path):
    compile_and_run(tmp_path, r'''
#include "rpu_source_ops.h"
#include <cassert>
int main() {
  using namespace rpu_source_ops;
  auto g = gelu_plan(true, 0x40000000, 0x40100000, 71680, true, 1);
  assert(g && g->regs[1] == 0x10 && g->regs[3] == 0 && g->grid[0] == 18);
  assert(gelu_plan(true, 0x40000000, 0x40000000, 71680, false, 1));
  assert(!gelu_plan(false, 0, 0x100000, 71680, true, 1));
  assert(!gelu_plan(true, 0, 0x100000, 71680, true, 8));
  assert(!gelu_plan(true, 0, 0x100000, 17, true, 1));
  assert(!gelu_plan(true, 0, 32, 71680, true, 1));
  assert(!gelu_plan(true, 0x41000000, 0, 71680, true, 1));
  auto ln = layernorm_plan(true, 0x40000000, 0x40100000, 0x40200000,
                           0x40201000, 392, 1024, 1e-5, false, 1);
  assert(ln && ln->regs[14] == 0x00a8 && ln->grid[0] == 13);
  assert(!layernorm_plan(true, 0, 0x100000, 0x200000, 0x201000, 392, 1024, 1e-5, true, 1));
  assert(!layernorm_plan(true, 0, 0x100000, 0x200000, 0x201000, 392, 1024, 1e-5, false, 8));
  assert(!layernorm_plan(true, 0, 0, 0x200000, 0x201000, 392, 1024, 1e-5, false, 1));
  assert(!layernorm_plan(true, 0, 0x100000, 0x200000, 0x201000, 392, 1024, 0, false, 1));
}
''', sdk=True)


def test_rmsnorm_source_plan(tmp_path):
    include = Path(os.environ.get("RPU_OPS_INCLUDE_DIR", ROOT.parent / "rpu_ops/include"))
    abi = include / "rpu_ops/kernel_abi.hpp"
    if not abi.is_file() or "RPU_OPS_HAS_RMSNORM" not in abi.read_text():
        pytest.skip("RMSNorm-capable source SDK not selected")
    compile_and_run(tmp_path, r'''
#include "rpu_source_ops.h"
#include <cassert>
int main() {
  using namespace rpu_source_ops;
  auto p = rmsnorm_plan(true, 0x40000000, 0x40200000, 0x40400000, 320, 2048, 1e-6);
  assert(p && p->register_count == 11 && p->grid[0] == 20);
  assert(p->regs[1] == 0x20 && p->regs[3] == 0 && p->regs[5] == 0x40);
  assert(p->regs[10] == 0x0011 && p->cores == 8);
  auto a = rmsnorm_plan(true, 0, 0, 0x200000, 32, 1024, 1e-6 / 16);
  assert(a && a->regs[10] == 0x0001);
  assert(!rmsnorm_plan(false, 0, 0, 0x200000, 32, 1024, 1e-6));
  assert(!rmsnorm_plan(true, 0, 32, 0x200000, 32, 1024, 1e-6));
  assert(!rmsnorm_plan(true, 0, 0, 32, 32, 1024, 1e-6));
  assert(!rmsnorm_plan(true, 0x41000000, 0, 0x200000, 32, 1024, 1e-6));
  assert(!rmsnorm_plan(true, 0, 0, 0x200000, 31, 1024, 1e-6));
  assert(!rmsnorm_plan(true, 0, 0, 0x200000, 784, 1024, 1e-6));
  assert(!rmsnorm_plan(true, 0, 0, 0x200000, 32, 512, 1e-6));
  assert(!rmsnorm_plan(true, 0, 0, 0x200000, 32, 1024, 0));
  assert(!rmsnorm_plan(true, 0, 0, 0x200000, 32, 1024, 1e-20));
}
''', sdk=True)
