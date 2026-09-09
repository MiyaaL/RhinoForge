"""Regression checks for GDN fill extent and the cumsum workspace ABI."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("chunk", [64, 128, 192, 256, 320, 384])
def test_fixed_fill_extent_preserves_valid_rows_and_stays_in_guard(chunk):
    from rpu_backend.api.qwen3_5_cache import QWEN3_5_TOTAL_PADDING_CAP
    guard = min(chunk - 1, QWEN3_5_TOTAL_PADDING_CAP)
    for valid in range(max(1, chunk - guard), chunk + 1):
        # All seven possible fill destinations own this contiguous extent.
        storage = [1] * (chunk + guard) + [7] * 16
        storage[valid:valid + guard] = [0] * guard
        assert storage[:valid] == [1] * valid
        assert storage[valid:chunk] == [0] * (chunk - valid)
        assert storage[chunk + guard:] == [7] * 16


def test_native_gdn_declares_guarded_targets_and_full_workspace():
    cpp = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    assert "kGdnPrefillPaddingCap = 127" in cpp
    assert "CS + std::min(CS - 1, kGdnPrefillPaddingCap)" in cpp
    import re
    for name in ("query", "key", "value", "q_rep", "k_rep", "beta", "g"):
        assert re.search(r'BP\("gdn_c_' + name + r'",\s+guarded_cs \*', cpp)
    assert 'BP("gdn_c_ws",     64 * 17 * 16, 4, 4)' in cpp
    assert "TORCH_CHECK(L - Lv <= np" in cpp
    fill = cpp.split("const int64_t np = std::min(L - 1, kGdnPrefillPaddingCap);", 1)[1].split(
        "// ===== Phase 4", 1)[0]
    assert fill.count("rpu_launch_fill_spm_kernel(") == 5
    assert 'D("gdn_c_decay")' not in fill
    assert ", nc, 0," not in fill


def test_production_fill_rejects_oversized_grid_before_enqueue(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native fill contract test")
    source = (ROOT / "src/ops/rpu_memcpy.cpp").read_text()
    function = "void rpu_launch_fill_spm_kernel(" + source.split(
        "void rpu_launch_fill_spm_kernel(", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    harness = r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("guard"); } while (0)
namespace c10 { struct Half { uint16_t x = 0; }; }
struct SpmAllocator { static constexpr int NUM_CORES=8; static constexpr size_t SPM_USABLE=8*1024*1024; };
struct Alloc { bool is_initialized() { return true; } uint32_t addr(int,int) { return 0; } } SPM_ALLOC;
struct Kernel_t { void set_regs(int,uint16_t) {} } kernel;
enum class GraphKernelNoDdrProof { FillSpm };
struct RpuKernelGraph {
 static RpuKernelGraph& active() { static RpuKernelGraph g; return g; }
 void stage_kernel_no_ddr(const char*,GraphKernelNoDdrProof) {}
 Kernel_t* get_kernel_reset(const char*) { return &kernel; }
};
int enqueues=0, last_grid=0;
struct Queue {
 void set_broadcast_mode(bool) {}
 void enqueu_kernel(Kernel_t&,std::initializer_list<uint16_t> grid,const std::vector<uint8_t>&) {
   ++enqueues; last_grid=*grid.begin();
 }
} queue;
Queue* GET_QUEUE(int) { return &queue; }
'''
    main = r'''
int main() {
 bool rejected=false;
 try { rpu_launch_fill_spm_kernel(0,3072,{},8,0,81920); }
 catch (const std::runtime_error&) { rejected=true; }
 assert(rejected && enqueues==0);
 rpu_launch_fill_spm_kernel(0,3072,{},8,0,0);
 assert(enqueues==1 && last_grid==2);
 rpu_launch_fill_spm_kernel(0,127*256,{},8,0,0);
 assert(enqueues==2 && last_grid==16);
 rpu_launch_fill_spm_kernel(0,2048,{},8,0,2048);
 assert(enqueues==3 && last_grid==1);
}
'''
    cpp = tmp_path / "fill-contract.cpp"
    cpp.write_text(harness + function + main)
    executable = tmp_path / "fill-contract"
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
