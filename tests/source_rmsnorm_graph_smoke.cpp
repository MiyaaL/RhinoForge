// Test-only host hook. No device-program implementation belongs in this repo.
#include "rpu_ops.h"
#include "rpu_source_ops.h"
#include <rpu_ops/fp16.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

extern "C" __attribute__((visibility("default"))) int rpu_test_source_rmsnorm_graph() {
  try {
    using namespace rhino_lkn;
    constexpr unsigned rows = 320, cols = 2048, n = rows * cols;
    constexpr unsigned input = 256, weight = input + n * 2 + 256;
    constexpr unsigned output = weight + cols * 2 + 256;
    constexpr unsigned end = output + n * 2 + 256;
    std::vector<std::unique_ptr<GlobalSPM_t>> banks;
    for (int core = 0; core < 8; ++core)
      banks.emplace_back(std::make_unique<GlobalSPM_t>(8 * 1024 * 1024,
          read_write, kStride32B, 2, core));
    std::array<std::vector<uint16_t>, 8> x, w, retained;
    for (int core = 0; core < 8; ++core) {
      x[core].resize(n); w[core].resize(cols);
      for (unsigned i = 0; i < cols; ++i)
        w[core][i] = rpu_ops::fp32_to_fp16(.8F + ((i + core) % 17) * .01F);
    }
    KernelCache::instance().initialize();
    RpuGraphCache cache(2, true);
    GraphSignature signature;
    signature.op_id = 0x52505502; signature.shapes = {rows, cols};
    signature.dtypes = {1}; signature.op_id_str = "source_rmsnorm_graph_test";
    for (unsigned iteration = 0; iteration < 21; ++iteration) {
      const unsigned perturbation = iteration == 1 ? 47 : 0;
      for (int core = 0; core < 8; ++core) {
        for (unsigned i = 0; i < n; ++i)
          x[core][i] = rpu_ops::fp32_to_fp16(
              (int((i * 13 + core * 19 + perturbation) % 257) - 128) / 64.F);
        auto* p = static_cast<unsigned char*>(banks[core]->get_cpu_ptr());
        std::memset(p, 0xa5, end);
        std::memcpy(p + input, x[core].data(), n * 2);
        std::memcpy(p + weight, w[core].data(), cols * 2);
        std::memset(p + output, 0x7e, n * 2);
      }
      asm volatile("dsb sy" ::: "memory");
      auto& graph = cache.get_or_create(signature);
      graph.begin(signature);
      const bool replay = graph.state() == RpuKernelGraph::State::REPLAYING;
      TORCH_CHECK(replay == (iteration > 0), "unexpected RMSNorm Graph recapture");
      const auto base = static_cast<uint32_t>(banks[0]->get_rpu_addr());
      rpu_launch_rmsnorm_spm_kernel(base + input, base + output, base + weight,
                                   rows, cols, 1e-6);
      graph.end(); cache.touch_entry(signature, replay);
      asm volatile("dsb sy" ::: "memory");
      for (int core = 0; core < 8; ++core) {
        // SPM is mapped device memory. libc memcmp may issue unaligned loads
        // there on AArch64; take aligned scalar reads and compare host memory.
        const auto* mapped = static_cast<const volatile uint32_t*>(banks[core]->get_cpu_ptr());
        std::vector<uint32_t> snapshot_words(end / sizeof(uint32_t));
        for (unsigned i = 0; i < snapshot_words.size(); ++i) snapshot_words[i] = mapped[i];
        const auto* p = reinterpret_cast<const unsigned char*>(snapshot_words.data());
        TORCH_CHECK(!std::memcmp(p + input, x[core].data(), n * 2) &&
                    !std::memcmp(p + weight, w[core].data(), cols * 2),
                    "RMSNorm Graph mutated input/weight");
        for (unsigned i = 0; i < end; ++i) {
          const bool data = (i >= input && i < input + n * 2) ||
                            (i >= weight && i < weight + cols * 2) ||
                            (i >= output && i < output + n * 2);
          TORCH_CHECK(data || p[i] == 0xa5, "RMSNorm Graph guard modified");
        }
        std::vector<uint16_t> result(n);
        std::memcpy(result.data(), p + output, n * 2);
        if (iteration == 0) retained[core] = result;
        if (iteration == 1)
          TORCH_CHECK(result != retained[core], "RMSNorm Graph ignored new values");
        if (iteration >= 2)
          TORCH_CHECK(result == retained[core], "RMSNorm Graph repeated input changed bits");
        double err2 = 0, ref2 = 0;
        for (unsigned row = 0; row < rows; ++row) {
          double ss = 0;
          for (unsigned col = 0; col < cols; ++col) {
            const float value = rpu_ops::fp16_to_fp32(x[core][row * cols + col]);
            ss += double(value) * value;
          }
          const double inv = 1 / std::sqrt(ss / cols + rpu_ops::fp16_to_fp32(0x0011));
          for (unsigned col = 0; col < cols; ++col) {
            const auto i = row * cols + col;
            const double expected = rpu_ops::fp16_to_fp32(x[core][i]) * inv *
                                    rpu_ops::fp16_to_fp32(w[core][col]);
            const double actual = rpu_ops::fp16_to_fp32(result[i]);
            TORCH_CHECK(std::isfinite(actual) &&
                        std::abs(actual - expected) <= .003 + .004 * std::abs(expected),
                        "RMSNorm Graph anchor mismatch");
            err2 += (actual - expected) * (actual - expected);
            ref2 += expected * expected;
          }
        }
        TORCH_CHECK(std::sqrt(err2 / ref2) <= .003, "RMSNorm Graph anchor relative L2");
      }
      const auto snapshot = cache.snapshot();
      TORCH_CHECK(cache.size() == 1 && cache.cache_invariant_ok() && snapshot.size() == 1 &&
                  snapshot[0].recapture_count == 0 && snapshot[0].replay_count == iteration &&
                  snapshot[0].kernel_count == 1 && snapshot[0].segment_count == 1,
                  "RMSNorm Graph lifecycle invariant failed");
      TORCH_CHECK(graph.dump_replay_plan().find(rpu_source_ops::configuration().rmsnorm
                    ? "rmsnorm_f16" : "llama_rms_norm") != std::string::npos,
                  "RMSNorm Graph selected a different implementation");
    }
    cache.clear();
    std::cout << "PASS: eight-core RMSNorm, one BUILD + 20 REPLAY, A/B/A bits, anchor, guards\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
