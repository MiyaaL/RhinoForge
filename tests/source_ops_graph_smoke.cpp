// Test-only native hook, built only with RPU_BUILD_SOURCE_OPS_GRAPH_TEST=ON.
// Exercise real source/reference wrappers and the normal retained Graph cache.
#include "rpu_ops.h"
#include "rpu_source_ops.h"
#include "rpu_eltwise.h"
#include <rpu_ops/elementwise_reference.hpp>
#include <rpu_ops/layernorm_reference.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

extern "C" __attribute__((visibility("default"))) int rpu_test_source_ops_graph() {
  try {
    using namespace rhino_lkn;
    constexpr unsigned gn = 140 * 512, rows = 392, cols = 1024, ln = rows * cols;
    LocalSPM_t gx(gn * 2, read_write, kStride32B, 32, 0), gy(gn * 2, read_write, kStride32B, 32, 0);
    LocalSPM_t lx(ln * 2, read_write, kStride32B, 32, 0), ly(ln * 2, read_write, kStride32B, 32, 0);
    LocalSPM_t w(cols * 2, read_write, kStride32B, 32, 0), b(cols * 2, read_write, kStride32B, 32, 0);
    auto address = [](const LocalSPM_t& p) {
      return rpu_ops::abi::global_spm_base + static_cast<std::uint32_t>(p.get_rpu_addr());
    };
    std::vector<std::uint16_t> input_g(gn), input_l(ln), weight(cols), bias(cols);
    std::vector<std::uint16_t> output_g(gn), output_l(ln), first_g, first_l;
    for (unsigned i = 0; i < cols; ++i) {
      weight[i] = rpu_ops::fp32_to_fp16(0.8F + (i % 17) * 0.01F);
      bias[i] = rpu_ops::fp32_to_fp16((int(i % 13) - 6) * 0.01F);
    }
    std::memcpy(w.get_cpu_ptr(), weight.data(), cols * 2);
    std::memcpy(b.get_cpu_ptr(), bias.data(), cols * 2);
    KernelCache::instance().initialize();
    RpuGraphCache cache(2, true);
    GraphSignature signature;
    signature.op_id = 0x52505501; signature.shapes = {140, 512, rows, cols};
    signature.dtypes = {1}; signature.op_id_str = "source_ops_graph_test";
    for (unsigned iteration = 0; iteration < 3; ++iteration) {
      const unsigned offset = iteration == 1 ? 47 : 0;
      for (unsigned i = 0; i < gn; ++i)
        input_g[i] = rpu_ops::fp32_to_fp16((int((i + offset) % 251) - 125) / 32.0F);
      for (unsigned i = 0; i < ln; ++i)
        input_l[i] = rpu_ops::fp32_to_fp16((int((i * 13 + offset) % 257) - 128) / 64.0F);
      std::memcpy(gx.get_cpu_ptr(), input_g.data(), gn * 2);
      std::memcpy(lx.get_cpu_ptr(), input_l.data(), ln * 2);
      asm volatile("dsb sy" ::: "memory");
      auto& graph = cache.get_or_create(signature);
      graph.begin(signature);
      const bool replay = graph.state() == RpuKernelGraph::State::REPLAYING;
      TORCH_CHECK(replay == (iteration > 0), "unexpected Graph recapture");
      rpu_launch_eltwise_unary_spm_kernel(address(gx), address(gy), gn,
                                         ValuOpType::SIGMOID, GeluMode::TANH, 1);
      rpu_launch_layernorm_spm_kernel(address(lx), address(ly), address(w), address(b),
                                      rows, cols, 1e-5, false, 0, 1);
      graph.end(); cache.touch_entry(signature, replay);
      asm volatile("dsb sy" ::: "memory");
      std::memcpy(output_g.data(), gy.get_cpu_ptr(), gn * 2);
      std::memcpy(output_l.data(), ly.get_cpu_ptr(), ln * 2);
      TORCH_CHECK(std::memcmp(input_g.data(), gx.get_cpu_ptr(), gn * 2) == 0 &&
                  std::memcmp(input_l.data(), lx.get_cpu_ptr(), ln * 2) == 0,
                  "Graph input mutated");
      for (unsigned i = 0; i < gn; ++i) {
        const auto expected = rpu_ops::gelu_anchor(rpu_ops::fp16_to_fp32(input_g[i]),
                                                   rpu_ops::GeluApproximation::Tanh);
        const auto actual = rpu_ops::fp16_to_fp32(output_g[i]);
        TORCH_CHECK(std::isfinite(actual) && std::abs(actual - expected) <= 0.005F + 0.003F * std::abs(expected),
                    "Graph GELU parity failure");
      }
      const auto anchor = rpu_ops::layernorm_fp32_anchor(input_l, weight, rows, cols, bias, 1e-5F);
      for (unsigned i = 0; i < ln; ++i) {
        const auto actual = rpu_ops::fp16_to_fp32(output_l[i]);
        TORCH_CHECK(std::isfinite(actual) && std::abs(actual - anchor[i]) <=
                    rpu_ops::kLayerNormAnchorAtol + rpu_ops::kLayerNormAnchorRtol * std::abs(anchor[i]),
                    "Graph LayerNorm parity failure");
      }
      if (iteration == 0) { first_g = output_g; first_l = output_l; }
      if (iteration == 1) TORCH_CHECK(first_g != output_g && first_l != output_l, "stale Graph input");
      if (iteration == 2) TORCH_CHECK(first_g == output_g && first_l == output_l, "A/B/A repeatability failure");
      const auto snapshot = cache.snapshot();
      TORCH_CHECK(cache.size() == 1 && cache.cache_invariant_ok() && snapshot.size() == 1 &&
                  snapshot[0].recapture_count == 0 && snapshot[0].replay_count == iteration &&
                  snapshot[0].kernel_count == 2 && snapshot[0].segment_count == 1,
                  "Graph cache/lifecycle invariant failed");
      const auto plan = graph.dump_replay_plan();
      const auto& selected = rpu_source_ops::configuration();
      TORCH_CHECK(plan.find(selected.gelu ? "gelu_f16" : "unary_gelu_tanh") != std::string::npos &&
                  plan.find(selected.layernorm ? "layernorm_f16" : "layer_norm") != std::string::npos,
                  "Graph did not retain the selected implementation names");
    }
    cache.clear();
    std::cout << "PASS: source/reference Graph, A/B/A parity, one BUILD + two REPLAY, one stable cache entry\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
