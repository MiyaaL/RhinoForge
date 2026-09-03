// Board harness for the production-shaped handwritten GEMM+SiLU candidate.
//
// This is a verifier-owned C++ harness, not a device program and not a
// release asset.  It deliberately accepts only a caller-supplied binary
// (`--binary`); hxcc output is never generated or written by this file.  The
// fixed test contract is M=32, global N=2048, K=1024, eight col-partitioned
// cores, and grid.x=2.  Each core receives relative LocalSPM offsets while
// the two transformed FP16 weights are HostDDR-resident.  The custom source
// ABI uses pointer slots 0/2/4/6 as input/gate/output/up; slot 6 is therefore
// not a release bias pointer.  Keep this distinction explicit when reviewing
// any result from the harness.

#include "rhino_launch_buffer.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rhino_lkn::HostDDR_t;
using rhino_lkn::Kernel_t;
using rhino_lkn::LocalSPM_t;
using rhino_lkn::Program_t;
using rhino_lkn::Queue_t;

constexpr int kM = 32;
constexpr int kN = 2048;
constexpr int kK = 1024;
constexpr int kCores = 8;
constexpr int kLocalN = kN / kCores;
constexpr int kGridX = 2;
constexpr std::size_t kInputBytes = static_cast<std::size_t>(kM) * kK * 2;
constexpr std::size_t kWeightBytes = static_cast<std::size_t>(kN) * kK * 2;
constexpr std::size_t kOutputBytes = static_cast<std::size_t>(kM) * kLocalN * 2;

struct Options {
  std::string binary;
  std::string kernel;
  std::string trace;
  int warmup = 5;
  int iters = 30;
};

[[noreturn]] void usage(const char *program, int status) {
  std::FILE *stream = status == 0 ? stdout : stderr;
  std::fprintf(stream,
               "Usage: %s --binary FILE --kernel NAME [--warmup N] [--iters N] "
               "[--trace FILE]\n",
               program);
  std::exit(status);
}

int parse_count(const char *text, const char *name) {
  if (text == nullptr || *text == '\0' || *text == '-')
    throw std::runtime_error(std::string("invalid ") + name);
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 || value > 1000000)
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--help" || argument == "-h") usage(argv[0], 0);
    if (i + 1 >= argc)
      throw std::runtime_error("missing value after " + argument);
    const char *value = argv[++i];
    if (argument == "--binary") options.binary = value;
    else if (argument == "--kernel") options.kernel = value;
    else if (argument == "--warmup") options.warmup = parse_count(value, "warmup");
    else if (argument == "--iters") options.iters = parse_count(value, "iters");
    else if (argument == "--trace") options.trace = value;
    else throw std::runtime_error("unknown option " + argument);
  }
  if (options.binary.empty() || options.kernel.empty())
    throw std::runtime_error("--binary and --kernel are required");
  if (options.iters <= 0) throw std::runtime_error("iters must be positive");
  return options;
}

void check(std::uint32_t status, const char *what) {
  if (status != 0)
    throw std::runtime_error(std::string(what) + " failed: " +
                             std::to_string(status));
}

std::size_t align_up(std::size_t value, std::size_t alignment = 256) {
  return (value + alignment - 1) / alignment * alignment;
}

std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
  std::uint32_t fraction = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    fraction |= 0x800000U;
    const int shift = 14 - exponent;
    std::uint32_t result = fraction >> shift;
    const std::uint32_t halfway = 1U << (shift - 1);
    if ((fraction & halfway) &&
        ((fraction & (halfway - 1U)) || (result & 1U))) ++result;
    return static_cast<std::uint16_t>(sign | result);
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  std::uint32_t result = fraction >> 13;
  const std::uint32_t discarded = fraction & 0x1fffU;
  if (discarded > 0x1000U ||
      (discarded == 0x1000U && (result & 1U))) ++result;
  if (result == 0x400U) {
    result = 0;
    ++exponent;
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10) | result);
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t fraction = value & 0x3ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) bits = sign;
    else {
      int shift = 0;
      while ((fraction & 0x400U) == 0) {
        fraction <<= 1;
        ++shift;
      }
      fraction &= 0x3ffU;
      bits = sign | static_cast<std::uint32_t>(127 - 15 - shift) << 23 |
             fraction << 13;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | fraction << 13;
  } else {
    bits = sign | (exponent + 127 - 15) << 23 | fraction << 13;
  }
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::vector<std::uint16_t> make_data(std::size_t count, std::uint32_t seed) {
  std::vector<std::uint16_t> result(count);
  for (auto &value : result) {
    seed = seed * 1664525U + 1013904223U;
    const float sample =
        (static_cast<float>((seed >> 8) & 0xffffU) / 65535.0F - 0.5F) * 0.1F;
    value = float_to_half(sample);
  }
  return result;
}

// Match tp_col_swizzle_mc_weight for FP16 and eight cores.  The returned
// vector is only copied into HostDDR; no weight bytes are printed or persisted.
std::vector<std::uint16_t> col_swizzle(const std::vector<std::uint16_t> &weight) {
  constexpr std::size_t vector_width = 16;
  const std::size_t nblocks = kN / vector_width / kCores;
  const std::size_t kblocks = kK / vector_width;
  std::vector<std::uint16_t> result(weight.size());
  for (std::size_t core = 0; core < static_cast<std::size_t>(kCores); ++core)
    for (std::size_t block = 0; block < nblocks; ++block)
      for (std::size_t kblock = 0; kblock < kblocks; ++kblock)
        for (std::size_t ni = 0; ni < vector_width; ++ni)
          for (std::size_t ke = 0; ke < vector_width; ++ke) {
            const std::size_t source_row =
                (core * nblocks + block) * vector_width + ni;
            const std::size_t source = source_row * kK + kblock * vector_width + ke;
            const std::size_t destination =
                ((((block * kblocks + kblock) * kCores + core) * vector_width + ni) *
                 vector_width + ke);
            result[destination] = weight[source];
          }
  return result;
}

std::vector<std::uint16_t> reference(const std::vector<std::uint16_t> &input,
                                     const std::vector<std::uint16_t> &gate,
                                     const std::vector<std::uint16_t> &up) {
  std::vector<std::uint16_t> result(static_cast<std::size_t>(kM) * kN);
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      float gate_value = 0.0F;
      float up_value = 0.0F;
      for (int k = 0; k < kK; ++k) {
        gate_value += half_to_float(input[static_cast<std::size_t>(m) * kK + k]) *
                      half_to_float(gate[static_cast<std::size_t>(n) * kK + k]);
        up_value += half_to_float(input[static_cast<std::size_t>(m) * kK + k]) *
                    half_to_float(up[static_cast<std::size_t>(n) * kK + k]);
      }
      const float gate_half = half_to_float(float_to_half(gate_value));
      const float up_half = half_to_float(float_to_half(up_value));
      const float sigmoid_half = half_to_float(
          float_to_half(1.0F / (1.0F + std::exp(-gate_half))));
      const float silu_half = half_to_float(float_to_half(gate_half * sigmoid_half));
      result[static_cast<std::size_t>(m) * kN + n] =
          float_to_half(silu_half * up_half);
    }
  }
  return result;
}

void set_relative_pointer(Kernel_t &kernel, std::uint32_t register_index,
                          std::uint32_t offset) {
  check(kernel.set_regs(register_index,
                        static_cast<std::uint16_t>(offset & 0xffffU)),
        "set pointer low half");
  check(kernel.set_regs(register_index + 1,
                        static_cast<std::uint16_t>((offset >> 16) & 0xffffU)),
        "set pointer high half");
}

double percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      quantile * static_cast<double>(values.size() - 1));
  return values[index];
}

struct TraceSummary {
  std::size_t compute_events = 0;
  std::uint64_t duration_cycles = 0;
  double duration_us = 0.0;
  std::uint64_t frequency_hz = 0;
};

TraceSummary summarize_trace(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read trace output");
  const std::string text((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
  TraceSummary summary;
  const std::regex cycle_pattern("\"duration_cycles\":([0-9]+)");
  for (std::sregex_iterator it(text.begin(), text.end(), cycle_pattern), end;
       it != end; ++it) {
    ++summary.compute_events;
    summary.duration_cycles = std::max(
        summary.duration_cycles,
        static_cast<std::uint64_t>(std::stoull((*it)[1].str())));
  }
  const std::regex us_pattern("\"duration_us\":([0-9]+(?:\\.[0-9]+)?)");
  for (std::sregex_iterator it(text.begin(), text.end(), us_pattern), end;
       it != end; ++it) {
    summary.duration_us = std::max(summary.duration_us, std::stod((*it)[1].str()));
  }
  // SDK revisions have emitted this field both as a quoted decimal string and
  // as a JSON number.  Accept either representation, but never infer a
  // frequency when the field is absent.
  const std::regex frequency_pattern("\"frequency_hz\"[^0-9]*([0-9]+)");
  std::smatch frequency_match;
  if (std::regex_search(text, frequency_match, frequency_pattern))
    summary.frequency_hz = std::stoull(frequency_match[1].str());
  return summary;
}

void configure_kernel(Kernel_t &kernel, const std::string &name,
                      const HostDDR_t &gate, const HostDDR_t &up,
                      std::uint32_t output_offset) {
  check(kernel.reset_regs(), "reset kernel registers");
  // Relative LocalSPM offsets are broadcast to every selected core.  The
  // custom source ABI maps regs 0/2/4/6 to input/gate/output/up.
  set_relative_pointer(kernel, 0, 0);
  set_relative_pointer(kernel, 4, output_offset);
  check(kernel.set_regs(2, gate), "set gate DDR register");
  check(kernel.set_regs(6, up), "set up DDR register");
  kernel.set_name(name.c_str());
  kernel.set_op_type("matmul");
}

void add_broadcast_kernel(Queue_t &queue, Kernel_t &kernel) {
  std::vector<std::uint8_t> core_ids;
  core_ids.reserve(kCores);
  for (int core = 0; core < kCores; ++core)
    core_ids.push_back(static_cast<std::uint8_t>(core));
  check(queue.add_kernel_mutable(kernel, {kGridX, 1, 1}, core_ids),
        "add broadcast kernel");
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto input = make_data(kInputBytes / sizeof(std::uint16_t), 11U);
    const auto gate = make_data(kWeightBytes / sizeof(std::uint16_t), 17U);
    const auto up = make_data(kWeightBytes / sizeof(std::uint16_t), 23U);
    const auto expected = reference(input, gate, up);
    const auto gate_swizzled = col_swizzle(gate);
    const auto up_swizzled = col_swizzle(up);

    Program_t program;
    check(program.create_with_binary_file(options.binary.c_str(),
                                          options.kernel.c_str()),
          "load binary/kernel");
    HostDDR_t gate_ddr(kWeightBytes, rhino_lkn::read_write,
                       rhino_lkn::kStride256B, 256);
    HostDDR_t up_ddr(kWeightBytes, rhino_lkn::read_write,
                     rhino_lkn::kStride256B, 256);
    std::memcpy(gate_ddr.get_cpu_ptr(), gate_swizzled.data(), kWeightBytes);
    std::memcpy(up_ddr.get_cpu_ptr(), up_swizzled.data(), kWeightBytes);
    check(gate_ddr.flush(), "flush gate DDR");
    check(up_ddr.flush(), "flush up DDR");
    check(gate_ddr.end_cpu_access(0, kWeightBytes), "release gate DDR");
    check(up_ddr.end_cpu_access(0, kWeightBytes), "release up DDR");

    const std::size_t output_offset = align_up(kInputBytes);
    const std::size_t spm_bytes = output_offset + kOutputBytes;
    std::vector<std::unique_ptr<LocalSPM_t>> spm;
    spm.reserve(kCores);
    for (int core = 0; core < kCores; ++core) {
      spm.emplace_back(std::make_unique<LocalSPM_t>(
          spm_bytes, rhino_lkn::read_write, rhino_lkn::kStride256B, 256, core));
      auto *base = static_cast<std::uint8_t *>(spm.back()->get_cpu_ptr());
      std::memcpy(base, input.data(), kInputBytes);
      std::vector<std::uint16_t> zeros(kOutputBytes / sizeof(std::uint16_t), 0);
      std::memcpy(base + output_offset, zeros.data(), kOutputBytes);
    }
    __sync_synchronize();

    Kernel_t kernel(program, options.kernel.c_str());
    configure_kernel(kernel, options.kernel, gate_ddr, up_ddr,
                     static_cast<std::uint32_t>(output_offset));
    Queue_t queue(kCores);
    queue.set_flush_icache(false);
    queue.set_broadcast_mode(true);
    queue.set_enable_hw_perf(false);
    add_broadcast_kernel(queue, kernel);
    check(queue.build_batch(), "build batch");
    check(queue.enqueue_batch(true), "correctness replay");
    __sync_synchronize();

    std::size_t mismatches = 0;
    float max_abs = 0.0F;
    double sum_abs = 0.0;
    for (int core = 0; core < kCores; ++core) {
      const auto *actual = static_cast<const std::uint16_t *>(
          static_cast<const void *>(static_cast<const std::uint8_t *>(
                                        spm[core]->get_cpu_ptr()) + output_offset));
      for (int m = 0; m < kM; ++m) {
        for (int n = 0; n < kLocalN; ++n) {
          const std::size_t local_index = static_cast<std::size_t>(m) * kLocalN + n;
          const std::size_t global_index =
              static_cast<std::size_t>(m) * kN + core * kLocalN + n;
          const float difference =
              std::fabs(half_to_float(actual[local_index]) -
                        half_to_float(expected[global_index]));
          max_abs = std::max(max_abs, difference);
          sum_abs += difference;
          if (difference > 0.02F) ++mismatches;
        }
      }
    }
    const double mean_abs = sum_abs / static_cast<double>(kM * kN);
    std::printf("schema=rhinoforge-handwritten-fusion-v1 kernel=%s shape=M%dN%dK%d "
                "cores=%d grid_x=%d broadcast=1\n",
                options.kernel.c_str(), kM, kN, kK, kCores, kGridX);
    std::printf("parity=%s mismatches=%zu max_abs=%g mean_abs=%g\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches,
                static_cast<double>(max_abs), mean_abs);
    if (mismatches != 0) return 3;

    for (int i = 0; i < options.warmup; ++i)
      check(queue.enqueue_batch(true), "warmup replay");
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iters));
    for (int i = 0; i < options.iters; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      check(queue.enqueue_batch(true), "timed replay");
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::printf("replay_host_us min=%g p50=%g p95=%g mean=%g samples=%zu\n",
                *std::min_element(samples.begin(), samples.end()),
                percentile(samples, 0.50), percentile(samples, 0.95), mean,
                samples.size());

    if (!options.trace.empty()) {
      // A fresh queue keeps perf tracing outside the reported timing samples.
      Kernel_t trace_kernel(program, options.kernel.c_str());
      configure_kernel(trace_kernel, options.kernel, gate_ddr, up_ddr,
                       static_cast<std::uint32_t>(output_offset));
      Queue_t trace_queue(kCores);
      trace_queue.set_flush_icache(false);
      trace_queue.set_broadcast_mode(true);
      trace_queue.set_enable_hw_perf(true);
      add_broadcast_kernel(trace_queue, trace_kernel);
      check(trace_queue.build_batch(), "build trace batch");
      check(trace_queue.enqueue_batch(true), "trace replay");
      check(trace_queue.dump_hw_perf_chrome(options.trace.c_str()),
            "dump trace");
      const TraceSummary summary = summarize_trace(options.trace);
      std::printf("trace_summary compute_events=%zu duration_cycles=%llu "
                  "duration_us=%g frequency_hz=%llu path=%s\n",
                  summary.compute_events,
                  static_cast<unsigned long long>(summary.duration_cycles),
                  summary.duration_us,
                  static_cast<unsigned long long>(summary.frequency_hz),
                  options.trace.c_str());
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 2;
  }
}
