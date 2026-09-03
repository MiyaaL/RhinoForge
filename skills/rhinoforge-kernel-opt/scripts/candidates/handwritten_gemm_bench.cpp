// Standalone host harness for a source-built RPU GEMM candidate.
//
// This deliberately targets the small ABI documented by hxcc's 32x32x64
// example (three LocalSPM pointers).  It is not the production parallel-linear
// ABI used by RhinoForge.  Keeping the harness separate makes a source
// candidate measurable without silently substituting a release .ref or
// claiming that the candidate is already admitted by the runtime manifest.

#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rhino_launch_buffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using rhino_lkn::Kernel_t;
using rhino_lkn::HostDDR_t;
using rhino_lkn::LocalSPM_t;
using rhino_lkn::Program_t;
using rhino_lkn::Queue_t;

constexpr int kM = 32;
constexpr int kN = 32;
constexpr int kK = 64;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);

struct Options {
  std::string ref;
  std::string kernel;
  std::string trace;
  int warmup = 5;
  int iters = 30;
  bool weight_ddr = false;
  int ddr_stride = 256;
};

[[noreturn]] void usage(const char *argv0, int status) {
  std::FILE *out = status == 0 ? stdout : stderr;
  std::fprintf(out,
               "Usage: %s --binary FILE --kernel NAME [--warmup N] [--iters N] "
               "[--weight-ddr] [--ddr-stride {16,32,256}] [--trace FILE]\n",
               argv0);
  std::fprintf(out, "  --binary FILE  hxcc-generated candidate binary to load\n"
                   "  --ref FILE     legacy alias for --binary\n");
  std::exit(status);
}

int parse_positive(const char *text, const char *name) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 || value > 1000000)
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") usage(argv[0], 0);
    if (arg == "--weight-ddr") {
      out.weight_ddr = true;
      continue;
    }
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
    const char *value = argv[++i];
    if (arg == "--ref" || arg == "--binary") out.ref = value;
    else if (arg == "--kernel") out.kernel = value;
    else if (arg == "--warmup") out.warmup = parse_positive(value, "warmup");
    else if (arg == "--iters") out.iters = parse_positive(value, "iters");
    else if (arg == "--ddr-stride") {
      out.ddr_stride = parse_positive(value, "ddr-stride");
      if (out.ddr_stride != 16 && out.ddr_stride != 32 &&
          out.ddr_stride != 256)
        throw std::runtime_error("ddr-stride must be 16, 32, or 256");
    }
    else if (arg == "--trace") out.trace = value;
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (out.ref.empty() || out.kernel.empty())
    throw std::runtime_error("--binary/--ref and --kernel are required");
  if (out.iters == 0) throw std::runtime_error("iters must be positive");
  return out;
}

void check(std::uint32_t status, const char *what) {
  if (status != 0) throw std::runtime_error(std::string(what) + " failed: " +
                                             std::to_string(status));
}

// Round-to-nearest-even conversion, matching the independent CPU reference.
std::uint16_t to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
  std::uint32_t fraction = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    fraction |= 0x800000U;
    const int shift = 14 - exponent;
    std::uint32_t out = fraction >> shift;
    const std::uint32_t halfway = 1U << (shift - 1);
    if ((fraction & halfway) &&
        ((fraction & (halfway - 1U)) || (out & 1U)))
      ++out;
    return static_cast<std::uint16_t>(sign | out);
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  std::uint32_t out = fraction >> 13;
  const std::uint32_t discarded = fraction & 0x1fffU;
  if (discarded > 0x1000U ||
      (discarded == 0x1000U && (out & 1U)))
    ++out;
  if (out == 0x400U) {
    out = 0;
    ++exponent;
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  return static_cast<std::uint16_t>(sign |
                                    (static_cast<std::uint32_t>(exponent) << 10) |
                                    out);
}

float from_half(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t fraction = value & 0x3ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) bits = sign;
    else {
      int shift = -1;
      do {
        ++shift;
        fraction <<= 1;
      } while ((fraction & 0x400U) == 0);
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

void set_pointer(Kernel_t &kernel, std::uint32_t index, std::uint64_t address) {
  check(kernel.set_regs(index, static_cast<std::uint16_t>(address & 0xffffU)),
        "set pointer low half");
  check(kernel.set_regs(index + 1,
                        static_cast<std::uint16_t>((address >> 16) & 0xffffU)),
        "set pointer high half");
}

std::vector<std::uint16_t> make_input(std::size_t count, std::uint32_t seed) {
  std::vector<std::uint16_t> result(count);
  std::uint32_t state = seed;
  for (auto &element : result) {
    state = state * 1664525U + 1013904223U;
    const float value = (static_cast<float>((state >> 8) & 0xffffU) /
                             65535.0F -
                         0.5F) *
                        0.25F;
    element = to_half(value);
  }
  return result;
}

std::vector<std::uint16_t> reference(const std::vector<std::uint16_t> &a,
                                     const std::vector<std::uint16_t> &b) {
  std::vector<std::uint16_t> result(kM * kN, 0);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n) {
      float accumulator = 0.0F;
      for (int k = 0; k < kK; ++k)
        accumulator += from_half(a[m * kK + k]) *
                       from_half(b[n * kK + k]);
      result[m * kN + n] = to_half(accumulator);
    }
  return result;
}

double percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      quantile * static_cast<double>(values.size() - 1));
  return values[index];
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto host_a = make_input(kM * kK, 11U);
    const auto host_b = make_input(kN * kK, 23U);
    const auto expected = reference(host_a, host_b);

    Program_t program;
    check(program.create_with_binary_file(options.ref, options.kernel.c_str()),
          "create program");
    Kernel_t kernel(program, options.kernel.c_str());
    kernel.set_op_type("matmul");
    Queue_t queue(1);
    queue.set_broadcast_mode(false);

    LocalSPM_t spm_a(kM * kK * kHalfBytes, rhino_lkn::read_write,
                     rhino_lkn::kStride32B, 2, 0);
    std::unique_ptr<LocalSPM_t> spm_b;
    std::unique_ptr<HostDDR_t> ddr_b;
    if (options.weight_ddr) {
      const auto ddr_stride =
          options.ddr_stride == 16
              ? rhino_lkn::kStride16B
              : options.ddr_stride == 32 ? rhino_lkn::kStride32B
                                          : rhino_lkn::kStride256B;
      ddr_b = std::make_unique<HostDDR_t>(kN * kK * kHalfBytes,
                                          rhino_lkn::read_write,
                                          ddr_stride, 256);
      std::memcpy(ddr_b->get_cpu_ptr(), host_b.data(), host_b.size() * kHalfBytes);
      check(ddr_b->flush(), "flush DDR weight");
      check(ddr_b->end_cpu_access(0, host_b.size() * kHalfBytes),
            "release CPU DDR weight access");
    } else {
      spm_b = std::make_unique<LocalSPM_t>(
          kN * kK * kHalfBytes, rhino_lkn::read_write,
          rhino_lkn::kStride32B, 2, 0);
      std::memcpy(spm_b->get_cpu_ptr(), host_b.data(), host_b.size() * kHalfBytes);
    }
    LocalSPM_t spm_c(kM * kN * kHalfBytes, rhino_lkn::read_write,
                     rhino_lkn::kStride32B, 2, 0);
    std::vector<std::uint16_t> zero(kM * kN, 0);
    std::memcpy(spm_a.get_cpu_ptr(), host_a.data(), host_a.size() * kHalfBytes);
    std::memcpy(spm_c.get_cpu_ptr(), zero.data(), zero.size() * kHalfBytes);
    __sync_synchronize();

    set_pointer(kernel, 0, spm_c.get_rpu_addr());
    set_pointer(kernel, 2, spm_a.get_rpu_addr());
    if (options.weight_ddr)
      check(kernel.set_regs(4, *ddr_b), "set typed DDR weight");
    else
      set_pointer(kernel, 4, spm_b->get_rpu_addr());
    check(queue.add_kernel_mutable(kernel, {1, 1, 1}, {0}), "add kernel");
    if (!options.trace.empty()) queue.set_enable_hw_perf(true);
    check(queue.build_batch(), "build batch");
    check(queue.enqueue_batch(true), "correctness launch");
    __sync_synchronize();

    const auto *actual = static_cast<const std::uint16_t *>(spm_c.get_cpu_ptr());
    std::size_t mismatches = 0;
    float max_abs = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float difference =
          std::fabs(from_half(actual[i]) - from_half(expected[i]));
      max_abs = std::max(max_abs, difference);
      if (difference > 0.001F) ++mismatches;
    }
    if (mismatches != 0) {
      std::fprintf(stderr, "parity=FAIL mismatches=%zu max_abs=%g\n", mismatches,
                   static_cast<double>(max_abs));
      return 2;
    }

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
    if (!options.trace.empty()) check(queue.dump_hw_perf_chrome(options.trace.c_str()),
                                      "dump trace");
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    std::printf("schema=rhinoforge-handwritten-gemm-v1 kernel=%s m=%d n=%d k=%d "
                "weight_residency=%s ddr_stride=%d parity=PASS max_abs=%g warmup=%d iters=%d host_us_min=%g "
                "host_us_p50=%g host_us_mean=%g host_us_p95=%g\n",
                options.kernel.c_str(), kM, kN, kK,
                options.weight_ddr ? "HostDDR" : "LocalSPM",
                options.ddr_stride,
                static_cast<double>(max_abs),
                options.warmup, options.iters,
                *std::min_element(samples.begin(), samples.end()),
                percentile(samples, 0.50), mean, percentile(samples, 0.95));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
