// Host harness for the fixed-shape SPM source candidates in this directory.
// It deliberately uses the three-pointer ABI from the hxcc manual, rather
// than the release parallel_linear ABI.  Generated .ref files and traces must
// be kept in a private build directory (for example /dev/shm).

#include "rhino_launch_buffer.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Options {
  std::string ref;
  std::string kernel;
  std::string trace;
  int m = 32;
  int n = 256;
  int k = 1024;
  int warmup = 5;
  int iters = 30;
  bool weight_ddr = false;
  int ddr_stride = 256;
  std::string ddr_binding = "typed";
};

[[noreturn]] void usage(const char *argv0, int status) {
  std::FILE *out = status == 0 ? stdout : stderr;
  std::fprintf(out,
               "Usage: %s --binary FILE --kernel NAME --n N --k K "
               "[--weight-ddr] [--ddr-stride {1,16,32,256}] "
               "[--ddr-binding {typed,raw,raw-shift256}] "
               "[--warmup N] [--iters N] [--trace FILE]\n",
               argv0);
  std::fprintf(out, "  --binary FILE  hxcc-generated candidate binary to load\n"
                   "  --ref FILE     legacy alias for --binary\n");
  std::exit(status);
}

int positive(const char *text, const char *name) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 || value > 65535)
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<int>(value);
}

Options parse(int argc, char **argv) {
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
    else if (arg == "--m") out.m = positive(value, "m");
    else if (arg == "--n") out.n = positive(value, "n");
    else if (arg == "--k") out.k = positive(value, "k");
    else if (arg == "--warmup") out.warmup = positive(value, "warmup");
    else if (arg == "--iters") out.iters = positive(value, "iters");
    else if (arg == "--ddr-stride") {
      out.ddr_stride = positive(value, "ddr-stride");
      if (out.ddr_stride != 1 && out.ddr_stride != 16 &&
          out.ddr_stride != 32 && out.ddr_stride != 256)
        throw std::runtime_error("ddr-stride must be 1, 16, 32, or 256");
    }
    else if (arg == "--ddr-binding") {
      out.ddr_binding = value;
      if (out.ddr_binding != "typed" && out.ddr_binding != "raw" &&
          out.ddr_binding != "raw-shift256")
        throw std::runtime_error(
            "ddr-binding must be typed, raw, or raw-shift256");
    }
    else if (arg == "--trace") out.trace = value;
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (out.ref.empty() || out.kernel.empty())
    throw std::runtime_error("--binary/--ref and --kernel are required");
  if (out.m != 32 || (out.n != 256 && out.n != 1024) ||
      (out.k != 1024 && out.k != 256))
    throw std::runtime_error("supported shapes are 32x256x1024 and 32x1024x256");
  if ((out.n == 256 && out.k != 1024) ||
      (out.n == 1024 && out.k != 256))
    throw std::runtime_error("N/K must match the selected fixed candidate");
  return out;
}

void check(std::uint32_t status, const char *what) {
  if (status != 0)
    throw std::runtime_error(std::string(what) + " failed: " +
                             std::to_string(status));
}

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

std::vector<std::uint16_t> make_values(std::size_t count, std::uint32_t seed) {
  std::vector<std::uint16_t> result(count);
  std::uint32_t state = seed;
  for (auto &value : result) {
    state = state * 1664525U + 1013904223U;
    value = to_half((static_cast<float>((state >> 8) & 0xffffU) /
                     65535.0F -
                     0.5F) *
                    0.125F);
  }
  return result;
}

std::vector<float> reference(const std::vector<std::uint16_t> &a,
                             const std::vector<std::uint16_t> &b,
                             const Options &o) {
  std::vector<float> result(static_cast<std::size_t>(o.m) * o.n, 0.0F);
  for (int m = 0; m < o.m; ++m)
    for (int n = 0; n < o.n; ++n) {
      float sum = 0.0F;
      for (int k = 0; k < o.k; ++k)
        sum += from_half(a[static_cast<std::size_t>(m) * o.k + k]) *
               from_half(b[static_cast<std::size_t>(n) * o.k + k]);
      result[static_cast<std::size_t>(m) * o.n + n] = sum;
    }
  return result;
}

double percentile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(q * (values.size() - 1))];
}
}  // namespace

int main(int argc, char **argv) {
  try {
    const Options o = parse(argc, argv);
    const auto a = make_values(static_cast<std::size_t>(o.m) * o.k, 11U);
    const auto b = make_values(static_cast<std::size_t>(o.n) * o.k, 23U);
    const auto expected = reference(a, b, o);
    const std::size_t a_bytes = a.size() * sizeof(std::uint16_t);
    const std::size_t b_bytes = b.size() * sizeof(std::uint16_t);
    const std::size_t c_bytes = expected.size() * sizeof(std::uint16_t);

    Program_t program;
    check(program.create_with_binary_file(o.ref, o.kernel.c_str()),
          "create program");
    Kernel_t kernel(program, o.kernel.c_str());
    kernel.set_op_type("matmul");
    Queue_t queue(1);
    queue.set_broadcast_mode(false);
    LocalSPM_t spm_a(a_bytes, rhino_lkn::read_write, rhino_lkn::kStride32B, 2, 0);
    // Keep both residency modes in one immutable harness.  LocalSPM isolates
    // the loop/compute schedule; --weight-ddr exercises the compiler's
    // address-space-qualified __DDR pointer and includes DDR traffic in the
    // device critical path.  Neither mode is a release asset by itself.
    std::unique_ptr<LocalSPM_t> spm_b;
    std::unique_ptr<HostDDR_t> ddr_b;
    if (o.weight_ddr) {
      const auto ddr_stride =
          o.ddr_stride == 1
              ? rhino_lkn::kStride1B
              : o.ddr_stride == 16
                    ? rhino_lkn::kStride16B
                    : o.ddr_stride == 32 ? rhino_lkn::kStride32B
                                          : rhino_lkn::kStride256B;
      ddr_b = std::make_unique<HostDDR_t>(b_bytes, rhino_lkn::read_write,
                                          ddr_stride, 256);
      std::memcpy(ddr_b->get_cpu_ptr(), b.data(), b_bytes);
      check(ddr_b->flush(), "flush DDR weight");
      check(ddr_b->end_cpu_access(0, b_bytes), "release CPU DDR weight access");
#ifdef RHINOFORGE_DEBUG_ADDRESSES
      std::fprintf(stderr, "ddr_cpu=%p ddr_rpu=0x%llx ddr_stride=%d bytes=%zu\n",
                   ddr_b->get_cpu_ptr(),
                   static_cast<unsigned long long>(ddr_b->get_rpu_addr()),
                   o.ddr_stride, b_bytes);
#endif
    } else {
      spm_b = std::make_unique<LocalSPM_t>(
          b_bytes, rhino_lkn::read_write, rhino_lkn::kStride32B, 2, 0);
      std::memcpy(spm_b->get_cpu_ptr(), b.data(), b_bytes);
    }
    LocalSPM_t spm_c(c_bytes, rhino_lkn::read_write, rhino_lkn::kStride32B, 2, 0);
    std::vector<std::uint16_t> zero(expected.size(), 0);
    std::memcpy(spm_a.get_cpu_ptr(), a.data(), a_bytes);
    std::memcpy(spm_c.get_cpu_ptr(), zero.data(), c_bytes);
    __sync_synchronize();
    set_pointer(kernel, 0, spm_c.get_rpu_addr());
    set_pointer(kernel, 2, spm_a.get_rpu_addr());
    if (o.weight_ddr) {
      if (o.ddr_binding == "typed") {
        check(kernel.set_regs(4, *ddr_b), "set typed DDR weight");
      } else {
        std::uint64_t address = ddr_b->get_rpu_addr();
        if (o.ddr_binding == "raw-shift256") address >>= 8;
        set_pointer(kernel, 4, address);
      }
    } else {
      set_pointer(kernel, 4, spm_b->get_rpu_addr());
    }
#if defined(RHINOFORGE_DEBUG_ADDRESSES) && RHINO_LAUNCH_HAS_DEV_PROGRAM_API
    kernel.print_kernel_info();
#endif
    check(queue.add_kernel_mutable(kernel, {1, 1, 1}, {0}), "add kernel");
    if (!o.trace.empty()) queue.set_enable_hw_perf(true);
    check(queue.build_batch(), "build batch");
    check(queue.enqueue_batch(true), "correctness launch");
    __sync_synchronize();
    const auto *actual = static_cast<const std::uint16_t *>(spm_c.get_cpu_ptr());
    std::size_t mismatches = 0;
    float max_abs = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float diff = std::fabs(from_half(actual[i]) - expected[i]);
      max_abs = std::max(max_abs, diff);
      if (diff > 0.02F) ++mismatches;
    }
    if (mismatches != 0) {
      std::fprintf(stderr, "parity=FAIL mismatches=%zu max_abs=%g\n", mismatches,
                   static_cast<double>(max_abs));
      std::size_t printed = 0;
      for (std::size_t i = 0; i < expected.size() && printed < 8; ++i) {
        const float diff = std::fabs(from_half(actual[i]) - expected[i]);
        if (diff > 0.02F) {
          std::fprintf(stderr, "first_mismatch index=%zu actual=%g expected=%g diff=%g\n",
                       i, static_cast<double>(from_half(actual[i])),
                       static_cast<double>(expected[i]), static_cast<double>(diff));
          ++printed;
        }
      }
      return 2;
    }
    for (int i = 0; i < o.warmup; ++i) check(queue.enqueue_batch(true), "warmup");
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(o.iters));
    for (int i = 0; i < o.iters; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      check(queue.enqueue_batch(true), "timed replay");
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    if (!o.trace.empty()) check(queue.dump_hw_perf_chrome(o.trace.c_str()),
                                 "dump trace");
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    std::printf("schema=rhinoforge-handwritten-spm-gemm-v1 kernel=%s m=%d n=%d "
                "k=%d weight_residency=%s ddr_stride=%d ddr_binding=%s parity=PASS max_abs=%g host_us_min=%g host_us_p50=%g "
                "host_us_mean=%g host_us_p95=%g\n",
                o.kernel.c_str(), o.m, o.n, o.k,
                o.weight_ddr ? "HostDDR" : "LocalSPM",
                o.ddr_stride,
                o.ddr_binding.c_str(),
                static_cast<double>(max_abs),
                *std::min_element(samples.begin(), samples.end()),
                percentile(samples, 0.50), mean, percentile(samples, 0.95));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
