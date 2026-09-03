// Standalone, device-resident FP16 GEMM benchmark for the Rhino Launch SDK.
//
// This is intentionally independent of torch.rpu and of RhinoForge's Graph
// wrapper.  It loads one kernel from an already released operator-library REF,
// places the transformed weight in HostDDR and the per-core input/output
// windows in directly mapped LocalSPM once, builds one Launch batch, and
// measures only synchronous batch replays.  Consequently the reported replay
// samples are not end-to-end model timings and do not include DDR↔SPM staging
// or output materialization.  The optional Chrome trace is collected in a
// separate diagnostic batch.
//
// The --m/--n/--k arguments are the *global* linear dimensions (weight is
// [N,K]).  partition=1 is column parallel (local N=N/cores), and partition=0
// is row parallel (local K=K/cores).  For the Wall decode contracts:
//   q/k/v/gate/up: --m 32 --n 2048 --k 1024 --partition 1
//   o/down:        --m 32 --n 1024 --k 2048 --partition 0
// They therefore print local tiles [32,256,1024] and [32,1024,256].
//
// The implementation uses the same set_parallel_linear_regs ABI as
// RhinoForge's host wrapper.  It is deliberately kept here (rather than
// including the private PyTorch extension) so it can be built directly on an
// RPU image with only the Launch SDK, rpu_api, Hxil, and HxMem libraries.

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
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using rhino_lkn::HostDDR_t;
using rhino_lkn::Kernel_t;
using rhino_lkn::LocalSPM_t;
using rhino_lkn::Program_t;
using rhino_lkn::Queue_t;

constexpr std::uint32_t kMaxCores = 8;
constexpr std::size_t kAlign = 256;
constexpr int kDefaultCores = 8;
// LocalSPM_t::get_rpu_addr() intentionally returns a core-relative offset.  A
// broadcast ParallelLinear packet is therefore programmed with that offset;
// the hardware resolves it against the executing core's local SPM.  Using a
// A GlobalSPM_t address here is a subtle ABI error: it makes core 0 appear
// right while every other core reads/writes the wrong window.
constexpr int kTiles[] = {128, 112, 96, 80, 64, 48, 32};

struct Options {
  std::uint64_t m = 32;
  std::uint64_t n = 2048;
  std::uint64_t k = 1024;
  int partition = 1;
  int cores = kDefaultCores;
  int tile = 0;       // zero means use the exact release-table entry when known
  bool acc32 = false;
  std::uint64_t warmup = 5;
  std::uint64_t iters = 30;
  bool verify = true;
  // Address/broadcast knobs are diagnostic-only.  The production wrapper
  // normally emits one core-0 unified address and relies on its broadcast
  // contract; the standalone harness keeps explicit alternatives so an ABI
  // mismatch cannot be hidden by a failed multi-core parity check.
  bool broadcast = true;
  bool typed_weight = false;
  std::string address_mode = "relative";  // relative, core0, per-core (diagnostic)
  std::string submit_mode = "mutable";  // mutable, immutable, immediate
  bool diagnose = false;
  std::string ref;
  std::string kernel;
  std::string trace;
  std::string raw;
  std::uint64_t seed = 17;
};

[[noreturn]] void usage(const char *argv0, int status) {
  std::FILE *out = status == 0 ? stdout : stderr;
  std::fprintf(
      out,
      "Usage: %s --ref FILE [options]\n"
      "  --m M --n N --k K        global GEMM dimensions (default 32,2048,1024)\n"
      "  --partition 0|1          0=row (split K), 1=col (split N)\n"
      "  --cores C                active cores, 1..8 (default 8)\n"
      "  --tile {128,112,96,80,64,48,32}\n"
      "                           n tile; omit for exact release-table pick\n"
      "  --acc32                  use FP16-weight ACC32 family\n"
      "  --broadcast 0|1          queue broadcast mode (default 1)\n"
      "  --typed-weight           set regs2/3 through HostDDR_t metadata\n"
      "  --address-mode MODE      core0|relative|per-core (default relative)\n"
      "                           diagnostic address ABI override\n"
      "  --submit-mode MODE       mutable|immutable|immediate (default mutable)\n"
      "  --diagnose               print validation indices only (default quiet)\n"
      "  --kernel NAME            override derived manifest name\n"
      "  --warmup W --iters I     replay warmups/samples (default 5/30)\n"
      "  --trace FILE             collect one separate HW Chrome trace\n"
      "  --raw FILE               write raw replay samples as JSON\n"
      "  --no-verify               skip host-side output parity check\n"
      "  --seed S                 deterministic input/weight seed\n"
      "  -h, --help               show this help\n\n"
      "Dimensions are global.  Wall qkv: M=32,N=2048,K=1024,col -> "
      "local [32,256,1024].  Wall o/down: M=32,N=1024,K=2048,row -> "
      "local [32,1024,256].\n",
      argv0);
  std::exit(status);
}

std::uint64_t parse_u64(const char *text, const char *name) {
  if (text == nullptr || *text == '\0' || *text == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " +
                             (text ? text : "<null>"));
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || value == 0) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::uint64_t>(value);
}

int parse_i(const char *text, const char *name) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options o;
  const char *env_ref = std::getenv("RPU_OPLIB_REF");
  if (env_ref != nullptr) o.ref = env_ref;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") usage(argv[0], 0);
    if (arg == "--acc32") {
      o.acc32 = true;
      continue;
    }
    if (arg == "--no-verify") {
      o.verify = false;
      continue;
    }
    if (arg == "--typed-weight") {
      o.typed_weight = true;
      continue;
    }
    if (arg == "--diagnose") {
      o.diagnose = true;
      continue;
    }
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
    const char *v = argv[++i];
    if (arg == "--m") o.m = parse_u64(v, "m");
    else if (arg == "--n") o.n = parse_u64(v, "n");
    else if (arg == "--k") o.k = parse_u64(v, "k");
    else if (arg == "--partition") o.partition = parse_i(v, "partition");
    else if (arg == "--cores") o.cores = parse_i(v, "cores");
    else if (arg == "--tile") o.tile = parse_i(v, "tile");
    else if (arg == "--warmup") o.warmup = parse_u64(v, "warmup");
    else if (arg == "--iters") o.iters = parse_u64(v, "iters");
    else if (arg == "--broadcast") {
      const int mode = parse_i(v, "broadcast");
      if (mode != 0 && mode != 1)
        throw std::runtime_error("broadcast must be 0 or 1");
      o.broadcast = mode != 0;
    }
    else if (arg == "--address-mode") o.address_mode = v;
    else if (arg == "--submit-mode") o.submit_mode = v;
    else if (arg == "--ref") o.ref = v;
    else if (arg == "--kernel") o.kernel = v;
    else if (arg == "--trace") o.trace = v;
    else if (arg == "--raw") o.raw = v;
    else if (arg == "--seed") o.seed = parse_u64(v, "seed");
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (o.ref.empty()) throw std::runtime_error("--ref (or RPU_OPLIB_REF) is required");
  if (o.partition != 0 && o.partition != 1)
    throw std::runtime_error("partition must be 0 or 1");
  if (o.cores < 1 || o.cores > static_cast<int>(kMaxCores))
    throw std::runtime_error("cores must be in [1,8]");
  if (o.partition == 0 && o.k % static_cast<std::uint64_t>(o.cores) != 0)
    throw std::runtime_error("row partition requires K divisible by cores");
  if (o.partition == 1 && o.n % static_cast<std::uint64_t>(o.cores) != 0)
    throw std::runtime_error("col partition requires N divisible by cores");
  if (o.m > 65535 || o.n > 65535 || o.k > 65535)
    throw std::runtime_error("M/N/K must fit the 16-bit register ABI");
  if (o.tile != 0 && std::find(std::begin(kTiles), std::end(kTiles), o.tile) ==
                         std::end(kTiles))
    throw std::runtime_error("tile must be one of 128,112,96,80,64,48,32");
  if (o.address_mode != "core0" && o.address_mode != "relative" &&
      o.address_mode != "per-core")
    throw std::runtime_error("address-mode must be core0, relative, or per-core");
  if (o.address_mode == "per-core" && o.broadcast)
    throw std::runtime_error("per-core address mode requires --broadcast 0");
  if (o.submit_mode != "mutable" && o.submit_mode != "immutable" &&
      o.submit_mode != "immediate")
    throw std::runtime_error(
        "submit-mode must be mutable, immutable, or immediate");
  return o;
}

void check_status(std::uint32_t status, const char *what) {
  if (status != 0)
    throw std::runtime_error(std::string(what) + " failed: " +
                             std::to_string(status));
}

std::size_t align_up(std::size_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }

// FP32 <-> IEEE binary16 conversion.  Inputs are deliberately finite and
// small, but these routines also handle subnormals, infinities and NaNs so a
// failed parity run is not hidden by an implementation-defined conversion.
std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  int exp = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
  std::uint32_t frac = bits & 0x7fffffU;
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    frac |= 0x800000U;
    const int shift = 14 - exp;
    std::uint32_t out = frac >> shift;
    const std::uint32_t halfway = 1U << (shift - 1);
    if ((frac & halfway) && ((frac & (halfway - 1U)) || (out & 1U))) ++out;
    return static_cast<std::uint16_t>(sign | out);
  }
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  std::uint32_t out_frac = frac >> 13;
  const std::uint32_t discarded = frac & 0x1fffU;
  if (discarded > 0x1000U ||
      (discarded == 0x1000U && (out_frac & 1U))) {
    ++out_frac;
  }
  if (out_frac == 0x400U) {
    out_frac = 0;
    ++exp;
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                    out_frac);
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  std::uint32_t exp = (h >> 10) & 0x1fU;
  std::uint32_t frac = h & 0x3ffU;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (frac == 0) bits = sign;
    else {
      // Normalize the half subnormal.
      int e = -1;
      do {
        ++e;
        frac <<= 1;
      } while ((frac & 0x400U) == 0);
      frac &= 0x3ffU;
      bits = sign | static_cast<std::uint32_t>(127 - 15 - e) << 23 |
             frac << 13;
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000U | (frac << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (frac << 13);
  }
  float out = 0;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// A tiny deterministic PRNG avoids libc distribution differences between
// board images.  Values are converted to FP16 before being copied to device,
// and the reference uses those converted values exactly.
std::uint32_t next_u32(std::uint64_t &state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return static_cast<std::uint32_t>((state * 2685821657736338717ULL) >> 32);
}

struct Problem {
  std::size_t local_k = 0;
  std::size_t local_n = 0;
  std::size_t input_elems = 0;
  std::size_t output_elems = 0;
  std::size_t input_bytes = 0;
  std::size_t output_offset = 0;
  std::size_t spm_bytes = 0;
};

Problem make_problem(const Options &o) {
  Problem p;
  p.local_k = o.partition == 0 ? o.k / static_cast<std::size_t>(o.cores) : o.k;
  p.local_n = o.partition == 0 ? o.n : o.n / static_cast<std::size_t>(o.cores);
  p.input_elems = static_cast<std::size_t>(o.m) * p.local_k;
  p.output_elems = static_cast<std::size_t>(o.m) * p.local_n;
  p.input_bytes = p.input_elems * sizeof(std::uint16_t);
  p.output_offset = align_up(p.input_bytes);
  p.spm_bytes = align_up(p.output_offset + p.output_elems * sizeof(std::uint16_t));
  if (p.spm_bytes == 0 || p.spm_bytes > 8U * 1024U * 1024U)
    throw std::runtime_error("per-core SPM allocation exceeds 8 MiB");
  return p;
}

// Apply the exact Python weight transformations from runtime/weights.py.
// Both transforms operate on 16-element (32-byte) vectors.  Keeping this code
// explicit makes the harness usable without importing torch or silently
// benchmarking an unswizzled matrix.
std::vector<std::uint16_t> swizzle_weight(const std::vector<std::uint16_t> &w,
                                          std::size_t n, std::size_t k,
                                          int partition, int cores) {
  constexpr std::size_t vec = 16;  // FP16 values per 32-byte line
  if (n % vec || k % vec) throw std::runtime_error("N and K must be 16-aligned");
  std::vector<std::uint16_t> out(w.size());
  if (partition == 1) {
    if (n % (vec * static_cast<std::size_t>(cores)))
      throw std::runtime_error("col swizzle requires N divisible by 16*cores");
    const std::size_t nb = n / vec / static_cast<std::size_t>(cores);
    const std::size_t kb = k / vec;
    // v[core, nblock, ni, kblock, ke] -> permute(1,3,0,2,4).
    for (std::size_t core = 0; core < static_cast<std::size_t>(cores); ++core)
      for (std::size_t b = 0; b < nb; ++b)
        for (std::size_t q = 0; q < kb; ++q)
          for (std::size_t ni = 0; ni < vec; ++ni)
            for (std::size_t ke = 0; ke < vec; ++ke) {
              const std::size_t src_row = (core * nb + b) * vec + ni;
              const std::size_t src = src_row * k + q * vec + ke;
              const std::size_t dst = ((((b * kb + q) * cores + core) * vec + ni) *
                                       vec + ke);
              out[dst] = w[src];
            }
  } else {
    if (k % (vec * static_cast<std::size_t>(cores)))
      throw std::runtime_error("row swizzle requires K divisible by 16*cores");
    const std::size_t nb = n / vec;
    const std::size_t kb = k / vec / static_cast<std::size_t>(cores);
    // v[nblock, ni, core, kblock, ke] -> permute(0,3,2,1,4).
    for (std::size_t b = 0; b < nb; ++b)
      for (std::size_t q = 0; q < kb; ++q)
        for (std::size_t core = 0; core < static_cast<std::size_t>(cores); ++core)
          for (std::size_t ni = 0; ni < vec; ++ni)
            for (std::size_t ke = 0; ke < vec; ++ke) {
              const std::size_t src_row = b * vec + ni;
              const std::size_t src = src_row * k +
                                      (core * kb + q) * vec + ke;
              const std::size_t dst = ((((b * kb + q) * cores + core) * vec + ni) *
                                       vec + ke);
              out[dst] = w[src];
            }
  }
  return out;
}

int default_m_tile(bool acc32, int n_tile) {
  if (!acc32) {
    switch (n_tile) {
      case 128: return 320;
      case 112: return 352;
      case 96: return 384;
      case 80: return 432;
      case 64: return 480;
      case 48: return 528;
      case 32: return 608;
    }
  } else {
    switch (n_tile) {
      case 128: return 208;
      case 112: return 240;
      case 96: return 272;
      case 80: return 304;
      case 64: return 352;
      case 48: return 416;
      case 32: return 496;
    }
  }
  throw std::runtime_error("unsupported n tile");
}

// The complete 285-entry tables live in RhinoForge's host wrapper.  The
// standalone tool only needs to encode the two exact Wall decode contracts
// (plus M=1 controls); explicit --tile still permits every released variant.
// Keep the fallback conservative and label it in the output rather than
// pretending that a nearby tile is an exact selector result.
int release_tile_for_shape(const Options &o, const Problem &p,
                           bool *exact) {
  *exact = true;
  if (!o.acc32) {
    if (o.m == 32 && p.local_n == 256 && p.local_k == 1024) return 32;
    if (o.m == 32 && p.local_n == 1024 && p.local_k == 256) return 64;
    if (o.m == 1 && p.local_n == 256 && p.local_k == 1024) return 48;
    if (o.m == 1 && p.local_n == 1024 && p.local_k == 256) return 112;
  } else {
    if (o.m == 32 && p.local_n == 256 && p.local_k == 1024) return 112;
    if (o.m == 32 && p.local_n == 1024 && p.local_k == 256) return 32;
    if (o.m == 1 && p.local_n == 256 && p.local_k == 1024) return 80;
    if (o.m == 1 && p.local_n == 1024 && p.local_k == 256) return 32;
  }
  *exact = false;
  return 128;
}

std::string derived_kernel_name(const Options &o, int n_tile) {
  const int mt = default_m_tile(o.acc32, n_tile);
  if (o.acc32) return "parallel_linear_acc32_m" + std::to_string(mt) +
                         "n" + std::to_string(n_tile) + "k128";
  return "parallel_linear_m" + std::to_string(mt) + "n" +
         std::to_string(n_tile) + "k128";
}

// This is byte-for-byte the host wrapper ABI in src/ops/rpu_linear_tiling.h.
void set_u32_pair(Kernel_t &kernel, std::uint32_t lo, std::uint64_t value) {
  check_status(kernel.set_regs(lo, static_cast<std::uint16_t>(value & 0xffffU)),
               "set register low half");
  check_status(kernel.set_regs(lo + 1,
                               static_cast<std::uint16_t>((value >> 16) & 0xffffU)),
               "set register high half");
}

void set_parallel_linear_regs(Kernel_t &kernel, std::uint32_t input_addr,
                              std::uint64_t weight_addr,
                              std::uint32_t output_addr, const Problem &p,
                              const Options &o) {
  if ((weight_addr >> 8) > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("DDR weight address does not fit shifted ABI");
  check_status(kernel.reset_regs(), "reset GEMM registers");
  set_u32_pair(kernel, 0, input_addr);
  // The generated wrapper stores DDR addresses in 256-byte units.
  set_u32_pair(kernel, 2, weight_addr >> 8);
  set_u32_pair(kernel, 4, output_addr);
  set_u32_pair(kernel, 6, 0);  // no bias
  check_status(kernel.set_regs(8, static_cast<std::uint16_t>(o.m)),
               "set local M");
  check_status(kernel.set_regs(9, static_cast<std::uint16_t>(0)), "set M high");
  set_u32_pair(kernel, 10, p.local_n);
  check_status(kernel.set_regs(12, static_cast<std::uint16_t>(p.local_k)),
               "set local K");
  check_status(kernel.set_regs(13, static_cast<std::uint16_t>(p.local_k / 16)),
               "set K vectors");
  check_status(kernel.set_regs(14, static_cast<std::uint16_t>(0)), "set bias flag");
  check_status(kernel.set_regs(15, static_cast<std::uint16_t>(o.cores)),
               "set core count");
  set_u32_pair(kernel, 16, p.local_k * sizeof(std::uint16_t));
  set_u32_pair(kernel, 18, p.local_n * sizeof(std::uint16_t));
  set_u32_pair(kernel, 20, 0);  // no quantization scale
  check_status(kernel.set_regs(22, static_cast<std::uint16_t>(o.partition)),
               "set partition");
  check_status(kernel.set_regs(23, static_cast<std::uint16_t>(2)),
               "set FP16 dtype mode");
}

std::vector<std::uint16_t> make_canonical_input(const Options &o) {
  const std::size_t elems = static_cast<std::size_t>(o.m) * o.k;
  std::vector<std::uint16_t> x(elems);
  std::uint64_t state = o.seed ^ 0x9e3779b97f4a7c15ULL;
  for (auto &v : x) {
    // [-0.125,0.125], with a nontrivial but bounded pattern.
    const float f = (static_cast<float>(next_u32(state) & 0xffffU) / 65535.0F -
                     0.5F) * 0.25F;
    v = float_to_half(f);
  }
  return x;
}

std::vector<std::uint16_t> make_canonical_weight(const Options &o) {
  const std::size_t elems = static_cast<std::size_t>(o.n) * o.k;
  std::vector<std::uint16_t> w(elems);
  std::uint64_t state = (o.seed + 101) ^ 0xd1b54a32d192ed03ULL;
  for (auto &v : w) {
    const float f = (static_cast<float>(next_u32(state) & 0xffffU) / 65535.0F -
                     0.5F) * 0.20F;
    v = float_to_half(f);
  }
  return w;
}

// Pack one core's SPM input according to the partition contract.
std::vector<std::uint16_t> pack_input_for_core(const std::vector<std::uint16_t> &x,
                                               const Options &o, int core,
                                               const Problem &p) {
  std::vector<std::uint16_t> out(p.input_elems);
  if (o.partition == 1) {
    std::copy(x.begin(), x.end(), out.begin());
  } else {
    for (std::size_t row = 0; row < static_cast<std::size_t>(o.m); ++row) {
      const auto begin = x.begin() + row * static_cast<std::size_t>(o.k) +
                         static_cast<std::size_t>(core) * p.local_k;
      std::copy(begin, begin + p.local_k, out.begin() + row * p.local_k);
    }
  }
  return out;
}

std::vector<float> reference_output(const std::vector<std::uint16_t> &x,
                                    const std::vector<std::uint16_t> &w,
                                    const Options &o) {
  std::vector<float> out(static_cast<std::size_t>(o.m) * o.n, 0.0F);
  for (std::size_t row = 0; row < static_cast<std::size_t>(o.m); ++row)
    for (std::size_t col = 0; col < static_cast<std::size_t>(o.n); ++col) {
      float acc = 0.0F;
      for (std::size_t q = 0; q < static_cast<std::size_t>(o.k); ++q)
        acc += half_to_float(x[row * o.k + q]) * half_to_float(w[col * o.k + q]);
      out[row * o.n + col] = acc;
    }
  return out;
}

struct Verification {
  std::size_t mismatches = 0;
  float max_abs = 0.0F;
  float max_rel = 0.0F;
};

// Read the post-kernel bytes from the directly mapped per-core LocalSPM
// buffers.  This follows the SDK/manual ddr_to_spm/spm_to_ddr boundary:
// memcpy the mapped window and publish it with a host barrier.  CopyToDevice /
// CopyFromDevice on GlobalSPM_t do not select an individual bank on this
// release and can make a multi-core run look valid on core 0 while overwriting
// the other cores' staging windows.
Verification verify_outputs(
    const std::vector<std::unique_ptr<LocalSPM_t>> &readback,
    const Options &o, const Problem &p, const std::vector<float> &expected) {
  __sync_synchronize();
  Verification result;
  const float abs_tol = o.acc32 ? 0.035F : 0.08F;
  const std::size_t out_off = p.output_offset / sizeof(std::uint16_t);
  std::vector<float> accumulated(static_cast<std::size_t>(o.m) * o.n, 0.0F);
  for (int core = 0; core < o.cores; ++core) {
    const auto *base = static_cast<const std::uint16_t *>(
        readback[core]->get_cpu_ptr());
    if (o.partition == 1) {
      for (std::size_t row = 0; row < static_cast<std::size_t>(o.m); ++row)
        for (std::size_t col = 0; col < p.local_n; ++col)
          accumulated[row * o.n + static_cast<std::size_t>(core) * p.local_n + col] =
              half_to_float(base[out_off + row * p.local_n + col]);
    } else {
      for (std::size_t row = 0; row < static_cast<std::size_t>(o.m); ++row)
        for (std::size_t col = 0; col < static_cast<std::size_t>(o.n); ++col) {
          const float v = half_to_float(base[out_off + row * p.local_n + col]);
          // Row partition emits one partial sum per core.  Accumulate in a
          // float host buffer before comparing with the mathematical result.
          accumulated[row * o.n + col] += v;
        }
    }
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float actual = accumulated[i];
    const float diff = std::fabs(actual - expected[i]);
    const float rel = diff / std::max(std::fabs(expected[i]), 1.0e-6F);
    result.max_abs = std::max(result.max_abs, diff);
    result.max_rel = std::max(result.max_rel, rel);
    if (!std::isfinite(actual) || (diff > abs_tol && rel > 0.25F)) {
      if (o.diagnose && result.mismatches < 12) {
        std::fprintf(stderr, "verify_mismatch index=%zu row=%zu col=%zu\n", i,
                     i / static_cast<std::size_t>(o.n),
                     i % static_cast<std::size_t>(o.n));
      }
      ++result.mismatches;
    }
  }
  return result;
}

double percentile(std::vector<double> samples, double q) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const std::size_t index = static_cast<std::size_t>(q * (samples.size() - 1));
  return samples[index];
}

struct TraceSummary {
  std::size_t compute_events = 0;
  double compute_critical_us = 0.0;
};

TraceSummary trace_summary(const std::string &path) {
  std::ifstream in(path);
  if (!in) return {};
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  TraceSummary result;
  constexpr char cat_marker[] = "\"cat\": \"Compute\"";
  std::size_t cat_pos = 0;
  while ((cat_pos = text.find(cat_marker, cat_pos)) != std::string::npos) {
    const std::size_t object_end = text.find('}', cat_pos);
    const std::size_t phase = text.find("\"ph\": \"B\"", cat_pos);
    if (phase != std::string::npos &&
        (object_end == std::string::npos || phase < object_end)) {
      ++result.compute_events;
    }
    cat_pos += sizeof(cat_marker) - 1;
  }
  if (result.compute_events == 0) {
    constexpr char compact_cat_marker[] = "\"cat\":\"Compute\"";
    cat_pos = 0;
    while ((cat_pos = text.find(compact_cat_marker, cat_pos)) !=
           std::string::npos) {
      const std::size_t object_end = text.find('}', cat_pos);
      const std::size_t phase = text.find("\"ph\":\"B\"", cat_pos);
      if (phase != std::string::npos &&
          (object_end == std::string::npos || phase < object_end)) {
        ++result.compute_events;
      }
      cat_pos += sizeof(compact_cat_marker) - 1;
    }
  }
  // The r4 native trace contains one duration_us field on each Compute END
  // event.  Use the maximum rather than a sum: parallel core streams overlap,
  // so the critical-path device time is the meaningful GEMM duration.
  constexpr char marker[] = "\"duration_us\":";
  std::size_t pos = 0;
  double critical = 0.0;
  while ((pos = text.find(marker, pos)) != std::string::npos) {
    pos += sizeof(marker) - 1;
    char *end = nullptr;
    const double d = std::strtod(text.c_str() + pos, &end);
    if (end == text.c_str() + pos) break;
    if (d > 0.0) critical = std::max(critical, d);
    pos = static_cast<std::size_t>(end - text.c_str());
  }
  result.compute_critical_us = critical;
  return result;
}

std::string json_escape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (ch < 0x20U) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
          escaped += buf;
        } else {
          escaped.push_back(static_cast<char>(ch));
        }
    }
  }
  return escaped;
}

void write_raw_samples(const std::string &path, const Options &o,
                       const Problem &p, int n_tile, int m_tile,
                       const std::string &kernel_name, double build_us,
                       const Verification &verification,
                       const std::vector<double> &samples,
                       const TraceSummary &trace) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write raw sample file: " + path);
  out << std::setprecision(17);
  out << "{\n"
      << "  \"schema\": \"rhinoforge-pure-gemm-v1\",\n"
      << "  \"m\": " << o.m << ",\n"
      << "  \"n\": " << o.n << ",\n"
      << "  \"k\": " << o.k << ",\n"
      << "  \"partition\": " << o.partition << ",\n"
      << "  \"cores\": " << o.cores << ",\n"
      << "  \"local_n\": " << p.local_n << ",\n"
      << "  \"local_k\": " << p.local_k << ",\n"
      << "  \"tile_n\": " << n_tile << ",\n"
      << "  \"tile_m\": " << m_tile << ",\n"
      << "  \"acc32\": " << (o.acc32 ? "true" : "false") << ",\n"
      << "  \"kernel\": \"" << json_escape(kernel_name) << "\",\n"
      << "  \"warmup\": " << o.warmup << ",\n"
      << "  \"iterations\": " << o.iters << ",\n"
      << "  \"build_us\": " << build_us << ",\n"
      << "  \"verify_mismatches\": " << verification.mismatches << ",\n"
      << "  \"verify_max_abs\": " << verification.max_abs << ",\n"
      << "  \"verify_max_rel\": " << verification.max_rel << ",\n"
      << "  \"replay_host_us\": [";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) out << ", ";
    out << samples[i];
  }
  out << "],\n"
      << "  \"device_compute_events\": " << trace.compute_events << ",\n"
      << "  \"device_compute_critical_us\": "
      << trace.compute_critical_us << "\n"
      << "}\n";
  if (!out) throw std::runtime_error("failed writing raw sample file: " + path);
}

}  // namespace

int main(int argc, char **argv) try {
  using namespace rhino_lkn;
  const Options o = parse_options(argc, argv);
  const Problem p = make_problem(o);
  bool exact_tile = false;
  const int n_tile = o.tile == 0 ? release_tile_for_shape(o, p, &exact_tile)
                                 : o.tile;
  const std::string kernel_name =
      o.kernel.empty() ? derived_kernel_name(o, n_tile) : o.kernel;

  const auto canonical_x = make_canonical_input(o);
  const auto canonical_w = make_canonical_weight(o);
  const auto transformed_w = swizzle_weight(canonical_w, o.n, o.k, o.partition,
                                            o.cores);
  const auto expected = reference_output(canonical_x, canonical_w, o);

  Program_t program;
  check_status(program.create_with_binary_file(o.ref, kernel_name.c_str()),
               "load operator-library REF/kernel");
  const std::size_t weight_bytes = transformed_w.size() * sizeof(std::uint16_t);
  HostDDR_t weight(weight_bytes, read_write, kStride256B, 256);
  std::memcpy(weight.get_cpu_ptr(), transformed_w.data(), weight_bytes);
  check_status(weight.flush(), "flush weight DDR");

  // LocalSPM_t is the correct owner for a per-core scratch window.  Its CPU
  // mapping is directly coherent with the local RPU memory and its
  // get_rpu_addr() is a core-relative byte offset, exactly what the generated
  // broadcast kernel ABI consumes.  GlobalSPM_t looks tempting but its global
  // address form is *not* valid in this packet ABI.
  std::vector<std::unique_ptr<LocalSPM_t>> spm;
  spm.reserve(o.cores);
  for (int core = 0; core < o.cores; ++core) {
    spm.emplace_back(std::make_unique<LocalSPM_t>(
        p.spm_bytes, read_write, kStride256B, 256, core));
    const auto packed = pack_input_for_core(canonical_x, o, core, p);
    auto *host = static_cast<std::uint8_t *>(spm.back()->get_cpu_ptr());
    std::memcpy(host, packed.data(), p.input_bytes);
    std::fill(host + p.output_offset,
              host + p.output_offset + p.output_elems * sizeof(std::uint16_t),
              static_cast<std::uint8_t>(0x7f));
  }
  // Publish all direct-mapped writes before the first device launch.  This is
  // the same host boundary used by the SDK's ddr_to_spm helper; no DMA copy is
  // part of the timed batch.
  __sync_synchronize();

  const std::uint64_t base_addr = spm.front()->get_rpu_addr();
  if (base_addr > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("SPM address does not fit 32-bit kernel ABI");
  // The release wrapper passes a core-relative LocalSPM offset once in
  // broadcast mode.  For ABI diagnosis, --address-mode "relative" passes
  // only the offset, while the legacy "core0"/"per-core" modes intentionally
  // exercise absolute-address alternatives and are not release candidates.
  auto address_for = [&](int core, std::size_t offset) -> std::uint64_t {
    if (o.address_mode == "relative") return offset;
    const int owner = o.address_mode == "per-core" ? core : 0;
    return static_cast<std::uint64_t>(spm[owner]->get_rpu_addr()) + offset;
  };
  auto checked_u32 = [](std::uint64_t value, const char *what) -> std::uint32_t {
    if (value > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error(std::string(what) + " overflows 32-bit ABI");
    return static_cast<std::uint32_t>(value);
  };
  const std::uint32_t diagnostic_input_addr =
      checked_u32(address_for(0, 0), "SPM input address");
  const std::uint32_t diagnostic_output_addr =
      checked_u32(address_for(0, p.output_offset), "SPM output address");

  std::vector<std::unique_ptr<Kernel_t>> kernels;
  kernels.reserve(o.address_mode == "per-core" ? o.cores : 1);
  if (o.address_mode == "per-core") {
    for (int core = 0; core < o.cores; ++core) {
      auto k = std::make_unique<Kernel_t>(program, kernel_name.c_str());
      k->set_name((kernel_name + "_c" + std::to_string(core)).c_str());
      k->set_op_type("matmul");
      const auto in_addr = checked_u32(address_for(core, 0), "SPM input address");
      const auto out_addr = checked_u32(address_for(core, p.output_offset),
                                        "SPM output address");
      set_parallel_linear_regs(*k, in_addr, weight.get_rpu_addr(), out_addr, p, o);
      if (o.typed_weight)
        check_status(k->set_regs(2, weight), "set typed weight register");
      kernels.emplace_back(std::move(k));
    }
  } else {
    auto k = std::make_unique<Kernel_t>(program, kernel_name.c_str());
    k->set_name(kernel_name.c_str());
    k->set_op_type("matmul");
    set_parallel_linear_regs(*k, diagnostic_input_addr, weight.get_rpu_addr(),
                             diagnostic_output_addr, p, o);
    if (o.typed_weight)
      check_status(k->set_regs(2, weight), "set typed weight register");
    kernels.emplace_back(std::move(k));
  }

  const std::vector<std::uint8_t> core_ids = [&] {
    std::vector<std::uint8_t> ids;
    for (int c = 0; c < o.cores; ++c) ids.push_back(static_cast<std::uint8_t>(c));
    return ids;
  }();
  const std::vector<std::uint16_t> grid = {
      static_cast<std::uint16_t>((p.local_n + static_cast<std::size_t>(n_tile) - 1) /
                                 static_cast<std::size_t>(n_tile)),
      static_cast<std::uint16_t>((o.m + static_cast<std::uint64_t>(default_m_tile(o.acc32, n_tile)) - 1) /
                                 static_cast<std::uint64_t>(default_m_tile(o.acc32, n_tile))),
      1};

  Queue_t queue(static_cast<std::uint8_t>(o.cores));
  queue.set_broadcast_mode(o.broadcast);
  queue.set_enable_hw_perf(false);
  // Freeze queue controls before add/build.  The timing batch is deliberately
  // a stable Launch batch; changing a queue knob after BUILD would make the
  // replay boundary ambiguous and is not a valid optimization comparison.
  queue.set_flush_icache(false);
  const auto build_begin = std::chrono::steady_clock::now();
  if (o.submit_mode != "immediate") {
    if (o.address_mode == "per-core") {
      for (int core = 0; core < o.cores; ++core) {
        const std::vector<std::uint8_t> one_core = {
            static_cast<std::uint8_t>(core)};
        const std::uint32_t status =
            o.submit_mode == "mutable"
                ? queue.add_kernel_mutable(*kernels[core], grid, one_core)
                : queue.add_kernel(*kernels[core], grid, one_core);
        check_status(status, "add per-core GEMM");
      }
    } else {
      const std::uint32_t status =
          o.submit_mode == "mutable"
              ? queue.add_kernel_mutable(*kernels.front(), grid, core_ids)
              : queue.add_kernel(*kernels.front(), grid, core_ids);
      check_status(status, "add GEMM");
    }
    check_status(queue.build_batch(), "build GEMM batch");
  }
  const auto build_end = std::chrono::steady_clock::now();
  const double build_us =
      std::chrono::duration<double, std::micro>(build_end - build_begin).count();
  auto enqueue_once = [&](const char *what) {
    if (o.submit_mode == "immediate") {
      if (o.address_mode == "per-core") {
        for (int core = 0; core < o.cores; ++core) {
          const std::vector<std::uint8_t> one_core = {
              static_cast<std::uint8_t>(core)};
          check_status(queue.enqueue_kernel(*kernels[core], grid, one_core),
                       what);
        }
      } else {
        check_status(queue.enqueue_kernel(*kernels.front(), grid, core_ids),
                     what);
      }
    } else {
      check_status(queue.enqueue_batch(true), what);
    }
  };
  enqueue_once("correctness replay");

  Verification verification;
  if (o.verify) {
    verification = verify_outputs(spm, o, p, expected);
    std::printf("verify mismatches=%zu max_abs=%.7g max_rel=%.7g\n",
                verification.mismatches, verification.max_abs,
                verification.max_rel);
    if (verification.mismatches != 0) return 2;
  }

  for (std::uint64_t i = 0; i < o.warmup; ++i)
    enqueue_once("warmup replay");
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(o.iters));
  for (std::uint64_t i = 0; i < o.iters; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    enqueue_once("timed replay");
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const double p50 = percentile(samples, 0.50);
  const double p95 = percentile(samples, 0.95);
  const double mean = samples.empty() ? 0.0 : sum / samples.size();
  const double flops = 2.0 * static_cast<double>(o.m) * o.n * o.k;
  std::printf(
      "gemm_contract M=%llu N=%llu K=%llu partition=%d cores=%d local_M=%llu "
      "local_N=%zu local_K=%zu tile_n=%d tile_m=%d tile_source=%s acc32=%d kernel=%s\n",
      static_cast<unsigned long long>(o.m), static_cast<unsigned long long>(o.n),
      static_cast<unsigned long long>(o.k), o.partition, o.cores,
      static_cast<unsigned long long>(o.m), p.local_n, p.local_k, n_tile,
      default_m_tile(o.acc32, n_tile), o.tile == 0 ? (exact_tile ? "exact" : "fallback") : "explicit",
      o.acc32 ? 1 : 0, kernel_name.c_str());
  std::printf("launch_abi=local_spm_%s broadcast=%d cores=%d\n",
              o.address_mode.c_str(), o.broadcast ? 1 : 0, o.cores);
  // This is the Rhino Launch batch lifecycle, not a RhinoForge Graph.  Do not
  // print Graph cache/replay invariants here: those require the PyTorch Graph
  // executor and are measured by a separate harness.
  std::printf(
      "launch_batch submit_mode=%s build_batch_count=%d enqueue_count=%llu "
      "build_us=%.3f warmup=%llu iters=%llu graph_lifecycle=not_applicable\n",
      o.submit_mode.c_str(), o.submit_mode == "immediate" ? 0 : 1,
      static_cast<unsigned long long>(1 + o.warmup + o.iters), build_us,
      static_cast<unsigned long long>(o.warmup),
      static_cast<unsigned long long>(o.iters));
  std::printf(
      "replay_host_us min=%.3f p50=%.3f p95=%.3f mean=%.3f max=%.3f "
      "flops=%.0f achieved_TFLOP=%.6f\n",
      samples.empty() ? 0.0 : *std::min_element(samples.begin(), samples.end()),
      p50, p95, mean,
      samples.empty() ? 0.0 : *std::max_element(samples.begin(), samples.end()),
      flops, p50 > 0.0 ? flops / (p50 * 1.0e6) : 0.0);

  if (!o.trace.empty()) {
    // Trace a fresh build so enabling perf logging cannot contaminate the
    // measured replay samples.
    std::vector<std::unique_ptr<Kernel_t>> trace_kernels;
    trace_kernels.reserve(o.address_mode == "per-core" ? o.cores : 1);
    if (o.address_mode == "per-core") {
      for (int core = 0; core < o.cores; ++core) {
        auto k = std::make_unique<Kernel_t>(program, kernel_name.c_str());
        k->set_name((kernel_name + "_trace_c" + std::to_string(core)).c_str());
        k->set_op_type("matmul");
        set_parallel_linear_regs(
            *k, checked_u32(address_for(core, 0), "trace SPM input address"),
            weight.get_rpu_addr(),
            checked_u32(address_for(core, p.output_offset),
                        "trace SPM output address"),
            p, o);
        if (o.typed_weight)
          check_status(k->set_regs(2, weight),
                       "set typed trace weight register");
        trace_kernels.emplace_back(std::move(k));
      }
    } else {
      auto k = std::make_unique<Kernel_t>(program, kernel_name.c_str());
      k->set_name((kernel_name + "_trace").c_str());
      k->set_op_type("matmul");
      set_parallel_linear_regs(*k, diagnostic_input_addr, weight.get_rpu_addr(),
                               diagnostic_output_addr, p, o);
      if (o.typed_weight)
        check_status(k->set_regs(2, weight),
                     "set typed trace weight register");
      trace_kernels.emplace_back(std::move(k));
    }
    Queue_t trace_queue(static_cast<std::uint8_t>(o.cores));
    trace_queue.set_broadcast_mode(o.broadcast);
    trace_queue.set_enable_hw_perf(true);
    trace_queue.set_flush_icache(false);
    if (o.address_mode == "per-core") {
      for (int core = 0; core < o.cores; ++core) {
        const std::vector<std::uint8_t> one_core = {
            static_cast<std::uint8_t>(core)};
        check_status(trace_queue.add_kernel(*trace_kernels[core], grid, one_core),
                     "add per-core trace GEMM");
      }
    } else {
      check_status(trace_queue.add_kernel(*trace_kernels.front(), grid, core_ids),
                   "add trace GEMM");
    }
    check_status(trace_queue.build_batch(), "build trace GEMM");
    check_status(trace_queue.enqueue_batch(true), "trace replay");
    check_status(trace_queue.dump_hw_perf_chrome(o.trace.c_str()),
                 "dump HW trace");
    const TraceSummary trace = trace_summary(o.trace);
    const double device_us = trace.compute_critical_us;
    const double device_tflop =
        device_us > 0.0 ? flops / (device_us * 1.0e6) : 0.0;
    std::printf("device_trace=%s compute_events=%zu "
                "device_compute_critical_us=%.3f "
                "device_achieved_TFLOP=%.6f\n",
                o.trace.c_str(), trace.compute_events, device_us, device_tflop);
    if (!o.raw.empty())
      write_raw_samples(o.raw, o, p, n_tile,
                        default_m_tile(o.acc32, n_tile), kernel_name, build_us,
                        verification, samples, trace);
  } else if (!o.raw.empty()) {
    write_raw_samples(o.raw, o, p, n_tile,
                      default_m_tile(o.acc32, n_tile), kernel_name, build_us,
                      verification, samples, TraceSummary{});
  }
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return 1;
}
