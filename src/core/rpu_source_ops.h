#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rpu_source_ops {
struct Configuration { bool gelu = false, layernorm = false, rmsnorm = false; };
inline Configuration parse(const char* value) {
    Configuration config;
    if (!value || !*value) return config;
    const std::string text(value);
    std::size_t start = 0;
    while (true) {
        const auto comma = text.find(',', start);
        const auto token = text.substr(start, comma == std::string::npos ? comma : comma - start);
        if (token == "gelu") config.gelu = true;
        else if (token == "layernorm") config.layernorm = true;
        else if (token == "rmsnorm") config.rmsnorm = true;
        else throw std::invalid_argument("RPU_SOURCE_OPS accepts only gelu,layernorm,rmsnorm or an empty value");
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return config;
}
// Immutable for this process, including Graph BUILD/REPLAY and cache teardown.
inline const Configuration& configuration() {
    static const auto config = parse(std::getenv("RPU_SOURCE_OPS"));
    return config;
}
}  // namespace rpu_source_ops

#ifdef RPU_HAS_SOURCE_OPS
#include <rpu_ops/kernel_abi.hpp>
#include <rpu_ops/library.hpp>
#include <optional>

namespace rpu_source_ops {
// These paths are opt-in source integrations. Multi-core and unsupported
// layouts stay on the existing reference wrappers until separately validated.
inline std::optional<rpu_ops::abi::LaunchPlan> gelu_plan(
    bool enabled, std::uint32_t input, std::uint32_t output,
    std::int64_t elements, bool tanh, int cores) {
    if (!enabled || cores != 1 || elements <= 0 || elements > UINT32_MAX) return {};
    try {
        return rpu_ops::abi::gelu({rpu_ops::abi::local_spm_offset(output),
            rpu_ops::abi::local_spm_offset(input), static_cast<std::uint32_t>(elements),
            tanh ? rpu_ops::abi::GeluMode::Tanh : rpu_ops::abi::GeluMode::Erf});
    } catch (const std::invalid_argument&) { return {}; }
}
inline std::optional<rpu_ops::abi::LaunchPlan> layernorm_plan(
    bool enabled, std::uint32_t input, std::uint32_t output,
    std::uint32_t weight, std::uint32_t bias, std::int64_t rows,
    std::int64_t cols, double epsilon, bool has_skip, int cores) {
    if (!enabled || has_skip || cores != 1 || rows <= 0 || rows > 65535 ||
        cols <= 0 || cols > 65520) return {};
    try {
        return rpu_ops::abi::layernorm({rpu_ops::abi::local_spm_offset(output),
            rpu_ops::abi::local_spm_offset(input), rpu_ops::abi::local_spm_offset(weight),
            rpu_ops::abi::local_spm_offset(bias), static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols), true, static_cast<float>(epsilon)});
    } catch (const std::invalid_argument&) { return {}; }
}
#ifdef RPU_OPS_HAS_RMSNORM
// Diagnostic eight-core profile. The caller must reject a missing plan when
// selected: silently falling back would invalidate the stability experiment.
inline std::optional<rpu_ops::abi::LaunchPlan> rmsnorm_plan(
    bool enabled, std::uint32_t input, std::uint32_t output,
    std::uint32_t weight, std::int64_t rows, std::int64_t cols, double epsilon) {
    if (!enabled || rows <= 0 || rows > 768 || rows % 16 != 0 ||
        (cols != 128 && cols != 256 && cols != 1024 && cols != 2048)) return {};
    try {
        return rpu_ops::abi::rmsnorm({rpu_ops::abi::local_spm_offset(output),
            rpu_ops::abi::local_spm_offset(input), rpu_ops::abi::local_spm_offset(weight),
            static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols),
            static_cast<float>(epsilon)});
    } catch (const std::invalid_argument&) { return {}; }
}
#endif
}  // namespace rpu_source_ops
#endif
