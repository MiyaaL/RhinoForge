#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rpu_graph_detail {

// Cold, bounded overrides. Never interpret zero or malformed input as
// "unlimited"; the planner independently clamps each resource to SDK headroom.
inline size_t parse_segment_limit(const char* text, const char* name,
                                  size_t default_value, size_t maximum) {
    if (text == nullptr) return default_value;
    const std::string error = std::string(name) +
        " must be a decimal integer in [1, " + std::to_string(maximum) + "]";
    size_t value = 0;
    if (*text == '\0') {
        throw std::invalid_argument(error);
    }
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9' || value > maximum / 10) {
            throw std::invalid_argument(error);
        }
        value = value * 10 + static_cast<size_t>(*p - '0');
        if (value > maximum) {
            throw std::invalid_argument(error);
        }
    }
    if (value == 0) {
        throw std::invalid_argument(error);
    }
    return value;
}

inline size_t parse_segment_entry_budget(const char* text) {
    return parse_segment_limit(text, "RPU_GRAPH_MAX_SEGMENT_ENTRIES", 8192, 32768);
}

inline size_t effective_segment_entry_budget(size_t requested, size_t sdk_entries) {
    return std::max<size_t>(1, std::min(requested, sdk_entries / 2));
}

}  // namespace rpu_graph_detail
