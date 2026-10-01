#pragma once

#include <cstddef>

#include "ovo/core/common/image.hpp"
#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"

namespace ovo::core::ports {

struct Frame {
    FrameId id;
    Image<float> depth;  // metres, 0 = no measurement
};

class FrameSource {
public:
    virtual ~FrameSource() = default;

    [[nodiscard]] virtual const geometry::PinholeCamera& camera() const = 0;
    [[nodiscard]] virtual std::size_t size() const = 0;
    // Loads frame `index` (< size()). Throws std::runtime_error if it cannot be read.
    [[nodiscard]] virtual Frame frame(std::size_t index) const = 0;
};

}  // namespace ovo::core::ports
