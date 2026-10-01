#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "ovo/core/common/image.hpp"
#include "ovo/core/geometry/point_pixel_matcher.hpp"

namespace ovo::core::mapping {

struct Pixel {
    std::uint32_t u;
    std::uint32_t v;
};

// Where to create new points this frame. v1 copies the Python VanillaMapper (k_pooling + downscale):
// a pixel is free if no pixel in its dilationSize x dilationSize window is matched or lacks depth,
// and only pixels with u and v multiple of step are considered.
class Coverage {
public:
    Coverage(std::uint32_t dilationSize, std::uint32_t step);

    // Free pixels in row order (v, then u).
    [[nodiscard]] std::vector<Pixel> freePixels(std::span<const geometry::PointPixelMatch> matches,
                                                const Image<float>& depth) const;

private:
    std::uint32_t radius_;  // dilationSize / 2
    std::uint32_t step_;
};

}  // namespace ovo::core::mapping
