#include "ovo/core/mapping/coverage.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace ovo::core::mapping {

namespace {

// Is any pixel of the window around (u, v), clipped to the image, matched or without depth?
// Same result as the Python ~maxpool(~mask): maxpool pads with -inf, so outside the image never blocks.
bool windowBlocked(const Image<std::uint8_t>& matched, const Image<float>& depth, std::uint32_t u, std::uint32_t v,
                   std::uint32_t radius) {
    const std::uint32_t uMin = u > radius ? u - radius : 0;  // unsigned: never subtract below 0
    const std::uint32_t vMin = v > radius ? v - radius : 0;
    const std::uint32_t uMax = std::min(u + radius, depth.width() - 1);
    const std::uint32_t vMax = std::min(v + radius, depth.height() - 1);
    for (std::uint32_t wv = vMin; wv <= vMax; ++wv) {
        for (std::uint32_t wu = uMin; wu <= uMax; ++wu) {
            if (matched.at(wu, wv) != 0 || !(depth.at(wu, wv) > 0.f)) { return true; }
        }
    }
    return false;
}

}  // namespace

Coverage::Coverage(std::uint32_t dilationSize, std::uint32_t step) : radius_(dilationSize / 2), step_(step) {
    if (dilationSize % 2 == 0) { throw std::invalid_argument("Coverage: dilationSize must be odd (1 = no dilation)"); }
    if (step == 0) { throw std::invalid_argument("Coverage: step must be >= 1"); }
}

std::vector<Pixel> Coverage::freePixels(std::span<const geometry::PointPixelMatch> matches,
                                        const Image<float>& depth) const {
    Image<std::uint8_t> matched(depth.width(), depth.height());
    for (const geometry::PointPixelMatch& m : matches) {
        matched.at(m.u, m.v) = 1;  // at() asserts the match lies inside the image
    }

    // Only the step grid is checked, each pixel against its window: same as dilating the whole
    // image and then subsampling, without dilating pixels that are never read.
    std::vector<Pixel> free;
    for (std::uint32_t v = 0; v < depth.height(); v += step_) {
        for (std::uint32_t u = 0; u < depth.width(); u += step_) {
            if (!windowBlocked(matched, depth, u, v, radius_)) { free.push_back({u, v}); }
        }
    }
    return free;
}

}  // namespace ovo::core::mapping
