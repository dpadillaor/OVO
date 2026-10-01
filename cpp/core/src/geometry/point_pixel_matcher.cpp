#include "ovo/core/geometry/point_pixel_matcher.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ovo::core::geometry {

PointPixelMatcher::PointPixelMatcher(const PinholeCamera& camera, float maxDepthError)
    : camera_(camera), maxDepthError_(maxDepthError) {
    if (maxDepthError <= 0.f || !std::isfinite(maxDepthError)) {
        throw std::invalid_argument("PointPixelMatcher: maxDepthError must be positive and finite");
    }
}

std::vector<PointPixelMatch> PointPixelMatcher::match(std::span<const Eigen::Vector3f> pointsWorld,
                                                      const Pose& cameraPose, const Image<float>& depth) const {
    assert(camera_.height() == depth.height());
    assert(camera_.width() == depth.width());

    const long width = static_cast<long>(camera_.width());
    const long height = static_cast<long>(camera_.height());

    std::vector<PointPixelMatch> matches;
    for (std::size_t i = 0; i < pointsWorld.size(); ++i) {
        const Eigen::Vector3f pointCam = cameraPose.worldToCam(pointsWorld[i]);
        const Eigen::Vector2f pixel = camera_.project(pointCam);

        const long u = std::lround(pixel.x());
        const long v = std::lround(pixel.y());
        if (u < 0 || u >= width || v < 0 || v >= height) { continue; }

        const auto pixelU = static_cast<std::uint32_t>(u);
        const auto pixelV = static_cast<std::uint32_t>(v);
        const float measured = depth.at(pixelU, pixelV);
        if (!(measured > 0.f)) { continue; }

        if (std::abs(pointCam.z() - measured) < maxDepthError_) {
            matches.push_back({static_cast<std::uint32_t>(i), pixelU, pixelV});
        }
    }
    return matches;
}

}  // namespace ovo::core::geometry
