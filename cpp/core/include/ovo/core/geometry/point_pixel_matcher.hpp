#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "ovo/core/common/image.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/geometry/pose.hpp"

namespace ovo::core::geometry {

// A map point seen by the sensor: it projects inside the image and its depth agrees with the measured one.
struct PointPixelMatch {
    std::uint32_t pointIndex;  // position in pointsWorld, not a PointId
    std::uint32_t u;
    std::uint32_t v;
};

// Which candidate points does the sensor actually see in this frame? Stateless.
// Runs once per frame; the per-point loop lives inside match() (in the .cpp).
class PointPixelMatcher {
public:
    PointPixelMatcher(const PinholeCamera& camera, float maxDepthError);

    [[nodiscard]] std::vector<PointPixelMatch> match(std::span<const Eigen::Vector3f> pointsWorld,
                                                     const Pose& cameraPose, const Image<float>& depth) const;

private:
    PinholeCamera camera_;  // copy: cheap, and no dangling reference
    float maxDepthError_;   // metres
};

}  // namespace ovo::core::geometry
