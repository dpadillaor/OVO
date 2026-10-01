#pragma once

#include <array>

#include <Eigen/Core>

#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/geometry/pose.hpp"

namespace ovo::core::geometry {

class Frustum {
public:
    Frustum(const PinholeCamera& camera, const Pose& cameraPose, float minDepth, float maxDepth);

    // Depth range [minDepth - margin, maxDepth + margin]. If the margin would put near at or below 0,
    // near is minDepth / 2. Valid when minDepth == maxDepth (margin > 0 keeps near < far).
    [[nodiscard]] static Frustum withDepthMargin(const PinholeCamera& camera, const Pose& cameraPose,
                                                 float minDepth, float maxDepth, float margin);

    [[nodiscard]] bool contains(const Eigen::Vector3f& pointWorld) const noexcept {
        for (const Eigen::Vector4f& plane : planes_) {
            if (plane.head<3>().dot(pointWorld) + plane.w() < 0.f) { return false; }
        }
        return true;
    }

private:
    std::array<Eigen::Vector4f, 6> planes_;
};

}  // namespace ovo::core::geometry
