#include "ovo/core/geometry/frustum.hpp"

#include <cassert>
#include <cstddef>

#include <Eigen/Geometry>

namespace ovo::core::geometry {

namespace {

// Index of each face in planes_.
enum Face : std::size_t { kNear, kFar, kLeft, kRight, kTop, kBottom };

// The 4 image corners, unprojected at `depth` and taken to world.
// Order: top-left, top-right, bottom-left, bottom-right.
std::array<Eigen::Vector3f, 4> imageCornersInWorld(const PinholeCamera& camera, const Pose& cameraPose,
                                                   float depth) {
    const float w = static_cast<float>(camera.width());
    const float h = static_cast<float>(camera.height());
    const std::array<Eigen::Vector2f, 4> pixels{{{0.f, 0.f}, {w, 0.f}, {0.f, h}, {w, h}}};

    std::array<Eigen::Vector3f, 4> corners;
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        corners[i] = cameraPose.camToWorld(camera.unproject(pixels[i], depth));
    }
    return corners;
}

// Mean of the 8 corners. The frustum is convex, so this point is always inside:
// used to orient every normal.
Eigen::Vector3f centroid(const std::array<Eigen::Vector3f, 4>& nearCorners,
                         const std::array<Eigen::Vector3f, 4>& farCorners) {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (std::size_t i = 0; i < nearCorners.size(); ++i) {
        sum += nearCorners[i] + farCorners[i];
    }
    return sum / 8.f;
}

// Plane through p1, p2, p3 as (n.x, n.y, n.z, d), n unit length and facing pointInside:
// n . p + d >= 0 on the inner side.
Eigen::Vector4f planeFacingInward(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3,
                                  const Eigen::Vector3f& pointInside) {
    const Eigen::Vector3f normal = (p2 - p1).cross(p3 - p1).normalized();
    const float d = -normal.dot(p1);
    Eigen::Vector4f plane{normal.x(), normal.y(), normal.z(), d};
    if (plane.head<3>().dot(pointInside) + plane.w() < 0.f) { plane = -plane; }
    return plane;
}

}  // namespace

Frustum::Frustum(const PinholeCamera& camera, const Pose& cameraPose, float minDepth, float maxDepth) {
    assert(0.f < minDepth && minDepth < maxDepth);  // positive form: NaN fails too

    const auto nearCorners = imageCornersInWorld(camera, cameraPose, minDepth);
    const auto farCorners = imageCornersInWorld(camera, cameraPose, maxDepth);

    const Eigen::Vector3f centre = centroid(nearCorners, farCorners);

    planes_[kNear] = planeFacingInward(nearCorners[0], nearCorners[1], nearCorners[2], centre);
    planes_[kFar] = planeFacingInward(farCorners[0], farCorners[1], farCorners[2], centre);
    planes_[kLeft] = planeFacingInward(nearCorners[0], nearCorners[2], farCorners[0], centre);
    planes_[kRight] = planeFacingInward(nearCorners[1], nearCorners[3], farCorners[1], centre);
    planes_[kTop] = planeFacingInward(nearCorners[0], nearCorners[1], farCorners[0], centre);
    planes_[kBottom] = planeFacingInward(nearCorners[2], nearCorners[3], farCorners[2], centre);
}

}  // namespace ovo::core::geometry
