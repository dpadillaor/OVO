#include "ovo/core/mapping/dense_mapper.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "ovo/core/geometry/frustum.hpp"

namespace ovo::core::mapping {

namespace {

struct DepthRange {
    float min;
    float max;
};

// Smallest and largest measured depth (> 0, finite). Nothing if no pixel has a measurement.
std::optional<DepthRange> validDepthRange(const Image<float>& depth) {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    DepthRange range{kInf, 0.f};
    const std::size_t numPixels = std::size_t{depth.width()} * depth.height();
    for (std::size_t i = 0; i < numPixels; ++i) {
        const float d = depth.data()[i];
        if (d > 0.f && d < kInf) {
            range.min = std::min(range.min, d);
            range.max = std::max(range.max, d);
        }
    }
    if (!(range.max > 0.f)) { return std::nullopt; }
    return range;
}

// Map points inside the frustum, copied into one contiguous array for the matcher.
std::vector<Eigen::Vector3f> pointsInside(const geometry::Frustum& frustum, const GeometricMap& map) {
    std::vector<Eigen::Vector3f> inside;
    for (const PointBlock& block : map.blocks()) {
        for (const Eigen::Vector3f& pointWorld : block.xyzWorld()) {
            if (frustum.contains(pointWorld)) { inside.push_back(pointWorld); }
        }
    }
    return inside;
}

// Each pixel at its measured depth, in world coordinates.
std::vector<Eigen::Vector3f> unprojectToWorld(const std::vector<Pixel>& pixels, const Image<float>& depth,
                                              const geometry::PinholeCamera& camera,
                                              const geometry::Pose& cameraPose) {
    std::vector<Eigen::Vector3f> pointsWorld;
    pointsWorld.reserve(pixels.size());
    for (const Pixel& p : pixels) {
        const Eigen::Vector2f pixel{static_cast<float>(p.u), static_cast<float>(p.v)};
        pointsWorld.push_back(cameraPose.camToWorld(camera.unproject(pixel, depth.at(p.u, p.v))));
    }
    return pointsWorld;
}

}  // namespace

DenseMapper::DenseMapper(const geometry::PinholeCamera& camera, const Config& config)
    : camera_(camera),
      maxDepthError_(config.maxDepthError),
      matcher_(camera, config.maxDepthError),
      coverage_(config.dilationSize, config.step) {}

void DenseMapper::map(FrameId frameId, const Image<float>& depth, const geometry::Pose& cameraPose) {
    assert(depth.width() == camera_.width() && depth.height() == camera_.height());

    const std::optional<DepthRange> range = validDepthRange(depth);
    if (!range) { return; }  // no measurement at all: nothing to match, nothing to create

    std::vector<geometry::PointPixelMatch> matches;  // map points this frame already sees
    if (!map_.empty()) {
        // Widened by the match tolerance: a point up to maxDepthError in front of the nearest
        // (or behind the farthest) measurement can still match.
        const auto frustum =
            geometry::Frustum::withDepthMargin(camera_, cameraPose, range->min, range->max, maxDepthError_);
        matches = matcher_.match(pointsInside(frustum, map_), cameraPose, depth);
    }

    const std::vector<Pixel> freePixels = coverage_.freePixels(matches, depth);  // where nothing covers
    map_.addBlock(frameId, unprojectToWorld(freePixels, depth, camera_, cameraPose));
}

}  // namespace ovo::core::mapping
