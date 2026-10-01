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

    // 1. Map points this frame already sees.
    std::vector<geometry::PointPixelMatch> matches;
    if (!map_.empty()) {
        // Depth range widened by the tolerance: a point up to maxDepthError in front of the nearest
        // (or behind the farthest) measurement can still match. Also keeps near < far when min == max.
        const float nearDepth = std::max(range->min - maxDepthError_, range->min * 0.5f);
        const float farDepth = range->max + maxDepthError_;
        const geometry::Frustum frustum(camera_, cameraPose, nearDepth, farDepth);

        std::vector<Eigen::Vector3f> candidates;
        for (const PointBlock& block : map_.blocks()) {
            for (const Eigen::Vector3f& pointWorld : block.xyzWorld()) {
                if (frustum.contains(pointWorld)) { candidates.push_back(pointWorld); }
            }
        }
        matches = matcher_.match(candidates, cameraPose, depth);
    }

    // 2. Pixels nothing covers become new points.
    const std::vector<Pixel> freePixels = coverage_.freePixels(matches, depth);
    std::vector<Eigen::Vector3f> newPoints;
    newPoints.reserve(freePixels.size());
    for (const Pixel& p : freePixels) {
        const Eigen::Vector2f pixel{static_cast<float>(p.u), static_cast<float>(p.v)};
        newPoints.push_back(cameraPose.camToWorld(camera_.unproject(pixel, depth.at(p.u, p.v))));
    }
    map_.addBlock(frameId, std::move(newPoints));
}

}  // namespace ovo::core::mapping
