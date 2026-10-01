#pragma once

#include <cstdint>

#include "ovo/core/common/image.hpp"
#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/geometry/point_pixel_matcher.hpp"
#include "ovo/core/geometry/pose.hpp"
#include "ovo/core/mapping/coverage.hpp"
#include "ovo/core/mapping/geometric_map.hpp"

namespace ovo::core::mapping {

// Builds the dense point cloud from depth and camera pose (the Python VanillaMapper.map).
// Each call: find which map points this frame already sees, then create points where nothing covers.
// How often to call it (Python map_every) is the caller's decision.
class DenseMapper {
public:
    struct Config {
        float maxDepthError = 0.03f;     // metres, matcher tolerance
        std::uint32_t dilationSize = 3;  // Python k_pooling
        std::uint32_t step = 2;          // Python downscale
    };

    // Throws std::invalid_argument on a bad config (checked by PointPixelMatcher and Coverage).
    DenseMapper(const geometry::PinholeCamera& camera, const Config& config);

    void map(FrameId frameId, const Image<float>& depth, const geometry::Pose& cameraPose);

    [[nodiscard]] const GeometricMap& geometricMap() const noexcept { return map_; }

private:
    geometry::PinholeCamera camera_;
    float maxDepthError_;
    geometry::PointPixelMatcher matcher_;
    Coverage coverage_;
    GeometricMap map_;
};

}  // namespace ovo::core::mapping
