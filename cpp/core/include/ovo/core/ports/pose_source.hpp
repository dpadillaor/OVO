#pragma once

#include <optional>

#include "ovo/core/geometry/pose.hpp"
#include "ovo/core/ports/frame_source.hpp"

namespace ovo::core::ports {

// Input port: the camera pose of each frame. Ground truth today, a SLAM (ORB-SLAM3) later.
class PoseSource {
public:
    virtual ~PoseSource() = default;

    // Camera-to-world pose of `frame`, or nothing if there is no valid one (NaN in the GT, tracking lost...).
    // Takes the whole frame and is not const: a SLAM computes the pose from the images and updates its
    // state, so it needs the frames in order. Ground truth only looks at frame.id.
    [[nodiscard]] virtual std::optional<geometry::Pose> pose(const Frame& frame) = 0;
};

}  // namespace ovo::core::ports
