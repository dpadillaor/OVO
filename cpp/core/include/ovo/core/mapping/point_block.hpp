#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "ovo/core/common/strong_id.hpp"

namespace ovo::core::mapping {

// The points one keyframe created. Immutable. Ids are consecutive: point i has id firstId + i.
class PointBlock {
public:
    PointBlock(FrameId keyframe, PointId firstId, std::vector<Eigen::Vector3f> xyzWorld)
        : keyframe_(keyframe), firstId_(firstId), xyzWorld_(std::move(xyzWorld)) {}

    [[nodiscard]] FrameId keyframe() const noexcept { return keyframe_; }
    [[nodiscard]] std::size_t size() const noexcept { return xyzWorld_.size(); }
    [[nodiscard]] PointId pointId(std::size_t i) const noexcept {
        assert(i < size());
        return PointId{firstId_.value() + static_cast<std::uint32_t>(i)};
    }
    [[nodiscard]] std::span<const Eigen::Vector3f> xyzWorld() const noexcept { return xyzWorld_; }

private:
    FrameId keyframe_;
    PointId firstId_;
    std::vector<Eigen::Vector3f> xyzWorld_;
};

}  // namespace ovo::core::mapping
