#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/mapping/point_block.hpp"

namespace ovo::core::mapping {

// The dense point cloud: one PointBlock per keyframe that created points.
// Owns the PointId counter: ids start at 0, are consecutive and never reused.
class GeometricMap {
public:
    // Adds the points `keyframe` created and gives them the next free ids. No points, no block.
    void addBlock(FrameId keyframe, std::vector<Eigen::Vector3f> xyzWorld) {
        if (xyzWorld.empty()) { return; }
        const std::size_t n = xyzWorld.size();
        assert(n <= std::numeric_limits<std::uint32_t>::max() - nextId_.value());  // id overflow
        blocks_.emplace_back(keyframe, nextId_, std::move(xyzWorld));
        nextId_ = PointId{nextId_.value() + static_cast<std::uint32_t>(n)};
        numPoints_ += n;
    }

    [[nodiscard]] std::span<const PointBlock> blocks() const noexcept { return blocks_; }
    [[nodiscard]] std::size_t numPoints() const noexcept { return numPoints_; }
    [[nodiscard]] bool empty() const noexcept { return numPoints_ == 0; }

private:
    std::vector<PointBlock> blocks_;
    PointId nextId_{0};
    std::size_t numPoints_ = 0;
};

}  // namespace ovo::core::mapping
