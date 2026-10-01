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

class GeometricMap {
public:
    
    void addBlock(FrameId keyframe, std::vector<Eigen::Vector3f> xyzWorld) {
        if (xyzWorld.empty()) { return; }
        const std::size_t n = xyzWorld.size();
        assert(n <= std::numeric_limits<std::uint32_t>::max() - nextId_.value());
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
