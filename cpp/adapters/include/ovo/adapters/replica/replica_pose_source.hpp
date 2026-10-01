#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

#include "ovo/core/geometry/pose.hpp"
#include "ovo/core/ports/pose_source.hpp"

namespace ovo::adapters::replica {

// Ground-truth poses of a Replica scene: <sceneDir>/traj.txt, one 4x4 camera-to-world matrix per line
// (16 numbers, row by row). Line i is frame i.
class ReplicaPoseSource : public core::ports::PoseSource {
public:
    // Reads the whole file. Throws std::runtime_error if it is missing or a line does not have 16 numbers.
    explicit ReplicaPoseSource(const std::filesystem::path& sceneDir);

    [[nodiscard]] std::optional<core::geometry::Pose> pose(const core::ports::Frame& frame) override;

    [[nodiscard]] std::size_t size() const noexcept { return poses_.size(); }

private:
    std::vector<std::optional<core::geometry::Pose>> poses_;  // nullopt: non-finite values in that line
};

}  // namespace ovo::adapters::replica
