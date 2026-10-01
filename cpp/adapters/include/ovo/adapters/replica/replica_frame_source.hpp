#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/ports/frame_source.hpp"

namespace ovo::adapters::replica {

class ReplicaFrameSource : public core::ports::FrameSource {
public:
    ReplicaFrameSource(const std::filesystem::path& sceneDir, const core::geometry::PinholeCamera& camera,
                       float depthScale);

    [[nodiscard]] const core::geometry::PinholeCamera& camera() const override { return camera_; }
    [[nodiscard]] std::size_t size() const override { return depthPaths_.size(); }
    [[nodiscard]] core::ports::Frame frame(std::size_t index) const override;

private:
    core::geometry::PinholeCamera camera_;
    float depthScale_;
    std::vector<std::filesystem::path> depthPaths_;  // sorted: element i is frame i
};

}  // namespace ovo::adapters::replica
