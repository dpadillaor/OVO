#include "ovo/adapters/replica/replica_frame_source.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "stb_image.h"

namespace ovo::adapters::replica {

namespace {

// Reads a 16-bit single-channel depth PNG and returns it in metres (raw value / depthScale; 0 stays 0 = no measurement).
// Throws std::runtime_error with the path if the file cannot be read or does not match the camera.
core::Image<float> loadDepth(const std::filesystem::path& path, float depthScale,
                             const core::geometry::PinholeCamera& camera) {
    const std::string file = path.string();
    // stbi_load_16 would silently widen an 8-bit PNG: reject it instead.
    if (stbi_is_16_bit(file.c_str()) == 0) {
        throw std::runtime_error("loadDepth: not a 16-bit PNG: " + file);
    }

    // C API: stb allocates the pixels and we must free them with stbi_image_free. unique_ptr does it for us
    // when it goes out of scope, also if we throw below (RAII).
    int width = 0;
    int height = 0;
    int channels = 0;
    const std::unique_ptr<stbi_us, decltype(&stbi_image_free)> raw(
        stbi_load_16(file.c_str(), &width, &height, &channels, 1), &stbi_image_free);
    if (!raw) {
        const char* reason = stbi_failure_reason();  // may be null: never append a null char*
        throw std::runtime_error("loadDepth: cannot read " + file + " (" + (reason != nullptr ? reason : "unknown") + ")");
    }
    // We asked for 1 channel, so stb would silently turn an RGB image into gray: a depth map has one channel.
    if (channels != 1) {
        throw std::runtime_error("loadDepth: " + file + " has " + std::to_string(channels) + " channels, expected 1");
    }
    if (static_cast<std::uint32_t>(width) != camera.width() || static_cast<std::uint32_t>(height) != camera.height()) {
        throw std::runtime_error("loadDepth: " + file + " is " + std::to_string(width) + "x" + std::to_string(height) +
                                 ", camera expects " + std::to_string(camera.width()) + "x" +
                                 std::to_string(camera.height()));
    }

    core::Image<float> depth(camera.width(), camera.height());
    const std::size_t numPixels = std::size_t{camera.width()} * camera.height();
    for (std::size_t i = 0; i < numPixels; ++i) {
        depth.data()[i] = static_cast<float>(raw.get()[i]) / depthScale;
    }
    return depth;
}

}  // namespace

ReplicaFrameSource::ReplicaFrameSource(const std::filesystem::path& sceneDir,
                                       const core::geometry::PinholeCamera& camera, float depthScale)
    : camera_(camera), depthScale_(depthScale) {
    if (!(depthScale > 0.f && std::isfinite(depthScale))) {  // positive form: NaN fails too
        throw std::invalid_argument("ReplicaFrameSource: depthScale must be finite and > 0");
    }
    const std::filesystem::path resultsDir = sceneDir / "results";
    if (!std::filesystem::is_directory(resultsDir)) {
        throw std::runtime_error("ReplicaFrameSource: folder not found: " + resultsDir.string());
    }

    for (const auto& entry : std::filesystem::directory_iterator(resultsDir)) {
        const std::string name = entry.path().filename().string();
        if (name.starts_with("depth") && entry.path().extension() == ".png") { depthPaths_.push_back(entry.path()); }
    }
    if (depthPaths_.empty()) {
        throw std::runtime_error("ReplicaFrameSource: no depth*.png in " + resultsDir.string());
    }

    // directory_iterator has no order. Names are zero-padded (depth000042.png): alphabetical = frame order.
    std::sort(depthPaths_.begin(), depthPaths_.end());
}

core::ports::Frame ReplicaFrameSource::frame(std::size_t index) const {
    assert(index < size());
    return {core::FrameId{static_cast<std::uint32_t>(index)}, loadDepth(depthPaths_[index], depthScale_, camera_)};
}

}  // namespace ovo::adapters::replica
