#pragma once

#include <filesystem>

#include "ovo/core/mapping/geometric_map.hpp"

namespace ovo::adapters::ply {

// Writes every point of the map as a binary PLY (x y z float), for MeshLab / CloudCompare / Open3D.
// Throws std::runtime_error if the file cannot be written.
void writePly(const core::mapping::GeometricMap& map, const std::filesystem::path& path);

}  // namespace ovo::adapters::ply
