#include "ovo/adapters/ply/ply_writer.hpp"

#include <bit>
#include <fstream>
#include <stdexcept>

#include <Eigen/Core>

namespace ovo::adapters::ply {

namespace {

// The points are written as raw floats, which is only "binary_little_endian" on a little-endian CPU.
static_assert(std::endian::native == std::endian::little, "PLY writer assumes a little-endian CPU");
static_assert(sizeof(Eigen::Vector3f) == 3 * sizeof(float), "Vector3f must be 3 packed floats");

void writeHeader(std::ofstream& out, std::size_t numPoints) {
    out << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << numPoints << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "end_header\n";
}

}  // namespace

void writePly(const core::mapping::GeometricMap& map, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("writePly: cannot open " + path.string()); }

    writeHeader(out, map.numPoints());
    for (const core::mapping::PointBlock& block : map.blocks()) {
        const auto xyz = block.xyzWorld();
        // One write per block: its points are contiguous floats in memory.
        out.write(reinterpret_cast<const char*>(xyz.data()),
                  static_cast<std::streamsize>(xyz.size() * sizeof(Eigen::Vector3f)));
    }
    if (!out) { throw std::runtime_error("writePly: error while writing " + path.string()); }
}

}  // namespace ovo::adapters::ply
