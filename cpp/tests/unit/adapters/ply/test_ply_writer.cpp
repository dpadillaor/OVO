#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "ovo/adapters/ply/ply_writer.hpp"
#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/mapping/geometric_map.hpp"

using ovo::adapters::ply::writePly;
using ovo::core::FrameId;
using ovo::core::mapping::GeometricMap;

namespace {

std::string readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

const std::filesystem::path kOut = std::filesystem::temp_directory_path() / "ovo_test_map.ply";

}  // namespace

// Two blocks, three points: header announces 3 vertices, then 3 * 3 floats follow the header.
TEST(PlyWriter, WritesHeaderAndAllPoints) {
    GeometricMap map;
    map.addBlock(FrameId{0}, {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    map.addBlock(FrameId{10}, {{7.f, 8.f, 9.f}});
    writePly(map, kOut);

    const std::string file = readFile(kOut);
    const std::string endHeader = "end_header\n";
    const std::size_t headerEnd = file.find(endHeader) + endHeader.size();
    EXPECT_NE(file.find("element vertex 3\n"), std::string::npos);
    ASSERT_EQ(file.size() - headerEnd, 9 * sizeof(float));

    // memcpy, not a cast: the floats start after a text header, so they may not be aligned.
    float xyz[9];
    std::memcpy(xyz, file.data() + headerEnd, sizeof(xyz));
    EXPECT_EQ(xyz[0], 1.f);
    EXPECT_EQ(xyz[8], 9.f);
    std::filesystem::remove(kOut);
}

TEST(PlyWriter, ThrowsIfFileCannotBeOpened) {
    const GeometricMap map;
    EXPECT_THROW(writePly(map, "this/folder/does/not/exist/map.ply"), std::runtime_error);
}
