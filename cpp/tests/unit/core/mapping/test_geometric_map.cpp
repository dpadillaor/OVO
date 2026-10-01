#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/mapping/geometric_map.hpp"

using ovo::core::FrameId;
using ovo::core::PointId;
using ovo::core::mapping::GeometricMap;

namespace {

// n points, all at the origin: the tests here are about bookkeeping, not positions.
std::vector<Eigen::Vector3f> points(std::size_t n) { return std::vector<Eigen::Vector3f>(n, Eigen::Vector3f::Zero()); }

}  // namespace

TEST(GeometricMap, StartsEmpty) {
    const GeometricMap map;
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.numPoints(), std::size_t{0});
    EXPECT_TRUE(map.blocks().empty());
}

TEST(GeometricMapAddBlock, AddsOneBlockPerCall) {
    GeometricMap map;
    map.addBlock(FrameId{0}, points(3));
    map.addBlock(FrameId{10}, points(2));
    ASSERT_EQ(map.blocks().size(), std::size_t{2});
    EXPECT_EQ(map.blocks()[0].keyframe(), FrameId{0});
    EXPECT_EQ(map.blocks()[1].keyframe(), FrameId{10});
}

TEST(GeometricMapAddBlock, CountsPointsOfAllBlocks) {
    GeometricMap map;
    map.addBlock(FrameId{0}, points(3));
    map.addBlock(FrameId{10}, points(2));
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.numPoints(), std::size_t{5});
}

TEST(GeometricMapAddBlock, FirstBlockStartsAtIdZero) {
    GeometricMap map;
    map.addBlock(FrameId{0}, points(3));
    EXPECT_EQ(map.blocks()[0].pointId(0), PointId{0});
}

// Ids continue where the previous block ended: no gaps, no repeats.
TEST(GeometricMapAddBlock, NextBlockContinuesIds) {
    GeometricMap map;
    map.addBlock(FrameId{0}, points(3));   // ids 0, 1, 2
    map.addBlock(FrameId{10}, points(2));  // ids 3, 4
    EXPECT_EQ(map.blocks()[1].pointId(0), PointId{3});
    EXPECT_EQ(map.blocks()[1].pointId(1), PointId{4});
}

// A frame whose pixels were all covered creates nothing: no empty block, no id consumed.
TEST(GeometricMapAddBlock, IgnoresEmptyBlock) {
    GeometricMap map;
    map.addBlock(FrameId{0}, points(3));
    map.addBlock(FrameId{10}, points(0));
    map.addBlock(FrameId{20}, points(2));
    ASSERT_EQ(map.blocks().size(), std::size_t{2});
    EXPECT_EQ(map.blocks()[1].keyframe(), FrameId{20});
    EXPECT_EQ(map.blocks()[1].pointId(0), PointId{3});
}
