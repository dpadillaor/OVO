#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/mapping/point_block.hpp"

using ovo::core::FrameId;
using ovo::core::PointId;
using ovo::core::mapping::PointBlock;

// Three points created by keyframe 10, ids starting at 1000.
const std::vector<Eigen::Vector3f> kXyz{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}, {7.f, 8.f, 9.f}};

TEST(PointBlock, KeepsKeyframe) {
    const PointBlock block(FrameId{10}, PointId{1000}, kXyz);
    EXPECT_EQ(block.keyframe(), FrameId{10});
}

TEST(PointBlock, SizeIsNumberOfPoints) {
    const PointBlock block(FrameId{10}, PointId{1000}, kXyz);
    EXPECT_EQ(block.size(), std::size_t{3});
}

// An empty block is valid (a frame whose pixels were all covered); it just has no ids.
TEST(PointBlock, CanBeEmpty) {
    const PointBlock block(FrameId{10}, PointId{1000}, {});
    EXPECT_EQ(block.size(), std::size_t{0});
    EXPECT_TRUE(block.xyzWorld().empty());
}

TEST(PointBlock, KeepsPointsInOrder) {
    const PointBlock block(FrameId{10}, PointId{1000}, kXyz);
    const auto xyz = block.xyzWorld();
    ASSERT_EQ(xyz.size(), std::size_t{3});
    EXPECT_EQ(xyz[0], kXyz[0]);
    EXPECT_EQ(xyz[2], kXyz[2]);
}

// Ids are consecutive from firstId: point i has id firstId + i.
TEST(PointBlockPointId, FirstPointHasFirstId) {
    const PointBlock block(FrameId{10}, PointId{1000}, kXyz);
    EXPECT_EQ(block.pointId(0), PointId{1000});
}

TEST(PointBlockPointId, LastPointHasFirstIdPlusIndex) {
    const PointBlock block(FrameId{10}, PointId{1000}, kXyz);
    EXPECT_EQ(block.pointId(2), PointId{1002});
}

// The block takes the vector's memory instead of copying it: same buffer, source left empty.
TEST(PointBlock, TakesOwnershipOfMovedPoints) {
    std::vector<Eigen::Vector3f> xyz = kXyz;
    const Eigen::Vector3f* buffer = xyz.data();
    const PointBlock block(FrameId{10}, PointId{1000}, std::move(xyz));
    EXPECT_EQ(block.xyzWorld().data(), buffer);
}

// Index past the end would give an id from another block: a caller bug, so assert.
TEST(PointBlockDeathTest, PointIdRejectsIndexOutOfRange) {
    const PointBlock block(FrameId{10}, PointId{1000}, kXyz);
    EXPECT_DEBUG_DEATH(static_cast<void>(block.pointId(3)), "");
}
