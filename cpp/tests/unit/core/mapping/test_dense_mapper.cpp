#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <Eigen/Core>

#include "ovo/core/common/image.hpp"
#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/geometry/pose.hpp"
#include "ovo/core/mapping/dense_mapper.hpp"

using ovo::core::FrameId;
using ovo::core::Image;
using ovo::core::geometry::PinholeCamera;
using ovo::core::geometry::Pose;
using ovo::core::mapping::DenseMapper;

// Same camera as the geometry tests: pixel (50, 50) at depth z unprojects to (0, 0, z).
constexpr PinholeCamera::Params kTestParams{
    .fx = 100.f, .fy = 100.f, .cx = 50.f, .cy = 50.f, .width = 100, .height = 100};
const PinholeCamera kCamera{kTestParams};
constexpr std::size_t kNumPixels = std::size_t{kTestParams.width} * kTestParams.height;

// One point per pixel: every count below can be checked by hand.
constexpr DenseMapper::Config kEveryPixel{.maxDepthError = 0.03f, .dilationSize = 1, .step = 1};

namespace {

Image<float> uniformDepth(float metres) {
    Image<float> depth(kTestParams.width, kTestParams.height);
    std::fill(depth.data(), depth.data() + kNumPixels, metres);
    return depth;
}

}  // namespace

// --- Construction: the config is checked by the parts that use it.

TEST(DenseMapper, RejectsBadMatcherConfig) {
    EXPECT_THROW(DenseMapper(kCamera, {.maxDepthError = 0.f, .dilationSize = 3, .step = 2}), std::invalid_argument);
}

TEST(DenseMapper, RejectsBadCoverageConfig) {
    EXPECT_THROW(DenseMapper(kCamera, {.maxDepthError = 0.03f, .dilationSize = 2, .step = 2}), std::invalid_argument);
}

TEST(DenseMapper, StartsWithEmptyMap) {
    const DenseMapper mapper(kCamera, kEveryPixel);
    EXPECT_TRUE(mapper.geometricMap().empty());
}

// --- First frame: every free pixel becomes a point.

TEST(DenseMapperMap, FirstFrameCreatesOnePointPerPixel) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{0}, uniformDepth(2.f), Pose::identity());
    EXPECT_EQ(mapper.geometricMap().numPoints(), kNumPixels);
}

// Default config (Python): step 2 keeps 1 pixel in 4.
TEST(DenseMapperMap, DefaultConfigCreatesOnePointPerFourPixels) {
    DenseMapper mapper(kCamera, {});
    mapper.map(FrameId{0}, uniformDepth(2.f), Pose::identity());
    EXPECT_EQ(mapper.geometricMap().numPoints(), kNumPixels / 4);
}

TEST(DenseMapperMap, BlockBelongsToTheFrame) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{7}, uniformDepth(2.f), Pose::identity());
    ASSERT_EQ(mapper.geometricMap().blocks().size(), std::size_t{1});
    EXPECT_EQ(mapper.geometricMap().blocks()[0].keyframe(), FrameId{7});
}

// Pixel (50, 50) is element 50 * 100 + 50 (row order) and lies on the optical axis.
TEST(DenseMapperMap, PointLiesOnTheMeasuredSurface) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{0}, uniformDepth(2.f), Pose::identity());
    const auto xyz = mapper.geometricMap().blocks()[0].xyzWorld();
    EXPECT_TRUE(xyz[50 * 100 + 50].isApprox(Eigen::Vector3f{0.f, 0.f, 2.f}));
}

TEST(DenseMapperMap, PointIsInWorldFrame) {
    DenseMapper mapper(kCamera, kEveryPixel);
    const Pose cameraPose = Pose::fromCamToWorld(Eigen::Matrix3f::Identity(), {10.f, 0.f, 0.f});
    mapper.map(FrameId{0}, uniformDepth(2.f), cameraPose);
    const auto xyz = mapper.geometricMap().blocks()[0].xyzWorld();
    EXPECT_TRUE(xyz[50 * 100 + 50].isApprox(Eigen::Vector3f{10.f, 0.f, 2.f}));
}

TEST(DenseMapperMap, FrameWithoutDepthCreatesNothing) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{0}, uniformDepth(0.f), Pose::identity());
    EXPECT_TRUE(mapper.geometricMap().empty());
}

TEST(DenseMapperMap, PixelsWithoutDepthCreateNothing) {
    DenseMapper mapper(kCamera, kEveryPixel);
    Image<float> depth = uniformDepth(2.f);
    for (std::uint32_t v = 0; v < kTestParams.height; ++v) {
        for (std::uint32_t u = 50; u < kTestParams.width; ++u) { depth.at(u, v) = 0.f; }
    }
    mapper.map(FrameId{0}, depth, Pose::identity());
    EXPECT_EQ(mapper.geometricMap().numPoints(), kNumPixels / 2);
}

// --- Next frames: what is already in the map is not created again.

// Same view twice: every pixel is covered by its own point, nothing new. Includes the image border
// (pixel 0), which only matches if the frustum spans the whole pixel (see Frustum, pixel centres).
TEST(DenseMapperMap, SameViewAddsNothing) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{0}, uniformDepth(2.f), Pose::identity());
    mapper.map(FrameId{10}, uniformDepth(2.f), Pose::identity());
    EXPECT_EQ(mapper.geometricMap().numPoints(), kNumPixels);
    EXPECT_EQ(mapper.geometricMap().blocks().size(), std::size_t{1});
}

TEST(DenseMapperMap, SameViewAddsNothingWithDefaultConfig) {
    DenseMapper mapper(kCamera, {});
    mapper.map(FrameId{0}, uniformDepth(2.f), Pose::identity());
    mapper.map(FrameId{10}, uniformDepth(2.f), Pose::identity());
    EXPECT_EQ(mapper.geometricMap().numPoints(), kNumPixels / 4);
}

// The right half moved from 2 m to 3 m (something new in front of the old wall... or a new wall):
// those pixels do not match the old points, so only the right half gets new points.
TEST(DenseMapperMap, AddsPointsOnlyWhereTheSurfaceChanged) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{0}, uniformDepth(2.f), Pose::identity());
    Image<float> depth = uniformDepth(2.f);
    for (std::uint32_t v = 0; v < kTestParams.height; ++v) {
        for (std::uint32_t u = 50; u < kTestParams.width; ++u) { depth.at(u, v) = 3.f; }
    }
    mapper.map(FrameId{10}, depth, Pose::identity());
    ASSERT_EQ(mapper.geometricMap().blocks().size(), std::size_t{2});
    EXPECT_EQ(mapper.geometricMap().blocks()[1].size(), kNumPixels / 2);
}

// A map point 2 cm in front of the nearest measurement still matches it (|dz| < 3 cm). The frustum
// is widened by the tolerance; with Python's exact [min, max] these points would be duplicated.
TEST(DenseMapperMap, MatchesPointsJustInFrontOfNearestMeasurement) {
    DenseMapper mapper(kCamera, kEveryPixel);
    mapper.map(FrameId{0}, uniformDepth(1.98f), Pose::identity());
    mapper.map(FrameId{10}, uniformDepth(2.f), Pose::identity());
    EXPECT_EQ(mapper.geometricMap().blocks().size(), std::size_t{1});
}

TEST(DenseMapperDeathTest, RejectsDepthOfWrongSize) {
    DenseMapper mapper(kCamera, kEveryPixel);
    const Image<float> smallDepth(50, 50);
    EXPECT_DEBUG_DEATH(mapper.map(FrameId{0}, smallDepth, Pose::identity()), "");
}
