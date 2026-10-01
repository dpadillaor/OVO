#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ovo/core/common/image.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/geometry/point_pixel_matcher.hpp"
#include "ovo/core/geometry/pose.hpp"

using ovo::core::Image;
using namespace ovo::core::geometry;

// Same camera as the Frustum tests: with identity pose, point (x, y, z) projects to
// u = 50 + 100 x / z, v = 50 + 100 y / z. (0, 0, 2) lands on pixel (50, 50).
constexpr PinholeCamera::Params kTestParams{
    .fx = 100.f, .fy = 100.f, .cx = 50.f, .cy = 50.f, .width = 100, .height = 100};
const PinholeCamera kCamera{kTestParams};
constexpr float kMaxDepthError = 0.03f;  // 3 cm, as in the Python config

// Depth image with the same measured depth on every pixel.
Image<float> uniformDepth(float metres) {
    Image<float> depth(kTestParams.width, kTestParams.height);
    std::fill(depth.data(), depth.data() + std::size_t{kTestParams.width} * kTestParams.height, metres);
    return depth;
}

// --- Construction: the tolerance is configuration, so a bad value throws (like PinholeCamera).

TEST(PointPixelMatcher, RejectsZeroMaxDepthError) {
    EXPECT_THROW(PointPixelMatcher(kCamera, 0.f), std::invalid_argument);
}

TEST(PointPixelMatcher, RejectsNaNMaxDepthError) {
    EXPECT_THROW(PointPixelMatcher(kCamera, std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
}

TEST(PointPixelMatcher, RejectsInfiniteMaxDepthError) {
    EXPECT_THROW(PointPixelMatcher(kCamera, std::numeric_limits<float>::infinity()), std::invalid_argument);
}

// --- The basic match.

TEST(PointPixelMatcherMatch, MatchesPointWhereDepthAgrees) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 2.f}};
    const auto matches = matcher.match(points, Pose::identity(), uniformDepth(2.f));
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].pointIndex, 0u);
    EXPECT_EQ(matches[0].u, 50u);
    EXPECT_EQ(matches[0].v, 50u);
}

// u != v on purpose: catches u and v swapped in the output.
TEST(PointPixelMatcherMatch, ReportsPixelOfProjection) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.5f, -0.3f, 2.f}};  // u = 75, v = 35
    const auto matches = matcher.match(points, Pose::identity(), uniformDepth(2.f));
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].u, 75u);
    EXPECT_EQ(matches[0].v, 35u);
}

// pointIndex is the position in the input, so the caller can map it back to a PointId.
TEST(PointPixelMatcherMatch, ReportsIndexInInput) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 3.f}, {0.f, 0.f, 2.f}, {0.f, 0.f, 4.f}};
    const auto matches = matcher.match(points, Pose::identity(), uniformDepth(2.f));
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].pointIndex, 1u);
}

TEST(PointPixelMatcherMatch, EmptyInputGivesNoMatches) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points;
    EXPECT_TRUE(matcher.match(points, Pose::identity(), uniformDepth(2.f)).empty());
}

// Not the matcher's job to pick one point per pixel: every agreeing point is reported.
TEST(PointPixelMatcherMatch, SeveralPointsMayShareAPixel) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 2.f}, {0.f, 0.f, 2.01f}};
    EXPECT_EQ(matcher.match(points, Pose::identity(), uniformDepth(2.f)).size(), std::size_t{2});
}

// --- Depth tolerance: |z - measured| < maxDepthError.

TEST(PointPixelMatcherMatch, AcceptsDepthWithinTolerance) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 2.02f}};
    EXPECT_EQ(matcher.match(points, Pose::identity(), uniformDepth(2.f)).size(), std::size_t{1});
}

// The point is hidden behind the measured surface (occluded): not seen.
TEST(PointPixelMatcherMatch, RejectsPointBehindMeasuredSurface) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 2.5f}};
    EXPECT_TRUE(matcher.match(points, Pose::identity(), uniformDepth(2.f)).empty());
}

// The point floats in front of the measured surface: not seen either. Catches a missing abs().
TEST(PointPixelMatcherMatch, RejectsPointInFrontOfMeasuredSurface) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 1.5f}};
    EXPECT_TRUE(matcher.match(points, Pose::identity(), uniformDepth(2.f)).empty());
}

// Depth 0 means "no measurement". A point 1 cm from the camera is within 3 cm of 0,
// so without an explicit check it would match: the Python comment saying the check could be skipped is wrong.
TEST(PointPixelMatcherMatch, IgnoresPixelsWithoutDepth) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 0.01f}};
    EXPECT_TRUE(matcher.match(points, Pose::identity(), uniformDepth(0.f)).empty());
}

// --- Pixel: round to nearest (pixel centre on the integer, see progress.md), then check bounds.

// u = 50.7 must read pixel 51, not 50. Only pixel (51, 50) has depth, so floor gives no match.
TEST(PointPixelMatcherMatch, RoundsToNearestPixel) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    Image<float> depth(kTestParams.width, kTestParams.height);
    depth.at(51, 50) = 2.f;
    const std::vector<Eigen::Vector3f> points{{0.014f, 0.f, 2.f}};  // u = 50.7
    const auto matches = matcher.match(points, Pose::identity(), depth);
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].u, 51u);
}

// u = -0.4 rounds to pixel 0, which is inside the image.
TEST(PointPixelMatcherMatch, AcceptsProjectionRoundingIntoImage) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{-1.008f, 0.f, 2.f}};  // u = -0.4
    const auto matches = matcher.match(points, Pose::identity(), uniformDepth(2.f));
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].u, 0u);
}

// u = 99.6 rounds to 100, one past the last pixel (99): out.
TEST(PointPixelMatcherMatch, RejectsProjectionRoundingOutOfImage) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.992f, 0.f, 2.f}};  // u = 99.6
    EXPECT_TRUE(matcher.match(points, Pose::identity(), uniformDepth(2.f)).empty());
}

// u = -20: far off the image. Catches a negative coordinate cast straight to uint32 (wraps to a huge number).
TEST(PointPixelMatcherMatch, RejectsPointProjectingOffImage) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{-1.4f, 0.f, 2.f}};  // u = -20
    EXPECT_TRUE(matcher.match(points, Pose::identity(), uniformDepth(2.f)).empty());
}

// --- Camera pose.

// Camera moved 10 m along world x: the point must be taken to camera coordinates first.
TEST(PointPixelMatcherMatch, FollowsCameraPose) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const Pose cameraPose = Pose::fromCamToWorld(Eigen::Matrix3f::Identity(), {10.f, 0.f, 0.f});
    const std::vector<Eigen::Vector3f> points{{10.f, 0.f, 2.f}};
    const auto matches = matcher.match(points, cameraPose, uniformDepth(2.f));
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].u, 50u);
    EXPECT_EQ(matches[0].v, 50u);
}

// Round trip with a generic pose: a point created from pixel (30, 70) must match that same pixel.
// This is how the mapper recognises its own points in the next frame.
TEST(PointPixelMatcherMatch, MatchesPointUnprojectedFromPixel) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const Pose cameraPose = Pose::fromCamToWorld(
        Eigen::AngleAxisf(0.7f, Eigen::Vector3f{1.f, 2.f, 3.f}.normalized()).toRotationMatrix(), {0.5f, -1.f, 2.f});
    const std::vector<Eigen::Vector3f> points{cameraPose.camToWorld(kCamera.unproject({30.f, 70.f}, 2.5f))};
    const auto matches = matcher.match(points, cameraPose, uniformDepth(2.5f));
    ASSERT_EQ(matches.size(), std::size_t{1});
    EXPECT_EQ(matches[0].u, 30u);
    EXPECT_EQ(matches[0].v, 70u);
}

// --- Preconditions.

// A depth image of another size means pixels (u, v) do not correspond: a wiring bug, so assert.
TEST(PointPixelMatcherDeathTest, RejectsDepthOfWrongSize) {
    const PointPixelMatcher matcher(kCamera, kMaxDepthError);
    const std::vector<Eigen::Vector3f> points{{0.f, 0.f, 2.f}};
    const Image<float> smallDepth(50, 50);
    EXPECT_DEBUG_DEATH(static_cast<void>(matcher.match(points, Pose::identity(), smallDepth)), "");
}
