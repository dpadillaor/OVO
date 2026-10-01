#include <gtest/gtest.h>

#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ovo/core/geometry/frustum.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/geometry/pose.hpp"

using namespace ovo::core::geometry;

// Simple numbers on purpose: with identity pose and depth [1, 5] the frustum is
// 1 <= z <= 5, |x| <= 0.5z, |y| <= 0.5z (at z = 3: x and y in [-1.5, 1.5]).
// No tests exactly on a wall: d comes from the corners with rounding, so the result is not stable.
constexpr PinholeCamera::Params kTestParams{
    .fx = 100.f, .fy = 100.f, .cx = 50.f, .cy = 50.f, .width = 100, .height = 100};
const PinholeCamera kCamera{kTestParams};

// On the optical axis, inside on every plane: a single flipped normal makes this fail.
TEST(Frustum, ContainsPointOnOpticalAxis) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_TRUE(frustum.contains({0.f, 0.f, 3.f}));
}

// At z = 3 the sides are at +-1.5: (1.4, 1.4) is close to the corner but inside.
// Catches side planes that are too tight (e.g. built from the wrong corners).
TEST(Frustum, ContainsPointNearCorner) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_TRUE(frustum.contains({1.4f, 1.4f, 3.f}));
}

TEST(Frustum, RejectsPointBeforeNear) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({0.f, 0.f, 0.5f}));
}

TEST(Frustum, RejectsPointBeyondFar) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({0.f, 0.f, 6.f}));
}

// The side planes meet at the camera centre and open again behind it (an hourglass):
// only the near plane keeps points behind the camera out.
TEST(Frustum, RejectsPointBehindCamera) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({0.f, 0.f, -3.f}));
}

TEST(Frustum, RejectsPointLeft) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({-2.f, 0.f, 3.f}));
}

TEST(Frustum, RejectsPointRight) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({2.f, 0.f, 3.f}));
}

// Image v grows downwards, so negative y is the top of the image.
TEST(Frustum, RejectsPointAbove) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({0.f, -2.f, 3.f}));
}

TEST(Frustum, RejectsPointBelow) {
    const Frustum frustum(kCamera, Pose::identity(), 1.f, 5.f);
    EXPECT_FALSE(frustum.contains({0.f, 2.f, 3.f}));
}

// Camera moved 10 m along world x, not rotated: the frustum moves with it.
TEST(FrustumWithPose, FollowsTranslation) {
    const Pose cameraPose = Pose::fromCamToWorld(Eigen::Matrix3f::Identity(), {10.f, 0.f, 0.f});
    const Frustum frustum(kCamera, cameraPose, 1.f, 5.f);
    EXPECT_TRUE(frustum.contains({10.f, 0.f, 3.f}));
    EXPECT_FALSE(frustum.contains({0.f, 0.f, 3.f}));
}

// 180 degrees about y: the camera looks at world -z, so what was in front is now behind.
TEST(FrustumWithPose, FollowsRotation) {
    const Eigen::Matrix3f rotY180 = Eigen::Vector3f{-1.f, 1.f, -1.f}.asDiagonal();
    const Frustum frustum(kCamera, Pose::fromCamToWorld(rotY180, Eigen::Vector3f::Zero()), 1.f, 5.f);
    EXPECT_TRUE(frustum.contains({0.f, 0.f, -3.f}));
    EXPECT_FALSE(frustum.contains({0.f, 0.f, 3.f}));
}

// Generic pose (same as the Pose round-trip test): the frustum must agree with
// PinholeCamera::unproject + Pose::camToWorld, the chain it is built from.
const Pose kGenericPose = Pose::fromCamToWorld(
    Eigen::AngleAxisf(0.7f, Eigen::Vector3f{1.f, 2.f, 3.f}.normalized()).toRotationMatrix(),
    {0.5f, -1.f, 2.f});

TEST(FrustumWithPose, ContainsUnprojectedPixelInsideImage) {
    const Frustum frustum(kCamera, kGenericPose, 1.f, 5.f);
    const Eigen::Vector3f pointWorld = kGenericPose.camToWorld(kCamera.unproject({30.f, 70.f}, 2.5f));
    EXPECT_TRUE(frustum.contains(pointWorld));
}

// Pixel u = -20 is off the image: at z = 2.5 it gives x = -1.75, past the left side at -1.25.
TEST(FrustumWithPose, RejectsUnprojectedPixelOffImage) {
    const Frustum frustum(kCamera, kGenericPose, 1.f, 5.f);
    const Eigen::Vector3f pointWorld = kGenericPose.camToWorld(kCamera.unproject({-20.f, 50.f}, 2.5f));
    EXPECT_FALSE(frustum.contains(pointWorld));
}

// Precondition 0 < minDepth < maxDepth. A frame without valid depth is filtered out by the caller
// before building a Frustum, so a bad range here is a bug: assert.
// static_cast<void>: the temporary Frustum is discarded on purpose.
TEST(FrustumDeathTest, RejectsNonPositiveMinDepth) {
    EXPECT_DEBUG_DEATH(static_cast<void>(Frustum(kCamera, Pose::identity(), 0.f, 5.f)), "");
}

TEST(FrustumDeathTest, RejectsMinNotBelowMax) {
    EXPECT_DEBUG_DEATH(static_cast<void>(Frustum(kCamera, Pose::identity(), 5.f, 5.f)), "");
}

TEST(FrustumDeathTest, RejectsNaNDepth) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEBUG_DEATH(static_cast<void>(Frustum(kCamera, Pose::identity(), 1.f, nan)), "");
}
