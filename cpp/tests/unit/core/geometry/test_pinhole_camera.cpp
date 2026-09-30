#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <Eigen/Core>

#include "ovo/core/geometry/pinhole_camera.hpp"

using namespace ovo::core::geometry;

// Parameters from Replica cameras.
constexpr PinholeCamera::Params kReplica{
    .fx = 600.f, .fy = 600.f, .cx = 599.5f, .cy = 339.5f, .width = 1200, .height = 680};


TEST(PinholeCamera, StoresParams) {
    const PinholeCamera camera{kReplica};
    EXPECT_FLOAT_EQ(kReplica.fx, camera.fx());
    EXPECT_FLOAT_EQ(kReplica.fy, camera.fy());
    EXPECT_FLOAT_EQ(kReplica.cx, camera.cx());
    EXPECT_FLOAT_EQ(kReplica.cy, camera.cy());
    EXPECT_EQ(kReplica.width, camera.width());
    EXPECT_EQ(kReplica.height, camera.height());
}

TEST(PinholeCamera, RejectsZeroFocal) {
    auto p = kReplica;
    p.fx = 0.f;
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCamera, RejectsNegativeFocal) {
    auto p = kReplica;
    p.fy = -1.f;
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCamera, RejectsZeroSize) {
    auto p = kReplica;
    p.width = 0;
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCamera, RejectsNaNFocal) {
    auto p = kReplica;
    p.fx = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}
TEST(PinholeCamera, RejectsZeroHeight) {
    auto p = kReplica;
    p.height = 0;
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCamera, RejectsInfiniteFocal) {
    auto p = kReplica;
    p.fx = std::numeric_limits<float>::infinity();
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCamera, RejectsNaNPrincipalPoint) {
    auto p = kReplica;
    p.cx = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCamera, RejectsInfinitePrincipalPoint) {
    auto p = kReplica;
    p.cy = -std::numeric_limits<float>::infinity();
    EXPECT_THROW(PinholeCamera{p}, std::invalid_argument);
}

TEST(PinholeCameraProject, OpticalAxisGoesToPrincipalPoint) {
    const PinholeCamera camera{kReplica};
    const Eigen::Vector3f pointCam{0.f, 0.f, 5.f};
    const Eigen::Vector2f expectedPixel{599.5f, 339.5f};
    const Eigen::Vector2f projected = camera.project(pointCam);
    EXPECT_FLOAT_EQ(projected.x(), expectedPixel.x());
    EXPECT_FLOAT_EQ(projected.y(), expectedPixel.y());
}

TEST(PinholeCameraProject, KnownPoint) {
    const PinholeCamera camera{kReplica};
    // u = 600 * 1 / 2 + 599.5 ; v = 600 * 0.5 / 2 + 339.5
    const Eigen::Vector2f projected = camera.project({1.f, 0.5f, 2.f});
    EXPECT_FLOAT_EQ(projected.x(), 899.5f);
    EXPECT_FLOAT_EQ(projected.y(), 489.5f);
}

// fx != fy catches swapped fx/fy (with Replica fx == fy would hide it).
TEST(PinholeCameraProject, DifferentFocals) {
    const PinholeCamera camera{{.fx = 500.f, .fy = 400.f, .cx = 320.f, .cy = 240.f,
                                .width = 640, .height = 480}};
    // u = 500 * 1 / 2 + 320 ; v = 400 * 1 / 2 + 240
    const Eigen::Vector2f projected = camera.project({1.f, 1.f, 2.f});
    EXPECT_FLOAT_EQ(projected.x(), 570.f);
    EXPECT_FLOAT_EQ(projected.y(), 440.f);
}

// Two points on the same ray (one twice the other) land on the same pixel.
TEST(PinholeCameraProject, ScaleInvariant) {
    const PinholeCamera camera{kReplica};
    const Eigen::Vector3f near{1.f, 0.5f, 2.f};
    const Eigen::Vector2f a = camera.project(near);
    const Eigen::Vector2f b = camera.project(2.f * near);
    EXPECT_FLOAT_EQ(a.x(), b.x());
    EXPECT_FLOAT_EQ(a.y(), b.y());
}

// DeathTest suffix: GoogleTest runs these suites first (official recommendation).
// static_cast<void>: project is [[nodiscard]]; the result is discarded on purpose here.
TEST(PinholeCameraProjectDeathTest, PointBehindCamera) {
    const PinholeCamera camera{kReplica};
    EXPECT_DEBUG_DEATH(static_cast<void>(camera.project({0.f, 0.f, -1.f})), "");
}

TEST(PinholeCameraUnproject, PrincipalPointGoesToOpticalAxis) {
    const PinholeCamera camera{kReplica};
    const Eigen::Vector3f point = camera.unproject({599.5f, 339.5f}, 3.f);
    EXPECT_FLOAT_EQ(point.x(), 0.f);
    EXPECT_FLOAT_EQ(point.y(), 0.f);
    EXPECT_FLOAT_EQ(point.z(), 3.f);
}

TEST(PinholeCameraUnproject, KnownPixel) {
    const PinholeCamera camera{kReplica};
    // x = (899.5 - 599.5) * 2 / 600 ; y = (489.5 - 339.5) * 2 / 600
    const Eigen::Vector3f point = camera.unproject({899.5f, 489.5f}, 2.f);
    EXPECT_FLOAT_EQ(point.x(), 1.f);
    EXPECT_FLOAT_EQ(point.y(), 0.5f);
    EXPECT_FLOAT_EQ(point.z(), 2.f);
}

// fx != fy catches swapped fx/fy (same reason as in project).
TEST(PinholeCameraUnproject, DifferentFocals) {
    const PinholeCamera camera{{.fx = 500.f, .fy = 400.f, .cx = 320.f, .cy = 240.f,
                                .width = 640, .height = 480}};
    const Eigen::Vector3f point = camera.unproject({570.f, 440.f}, 2.f);
    EXPECT_FLOAT_EQ(point.x(), 1.f);
    EXPECT_FLOAT_EQ(point.y(), 1.f);
    EXPECT_FLOAT_EQ(point.z(), 2.f);
}

// project then unproject with the same depth gives back the original point.
// EXPECT_NEAR: two chained operations may accumulate rounding.
TEST(PinholeCameraUnproject, RoundTripWithProject) {
    const PinholeCamera camera{kReplica};
    const Eigen::Vector3f original{0.3f, -0.7f, 2.5f};
    const Eigen::Vector3f back = camera.unproject(camera.project(original), original.z());
    EXPECT_NEAR(back.x(), original.x(), 1e-5f);
    EXPECT_NEAR(back.y(), original.y(), 1e-5f);
    EXPECT_NEAR(back.z(), original.z(), 1e-5f);
}

// Depth 0 means "no measurement" in RGB-D: unprojecting it is a caller bug.
TEST(PinholeCameraUnprojectDeathTest, NonPositiveDepth) {
    const PinholeCamera camera{kReplica};
    EXPECT_DEBUG_DEATH(static_cast<void>(camera.unproject({599.5f, 339.5f}, 0.f)), "");
}
