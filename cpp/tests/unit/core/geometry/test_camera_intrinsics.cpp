#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ovo/core/geometry/camera_intrinsics.hpp"

using namespace ovo::core::geometry;

// Parameters from Replica cameras.
constexpr CameraIntrinsics::Params kReplica{
    .fx = 600.f, .fy = 600.f, .cx = 599.5f, .cy = 339.5f, .width = 1200, .height = 680};


TEST(CameraIntrinsics, StoresParams) {
    const CameraIntrinsics camera{kReplica};
    EXPECT_FLOAT_EQ(kReplica.fx, camera.fx());
    EXPECT_FLOAT_EQ(kReplica.fy, camera.fy());
    EXPECT_FLOAT_EQ(kReplica.cx, camera.cx());
    EXPECT_FLOAT_EQ(kReplica.cy, camera.cy());
    EXPECT_EQ(kReplica.width, camera.width());
    EXPECT_EQ(kReplica.height, camera.height());
}

TEST(CameraIntrinsics, RejectsZeroFocal) {
    auto p = kReplica;
    p.fx = 0.f;
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}

TEST(CameraIntrinsics, RejectsNegativeFocal) {
    auto p = kReplica;
    p.fy = -1.f;
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}

TEST(CameraIntrinsics, RejectsZeroSize) {
    auto p = kReplica;
    p.width = 0;
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}

TEST(CameraIntrinsics, RejectsNaNFocal) {
    auto p = kReplica;
    p.fx = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}
TEST(CameraIntrinsics, RejectsZeroHeight) {
    auto p = kReplica;
    p.height = 0;
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}

TEST(CameraIntrinsics, RejectsInfiniteFocal) {
    auto p = kReplica;
    p.fx = std::numeric_limits<float>::infinity();
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}

TEST(CameraIntrinsics, RejectsNaNPrincipalPoint) {
    auto p = kReplica;
    p.cx = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}

TEST(CameraIntrinsics, RejectsInfinitePrincipalPoint) {
    auto p = kReplica;
    p.cy = -std::numeric_limits<float>::infinity();
    EXPECT_THROW(CameraIntrinsics{p}, std::invalid_argument);
}
