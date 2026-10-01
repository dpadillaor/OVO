#include <gtest/gtest.h>

#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ovo/core/geometry/pose.hpp"

using namespace ovo::core::geometry;

TEST(Pose, IdentityHasIdentityRotation) {
    const Pose cameraPose = Pose::identity();
    EXPECT_TRUE(cameraPose.rotation().isIdentity());
    EXPECT_TRUE(cameraPose.translation().isZero());
}

TEST(Pose, StoresRotationAndTranslation) {
    const Eigen::Vector3f t{1.f, 2.f, 3.f};
    const Eigen::Matrix3f R = (Eigen::Matrix3f() << 0.f, -1.f, 0.f,
                                                    1.f,  0.f, 0.f,
                                                    0.f,  0.f, 1.f).finished();
    const Pose cameraPose = Pose::fromCamToWorld(R, t);
    EXPECT_TRUE(cameraPose.rotation().isApprox(R));
    EXPECT_TRUE(cameraPose.translation().isApprox(t));
}

// 90 degrees around z: camera x axis -> world y, camera y axis -> world -x.
const Eigen::Matrix3f kRotZ90 = (Eigen::Matrix3f() << 0.f, -1.f, 0.f,
                                                      1.f,  0.f, 0.f,
                                                      0.f,  0.f, 1.f).finished();

TEST(PoseCamToWorld, IdentityLeavesPointUnchanged) {
    const Pose cameraPose = Pose::identity();
    const Eigen::Vector3f pointWorld = cameraPose.camToWorld({1.f, 2.f, 3.f});
    EXPECT_FLOAT_EQ(pointWorld.x(), 1.f);
    EXPECT_FLOAT_EQ(pointWorld.y(), 2.f);
    EXPECT_FLOAT_EQ(pointWorld.z(), 3.f);
}

// Camera at (2,0,0), not rotated: a point 1 m in front of it is at (2,0,1).
TEST(PoseCamToWorld, TranslationOnly) {
    const Pose cameraPose = Pose::fromCamToWorld(Eigen::Matrix3f::Identity(), {2.f, 0.f, 0.f});
    const Eigen::Vector3f pointWorld = cameraPose.camToWorld({0.f, 0.f, 1.f});
    EXPECT_FLOAT_EQ(pointWorld.x(), 2.f);
    EXPECT_FLOAT_EQ(pointWorld.y(), 0.f);
    EXPECT_FLOAT_EQ(pointWorld.z(), 1.f);
}

TEST(PoseCamToWorld, RotationOnly) {
    const Pose cameraPose = Pose::fromCamToWorld(kRotZ90, Eigen::Vector3f::Zero());
    const Eigen::Vector3f pointWorld = cameraPose.camToWorld({1.f, 0.f, 0.f});
    EXPECT_FLOAT_EQ(pointWorld.x(), 0.f);
    EXPECT_FLOAT_EQ(pointWorld.y(), 1.f);
    EXPECT_FLOAT_EQ(pointWorld.z(), 0.f);
}

// Rotate first, then translate: (1,0,0) -> (0,1,0) -> (0,1,0) + (1,2,3) = (1,3,3).
// Translating first would give a different point, so this catches the wrong order.
TEST(PoseCamToWorld, RotationAndTranslation) {
    const Pose cameraPose = Pose::fromCamToWorld(kRotZ90, {1.f, 2.f, 3.f});
    const Eigen::Vector3f pointWorld = cameraPose.camToWorld({1.f, 0.f, 0.f});
    EXPECT_FLOAT_EQ(pointWorld.x(), 1.f);
    EXPECT_FLOAT_EQ(pointWorld.y(), 3.f);
    EXPECT_FLOAT_EQ(pointWorld.z(), 3.f);
}

// Same camera as PoseCamToWorld.TranslationOnly, the other way around.
TEST(PoseWorldToCam, TranslationOnly) {
    const Pose cameraPose = Pose::fromCamToWorld(Eigen::Matrix3f::Identity(), {2.f, 0.f, 0.f});
    const Eigen::Vector3f pointCam = cameraPose.worldToCam({2.f, 0.f, 1.f});
    EXPECT_FLOAT_EQ(pointCam.x(), 0.f);
    EXPECT_FLOAT_EQ(pointCam.y(), 0.f);
    EXPECT_FLOAT_EQ(pointCam.z(), 1.f);
}

// Same camera as PoseCamToWorld.RotationAndTranslation, the other way around.
TEST(PoseWorldToCam, RotationAndTranslation) {
    const Pose cameraPose = Pose::fromCamToWorld(kRotZ90, {1.f, 2.f, 3.f});
    const Eigen::Vector3f pointCam = cameraPose.worldToCam({1.f, 3.f, 3.f});
    EXPECT_FLOAT_EQ(pointCam.x(), 1.f);
    EXPECT_FLOAT_EQ(pointCam.y(), 0.f);
    EXPECT_FLOAT_EQ(pointCam.z(), 0.f);
}

// The camera centre is the origin of the camera's own frame.
TEST(PoseWorldToCam, CenterGoesToOrigin) {
    const Pose cameraPose = Pose::fromCamToWorld(kRotZ90, {1.f, 2.f, 3.f});
    const Eigen::Vector3f pointCam = cameraPose.worldToCam({1.f, 2.f, 3.f});
    EXPECT_FLOAT_EQ(pointCam.x(), 0.f);
    EXPECT_FLOAT_EQ(pointCam.y(), 0.f);
    EXPECT_FLOAT_EQ(pointCam.z(), 0.f);
}

// A generic rotation (not 90 degrees) so rounding shows up: EXPECT_NEAR.
TEST(PoseWorldToCam, RoundTripWithCamToWorld) {
    const Eigen::Matrix3f rotation =
        Eigen::AngleAxisf(0.7f, Eigen::Vector3f{1.f, 2.f, 3.f}.normalized()).toRotationMatrix();
    const Pose cameraPose = Pose::fromCamToWorld(rotation, {0.5f, -1.f, 2.f});
    const Eigen::Vector3f original{0.3f, -0.7f, 2.5f};
    const Eigen::Vector3f back = cameraPose.worldToCam(cameraPose.camToWorld(original));
    EXPECT_NEAR(back.x(), original.x(), 1e-5f);
    EXPECT_NEAR(back.y(), original.y(), 1e-5f);
    EXPECT_NEAR(back.z(), original.z(), 1e-5f);
}

TEST(PoseCenter, EqualsTranslation) {
    const Eigen::Vector3f translation{1.f, 2.f, 3.f};
    const Pose cameraPose = Pose::fromCamToWorld(kRotZ90, translation);
    const Eigen::Vector3f center = cameraPose.center();
    EXPECT_FLOAT_EQ(center.x(), translation.x());
    EXPECT_FLOAT_EQ(center.y(), translation.y());
    EXPECT_FLOAT_EQ(center.z(), translation.z());
}

TEST(PoseCenter, CameraOriginGoesToCenter) {
    const Pose cameraPose = Pose::fromCamToWorld(kRotZ90, {1.f, 2.f, 3.f});
    const Eigen::Vector3f originInWorld = cameraPose.camToWorld(Eigen::Vector3f::Zero());
    EXPECT_TRUE(originInWorld.isApprox(cameraPose.center()));
}

TEST(PoseDeathTest, RejectsNonOrthonormalRotation) {
    const Eigen::Matrix3f scaled = 2.f * kRotZ90;
    EXPECT_DEBUG_DEATH(static_cast<void>(Pose::fromCamToWorld(scaled, Eigen::Vector3f::Zero())), "");
}

TEST(PoseDeathTest, RejectsReflection) {
    const Eigen::Matrix3f mirrorZ = Eigen::Vector3f{1.f, 1.f, -1.f}.asDiagonal();
    EXPECT_DEBUG_DEATH(static_cast<void>(Pose::fromCamToWorld(mirrorZ, Eigen::Vector3f::Zero())), "");
}

TEST(PoseDeathTest, RejectsNaN) {
    const Eigen::Vector3f badTranslation{1.f, std::numeric_limits<float>::quiet_NaN(), 3.f};
    EXPECT_DEBUG_DEATH(static_cast<void>(Pose::fromCamToWorld(kRotZ90, badTranslation)), "");
}
