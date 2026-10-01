#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include "ovo/adapters/replica/replica_pose_source.hpp"
#include "ovo/core/common/image.hpp"
#include "ovo/core/common/strong_id.hpp"
#include "ovo/core/ports/frame_source.hpp"

using ovo::adapters::replica::ReplicaPoseSource;
using ovo::core::FrameId;
using ovo::core::Image;
using ovo::core::ports::Frame;

namespace {

// The pose source only looks at the id: a 1x1 depth is enough.
Frame frameWithId(std::uint32_t id) { return {FrameId{id}, Image<float>(1, 1)}; }

// Camera at (1, 2, 3), no rotation; and the same rotated 90 degrees about z.
const std::string kIdentityAt123 = "1 0 0 1  0 1 0 2  0 0 1 3  0 0 0 1";
const std::string kRotZ90At000 = "0 -1 0 0  1 0 0 0  0 0 1 0  0 0 0 1";

// A scene folder with a traj.txt holding `content`, deleted at the end of the test.
class TempTrajectory : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("ovo_test_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
    }
    void TearDown() override { std::filesystem::remove_all(dir_); }

    void write(const std::string& content) const { std::ofstream(dir_ / "traj.txt") << content; }

    std::filesystem::path dir_;
};

}  // namespace

TEST(ReplicaPoseSource, RejectsSceneWithoutTrajectory) {
    EXPECT_THROW(ReplicaPoseSource("this/folder/does/not/exist"), std::runtime_error);
}

TEST_F(TempTrajectory, ReadsOnePosePerLine) {
    write(kIdentityAt123 + "\n" + kRotZ90At000 + "\n");
    const ReplicaPoseSource poses(dir_);
    EXPECT_EQ(poses.size(), std::size_t{2});
}

// The matrix is camera-to-world, row by row: the translation is the 4th column (camera centre).
TEST_F(TempTrajectory, TranslationIsTheCameraCentre) {
    write(kIdentityAt123 + "\n");
    ReplicaPoseSource poses(dir_);
    const auto pose = poses.pose(frameWithId(0));
    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->center().isApprox(Eigen::Vector3f{1.f, 2.f, 3.f}));
}

// Row by row, not column by column: x of the camera goes to y of the world for a +90 degree turn about z.
TEST_F(TempTrajectory, ReadsRotationRowByRow) {
    write(kRotZ90At000 + "\n");
    ReplicaPoseSource poses(dir_);
    const auto pose = poses.pose(frameWithId(0));
    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->camToWorld({1.f, 0.f, 0.f}).isApprox(Eigen::Vector3f{0.f, 1.f, 0.f}));
}

TEST_F(TempTrajectory, LineIndexIsFrameId) {
    write(kIdentityAt123 + "\n" + kRotZ90At000 + "\n");
    ReplicaPoseSource poses(dir_);
    const auto pose = poses.pose(frameWithId(1));
    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->center().isApprox(Eigen::Vector3f::Zero()));
}

TEST_F(TempTrajectory, NoPoseForFrameBeyondTheFile) {
    write(kIdentityAt123 + "\n");
    ReplicaPoseSource poses(dir_);
    EXPECT_FALSE(poses.pose(frameWithId(5)).has_value());
}

// NaN in the GT: that frame has no pose (the Python skips it), the others still load.
TEST_F(TempTrajectory, NonFiniteLineHasNoPose) {
    write("nan 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1\n" + kIdentityAt123 + "\n");
    ReplicaPoseSource poses(dir_);
    EXPECT_FALSE(poses.pose(frameWithId(0)).has_value());
    EXPECT_TRUE(poses.pose(frameWithId(1)).has_value());
}

TEST_F(TempTrajectory, RejectsLineWithTooFewNumbers) {
    write("1 0 0 0  0 1 0 0\n");
    EXPECT_THROW(ReplicaPoseSource{dir_}, std::runtime_error);
}

TEST_F(TempTrajectory, IgnoresBlankLines) {
    write(kIdentityAt123 + "\n\n" + kRotZ90At000 + "\n\n");
    const ReplicaPoseSource poses(dir_);
    EXPECT_EQ(poses.size(), std::size_t{2});
}
