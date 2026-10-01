#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "ovo/adapters/replica/replica_frame_source.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"

using ovo::adapters::replica::ReplicaFrameSource;
using ovo::core::geometry::PinholeCamera;

namespace {

const PinholeCamera kCamera{{.fx = 600.f, .fy = 600.f, .cx = 599.5f, .cy = 339.5f, .width = 1200, .height = 680}};
constexpr float kDepthScale = 6553.5f;

// A fresh empty scene folder under the system temp dir, deleted at the end of the test.
class TempScene : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("ovo_test_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_ / "results");
    }
    void TearDown() override { std::filesystem::remove_all(dir_); }

    // Empty file: enough for listing; not a valid PNG.
    void touch(const std::string& name) const { std::ofstream(dir_ / "results" / name).put('x'); }

    std::filesystem::path dir_;
};

}  // namespace

TEST(ReplicaFrameSource, RejectsSceneWithoutResultsFolder) {
    EXPECT_THROW(ReplicaFrameSource("this/folder/does/not/exist", kCamera, kDepthScale), std::runtime_error);
}

// depthScale converts the raw uint16 to metres: 0 would give infinite depths, negative ones would flip them.
TEST_F(TempScene, RejectsNonPositiveDepthScale) {
    touch("depth000000.png");
    EXPECT_THROW(ReplicaFrameSource(dir_, kCamera, 0.f), std::invalid_argument);
    EXPECT_THROW(ReplicaFrameSource(dir_, kCamera, -1.f), std::invalid_argument);
}

TEST_F(TempScene, RejectsSceneWithoutDepthImages) {
    touch("frame000000.jpg");
    EXPECT_THROW(ReplicaFrameSource(dir_, kCamera, kDepthScale), std::runtime_error);
}

// Only depth*.png count: RGB frames and other files are ignored.
TEST_F(TempScene, CountsOnlyDepthImages) {
    touch("depth000000.png");
    touch("depth000001.png");
    touch("frame000000.jpg");
    touch("depth_notes.txt");
    const ReplicaFrameSource source(dir_, kCamera, kDepthScale);
    EXPECT_EQ(source.size(), std::size_t{2});
}

TEST_F(TempScene, KeepsTheCamera) {
    touch("depth000000.png");
    const ReplicaFrameSource source(dir_, kCamera, kDepthScale);
    EXPECT_EQ(source.camera().width(), 1200u);
    EXPECT_EQ(source.camera().fx(), 600.f);
}

// A file that is not a PNG: clear error, not a crash.
TEST_F(TempScene, FrameThrowsOnUnreadableImage) {
    touch("depth000000.png");
    const ReplicaFrameSource source(dir_, kCamera, kDepthScale);
    EXPECT_THROW(static_cast<void>(source.frame(0)), std::runtime_error);
}
