#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "ovo/adapters/replica/replica_frame_source.hpp"
#include "ovo/adapters/replica/replica_pose_source.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"

using ovo::adapters::replica::ReplicaFrameSource;
using ovo::core::geometry::PinholeCamera;

namespace {

// data/input/Datasets/Replica/office0 (OVO layout). OVO_DATASETS_DIR overrides the folder at run time.
std::filesystem::path office0() {
    const char* env = std::getenv("OVO_DATASETS_DIR");
    const std::filesystem::path datasets = env != nullptr ? env : OVO_DEFAULT_DATASETS_DIR;
    return datasets / "Replica" / "office0";
}

// data/working/configs/Replica/replica.yaml
const PinholeCamera kReplicaCamera{
    {.fx = 600.f, .fy = 600.f, .cx = 599.5f, .cy = 339.5f, .width = 1200, .height = 680}};
constexpr float kReplicaDepthScale = 6553.5f;

}  // namespace

class ReplicaOffice0 : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::is_directory(office0())) { GTEST_SKIP() << "Replica office0 not found at " << office0(); }
    }
};

TEST_F(ReplicaOffice0, Has2000Frames) {
    const ReplicaFrameSource source(office0(), kReplicaCamera, kReplicaDepthScale);
    EXPECT_EQ(source.size(), std::size_t{2000});
}

// Indoor scene: every measured depth is between a few cm and ~10 m, and most pixels have one.
TEST_F(ReplicaOffice0, FirstFrameHasIndoorDepthsInMetres) {
    const ReplicaFrameSource source(office0(), kReplicaCamera, kReplicaDepthScale);
    const auto frame = source.frame(0);
    EXPECT_EQ(frame.id.value(), 0u);
    ASSERT_EQ(frame.depth.width(), 1200u);
    ASSERT_EQ(frame.depth.height(), 680u);

    std::size_t measured = 0;
    float minDepth = 1e9f;
    float maxDepth = 0.f;
    const std::size_t numPixels = std::size_t{1200} * 680;
    for (std::size_t i = 0; i < numPixels; ++i) {
        const float d = frame.depth.data()[i];
        if (d > 0.f) {
            ++measured;
            minDepth = std::min(minDepth, d);
            maxDepth = std::max(maxDepth, d);
        }
    }
    EXPECT_GT(measured, numPixels / 2);
    EXPECT_GT(minDepth, 0.05f);
    EXPECT_LT(maxDepth, 10.f);
    std::cout << "office0 frame 0: " << measured << " measured pixels, depth [" << minDepth << ", " << maxDepth
              << "] m\n";
}

TEST_F(ReplicaOffice0, LastFrameLoads) {
    const ReplicaFrameSource source(office0(), kReplicaCamera, kReplicaDepthScale);
    EXPECT_EQ(source.frame(source.size() - 1).id.value(), 1999u);
}

// Every GT pose of office0 loads and passes Pose's checks (orthonormal, det +1, finite).
TEST_F(ReplicaOffice0, Loads2000ValidPoses) {
    ovo::adapters::replica::ReplicaPoseSource poses(office0());
    ASSERT_EQ(poses.size(), std::size_t{2000});
    for (std::uint32_t id = 0; id < 2000; ++id) {
        const ovo::core::ports::Frame frame{ovo::core::FrameId{id}, ovo::core::Image<float>(1, 1)};
        EXPECT_TRUE(poses.pose(frame).has_value()) << "frame " << id;
    }
}
