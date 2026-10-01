// ovo_dense_map: builds the dense point cloud of one Replica scene with ground-truth poses and writes it as PLY.
//
//   ovo_dense_map <sceneDir> <output.ply> [mapEvery = 10] [maxFrames = all]
//
// Composition root: the only place that knows every concrete piece (adapters + domain) and wires them.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ovo/adapters/ply/ply_writer.hpp"
#include "ovo/adapters/replica/replica_frame_source.hpp"
#include "ovo/adapters/replica/replica_pose_source.hpp"
#include "ovo/core/geometry/pinhole_camera.hpp"
#include "ovo/core/mapping/dense_mapper.hpp"
#include "ovo/core/ports/frame_source.hpp"
#include "ovo/core/ports/pose_source.hpp"

namespace {

using namespace ovo;

struct Options {
    std::filesystem::path sceneDir;
    std::filesystem::path outputPly;
    std::size_t mapEvery = 10;  // Python ovomapping.py map_every
    std::size_t maxFrames = 0;  // 0 = all
};

struct RunStats {
    std::size_t mappedFrames = 0;
    std::size_t framesWithoutPose = 0;
};

// TODO: read from data/working/configs/Replica/replica.yaml once there is a YAML reader.
const core::geometry::PinholeCamera kReplicaCamera{
    {.fx = 600.f, .fy = 600.f, .cx = 599.5f, .cy = 339.5f, .width = 1200, .height = 680}};
constexpr float kReplicaDepthScale = 6553.5f;

Options parseArguments(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        throw std::invalid_argument("usage: ovo_dense_map <sceneDir> <output.ply> [mapEvery = 10] [maxFrames = all]");
    }
    Options options{.sceneDir = argv[1], .outputPly = argv[2]};
    if (argc >= 4) { options.mapEvery = std::stoul(argv[3]); }
    if (argc >= 5) { options.maxFrames = std::stoul(argv[4]); }
    if (options.mapEvery == 0) { throw std::invalid_argument("mapEvery must be >= 1"); }
    return options;
}

// The orchestration loop: every mapEvery frames, load the frame, ask for its pose, map it.
RunStats mapScene(core::ports::FrameSource& frames, core::ports::PoseSource& poses,
                  core::mapping::DenseMapper& mapper, const Options& options) {
    const std::size_t numFrames =
        options.maxFrames == 0 ? frames.size() : std::min(options.maxFrames, frames.size());
    RunStats stats;
    for (std::size_t i = 0; i < numFrames; i += options.mapEvery) {
        const core::ports::Frame frame = frames.frame(i);
        const auto pose = poses.pose(frame);
        if (!pose) {
            ++stats.framesWithoutPose;
            continue;
        }
        mapper.map(frame.id, frame.depth, *pose);
        ++stats.mappedFrames;
        std::cout << "frame " << i << "/" << numFrames << ": " << mapper.geometricMap().numPoints() << " points\n";
    }
    return stats;
}

void run(const Options& options) {
    const auto start = std::chrono::steady_clock::now();

    adapters::replica::ReplicaFrameSource frames(options.sceneDir, kReplicaCamera, kReplicaDepthScale);
    adapters::replica::ReplicaPoseSource poses(options.sceneDir);
    core::mapping::DenseMapper mapper(frames.camera(), {});  // default config = Python VanillaMapper

    const RunStats stats = mapScene(frames, poses, mapper, options);
    adapters::ply::writePly(mapper.geometricMap(), options.outputPly);

    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Done: " << stats.mappedFrames << " frames mapped (" << stats.framesWithoutPose
              << " without pose), " << mapper.geometricMap().numPoints() << " points -> " << options.outputPly.string()
              << " in " << seconds << " s\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parseArguments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
