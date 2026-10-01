#include "ovo/adapters/replica/replica_pose_source.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace ovo::adapters::replica {

namespace {

using MatrixRow = std::array<double, 16>;  // a 4x4 matrix written row by row

bool isBlank(const std::string& line) { return line.find_first_not_of(" \t\r") == std::string::npos; }

// The 16 numbers of one line. Throws if there are fewer, or something that is not a number.
// std::from_chars, not `stream >> double`: the stream rejects "nan" / "inf", which the GT may contain.
MatrixRow parseMatrixLine(const std::string& line, std::size_t lineNumber, const std::filesystem::path& file) {
    std::istringstream tokens(line);
    MatrixRow m{};
    for (double& value : m) {
        std::string token;
        const bool ok = static_cast<bool>(tokens >> token) &&
                        std::from_chars(token.data(), token.data() + token.size(), value).ec == std::errc{};
        if (!ok) {
            throw std::runtime_error("ReplicaPoseSource: line " + std::to_string(lineNumber) + " of " + file.string() +
                                     " does not have 16 numbers");
        }
    }
    return m;
}

// Camera-to-world matrix -> Pose. Nothing if any value is NaN or inf (the Python skips those poses).
// The text has ~16 digits but Pose stores float: re-orthonormalise R (closest rotation, via SVD) so the
// rounding never trips Pose's orthonormality assert.
std::optional<core::geometry::Pose> toPose(const MatrixRow& m) {
    for (const double value : m) {
        if (!std::isfinite(value)) { return std::nullopt; }
    }
    Eigen::Matrix3d rotation;
    rotation << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10];
    const Eigen::Vector3d translation{m[3], m[7], m[11]};

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d closestRotation = svd.matrixU() * svd.matrixV().transpose();
    return core::geometry::Pose::fromCamToWorld(closestRotation.cast<float>(), translation.cast<float>());
}

}  // namespace

ReplicaPoseSource::ReplicaPoseSource(const std::filesystem::path& sceneDir) {
    const std::filesystem::path file = sceneDir / "traj.txt";
    std::ifstream in(file);
    if (!in) { throw std::runtime_error("ReplicaPoseSource: cannot open " + file.string()); }

    std::string line;
    for (std::size_t lineNumber = 1; std::getline(in, line); ++lineNumber) {
        if (isBlank(line)) { continue; }
        poses_.push_back(toPose(parseMatrixLine(line, lineNumber, file)));
    }
}

std::optional<core::geometry::Pose> ReplicaPoseSource::pose(const core::ports::Frame& frame) {
    const std::size_t index = frame.id.value();
    if (index >= poses_.size()) { return std::nullopt; }  // no GT for this frame
    return poses_[index];
}

}  // namespace ovo::adapters::replica
