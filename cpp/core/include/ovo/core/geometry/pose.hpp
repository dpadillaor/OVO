#pragma once

#include <Eigen/Core>

namespace ovo::core::geometry {

class Pose {
public:
    [[nodiscard]] static Pose fromCamToWorld(const Eigen::Matrix3f& rotation,
                                             const Eigen::Vector3f& translation) noexcept {
        return Pose(rotation, translation);
    }

    [[nodiscard]] static Pose identity() noexcept {
        return Pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
    }

    [[nodiscard]] const Eigen::Matrix3f& rotation() const noexcept { return rotation_; }
    [[nodiscard]] const Eigen::Vector3f& translation() const noexcept { return translation_; }

    [[nodiscard]] Eigen::Vector3f camToWorld(const Eigen::Vector3f& pointCam) const noexcept {
        return rotation_ * pointCam + translation_;
    }

    [[nodiscard]] Eigen::Vector3f worldToCam(const Eigen::Vector3f& pointWorld) const noexcept {
        return rotation_.transpose() * (pointWorld - translation_);
    }

private:
    Pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) noexcept
        : rotation_(rotation), translation_(translation) {}

    Eigen::Matrix3f rotation_;     // c2w rotation
    Eigen::Vector3f translation_;  // c2w translation = camera centre in world
};

}  // namespace ovo::core::geometry
