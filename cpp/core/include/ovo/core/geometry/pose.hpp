#pragma once

#include <cassert>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>

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

    [[nodiscard]] const Eigen::Vector3f& center() const noexcept { return translation_; }

private:
    Pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) noexcept
        : rotation_(rotation), translation_(translation) {
        assert((rotation_.transpose() * rotation_).isIdentity(1e-4f));  // orthonormal
        assert(std::abs(rotation_.determinant() - 1.f) < 1e-4f);        // no reflection
        assert(rotation_.allFinite() && translation_.allFinite());      // no NaN or inf
    }

    Eigen::Matrix3f rotation_;     // c2w rotation
    Eigen::Vector3f translation_;  // c2w translation = camera centre in world
};

}  // namespace ovo::core::geometry
