#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>

namespace ovo::core::geometry {

class PinholeCamera {
public:
    struct Params {
        float fx;
        float fy;
        float cx;
        float cy;
        std::uint32_t width;
        std::uint32_t height;
    };

    constexpr explicit PinholeCamera(const Params& p)
        : fx_(p.fx),
          fy_(p.fy),
          cx_(p.cx),
          cy_(p.cy),
          width_(p.width),
          height_(p.height),
          invFx_(1.f / p.fx),
          invFy_(1.f / p.fy) {
        if (!isPositiveFinite(fx_) || !isPositiveFinite(fy_)) { throw std::invalid_argument("PinholeCamera: fx and fy must be finite and > 0"); }
        if (!isFinite(cx_) || !isFinite(cy_))                 { throw std::invalid_argument("PinholeCamera: cx and cy must be finite"); }
        if (width_ == 0 || height_ == 0)                      { throw std::invalid_argument("PinholeCamera: width and height must be > 0"); }
    }

    [[nodiscard]] constexpr float fx() const noexcept { return fx_; }
    [[nodiscard]] constexpr float fy() const noexcept { return fy_; }
    [[nodiscard]] constexpr float cx() const noexcept { return cx_; }
    [[nodiscard]] constexpr float cy() const noexcept { return cy_; }
    [[nodiscard]] constexpr std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] constexpr std::uint32_t height() const noexcept { return height_; }

    [[nodiscard]] Eigen::Vector2f project(const Eigen::Vector3f& pointCam) const noexcept {
        assert(pointCam.z() > 0.f);
        const float invZ = 1.f / pointCam.z();
        const float u = fx_ * pointCam.x() * invZ + cx_;
        const float v = fy_ * pointCam.y() * invZ + cy_;
        return Eigen::Vector2f{u, v};
    }

    [[nodiscard]] Eigen::Vector3f unproject(const Eigen::Vector2f& pixel, float depth) const noexcept {
        assert(depth > 0.f);
        const float x = (pixel.x() - cx_) * depth * invFx_;
        const float y = (pixel.y() - cy_) * depth * invFy_;
        return Eigen::Vector3f{x, y, depth};
    }

private:
    static constexpr float kInf = std::numeric_limits<float>::infinity();
    static constexpr bool isFinite(float x) noexcept { return x > -kInf && x < kInf; }
    static constexpr bool isPositiveFinite(float x) noexcept { return x > 0.f && x < kInf; }

    float fx_;
    float fy_;
    float cx_;
    float cy_;
    std::uint32_t width_;
    std::uint32_t height_;
    float invFx_;  // 1 / fx, precomputed: unproject runs once per pixel
    float invFy_;  // 1 / fy
};

}  // namespace ovo::core::geometry
