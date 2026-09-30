#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ovo::core::geometry {

class CameraIntrinsics {
public:

    struct Params {
        float fx;
        float fy;
        float cx;
        float cy;
        std::uint32_t width;
        std::uint32_t height;
    };

    constexpr explicit CameraIntrinsics(const Params& p)
        : fx_(p.fx),
          fy_(p.fy),
          cx_(p.cx),
          cy_(p.cy),
          width_(p.width),
          height_(p.height) {
        if (!isPositiveFinite(fx_) || !isPositiveFinite(fy_)) { throw std::invalid_argument("CameraIntrinsics: fx y fy deben ser finitos y > 0"); }
        if (!isFinite(cx_) || !isFinite(cy_))                 { throw std::invalid_argument("CameraIntrinsics: cx y cy deben ser finitos"); }
        if (width_ == 0 || height_ == 0)                      { throw std::invalid_argument("CameraIntrinsics: width y height deben ser > 0"); }
    }

    [[nodiscard]] constexpr float fx() const noexcept { return fx_; }
    [[nodiscard]] constexpr float fy() const noexcept { return fy_; }
    [[nodiscard]] constexpr float cx() const noexcept { return cx_; }
    [[nodiscard]] constexpr float cy() const noexcept { return cy_; }
    [[nodiscard]] constexpr std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] constexpr std::uint32_t height() const noexcept { return height_; }

private:
    // Comparaciones escritas "en positivo" para que NaN dé false (cualquier comparación con NaN es false).
    // Sin <cmath>: std::isfinite no es constexpr en todos los compiladores.
    static constexpr float kInf = std::numeric_limits<float>::infinity();
    static constexpr bool isFinite(float x) noexcept { return x > -kInf && x < kInf; }
    static constexpr bool isPositiveFinite(float x) noexcept { return x > 0.f && x < kInf; }

    float fx_;
    float fy_;
    float cx_;
    float cy_;
    std::uint32_t width_;
    std::uint32_t height_;
};

}  // namespace ovo::core::geometry
