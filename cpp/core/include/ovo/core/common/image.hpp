#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ovo::core {

template <typename T>
class Image {
public:
    Image(std::uint32_t width, std::uint32_t height)
        : width_(width), height_(height), data_(static_cast<std::size_t>(width) * height) {
        assert(width > 0 && height > 0);
    }

    // Getters
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }

    // Accessors
    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    // Element access
    [[nodiscard]] T& at(std::uint32_t u, std::uint32_t v) noexcept {
        assert(u < width_ && v < height_);
        return data_[static_cast<std::size_t>(v) * width_ + u];
    }
    [[nodiscard]] const T& at(std::uint32_t u, std::uint32_t v) const noexcept {
        assert(u < width_ && v < height_);
        return data_[static_cast<std::size_t>(v) * width_ + u];
    }

private:
    std::uint32_t width_;
    std::uint32_t height_;
    std::vector<T> data_;
};

}  // namespace ovo::core
