#pragma once

#include <compare>
#include <cstdint>

namespace ovo::core {

template <typename Tag>
class StrongId {
public:
    constexpr explicit StrongId(std::uint32_t value) noexcept : value_(value) {}

    [[nodiscard]] constexpr std::uint32_t value() const noexcept { return value_; }

    constexpr auto operator<=>(const StrongId&) const = default;

private:
    std::uint32_t value_;
};

using FrameId    = StrongId<struct FrameTag>;
using PointId    = StrongId<struct PointTag>;
using InstanceId = StrongId<struct InstanceTag>;

} // namespace ovo::core
