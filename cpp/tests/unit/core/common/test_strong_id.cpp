#include <gtest/gtest.h>

#include <cstdint> 
#include <type_traits>

#include "ovo/core/common/strong_id.hpp"

using namespace ovo::core;


static_assert(sizeof(PointId) == sizeof(std::uint32_t));
static_assert(!std::is_convertible_v<FrameId, PointId>);
static_assert(!std::is_convertible_v<std::uint32_t, PointId>);
static_assert(PointId{3} < PointId{5});


TEST(StrongId, StoresValue) {
    const PointId p{42};
    EXPECT_EQ(p.value(), 42u);
}

TEST(StrongId, EqualWhenSameValue) {
    const PointId p1{42};
    const PointId p2{42};
    EXPECT_EQ(p1, p2);
}

TEST(StrongId, OrderedByValue) {
    const PointId p1{45};
    const PointId p2{50};
    EXPECT_LT(p1, p2);
}

TEST(StrongId, NotEqualWhenDifferentValue) {
    const PointId p1{45};
    const PointId p2{50};
    EXPECT_NE(p1, p2);
}
