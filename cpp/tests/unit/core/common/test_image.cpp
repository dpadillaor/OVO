#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "ovo/core/common/image.hpp"

using ovo::core::Image;

// Not square on purpose: a test that mixes up width and height (or u and v) must fail.
constexpr std::uint32_t kWidth = 4;
constexpr std::uint32_t kHeight = 3;

TEST(Image, HasRequestedSize) {
    const Image<float> image(kWidth, kHeight);
    EXPECT_EQ(image.width(), kWidth);
    EXPECT_EQ(image.height(), kHeight);
}

// Depth 0 means "no measurement": a new image must not hold garbage.
TEST(Image, StartsZeroed) {
    const Image<float> image(kWidth, kHeight);
    for (std::size_t i = 0; i < std::size_t{kWidth} * kHeight; ++i) {
        EXPECT_EQ(image.data()[i], 0.f) << "pixel " << i;
    }
}

TEST(ImageAt, ReadsBackWhatWasWritten) {
    Image<float> image(kWidth, kHeight);
    image.at(1, 2) = 7.f;
    EXPECT_EQ(image.at(1, 2), 7.f);
}

// Catches at(v, u) or a formula with u and v swapped.
TEST(ImageAt, DoesNotSwapUAndV) {
    Image<float> image(kWidth, kHeight);
    image.at(1, 2) = 7.f;
    EXPECT_EQ(image.at(2, 1), 0.f);
}

// Reading through a const reference must compile and see the same pixel (the const at()).
TEST(ImageAt, ReadsThroughConstReference) {
    Image<float> image(kWidth, kHeight);
    image.at(3, 1) = 5.f;
    const Image<float>& constImage = image;
    EXPECT_EQ(constImage.at(3, 1), 5.f);
}

// Row-major like numpy: pixel (u, v) is element v * width + u. cv::Mat and from_blob rely on it.
TEST(ImageLayout, IsRowMajor) {
    Image<float> image(kWidth, kHeight);
    image.at(3, 1) = 5.f;
    EXPECT_EQ(image.data()[1 * kWidth + 3], 5.f);
}

TEST(ImageLayout, LastPixelIsLastElement) {
    Image<float> image(kWidth, kHeight);
    image.at(kWidth - 1, kHeight - 1) = 9.f;
    EXPECT_EQ(image.data()[std::size_t{kWidth} * kHeight - 1], 9.f);
}

// Masks are uint8_t (not bool, see ADR-0003): the same template must work for them.
TEST(ImageTypes, WorksWithByteMask) {
    Image<std::uint8_t> mask(kWidth, kHeight);
    mask.at(2, 0) = 255;
    EXPECT_EQ(mask.at(2, 0), 255);
    EXPECT_EQ(mask.at(0, 0), 0);
}

// Preconditions: u < width, v < height, size > 0. Breaking them is a bug: assert.
// u and v are unsigned, so there is no negative case: -1 would wrap to a huge number, also out of range.
TEST(ImageDeathTest, AtRejectsUOutOfRange) {
    const Image<float> image(kWidth, kHeight);
    EXPECT_DEBUG_DEATH(static_cast<void>(image.at(kWidth, 0)), "");
}

TEST(ImageDeathTest, AtRejectsVOutOfRange) {
    const Image<float> image(kWidth, kHeight);
    EXPECT_DEBUG_DEATH(static_cast<void>(image.at(0, kHeight)), "");
}

TEST(ImageDeathTest, RejectsZeroSize) {
    EXPECT_DEBUG_DEATH(static_cast<void>(Image<float>(0, kHeight)), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(Image<float>(kWidth, 0)), "");
}
