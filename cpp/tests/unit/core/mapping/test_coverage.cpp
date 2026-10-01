#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ovo/core/common/image.hpp"
#include "ovo/core/geometry/point_pixel_matcher.hpp"
#include "ovo/core/mapping/coverage.hpp"

using ovo::core::Image;
using ovo::core::geometry::PointPixelMatch;
using ovo::core::mapping::Coverage;
using ovo::core::mapping::Pixel;

// Small 6x6 image with depth everywhere: every pixel can be checked by hand.
constexpr std::uint32_t kSize = 6;

Image<float> fullDepth() {
    Image<float> depth(kSize, kSize);
    std::fill(depth.data(), depth.data() + std::size_t{kSize} * kSize, 2.f);
    return depth;
}

bool contains(const std::vector<Pixel>& pixels, std::uint32_t u, std::uint32_t v) {
    return std::any_of(pixels.begin(), pixels.end(), [&](const Pixel& p) { return p.u == u && p.v == v; });
}

const std::vector<PointPixelMatch> kNoMatches;

// --- Construction: dilationSize and step are configuration, so a bad value throws.

TEST(Coverage, RejectsEvenDilationSize) {
    EXPECT_THROW(Coverage(2, 1), std::invalid_argument);
}

TEST(Coverage, RejectsZeroDilationSize) {
    EXPECT_THROW(Coverage(0, 1), std::invalid_argument);
}

TEST(Coverage, RejectsZeroStep) {
    EXPECT_THROW(Coverage(1, 0), std::invalid_argument);
}

// --- Without matches: only depth and the step grid decide.

TEST(CoverageFreePixels, EveryPixelWithDepthIsFree) {
    const Coverage coverage(1, 1);
    EXPECT_EQ(coverage.freePixels(kNoMatches, fullDepth()).size(), std::size_t{kSize * kSize});
}

TEST(CoverageFreePixels, PixelWithoutDepthIsNotFree) {
    const Coverage coverage(1, 1);
    Image<float> depth = fullDepth();
    depth.at(4, 1) = 0.f;
    const auto free = coverage.freePixels(kNoMatches, depth);
    EXPECT_EQ(free.size(), std::size_t{kSize * kSize - 1});
    EXPECT_FALSE(contains(free, 4, 1));
}

// Step 2 keeps u and v even: 3 x 3 = 9 of the 36 pixels (Python depth[::2, ::2]).
TEST(CoverageFreePixels, StepKeepsOnlyGridPixels) {
    const Coverage coverage(1, 2);
    const auto free = coverage.freePixels(kNoMatches, fullDepth());
    EXPECT_EQ(free.size(), std::size_t{9});
    EXPECT_TRUE(contains(free, 4, 2));
    EXPECT_FALSE(contains(free, 3, 2));
    EXPECT_FALSE(contains(free, 2, 3));
}

// Pixels keep their full-resolution coordinates (the mapper unprojects them with the same camera).
TEST(CoverageFreePixels, ReturnsFullResolutionCoordinatesInRowOrder) {
    const Coverage coverage(1, 2);
    const auto free = coverage.freePixels(kNoMatches, fullDepth());
    ASSERT_EQ(free.size(), std::size_t{9});
    EXPECT_EQ(free[1].u, 2u);  // (0,0), (2,0), (4,0), (0,2)...
    EXPECT_EQ(free[1].v, 0u);
    EXPECT_EQ(free[3].u, 0u);
    EXPECT_EQ(free[3].v, 2u);
}

// --- Matches block their window.

TEST(CoverageFreePixels, MatchBlocksItsOwnPixel) {
    const Coverage coverage(1, 1);
    const std::vector<PointPixelMatch> matches{{0, 2, 3}};
    const auto free = coverage.freePixels(matches, fullDepth());
    EXPECT_EQ(free.size(), std::size_t{kSize * kSize - 1});
    EXPECT_FALSE(contains(free, 2, 3));
}

// Dilation 3: the match at (2, 2) blocks (1..3, 1..3), nothing further.
TEST(CoverageFreePixels, MatchBlocksItsWindow) {
    const Coverage coverage(3, 1);
    const std::vector<PointPixelMatch> matches{{0, 2, 2}};
    const auto free = coverage.freePixels(matches, fullDepth());
    EXPECT_EQ(free.size(), std::size_t{kSize * kSize - 9});
    EXPECT_FALSE(contains(free, 1, 1));
    EXPECT_FALSE(contains(free, 3, 3));
    EXPECT_TRUE(contains(free, 4, 2));
    EXPECT_TRUE(contains(free, 2, 0));
}

// A match on the corner: the window is clipped to the image (no unsigned wrap, no crash).
TEST(CoverageFreePixels, WindowIsClippedAtImageBorder) {
    const Coverage coverage(3, 1);
    const std::vector<PointPixelMatch> matches{{0, 0, 0}};
    const auto free = coverage.freePixels(matches, fullDepth());
    EXPECT_EQ(free.size(), std::size_t{kSize * kSize - 4});  // (0..1, 0..1)
    EXPECT_FALSE(contains(free, 1, 1));
    EXPECT_TRUE(contains(free, 2, 0));
}

// Python dilates the "not free" mask, which includes pixels without depth: a hole also blocks its neighbours.
TEST(CoverageFreePixels, PixelWithoutDepthBlocksItsWindow) {
    const Coverage coverage(3, 1);
    Image<float> depth = fullDepth();
    depth.at(2, 2) = 0.f;
    const auto free = coverage.freePixels(kNoMatches, depth);
    EXPECT_FALSE(contains(free, 3, 3));
    EXPECT_TRUE(contains(free, 4, 4));
}

// Dilation then subsampling: grid pixel (2, 2) is blocked by a match on its neighbour (3, 3), off the grid.
TEST(CoverageFreePixels, OffGridMatchBlocksGridNeighbour) {
    const Coverage coverage(3, 2);
    const std::vector<PointPixelMatch> matches{{0, 3, 3}};
    const auto free = coverage.freePixels(matches, fullDepth());
    EXPECT_FALSE(contains(free, 2, 2));
    EXPECT_FALSE(contains(free, 4, 4));
    EXPECT_TRUE(contains(free, 0, 0));
}

// A match outside the image is a caller bug (the matcher never returns one): assert.
TEST(CoverageDeathTest, RejectsMatchOutsideImage) {
    const Coverage coverage(1, 1);
    const std::vector<PointPixelMatch> matches{{0, kSize, 0}};
    EXPECT_DEBUG_DEATH(static_cast<void>(coverage.freePixels(matches, fullDepth())), "");
}
