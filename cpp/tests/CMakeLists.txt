# Unit tests: GoogleTest + GoogleMock, fetched with FetchContent (pinned version).
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Core unit tests: fast, no data, no external dependencies.
# Folder mirrors the header layout. integration/ and regression/ will get their own executables.
add_executable(ovo_unit_tests
    unit/core/common/test_strong_id.cpp
    unit/core/common/test_image.cpp
    unit/core/geometry/test_pinhole_camera.cpp
    unit/core/geometry/test_pose.cpp
    unit/core/geometry/test_frustum.cpp
    unit/core/geometry/test_point_pixel_matcher.cpp
    unit/core/mapping/test_point_block.cpp
    unit/core/mapping/test_geometric_map.cpp)
target_link_libraries(ovo_unit_tests PRIVATE ovo::core GTest::gtest_main GTest::gmock)
ovo_set_warnings(ovo_unit_tests)
gtest_discover_tests(ovo_unit_tests)
