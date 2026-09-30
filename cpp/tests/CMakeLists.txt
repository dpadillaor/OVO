# Tests unitarios: GoogleTest + GoogleMock, descargado con FetchContent (versión fijada).
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Un ejecutable de tests para core. La carpeta de tests replica la de headers.
add_executable(ovo_core_tests
    core/common/test_strong_id.cpp
    core/geometry/test_camera_intrinsics.cpp
    core/geometry/test_pose.cpp
    core/geometry/test_frustum.cpp
    core/geometry/test_point_pixel_matcher.cpp)
target_link_libraries(ovo_core_tests PRIVATE ovo::core GTest::gtest_main GTest::gmock)
ovo_set_warnings(ovo_core_tests)
gtest_discover_tests(ovo_core_tests)
