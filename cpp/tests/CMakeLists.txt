# Tests unitarios: GoogleTest + GoogleMock, descargado con FetchContent (versión fijada).
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Tests unitarios de core: rápidos, sin datos ni dependencias externas.
# La carpeta replica la de headers. integration/ y regression/ tendrán su propio ejecutable.
add_executable(ovo_unit_tests
    unit/core/common/test_strong_id.cpp
    unit/core/geometry/test_camera_intrinsics.cpp
    unit/core/geometry/test_pose.cpp
    unit/core/geometry/test_frustum.cpp
    unit/core/geometry/test_point_pixel_matcher.cpp)
target_link_libraries(ovo_unit_tests PRIVATE ovo::core GTest::gtest_main GTest::gmock)
ovo_set_warnings(ovo_unit_tests)
gtest_discover_tests(ovo_unit_tests)
