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
    unit/core/mapping/test_geometric_map.cpp
    unit/core/mapping/test_coverage.cpp
    unit/core/mapping/test_dense_mapper.cpp
    unit/adapters/replica/test_replica_frame_source.cpp
    unit/adapters/replica/test_replica_pose_source.cpp
    unit/adapters/ply/test_ply_writer.cpp)
target_link_libraries(ovo_unit_tests PRIVATE ovo::core ovo::adapters GTest::gtest_main GTest::gmock)
ovo_set_warnings(ovo_unit_tests)
gtest_discover_tests(ovo_unit_tests)

# Integration tests: real adapters on real data (Replica office0). Skipped when the data is not there.
add_executable(ovo_integration_tests
    integration/test_replica_dataset.cpp)
target_link_libraries(ovo_integration_tests PRIVATE ovo::adapters GTest::gtest_main)
target_compile_definitions(ovo_integration_tests PRIVATE
    OVO_DEFAULT_DATASETS_DIR="${PROJECT_SOURCE_DIR}/../data/input/Datasets")
ovo_set_warnings(ovo_integration_tests)
gtest_discover_tests(ovo_integration_tests)
