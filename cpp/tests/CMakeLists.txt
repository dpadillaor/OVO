# Tests unitarios: GoogleTest + GoogleMock, descargado con FetchContent (versión fijada).
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Un ejecutable de tests por capa. Añade cada test_*.cpp nuevo a la lista.
add_executable(ovo_core_tests
    core/test_strong_id.cpp)
target_link_libraries(ovo_core_tests PRIVATE ovo::core GTest::gtest_main GTest::gmock)
ovo_set_warnings(ovo_core_tests)
gtest_discover_tests(ovo_core_tests)
