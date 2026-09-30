# External dependencies of core.
# Strategy: use the system install if present (Linux: apt/conda); otherwise fetch a pinned
# version. SYSTEM: its headers do not get our strict warnings.
include(FetchContent)

# --- Eigen (header-only): vectors, matrices, poses ---
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found on the system: fetching 3.4.0")
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM)
    FetchContent_MakeAvailable(eigen)
endif()
