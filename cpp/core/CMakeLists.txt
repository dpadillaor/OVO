# ovo_core: the domain. Pure C++: no I/O, no models, no SLAM.
# STATIC library: headers in include/, cold code (built once per frame, not per point) in src/.
add_library(ovo_core STATIC
    src/geometry/frustum.cpp)
add_library(ovo::core ALIAS ovo_core)

# PUBLIC: ovo_core needs them to compile its .cpp files, and whoever links it needs them for the headers.
target_include_directories(ovo_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(ovo_core PUBLIC Eigen3::Eigen)
ovo_set_warnings(ovo_core)
