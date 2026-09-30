# ovo_core: the domain. Pure C++: no I/O, no models, no SLAM.
# Header-only for now (INTERFACE). Becomes STATIC once there are .cpp files.
add_library(ovo_core INTERFACE)
add_library(ovo::core ALIAS ovo_core)

target_include_directories(ovo_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(ovo_core INTERFACE Eigen3::Eigen)
