# ovo_core: el dominio. C++ puro: sin I/O, sin modelos, sin SLAM.
# De momento solo headers (INTERFACE). Pasará a STATIC cuando haya .cpp.
add_library(ovo_core INTERFACE)
add_library(ovo::core ALIAS ovo_core)

target_include_directories(ovo_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
