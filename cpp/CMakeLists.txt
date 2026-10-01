# OVO C++: root build.
cmake_minimum_required(VERSION 3.24)
project(ovo_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MinGW: link the C++ runtime statically. Otherwise the .exe loads the first
# libstdc++-6.dll found on PATH (conda, Git, Strawberry...) and, if it is a
# different version, dies at startup with 0xc0000139.
if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Dependencies)

add_subdirectory(core)
add_subdirectory(adapters)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
