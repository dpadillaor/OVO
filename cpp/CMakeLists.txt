# OVO C++: build raíz.
cmake_minimum_required(VERSION 3.24)
project(ovo_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MinGW: enlaza el runtime de C++ en estático. Si no, el .exe carga la primera
# libstdc++-6.dll que encuentre en el PATH (conda, Git, Strawberry...) y, si es
# de otra versión, muere al arrancar con 0xc0000139.
if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

add_subdirectory(core)

enable_testing()
add_subdirectory(tests)
