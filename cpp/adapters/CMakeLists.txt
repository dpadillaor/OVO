# Adapters: implementations of the core ports (files, datasets, viewers). They depend on core, never the reverse.

# stb_image implementation, built alone and without our warnings (third-party code).
add_library(ovo_stb_image STATIC src/stb_image_impl.cpp)
target_include_directories(ovo_stb_image SYSTEM PUBLIC ${stb_SOURCE_DIR})

add_library(ovo_adapters STATIC
    src/replica/replica_frame_source.cpp
    src/replica/replica_pose_source.cpp
    src/ply/ply_writer.cpp)
add_library(ovo::adapters ALIAS ovo_adapters)
target_include_directories(ovo_adapters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ovo_adapters PUBLIC ovo::core PRIVATE ovo_stb_image)
ovo_set_warnings(ovo_adapters)
