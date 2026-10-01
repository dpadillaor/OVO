# Executables (composition roots): wire adapters and domain together. No logic of their own beyond the loop.
add_executable(ovo_dense_map dense_map.cpp)
target_link_libraries(ovo_dense_map PRIVATE ovo::adapters)
ovo_set_warnings(ovo_dense_map)
