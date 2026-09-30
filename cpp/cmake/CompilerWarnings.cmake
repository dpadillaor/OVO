# ovo_set_warnings(<target>): project warning policy.
# Applied per target so external dependencies (GoogleTest, Eigen) are not affected.
option(OVO_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

function(ovo_set_warnings target)
    set(warnings
        -Wall -Wextra -Wpedantic
        -Wshadow -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Woverloaded-virtual -Wold-style-cast)
    if(OVO_WARNINGS_AS_ERRORS)
        list(APPEND warnings -Werror)
    endif()
    target_compile_options(${target} PRIVATE ${warnings})
endfunction()
