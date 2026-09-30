# ovo_set_warnings(<target>): política de warnings del proyecto.
# Se aplica por target para no afectar a dependencias externas (GoogleTest, Eigen).
option(OVO_WARNINGS_AS_ERRORS "Tratar warnings como errores" ON)

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
