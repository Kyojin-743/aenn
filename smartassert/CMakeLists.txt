
add_library(smartassert STATIC
    src/smartassert.cpp
)

target_include_directories(smartassert
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)