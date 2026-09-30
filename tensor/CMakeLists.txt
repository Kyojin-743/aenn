#AENN/tensor/CMakeLists.txt

#==========================================================
# Tensor Library
#==========================================================

add_library(tensor STATIC
    src/tensor.cpp
)

target_include_directories(tensor 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tensor 
    PRIVATE
        smartassert)

#==========================================================
# Tensor Test Stuff
#==========================================================
add_executable(test_tensor
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        tensor
        smartassert
        gtest_main  
)

include(GoogleTest)
gtest_discover_tests(test_tensor)