#AENN/examples/simple_interactive/

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

add_library(simple_model_cuda STATIC
    simple_model.cu
)

target_link_libraries(simple_model_cuda 
    PRIVATE
        smartassert 
        tensor 
        cuda_tensor
        CUDA::cudart 
        CUDA::cublas
)

set_target_properties(simple_model_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(simple_model_cuda
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


add_executable(simple_interactive
    simple_interactive.cpp
)

target_link_libraries(simple_interactive
    PRIVATE
        tensor
        smartassert
        imgui
        implot
        ImGuiFileDialog
        cuda_tensor
        simple_model_cuda
)

target_include_directories(simple_interactive
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
