#AENN/cuda_tensor/

add_library(cuda_tensor STATIC
    src/cuda_matmul.cu
)

target_include_directories(cuda_tensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cuda_tensor PRIVATE tensor CUDA::cudart CUDA::cublas)

set_target_properties(cuda_tensor PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)