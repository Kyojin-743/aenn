#AENN/
cmake_minimum_required(VERSION 3.16)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "")
    if(NOT DEFINED VCPKG_TARGET_TRIPLET)
        set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "" FORCE)
    endif()
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(
        $<$<CONFIG:Debug>:/MTd>
        $<$<CONFIG:Release>:/MT>
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<CONFIG:Release>:-O3 -march=native -flto>)
    add_link_options($<$<CONFIG:Release>:-flto>)
elseif(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2 /GL>)
    add_link_options($<$<CONFIG:Release>:/LTCG>)
endif()

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
endif()

project(AENN CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

add_subdirectory(thirdparty/googletest)
add_subdirectory(thirdparty/imgui)
add_subdirectory(thirdparty/implot)
add_subdirectory(thirdparty/ImGuiFileDialog)

add_subdirectory(smartassert)
add_subdirectory(tensor)
add_subdirectory(autograd)
add_subdirectory(examples)
add_subdirectory(cuda_tensor)