#AENN/
cmake_minimum_required(VERSION 3.16)


if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "")
endif()

project(AENN CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

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
