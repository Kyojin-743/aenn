#AENN/examples/simple_interactive/

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
)

target_include_directories(simple_interactive
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)