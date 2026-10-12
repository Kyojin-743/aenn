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
