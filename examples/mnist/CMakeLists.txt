add_executable(mnist
    mnist.cpp
)

target_link_libraries(mnist
    PRIVATE
        smartassert
        tensor
        autograd
)