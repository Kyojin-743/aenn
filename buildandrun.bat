cmake . -B build
cmake --build build
cd build
@REM  ctest --output-on-failure
.\tensor\Debug\test_tensor.exe
cd ..
@REM .\build\examples\mnist\Debug\mnist.exe