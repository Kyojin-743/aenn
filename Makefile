CPPFLAGS:= -Wextra -Wall -Werror -pedantic -std=c++20 -g -lm

.phony : all

all: xor mnist

xor: test_xor.cpp
	g++ .\test_xor.cpp -o xor.exe $(CPPFLAGS)

mnist: test_mnist.cpp
	g++ -O3 .\test_mnist.cpp -o mnist.exe $(CPPFLAGS)

clean:
	rm *.o *.exe



