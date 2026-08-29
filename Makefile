BUILDDIR:=bin
RELEASEDIR:=$(BUILDDIR)/release
DEBUGDIR:=$(BUILDDIR)/debug

SRCDIR:=src
INCDIR:=inc 
THIRDPARTY:=thirdparty
TESTDIR:=tests

SRCS := $(wildcard $(SRCDIR)/*.cpp)
INCS := $(wildcard $(INCDIR)/*.h)

CC:= g++
CCFLAGS:= -std=c++23 -I $(INCDIR) -I $(THIRDPARTY)/inc 

DEBUGFLAGS:= -O0 -g -Wall -Wpedantic -Wno-unused-const-variable -Wno-comment $(CCFLAGS)
RELEASEFLAGS:= -O3 $(CCFLAGS)

.phony:= tests all 

all: tests

tests: auto-grad 

auto-grad: $(SRCS) $(INCS)	$(TESTDIR)/auto-grad.cpp
	mkdir -p $(DEBUGDIR)
	$(CC) $(SRCS) $(TESTDIR)/auto-grad.cpp -o $(DEBUGDIR)/auto-grad $(DEBUGFLAGS)

clean:
	rm -rf bin/