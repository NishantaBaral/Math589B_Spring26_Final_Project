CXX = g++
CXXFLAGS = -O0 -std=c++17 -Wall

EIGEN_INC ?= /usr/include/eigen3

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) -I$(EIGEN_INC) $< -o $@

clean:
	rm -f main
