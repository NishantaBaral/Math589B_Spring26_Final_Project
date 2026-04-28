NVCC = nvcc
CFLAGS = -O3

TARGET = solver

all: $(TARGET)

$(TARGET): src/main.cpp src/solver.cu src/solver.hpp
	$(NVCC) $(CFLAGS) src/main.cpp src/solver.cu -o $(TARGET)

clean:
	rm -f $(TARGET)