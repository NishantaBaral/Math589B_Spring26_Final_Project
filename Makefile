TARGET = solver
NVCC   = nvcc
EIGEN_INC ?= /usr/include/eigen3
NVCCFLAGS = -O2 -I$(EIGEN_INC)
SRC = src/main.cu src/solver.cu
all:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)
clean:
	rm -f $(TARGET)