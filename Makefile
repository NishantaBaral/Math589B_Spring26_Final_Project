TARGET = solver
NVCC   = nvcc
NVCCFLAGS = -O2


SRC = src/main.cu src/solver.cu

all:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
