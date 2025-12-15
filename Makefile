ARCH ?= sm_35

HOST_COMP ?= mpicc


NVCC = nvcc

ifeq ($(ARCH),sm_35)
    NVCC_FLAGS = -gencode arch=compute_35,code=sm_35 -O3 -Xcompiler -fopenmp
else ifeq ($(ARCH),sm_60)
    NVCC_FLAGS = -gencode arch=compute_60,code=sm_60 -O3 -Xcompiler -fopenmp
else
    NVCC_FLAGS = -arch=$(ARCH) -O3 -Xcompiler -fopenmp
endif

CUDA_LIBS = -lcudart -lstdc++ -lm

CUDA_SRC = main_mpi_cuda.cpp
CUDA_OBJ = main_mpi_cuda.o
CUDA_EXEC = main_mpi_cuda


all: $(CUDA_EXEC)


$(CUDA_EXEC): $(CUDA_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS) -ccbin $(HOST_COMP)

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $< -ccbin $(HOST_COMP)

clean:
	rm -f $(CUDA_OBJ) $(CUDA_EXEC) *.o

.PHONY: all clean

