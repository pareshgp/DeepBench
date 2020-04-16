SOURCE_DIR?=.
BIN_DIR?=bin
MKDIR=mkdir -p

#hipcc
HIPCC=/opt/rocm/bin/hipcc

#BLAS
ROCBLAS_LIB=rocblas

#CONV
MIOPEN_LIB?=MIOpen

#DeepBench
DEEPBENCH_INC=${SOURCE_DIR}/../kernels
MIOPEN_INC=/opt/rocm/miopen/include

all: conv rnn gemm 

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(DEEPBENCH_INC) -I$(MIOPEN_INC) -L/opt/rocm/miopen/lib -l$(MIOPEN_LIB) $(OPT)  -std=c++11 --amdgpu-target=gfx906

batchnorm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/batchnorm_bench_rocm.cpp -o $(BIN_DIR)/batchnorm_bench -I$(DEEPBENCH_INC) -I$(MIOPEN_INC) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx906

rnn:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_rocm.cpp -o $(BIN_DIR)/rnn_bench -I$(DEEPBENCH_INC) -I$(MIOPEN_INC) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx906

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(DEEPBENCH_INC) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx906

clean:
	rm -rf $(BIN_DIR)
