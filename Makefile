EXECUTABLES = test-whole-svd

all: $(EXECUTABLES)

ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif


COMPILER_OPTIONS = gcc -std=gnu99 -O3 -fopenmp

test-whole-svd : test-whole-svd.c matrix_helper.o parallel-twisted.o svd_gpu.o cl-helper.o bidiag_par.o Calculations-Parallel.o
	$(COMPILER_OPTIONS) -o$@ $^ -lrt -lm $(CL_CFLAGS) $(CL_LDFLAGS) -lOpenCL	


%.o : %.c %.h cl-helper.c
		gcc -c  -std=gnu99 -O3 $< -lm -fopenmp -lOpenCL $(CL_CFLAGS) $(CL_LDFLAGS)