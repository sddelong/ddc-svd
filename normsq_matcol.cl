#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void normsq_matcol(
    __global const double * mat, 
    __global double *scratch,
    __global double *temp, 
    unsigned int m, unsigned int n,
    unsigned int col, unsigned int offset, unsigned int work_per_item )
{
	// find local work group dimensions and location
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	
	int si = (gi*gdim0+li)*work_per_item+offset; // start for global read
	int sl = li*work_per_item; // starting place for local storage
	int comp = m-si; // compare i to this to not run off array/ column
	int cs = m*col; // start of column

	if (li+gi == 0)
		temp[0] = mat[cs+offset];
	
	__local double loc [LOC_SIZE];

	for (int i=0; i < work_per_item; i++){
		loc[sl+i] = (i < comp) ? mat[cs+si+i] : 0;
	}
	
	double s = 0;
	
	for (int i=0; i < work_per_item; i++){
		s += loc[sl+i]*loc[sl+i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	loc[li] = s;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int i=gdim0/2; i>0; i>>=1){
		if (li < i){
			loc[li] += loc[li+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write to scratch
	if (li == 0)
		scratch[gi] = loc[0];


}
