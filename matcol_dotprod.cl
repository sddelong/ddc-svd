#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void matcol_dotprod(
    __global const double * mat_a, 
    __global const double * mat_b, 
    __global double *scratch, 
    unsigned int m, unsigned int n,
    unsigned int col_a, unsigned int col_b, unsigned int offset, 
    unsigned int work_per_item, unsigned int write_offset )
{
	// find local work group dimensions and location
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	
	int si = (gi*gdim0+li)*work_per_item+offset; // start for global read
	int sl = li*work_per_item; // starting place for local storage
	int comp = m-si; // compare i to this to not run off array/ column
	int cs_a = m*col_a; // start of column for mat_a
	int cs_b = m*col_b; // start of column for mat_b
	
	__local double loc_a [LOC_SIZE];
	__local double loc_b [LOC_SIZE];

	for (int i=0; i < work_per_item; i++){
		loc_a[sl+i] = (i < comp) ? mat_a[cs_a+si+i] : 0;
		loc_b[sl+i] = (i < comp) ? mat_b[cs_b+si+i] : 0;
	}
	
	double s = 0;
	
	for (int i=0; i < work_per_item; i++){
		s += loc_a[sl+i]*loc_b[sl+i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	loc_a[li] = s;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int i=gdim0/2; i>0; i>>=1){
		if (li < i){
			loc_a[li] += loc_a[li+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write to scratch
	if (li == 0)
		scratch[gi+write_offset] = loc_a[0];
	


}
