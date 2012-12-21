#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void matrow_dotprod(
    __global const double * mat_a, 
    __global const double * mat_b, 
    __global double *scratch, 
    unsigned int m, unsigned int n,
    unsigned int row_a, unsigned int row_b, unsigned int offset, 
    unsigned int work_per_item, unsigned int write_offset )
{
	// find local work group dimensions and location
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	
	int si_a = ((gi*gdim0+li)*work_per_item+offset)*m+row_a; // start for global read from mat_a
	int si_b = ((gi*gdim0+li)*work_per_item+offset)*m+row_b; // start for global read from mat_b
	
	int sl = li*work_per_item; // starting place for local storage
	int comp_a = row_a+m*(n-1)+1; // compare i*m to this to not run off array/ row
	int comp_b = row_b+m*(n-1)+1; // compare i*m to this to not run off array/ row
	
	
	__local double loc_a [LOC_SIZE];
	__local double loc_b [LOC_SIZE];
	
	for (int i=0; i < work_per_item; i++){
		loc_a[sl+i] = (si_a+i*m < comp_a) ? mat_a[si_a+i*m] : 0;
		loc_b[sl+i] = (si_b+i*m < comp_b) ? mat_b[si_b+i*m] : 0;
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
