#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void update_right_hh(
    __global const double * mat_a, 
    __global double * mat_b, 
    __global double *dot_prods, 
    unsigned int m, int n,
    unsigned int row_a, unsigned int row_b, unsigned int offset, 
    unsigned int dot_prod_loc, int work_per_item)
{
	// find local work group dimensions and location
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	double dot_prod = dot_prods[dot_prod_loc];
	
	
	int scol = ((gi*gdim0+li)*work_per_item+offset); // start for global reads/writes
	int si_a = scol*m+row_a; // start for global read from mat_a
	int si_b = scol*m+row_b; // start for global write to mat_b
	int endi = ( n-scol < work_per_item ) ? n-scol:work_per_item;
	
	for (int i=0; i < endi; i++){
		mat_b[si_b+i*m] -= 2*dot_prod*mat_a[si_a+i*m];
	}
	
}
