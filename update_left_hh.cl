#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void update_left_hh(
    __global const double * mat_a, 
    __global double * mat_b, 
    __global double *dot_prods, 
    unsigned int m, unsigned int n,
    unsigned int col_a, unsigned int col_b, unsigned int offset, 
    int dot_prod_loc, unsigned int work_per_item)
{
	double dot_prod = dot_prods[dot_prod_loc];
	
	// find local work group dimensions and location
	
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	int si = (gi*gdim0+li)*work_per_item+offset; // start for global work
	int endi = (m < si+work_per_item) ? m:si+work_per_item;
	int cs_a = m*col_a; // start of column for mat_a
	int cs_b = m*col_b; // start of column for mat_b

	for (int i=si; i < endi; i++){
		mat_b[cs_b+i] -= 2*dot_prod*mat_a[cs_a+i];
	}


}
