#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void update_scale_matcol(
    __global double * mat, 
    __global double * alpha, 
    __global const double * norm_storage, 
    __global const double * temp, 
    unsigned int m, unsigned int n,
    unsigned int col, unsigned int offset, unsigned int work_per_item )
{
	double norm = norm_storage[0];
	double x1 = temp[0];
	norm = sqrt(norm);
	// find local work group dimensions and location
	
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	int si = (gi*gdim0+li)*work_per_item+offset; // start for global work
	int endi = (m < si+work_per_item) ? m:si+work_per_item;
	int cs = m*col; // start of column

	int sign = ( x1 < 0 ) ? -1:1;
	
	if (li+gi == 0){
		mat[cs+offset] += sign*norm;
		alpha[offset] = -sign*norm;
	}
		
	norm = sqrt(2.0)*sqrt(norm*norm + fabs(norm*x1));
	
	for (int i=si; i < endi; i++){
		mat[cs+i] /= norm;
	}
	

}

