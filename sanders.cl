#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void sanders(
    __global double * mat, 
    __global double * vec, 
    unsigned int m, int n,
    unsigned int mat_loc, unsigned int vec_loc)
{
	int li = get_local_id(0);
	int gi = get_group_id(0);
	if ( gi+li == 0 ){
		vec[vec_loc] = mat[mat_loc];
		mat[mat_loc] = 0.0;
	}
}
