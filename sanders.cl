#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

/**********************************************************
 * kernel: sanders
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: This function is used for edge cases in which there is
 * no reflection on either the left or right for the current step. In
 * this case, 0 is stored for the reflection vector and the value of
 * A is stored in the corresponding location in B (the bidiagonal
 * matrix). The result is 
 * 
 *      vec[vec_loc] = mat [mat_loc]
 *      mat[mat_loc] --> 0
 * 
 * where vec is alpha or beta.
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 *  mat_loc - as above
 * 	vec_loc - as above
 * 	mat - the on-chip matrix A
 *  vec - as above
 * 
 * 
 **********************************************************/

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
