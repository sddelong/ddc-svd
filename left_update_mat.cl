#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

/**********************************************************
 * kernel: left_update_mat
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: This kernel is used to apply the reflection vector
 * v stored in the given partial column of mat to the submatrix to its
 * right. The inner products of v with the columns of that submatrix 
 * are given in the scratch vector. Let c be a column in that submatrix.
 * The result of this kernel is 
 * 
 *           c --> (I - 2 v v*) c
 *  
 * Input: as above, and
 * 
 * 	mat - the matrix whose partial columns are updated
 *  scratch - where the dot products are stored
 * 	m - number of rows in mat
 *  n - number of cols in mat
 *  col - the column number of the partial column which is the reflector
 *  offset - the starting point of the reflector in its column
 * 
 **********************************************************/

__kernel void left_update_mat(
    __global double * mat, 
    __global double *scratch, 
    int m, int n,
    int col, int row_offset)
{
	// find global work item location
	const int gli0 = get_global_id(0);
	const int gli1 = get_global_id(1);
	const int my_row = gli0+row_offset;
	const int my_col = gli1+col+1;
	const bool do_stuff = ( my_col < n && my_row < m);
	if (do_stuff){
		mat[my_col*m+ my_row] -= 2*scratch[gli1]*mat[col*m+my_row];
	}

}
