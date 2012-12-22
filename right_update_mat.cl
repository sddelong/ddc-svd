#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

/**********************************************************
 * kernel: right_update_mat
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: This kernel is used to apply the reflection vector
 * v stored in the given partial row of mat to the submatrix below that
 * row. The inner products of v with the rows of that submatrix 
 * are given in the scratch vector. Let r be a row in that submatrix.
 * The result of this kernel is 
 * 
 *           r --> (I - 2 v v*) r
 *  
 * Input: as above, and
 * 
 * 	mat - the matrix whose partial rows are updated
 *  scratch - where the dot products are stored
 * 	m - number of rows in mat
 *  n - number of cols in mat
 *  row - the row number of the partial row which is the reflector
 *  col_offset - the starting point of the reflector in its row
 * 
 **********************************************************/

__kernel void right_update_mat(
    __global double * mat, 
    __global double *scratch, 
    int m, int n,
    int row, int col_offset)
{
	// find global work item location
	const int gli0 = get_global_id(0);
	const int gli1 = get_global_id(1);
	const int my_row = gli0+row+1;
	const int my_col = gli1+col_offset;
	const bool do_stuff = ( my_col < n && my_row < m);
	if (do_stuff){
		mat[my_col*m+ my_row] -= 2*scratch[gli0]*mat[my_col*m+row];
	}

}
