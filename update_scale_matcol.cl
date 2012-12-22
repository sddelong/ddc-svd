#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

/**********************************************************
 * Kernel: update_scale_matcol
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: This kernel creates the left Householder reflector for
 * the current step and stores it in place. Let c be the partial column
 * starting at mat[col,col]. The result of this function is
 * 
 * 		alpha[col] -->   - sign(c[0]) || c ||
 *      c --> normalize(sign(c[0]) || c || e_0 + c)
 * 
 * where || c || indicates the 2-norm of the partial column rc. The vector
 * e_0 is the standard basis vector with 1 in its first entry. The norm 
 * of the partial column is stored in norm_storage.
 * 
 * Input: as above, and
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 *  mat - the matrix whose partial column is being updated
 *  alpha - the diagonal of the bidiagonal matrix
 *  temp - holds the original value of the first entry of c
 *  col - the column number of the partial column which will be the reflector
 *  offset - the starting point of that partial column
 *  work_per_item - how many entries each work item updates. 
 * 
 **********************************************************/

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
	
	int row_chunks = (endi-si)/8;
	
	for (int i=0; i< row_chunks*8; i+=8){
		mat[cs+si+i] /= norm;
		mat[cs+si+i+1] /= norm;
		mat[cs+si+i+2] /= norm;
		mat[cs+si+i+3] /= norm;
		mat[cs+si+i+4] /= norm;
		mat[cs+si+i+5] /= norm;
		mat[cs+si+i+6] /= norm;
		mat[cs+si+i+7] /= norm;
	}
	
	for (int i=row_chunks*8+si; i < endi; i++){
		mat[cs+i] /= norm;
	}
	

}

