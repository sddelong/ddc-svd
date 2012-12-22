#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

/**********************************************************
 * Kernel: update_scale_matrow
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: This kernel creates the right Householder reflector for
 * the current step and stores it in place. Let r be the partial row
 * starting at mat[row,row+1]. The result of this function is
 * 
 * 		beta[row] -->   - sign(r[0]) || r ||
 *      r --> normalize(sign(r[0]) || r || e_0 + r)
 * 
 * where || r || indicates the 2-norm of the partial row r. The vector
 * e_0 is the standard basis vector with 1 in its first entry. The norm 
 * of the partial row is stored in norm_storage.
 * 
 * Input: as above, and
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 *  mat - the matrix whose partial row is being updated
 *  beta - the superdiagonal of the bidiagonal matrix
 *  temp - holds the original value of the first entry of r
 *  row - the row number of the partial row which will be the reflector
 *  offset - the starting point of that partial row
 *  work_per_item - how many entries each work item updates. 
 * 
 **********************************************************/

__kernel void update_scale_matrow(
    __global double * mat, 
    __global double * beta, 
    __global const double * norm_storage, 
    __global const double * temp, 
    unsigned int m, unsigned int n,
    unsigned int row, unsigned int offset, unsigned int work_per_item )
{
	double norm = norm_storage[0];
	double x1 = temp[0];
	norm = sqrt(norm);
	// find local work group dimensions and location
	int gdim0 = get_local_size(0);
	int li = get_local_id(0);
	int gi = get_group_id(0);
	
	
	int si = ((gi*gdim0+li)*work_per_item+offset)*m+row; // start for global read
	int endi = (m*(n-1)+row+1< si+1+(work_per_item-1)*m) ? m*(n-1)+row+1:si+1+(work_per_item-1)*m;
	
	int sign = ( x1 < 0 ) ? -1:1;
	
	if (li+gi == 0){
		mat[row+offset*m] += sign*norm;
		beta[offset-1] = -sign*norm;
	}
	
	norm = sqrt(2.0)*sqrt(norm*norm + fabs(norm*x1));
	
	for (int i=si; i < endi; i+=m){
		mat[i] /= norm;
	}
	

}

