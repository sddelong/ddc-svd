#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

/**********************************************************
 * kernel: left_dotprods
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: This kernel is used to compute the inner products of 
 * the given partial column (reflector) with the columns to its right. 
 * It writes the resulting inner products to the scratch vector at 
 * position i where the inner product is taken with the partial column 
 * i+1 to the right of the reflector. 
 * 
 * Input: as above, and
 * 
 * 	mat - the matrix whose columns are multiplied
 *  scratch - where the dot products are written
 * 	m - number of rows in mat
 *  n - number of cols in mat
 *  col - the column number of the partial column which is the reflector
 *  offset - the starting point of the reflector in its column
 * 
 **********************************************************/

__kernel void left_dotprods(
    __global const double * mat, 
    __global double *scratch, 
    const int m, const int n,
    const int col, const int row_offset)
{
	// find local work group/item location
	const int li0 = get_local_id(0);
	const int li1 = get_local_id(1);
	const int gi1 = get_group_id(1);
	const int gli1 = get_global_id(1);
	
	__local double loc_col[LOC_SIZE0];
	__local double loc_mat[LOC_SIZE0*LOC_SIZE1];
	
	double s=0;
	const int top_of_col = m*col+row_offset;
	const int loc_mat_entry = li0+LOC_SIZE0*li1;
	const int top_left_global = m*(col+1+gi1*LOC_SIZE1)+row_offset;
	const int row_chunks = (m-row_offset)/LOC_SIZE0;
	
	const bool do_stuff_col = (col+1+gi1*LOC_SIZE1+li1 < n);
	
	for (int i=0; i< row_chunks; i++){
		if (li1 == 0)
			loc_col[li0] = mat[top_of_col+i*LOC_SIZE0+li0];
		loc_mat[loc_mat_entry] = do_stuff_col ? mat[top_left_global+i*LOC_SIZE0+li0+li1*m]:0;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (li0 == 0 && do_stuff_col){
			for (int j=0; j<LOC_SIZE0; j+=8){
				s += loc_mat[loc_mat_entry+j]*loc_col[j];
				s += loc_mat[loc_mat_entry+j+1]*loc_col[j+1];
				s += loc_mat[loc_mat_entry+j+2]*loc_col[j+2];
				s += loc_mat[loc_mat_entry+j+3]*loc_col[j+3];
				s += loc_mat[loc_mat_entry+j+4]*loc_col[j+4];
				s += loc_mat[loc_mat_entry+j+5]*loc_col[j+5];
				s += loc_mat[loc_mat_entry+j+6]*loc_col[j+6];
				s += loc_mat[loc_mat_entry+j+7]*loc_col[j+7];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	const bool do_stuff_row = (row_chunks*LOC_SIZE0+li0 < m-row_offset);
	if (row_chunks*LOC_SIZE0 < m-row_offset){
		if (li1  == 0)
			loc_col[li0] = do_stuff_row ? mat[top_of_col+row_chunks*LOC_SIZE0+li0]:0;
		loc_mat[loc_mat_entry] = (do_stuff_row && do_stuff_col) ? mat[top_left_global+row_chunks*LOC_SIZE0+li0+li1*m]:0;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (li0 == 0 && do_stuff_col){
			for (int j=0; j<LOC_SIZE0; j+=8){
				s += loc_mat[loc_mat_entry+j]*loc_col[j];
				s += loc_mat[loc_mat_entry+j+1]*loc_col[j+1];
				s += loc_mat[loc_mat_entry+j+2]*loc_col[j+2];
				s += loc_mat[loc_mat_entry+j+3]*loc_col[j+3];
				s += loc_mat[loc_mat_entry+j+4]*loc_col[j+4];
				s += loc_mat[loc_mat_entry+j+5]*loc_col[j+5];
				s += loc_mat[loc_mat_entry+j+6]*loc_col[j+6];
				s += loc_mat[loc_mat_entry+j+7]*loc_col[j+7];
			}
		}
	}
	
	if (do_stuff_col && li0 == 0){
		scratch[gli1] = s;
	}
		
}
