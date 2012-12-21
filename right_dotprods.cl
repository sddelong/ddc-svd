#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void right_dotprods(
    __global const double * mat, 
    __global double *scratch, 
    const int m, const int n,
    const int row, const int col_offset)
{
	// find local work group/item location
	const int li0 = get_local_id(0);
	const int li1 = get_local_id(1);
	const int gi0 = get_group_id(0);
	const int gli0 = get_global_id(0);
	
	__local double loc_row[LOC_SIZE1];
	__local double loc_mat[LOC_SIZE0*LOC_SIZE1];
	
	double s=0;
	const int left_of_row = m*col_offset+row;
	const int loc_mat_entry = li1+LOC_SIZE1*li0;
	const int top_left_global = m*col_offset+row+1+gi0*LOC_SIZE0;
	const int col_chunks = (n-col_offset)/LOC_SIZE1;
	
	const bool do_stuff_row = (row+1+gi0*LOC_SIZE0+li0 < m);
	
	for (int i=0; i< col_chunks; i++){
		if (li0 == 0)
			loc_row[li1] = mat[left_of_row+m*(i*LOC_SIZE1+li1)];
		loc_mat[loc_mat_entry] = do_stuff_row ? mat[top_left_global+i*LOC_SIZE1*m+li0+li1*m]:0;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (li1 == 0 && do_stuff_row){
			for (int j=0; j<LOC_SIZE1; j+=8){
				s += loc_mat[loc_mat_entry+j]*loc_row[j];
				s += loc_mat[loc_mat_entry+j+1]*loc_row[j+1];
				s += loc_mat[loc_mat_entry+j+2]*loc_row[j+2];
				s += loc_mat[loc_mat_entry+j+3]*loc_row[j+3];
				s += loc_mat[loc_mat_entry+j+4]*loc_row[j+4];
				s += loc_mat[loc_mat_entry+j+5]*loc_row[j+5];
				s += loc_mat[loc_mat_entry+j+6]*loc_row[j+6];
				s += loc_mat[loc_mat_entry+j+7]*loc_row[j+7];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	const bool do_stuff_col = (col_chunks*LOC_SIZE1+li1 < n-col_offset);
	if (col_chunks*LOC_SIZE1 < n-col_offset){
		if (li0  == 0)
			loc_row[li1] = do_stuff_col ? mat[left_of_row+m*(col_chunks*LOC_SIZE1+li1)]:0;
		loc_mat[loc_mat_entry] = (do_stuff_row && do_stuff_col) ? mat[top_left_global+col_chunks*LOC_SIZE1*m+li0+li1*m]:0;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (li1 == 0 && do_stuff_row){
			for (int j=0; j<LOC_SIZE1; j+=8){
				s += loc_mat[loc_mat_entry+j]*loc_row[j];
				s += loc_mat[loc_mat_entry+j+1]*loc_row[j+1];
				s += loc_mat[loc_mat_entry+j+2]*loc_row[j+2];
				s += loc_mat[loc_mat_entry+j+3]*loc_row[j+3];
				s += loc_mat[loc_mat_entry+j+4]*loc_row[j+4];
				s += loc_mat[loc_mat_entry+j+5]*loc_row[j+5];
				s += loc_mat[loc_mat_entry+j+6]*loc_row[j+6];
				s += loc_mat[loc_mat_entry+j+7]*loc_row[j+7];
			}
		}
	}
	
	if (do_stuff_row && li1 == 0){
		scratch[gli0] = s;
	}
		
}
