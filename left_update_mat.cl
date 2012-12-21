#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

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
