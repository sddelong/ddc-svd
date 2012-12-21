#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_helper.h"
#include "cl-helper.h"
#include "bidiag_par.h"

void bidiag_par(int m, int n, double*restrict A, double*restrict alpha, double*restrict beta){
			
/**********************************************************
 * Function: bidiag_par
 * 
 * Description: reduces the input matrix to bidiagonal form
 * using a parallel algorithm. The reduction is accomplished via
 * orthogonal transformations.
 * 
 * Input
 * 
 * 	m - the number of rows in the matrix A
 * 	n - the number of columns in the matrix A
 * 	A - the matrix, given in column major order
 * 
 * Output
 * 	
 * 	alpha - the diagonal of the resulting bidiagonal matrix
 * 			its length is min(m,n)
 * 	beta - the superdiagonal of the resulting bidiagonal matrix
 * 			its length is min(m,n)-1 if n <= m or min(m,n) if n > m
 * 	A - has been overwritten with the information necessary to 
 * 		reconstruct the orthogonal transformations
 * 
 * **********************************************************/
 
	int max_mn = m;
	int mn = n;
	int len_beta;
	if (m < n){
		max_mn = n;
		mn = m;
		len_beta = mn;
	}
	else{
		len_beta = mn-1;
	}	
	
	/*******************
	 * 
	 * OPENCL WORK GROUP SIZE PARAMETERS
	 * 
	 * *****************/
	 
	 // Big reduction kernels
	 int wgdim_red = 32;
	 int work_per_item_red = 16;
	 
	 // Big update kernels
	 int wgdim_upd = 16;
	 int work_per_item_upd = 16;
	 
	 // Tiny no reflect kernel (for edge cases)
	 int wgdim_nr = 1;
	 int work_per_item_nr = 1;
	 
	 // Right and left dot product kernels
	 size_t ldim_dp[] = { 16, 16 };
	 
	 // Matrix update kernels
	 size_t ldim_update_mat[] = { 16, 16 };
	 
	/**************************
	 * OPENCL CONTEXT, QUEUE
	 * ************************/
 
	print_platforms_devices();

	cl_context ctx;
	cl_command_queue queue;
		
	create_context_on("Advanced Micro Devices, Inc.", "Capeverde", 0, &ctx, &queue, 0); 
	//create_context_on("Advanced Micro Devices, Inc.", "AMD FX(tm)-4100 Quad-Core Processor", 0, &ctx, &queue, 0); 
	//	create_context_on("Intel", NULL, 0, &ctx, &queue, 0); 
	//create_context_on("NVIDIA Corporation", NULL, 0, &ctx, &queue, 0); 
	// pointer to the text of the kernel file

	char *knl_text;

	// options string for kernel from string, tells the kernel its local dimensions
	
	// Big reduction kernels
	const char base[] = "-DLOC_SIZE=";
	char options_str_reduction [ 20 ];
	int size = work_per_item_red*wgdim_red;
	sprintf(options_str_reduction, "%s%d", base, size);  
	
	// Update a column or row
	char options_str_update [ 20 ];
	size = wgdim_upd; //only need to store a copy of the innerprod
	sprintf(options_str_update, "%s%d", base, size);  
	
	// Dot product kernels
	const char base0[] = "-DLOC_SIZE0=";
	const char base1[] = "-DLOC_SIZE1=";
	char options_str_dp [ 40 ];
	size = ldim_dp[0];
	sprintf(options_str_dp, "%s%d ", base0, size);  
	size = ldim_dp[1];
	sprintf(options_str_dp, "%s%s%d ", options_str_dp,base1, size);  
	
	// Matrix update kernels
	char options_str_update_mat [ 40 ];
	size = ldim_update_mat[0];
	sprintf(options_str_update_mat, "%s%d ", base0, size);  
	size = ldim_update_mat[1];
	sprintf(options_str_update_mat, "%s%s%d ", options_str_update_mat,base1, size);  
	
	// norm square mat col kernel [reduction]
	knl_text = read_file("normsq_matcol.cl");	
	cl_kernel knl_normsq_matcol = kernel_from_string(ctx, knl_text, 
		"normsq_matcol", options_str_reduction);
	free(knl_text);  
	
	// norm square mat row kernel [reduction]
	knl_text = read_file("normsq_matrow.cl");	
	cl_kernel knl_normsq_matrow = kernel_from_string(ctx, knl_text, 
		"normsq_matrow", options_str_reduction);
	free(knl_text);  
	
	// sum kernel [reduction]
	knl_text = read_file("sum.cl");	
	cl_kernel knl_sum = kernel_from_string(ctx, knl_text, 
		"sum", options_str_reduction);
	free(knl_text); 
	
	// matcol_dotprod kernel [reduction]
	knl_text = read_file("matcol_dotprod.cl");	
	cl_kernel knl_matcol_dotprod = kernel_from_string(ctx, knl_text, 
		"matcol_dotprod", options_str_reduction);
	free(knl_text); 
	
	// matrow_dotprod kernel [reduction]
	knl_text = read_file("matrow_dotprod.cl");	
	cl_kernel knl_matrow_dotprod = kernel_from_string(ctx, knl_text, 
		"matrow_dotprod", options_str_reduction);
	free(knl_text); 

	// update and scale mat col [update]
	knl_text = read_file("update_scale_matcol.cl");	
	cl_kernel knl_update_scale_matcol = kernel_from_string(ctx, knl_text, 
		"update_scale_matcol", options_str_update);
	free(knl_text); 
	
	// update and scale mat row [update]
	knl_text = read_file("update_scale_matrow.cl");	
	cl_kernel knl_update_scale_matrow = kernel_from_string(ctx, knl_text, 
		"update_scale_matrow", options_str_update);
	free(knl_text); 
	
	// left householder update [update]
	knl_text = read_file("update_left_hh.cl");	
	cl_kernel knl_update_left_hh = kernel_from_string(ctx, knl_text, 
		"update_left_hh", options_str_update);
	free(knl_text); 
	
	// right householder update [update]
	knl_text = read_file("update_right_hh.cl");	
	cl_kernel knl_update_right_hh = kernel_from_string(ctx, knl_text, 
		"update_right_hh", options_str_update);
	free(knl_text); 
	
	// no reflect [single workitem/group]
	knl_text = read_file("sanders.cl");	
	cl_kernel knl_sanders = kernel_from_string(ctx, knl_text, 
		"sanders", NULL);
	free(knl_text); 
	
	// left_dotprods [dot_prods]
	knl_text = read_file("left_dotprods.cl");	
	cl_kernel knl_left_dotprods = kernel_from_string(ctx, knl_text, 
		"left_dotprods", options_str_dp);
	free(knl_text); 
	
	// right_dotprods [dot_prods]
	knl_text = read_file("right_dotprods.cl");	
	cl_kernel knl_right_dotprods = kernel_from_string(ctx, knl_text, 
		"right_dotprods", options_str_dp);
	free(knl_text); 
	
	// left update mat [update_mat]
	knl_text = read_file("left_update_mat.cl");	
	cl_kernel knl_left_update_mat = kernel_from_string(ctx, knl_text, 
		"left_update_mat", options_str_update_mat);
	free(knl_text); 
	
	// right update mat [update_mat]
	knl_text = read_file("right_update_mat.cl");	
	cl_kernel knl_right_update_mat = kernel_from_string(ctx, knl_text, 
		"right_update_mat", options_str_update_mat);
	free(knl_text); 
	
	
	// --------------------------------------------------------------------------
	// allocate device memory
	// --------------------------------------------------------------------------
	cl_int status;

	// load vector
	cl_mem buf_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double) *m*n, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for Matrix A");

	// for alpha and beta on device
	cl_mem buf_alpha = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double) *mn, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for alpha");
	cl_mem buf_beta = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double) *len_beta, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for beta");
	// for work/storage
	cl_mem buf_scratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double) * max_mn, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for scratch work 1");
	cl_mem buf_scratch2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double) * max_mn, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for scratch work 2");
	cl_mem buf_scratch3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double) * max_mn, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for scratch work 3");
	cl_mem buf_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	  sizeof(double), 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer for temp");

	// --------------------------------------------------------------------------
	// transfer to device
	// --------------------------------------------------------------------------
	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
		queue, buf_A, /*blocking*/ CL_TRUE, /*offset*/ 0,
		sizeof(double) * m*n, A,
		0, NULL, NULL));
		
	
	int sign;
	double temp_norm;
	double x1;
	
	for (int i=0; i< mn-1; i++){
		
		find_reflector_and_scale_col(m, n, mn, i, 
				work_per_item_red, wgdim_red, work_per_item_upd, wgdim_upd,
				ctx, queue,
				knl_normsq_matcol, knl_update_scale_matcol, knl_sum,
				buf_A, buf_alpha,
				buf_scratch, buf_scratch2, buf_temp);
		
		
		/*left_householder_par(m,n,mn,i,
				work_per_item_red, wgdim_red, work_per_item_upd, wgdim_upd,
				ctx, queue,
				knl_matcol_dotprod, knl_update_left_hh, knl_sum,
				buf_A,
				buf_scratch, buf_scratch2, buf_scratch3);*/
				
		left_householder_par_2d(m,n,mn,i, ldim_dp, ldim_update_mat,
				ctx, queue,
				knl_left_dotprods, knl_left_update_mat,
				buf_A, buf_scratch);
															 // to the remainder of the matrix
		
		if ( i < n-2){
			
			find_reflector_and_scale_row(m, n, mn, i, 
				work_per_item_red, wgdim_red, work_per_item_upd, wgdim_upd,
				ctx, queue,
				knl_normsq_matrow, knl_update_scale_matrow, 
				knl_sum,
				buf_A, buf_beta,
				buf_scratch, buf_scratch2, buf_temp);
				
			
		
			/*right_householder_par(m,n,mn,i,
				work_per_item_red, wgdim_red, work_per_item_upd, wgdim_upd,
				ctx, queue,
				knl_matrow_dotprod, knl_update_right_hh, knl_sum,
				buf_A,
				buf_scratch, buf_scratch2, buf_scratch3);*/
			
			right_householder_par_2d(m,n,mn,i, ldim_dp, ldim_update_mat,
				ctx, queue,
				knl_right_dotprods, knl_right_update_mat,
				buf_A, buf_scratch);
		}
		else{
			int mat_loc = i+(i+1)*m;
			int vec_loc = i;
			
			// no reflection on right
			no_reflect(m,n,mat_loc,vec_loc,work_per_item_nr,wgdim_nr,
						ctx,queue,knl_sanders,
						buf_A,buf_beta);
			/*
			// no reflection on right
			beta[i] = A[i+(i+1)*m];
			A[i+(i+1)*m] = 0; */
		}
	
	}
		
	if ( n >= m+1 ){
		
		// do right householder mod to this row of A
		// no need to apply to the rest of the rows
		int temp_offset = mn-1;	
		find_reflector_and_scale_row(m, n, mn, temp_offset, 
			work_per_item_red, wgdim_red, work_per_item_upd, wgdim_upd,
			ctx, queue,
			knl_normsq_matrow, knl_update_scale_matrow, 
			knl_sum,
			buf_A, buf_beta,
			buf_scratch, buf_scratch2, buf_temp);
		
		int mat_loc = mn-1+(mn-1)*m;
		int vec_loc = mn-1;
		
		// no reflection on right
		no_reflect(m,n,mat_loc,vec_loc,work_per_item_nr,wgdim_nr,
					ctx,queue,knl_sanders,
					buf_A,buf_alpha);
		// no reflection on left
		/*alpha[mn-1] = A[mn-1+(mn-1)*m]; 
		A[mn-1 + (mn-1)*m] = 0; */
		
	}
	else {
		// do the left Householder reflection 
		int temp_offset = mn-1;
		find_reflector_and_scale_col(m, n, mn, temp_offset, 
				work_per_item_red, wgdim_red, work_per_item_upd, wgdim_upd,
				ctx, queue,
				knl_normsq_matcol, knl_update_scale_matcol, knl_sum,
				buf_A, buf_alpha,
				buf_scratch, buf_scratch2, buf_temp);
	}	
	
	// -----------------------------------------------
	// GET DATA BACK FROM DEVICE
	// -----------------------------------------------
	
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
			queue, buf_alpha, /*blocking*/ CL_TRUE, /*offset*/ 0,
			mn* sizeof(double), alpha,
			0, NULL, NULL));
	
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
			queue, buf_beta, /*blocking*/ CL_TRUE, /*offset*/ 0,
			len_beta* sizeof(double), beta,
			0, NULL, NULL));
			
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
			queue, buf_A, /*blocking*/ CL_TRUE, /*offset*/ 0,
			m*n* sizeof(double), A,
			0, NULL, NULL));
		
	CALL_CL_GUARDED(clFinish, (queue));
	
	// --------------------------------------------------------------------------
	// clean up OPENCL OBJECTS
	// --------------------------------------------------------------------------
	
	CALL_CL_GUARDED(clReleaseMemObject, (buf_A));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_scratch));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_scratch2));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_scratch3));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_temp));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_alpha));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_beta));
	CALL_CL_GUARDED(clReleaseKernel, (knl_normsq_matcol));
	CALL_CL_GUARDED(clReleaseKernel, (knl_update_scale_matcol));
	CALL_CL_GUARDED(clReleaseKernel, (knl_update_scale_matrow));
	CALL_CL_GUARDED(clReleaseKernel, (knl_normsq_matrow));
	CALL_CL_GUARDED(clReleaseKernel, (knl_matcol_dotprod));
	CALL_CL_GUARDED(clReleaseKernel, (knl_matrow_dotprod));
	CALL_CL_GUARDED(clReleaseKernel, (knl_update_left_hh));
	CALL_CL_GUARDED(clReleaseKernel, (knl_update_right_hh));
	CALL_CL_GUARDED(clReleaseKernel, (knl_sum));
	CALL_CL_GUARDED(clReleaseKernel, (knl_sanders));
	CALL_CL_GUARDED(clReleaseKernel, (knl_left_update_mat));
	CALL_CL_GUARDED(clReleaseKernel, (knl_left_dotprods));
	CALL_CL_GUARDED(clReleaseKernel, (knl_right_update_mat));
	CALL_CL_GUARDED(clReleaseKernel, (knl_right_dotprods));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));
	
		
	return;
}

void no_reflect( int m, int n, int mat_loc, int vec_loc,
				int work_per_item_nr, int wgdim_nr,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_sanders,
				cl_mem buf_mat, cl_mem buf_vec){
					
	size_t ldim[] = { wgdim_nr };
	size_t gdim[] = { wgdim_nr };
	
	SET_6_KERNEL_ARGS( knl_sanders, buf_mat, buf_vec, m, n, mat_loc, vec_loc);
	
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_sanders,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));	
			 
	return;
}
				

void left_householder_par(int m, int n, int mn, int offset,
				int work_per_item_red, int wgdim_red, int work_per_item_upd, int wgdim_upd,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_matcol_dotprod, cl_kernel knl_update_left_hh, cl_kernel knl_sum,
				cl_mem buf_A,
				cl_mem buf_scratch, cl_mem buf_scratch2, cl_mem buf_scratch3){
								
/************************************************************
 * Function: left_householder_par
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	offset - 
 * 	k - number of columns in submatrix
 *  v - reflection vector of length l
 * 	A - pointer to top left of submatrix
 * 
 * 
 ************************************************************/
	
	// KERNEL LAUNCH INFO FOR DOT PRODUCTS
	size_t ldim[] = { wgdim_red };
	size_t total_size = ((m-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red))*ldim[0];
	int starting_num_groups = ((m-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red));
	int current_num_groups;
	size_t gdim[] = { total_size };
	
	int write_offset;
	
	cl_mem in;
	cl_mem out;
	cl_mem swap;
	
	// step through columns and compute dot products
	for (int j=1; j<n-offset; j++){
		current_num_groups = starting_num_groups;
		// INITIAL REDUCTION FOR DOT PRODUCT CALCULATION
		// This pass multiplies the entries and adds them
		if (starting_num_groups == 1){
			out = buf_scratch3;
			write_offset = j-1;
		}
		else{
			out = buf_scratch;
			write_offset = 0;
		}
		// the problem is here .. if you change offset+j to offset+1 it "works"
		// previously it would fail when n >= 31
		int col2 = offset+j;
		SET_10_KERNEL_ARGS(knl_matcol_dotprod, buf_A, buf_A, out,
			m,n,offset,col2,offset,work_per_item_red,
			write_offset);
		gdim[0] = current_num_groups*ldim[0];
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_matcol_dotprod,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));	
			 
		in = buf_scratch;
		out = buf_scratch2;
		
		// Further reduction steps which simply add the numbers together
		// not necessary to multiply again
		while (current_num_groups > 1){		
			int temp = current_num_groups;
			current_num_groups = ((current_num_groups + ldim[0]*work_per_item_red - 1)
										/(ldim[0]*work_per_item_red));
			if (current_num_groups == 1){
				out = buf_scratch3;
				write_offset = j-1;
			}
			else{
				write_offset = 0;
			}
			SET_5_KERNEL_ARGS(knl_sum, in, out,temp,work_per_item_red,write_offset);
			// FIXME set to 1
			
			gdim[0] = current_num_groups*ldim[0];
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, knl_sum,
				 /*dimensions*/1, NULL, gdim, ldim,
				 0, NULL, NULL));
			
			swap = out;
			out = in;
			in = swap;
		} // dot product is now in buff_scratch3[j-1]
						
	}
	
	
	// KERNEL LAUNCH INFO FOR HOUSEHOLDER UPDATES
	ldim[0] = wgdim_upd;
	total_size = ((m-offset + ldim[0]*work_per_item_upd - 1)
		/(ldim[0]*work_per_item_upd))*ldim[0];
		
	gdim[0] = total_size;
		
	// UDPATE THE COLUMNS OF A 
	for (int j=1; j < n-offset; j++){
		int col2 = offset+j;
		int jm1 = j-1;
		SET_10_KERNEL_ARGS(knl_update_left_hh, buf_A, buf_A, buf_scratch3,
				m,n,offset,col2,offset,jm1,work_per_item_upd);
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_update_left_hh,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));
	}
		
	return;	

}

void left_householder_par_2d(int m, int n, int mn, int offset,
				size_t ldim_dp[], size_t ldim_update_mat[],
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_left_dotprods, cl_kernel knl_left_update_mat,
				cl_mem buf_A, cl_mem buf_scratch){
								
/************************************************************
 * Function: left_householder_par
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	offset - 
 * 	k - number of columns in submatrix
 *  v - reflection vector of length l
 * 	A - pointer to top left of submatrix
 * 
 * 
 ************************************************************/
	
	// KERNEL LAUNCH INFO FOR DOT PRODUCTS
	size_t total_size1 = ((n-offset-1 + ldim_dp[1] - 1)
		/(ldim_dp[1]))*ldim_dp[1];
	size_t gdim[] = { ldim_dp[0]*1, total_size1 };
	
	SET_6_KERNEL_ARGS(knl_left_dotprods, buf_A, buf_scratch,
		m,n,offset,offset);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_left_dotprods,
			 /*dimensions*/2, NULL, gdim, ldim_dp,
			 0, NULL, NULL));	
	
	// KERNEL LAUNCH INFO FOR MATRIX UPDATES	
	size_t total_size0 = ((m-offset+ ldim_update_mat[0]-1)/(ldim_update_mat[0])) * ldim_update_mat[0];
	total_size1 = ((n-offset-1 + ldim_update_mat[1] - 1)/(ldim_update_mat[1])) * ldim_update_mat[1];
		
	gdim[0] = total_size0;
	gdim[1] = total_size1;
	
	SET_6_KERNEL_ARGS(knl_left_update_mat, buf_A, buf_scratch,
			m,n,offset,offset);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, knl_left_update_mat,
		 /*dimensions*/2, NULL, gdim, ldim_update_mat,
		 0, NULL, NULL));
		 		
	return;	

}

void right_householder_par(int m, int n, int mn, int offset,
				int work_per_item_red, int wgdim_red, int work_per_item_upd, int wgdim_upd,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_matrow_dotprod, cl_kernel knl_update_right_hh, cl_kernel knl_sum,
				cl_mem buf_A,
				cl_mem buf_scratch, cl_mem buf_scratch2, cl_mem buf_scratch3){
								
/************************************************************
 * Function: right_householder_par
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	offset - 
 * 	k - number of columns in submatrix
 *  v - reflection vector of length l
 * 	A - pointer to top left of submatrix
 * 
 * 
 ************************************************************/
	
	// KERNEL LAUNCH INFO FOR DOT PRODUCTS
	size_t ldim[] = { wgdim_red };
	size_t total_size = ((n-1-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red))*ldim[0];
	int starting_num_groups = ((n-1-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red));
	int current_num_groups;
	size_t gdim[] = { total_size };
	
	int write_offset;
	
	cl_mem in;
	cl_mem out;
	cl_mem swap;
	
	// step through columns and compute dot products
	for (int j=1; j<m-offset; j++){
		current_num_groups = starting_num_groups;
		// INITIAL REDUCTION FOR DOT PRODUCT CALCULATION
		// This pass multiplies the entries and adds them
		if (starting_num_groups == 1){
			out = buf_scratch3;
			write_offset = j-1;
		}
		else{
			out = buf_scratch;
			write_offset = 0;
		}
		
		int row2 = offset+j;
		int start_col = offset+1;
		
		SET_10_KERNEL_ARGS(knl_matrow_dotprod, buf_A, buf_A, out,
			m,n,offset,row2,start_col,work_per_item_red,
			write_offset);
			
		gdim[0] = current_num_groups*ldim[0];
		
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_matrow_dotprod,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));	
			 
		in = buf_scratch;
		out = buf_scratch2;
		// Further reduction steps which simply add the numbers together
		// not necessary to multiply again
		while (current_num_groups > 1){		
			int temp = current_num_groups;
			current_num_groups = ((current_num_groups + ldim[0]*work_per_item_red - 1)
										/(ldim[0]*work_per_item_red));
			if (current_num_groups == 1){
				out = buf_scratch3;
				write_offset = j-1;
			}
			else{
				write_offset = 0;
			}
			SET_5_KERNEL_ARGS(knl_sum, in, out,temp,work_per_item_red,write_offset);
			// FIXME set to 1
			
			gdim[0] = current_num_groups*ldim[0];
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, knl_sum,
				 /*dimensions*/1, NULL, gdim, ldim,
				 0, NULL, NULL));
			
			swap = out;
			out = in;
			in = swap;
		} // dot product is now in buff_scratch3[j-1]
						
	}
	
	
	// KERNEL LAUNCH INFO FOR HOUSEHOLDER UPDATES
	ldim[0] = wgdim_upd;
	total_size = ((n-1-offset + ldim[0]*work_per_item_upd - 1)
		/(ldim[0]*work_per_item_upd))*ldim[0];
		
	gdim[0] = total_size;
	
	
	// UDPATE THE COLUMNS OF A 
	for (int j=1; j < m-offset; j++){
		int row2 = offset+j;
		int start_col = offset+1;
		int jm1 = j-1;
		SET_10_KERNEL_ARGS(knl_update_right_hh, buf_A, buf_A, buf_scratch3,
				m,n,offset,row2,start_col,jm1,work_per_item_upd);
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_update_right_hh,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));
	}
	
		
	return;	

}

void right_householder_par_2d(int m, int n, int mn, int offset,
				size_t ldim_dp[], size_t ldim_update_mat[],
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_right_dotprods, cl_kernel knl_right_update_mat,
				cl_mem buf_A, cl_mem buf_scratch){
								
/************************************************************
 * Function: left_householder_par
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	offset - 
 * 	k - number of columns in submatrix
 *  v - reflection vector of length l
 * 	A - pointer to top left of submatrix
 * 
 * 
 ************************************************************/
	
	// KERNEL LAUNCH INFO FOR DOT PRODUCTS
	size_t total_size0 = ((m-offset-1 + ldim_dp[0] - 1)
		/(ldim_dp[0]))*ldim_dp[0];
	size_t gdim[] = { total_size0, ldim_dp[1]*1 };
	
	int col_offset = offset+1;
	SET_6_KERNEL_ARGS(knl_right_dotprods, buf_A, buf_scratch,
		m,n,offset,col_offset);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_right_dotprods,
			 /*dimensions*/2, NULL, gdim, ldim_dp,
			 0, NULL, NULL));	
	
	// KERNEL LAUNCH INFO FOR MATRIX UPDATES	
	total_size0 = ((m-offset-1+ ldim_update_mat[0]-1)/(ldim_update_mat[0])) * ldim_update_mat[0];
	size_t total_size1 = ((n-col_offset + ldim_update_mat[1] - 1)/(ldim_update_mat[1])) * ldim_update_mat[1];
		
	gdim[0] = total_size0;
	gdim[1] = total_size1;
	
	SET_6_KERNEL_ARGS(knl_right_update_mat, buf_A, buf_scratch,
			m,n,offset,col_offset);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, knl_right_update_mat,
		 /*dimensions*/2, NULL, gdim, ldim_update_mat,
		 0, NULL, NULL));
		 		
	return;	

}


void form_u_par(int m, int n, const double*restrict A_mod, double*restrict U){

/************************************************************
 * Function: form_u
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	A_mod - pointer to top left of matrix storing reflection vectors (comes from
 *  the bidiag_seq routine)
 * 
 * Output:
 * 	U - the left orthogonal matrix in the bidiagonal decomposition
 * 
 * 				A = U B V^T
 * 
 * where B is bidiagonal
 * 
 ************************************************************/
 	
	int mn;
	int j_start;
	double inner_prod;
	if (m<n){
		mn = m;
	}
	else{
		mn = n;
	}
	
	// set the ith column of U to Ue_i
	for (int i=0; i < m; i++){
		set_vec_to_zero(m,U+i*m);
		U[i+i*m] = 1;
		if (i < mn-1 ){
			j_start = i;
		}
		else{
			j_start = mn-1;
		}
		for (int j=j_start; j >= 0; j--){
			inner_prod = dot_prod(m-j,U+j+i*m,A_mod+j+j*m);
			for (int k=j; k<m; k++){
				U[k+i*m] -= 2*A_mod[k+j*m]*inner_prod;
			}
		}
	}
		
	return;
}

void form_v_par(int m, int n, const double*restrict A_mod, double*restrict V){

/************************************************************
 * Function: form_v
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	A_mod - pointer to top left of matrix storing reflection vectors (comes from
 *  the bidiag_seq routine)
 * 
 * Output:
 * 	V - the right orthogonal matrix in the bidiagonal decomposition
 * 
 * 				A = U B V^T
 * 
 * where B is bidiagonal
 * 
 ************************************************************/
 	
	int mn;
	int j_start;
	int num_refs;
	double inner_prod;
	if (m<n){
		mn = m;
		num_refs = mn;
	}
	else{
		mn = n;
		num_refs = mn-1;
	}
	
	// set the ith column of V to Ve_i
	for (int i=0; i < n; i++){
		set_vec_to_zero(n,V+i*n);
		V[i+i*n] = 1;
		if (i < num_refs){
			j_start = i-1;
		}
		else{
			j_start = num_refs-1;
		}
		for (int j=j_start; j >= 0; j--){
			inner_prod = dot_prod_mat_row_with_vec(m,n,n-j-1,A_mod+j+(j+1)*m,V+j+1+i*n);
			for (int k=j+1; k<n; k++){
				V[k+i*n] -= 2*A_mod[j+k*m]*inner_prod;
			}
		}
	}
		
	return;
}


void find_reflector_and_scale_col(int m, int n, int mn, int col, 
				int work_per_item_red, int wgdim_red, int work_per_item_upd, int wgdim_upd,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_normsq_matcol,cl_kernel knl_update_scale_matcol, 
				cl_kernel knl_sum,
				cl_mem buf_A, cl_mem buf_alpha,
				cl_mem buf_scratch, cl_mem buf_scratch2, cl_mem buf_temp){
		
	int offset = col;
	size_t ldim[] = { wgdim_red };
	size_t total_size = ((m-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red))*ldim[0];
	int current_num_groups = ((m-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red));
	size_t gdim[] = { total_size };
	
	cl_mem in = buf_A;
	cl_mem out = buf_scratch;
	
	// INITIAL REDUCTION FOR NORM SQUARED CALCULATION
	// This pass squares the entries and adds them
	// It also outputs the first entry of the partial column of A to buf_temp
	SET_8_KERNEL_ARGS(knl_normsq_matcol, in, out, buf_temp,
		m,n,col,col,work_per_item_red);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, knl_normsq_matcol,
		 /*dimensions*/1, NULL, gdim, ldim,
		 0, NULL, NULL));
		 		 
	cl_mem swap;
	in = buf_scratch;
	out = buf_scratch2;
	
	int odd = 1;
	int write_offset = 0;
	
	// Further reduction steps which simply add the numbers together
	// not necessary to square again
	while (current_num_groups > 1){		
		SET_5_KERNEL_ARGS(knl_sum, in, out,current_num_groups,work_per_item_red,write_offset);
		// FIXME set to 1
		
		current_num_groups = ((current_num_groups + ldim[0]*work_per_item_red - 1)
									/(ldim[0]*work_per_item_red));
		gdim[0] = current_num_groups*ldim[0];
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_sum,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));
		
		odd++;
		swap = out;
		out = in;
		in = swap;
	} // norm is now stored in buff_scratch if odd is odd, buff_scratch2 if odd is even
	
	if ( odd%2 == 0)
		in = buf_scratch2;
	else
		in = buf_scratch;
				
		
	ldim[0] = wgdim_upd;
	total_size = ((m-offset + ldim[0]*work_per_item_upd - 1)
		/(ldim[0]*work_per_item_upd))*ldim[0];
		
	gdim[0] = total_size;
		
	// UDPATE THE COLUMN OF A 
	SET_9_KERNEL_ARGS(knl_update_scale_matcol, buf_A, buf_alpha, in, buf_temp,
		m,n,col,col,work_per_item_upd);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, knl_update_scale_matcol,
		 /*dimensions*/1, NULL, gdim, ldim,
		 0, NULL, NULL));
		 
	return;
	
}


void find_reflector_and_scale_row(int m, int n, int mn, int row, 
				int work_per_item_red, int wgdim_red, int work_per_item_upd, int wgdim_upd,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_normsq_matrow,cl_kernel knl_update_scale_matrow, 
				cl_kernel knl_sum,
				cl_mem buf_A, cl_mem buf_beta,
				cl_mem buf_scratch, cl_mem buf_scratch2, cl_mem buf_temp){
	
	int offset = row+1;
	size_t ldim[] = { wgdim_red };
	size_t total_size = ((n-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red))*ldim[0];
	int current_num_groups = ((n-offset + ldim[0]*work_per_item_red - 1)
		/(ldim[0]*work_per_item_red));
	size_t gdim[] = { total_size };
	
	cl_mem in = buf_A;
	cl_mem out = buf_scratch;
		
	// INITIAL REDUCTION FOR NORM SQUARED CALCULATION
	// This pass squares the entries and adds them
	// It also outputs the first entry of the partial row of A to buf_temp
	SET_8_KERNEL_ARGS(knl_normsq_matrow, in, out, buf_temp,
		m,n,row,offset,work_per_item_red);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, knl_normsq_matrow,
		 /*dimensions*/1, NULL, gdim, ldim,
		 0, NULL, NULL));
		 
	double norm;
	
	
		 
	cl_mem swap;
	in = buf_scratch;
	out = buf_scratch2;
	
	int odd = 1;
	int write_offset = 0;
	
	// Further reduction steps which simply add the numbers together
	// not necessary to square again
	while (current_num_groups > 1){	
		SET_5_KERNEL_ARGS(knl_sum, in, out,current_num_groups,work_per_item_red,write_offset);
		// FIXME set to 1
		
		current_num_groups = ((current_num_groups + ldim[0]*work_per_item_red - 1)
									/(ldim[0]*work_per_item_red));
		gdim[0] = current_num_groups*ldim[0];
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl_sum,
			 /*dimensions*/1, NULL, gdim, ldim,
			 0, NULL, NULL));
		
		odd++;
		swap = out;
		out = in;
		in = swap;
	} // norm is now stored in buff_scratch if odd is odd, buff_scratch2 if odd is even
	
	if ( odd%2 == 0)
		in = buf_scratch2;
	else
		in = buf_scratch;
				
		
	ldim[0] = wgdim_upd;
	total_size = ((n-offset + ldim[0]*work_per_item_upd - 1)
		/(ldim[0]*work_per_item_upd))*ldim[0];
		
	gdim[0] = total_size;
	
	// UDPATE THE ROW OF A 
	SET_9_KERNEL_ARGS(knl_update_scale_matrow, buf_A, buf_beta, in, buf_temp,
		m,n,row,offset,work_per_item_upd);
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, knl_update_scale_matrow,
		 /*dimensions*/1, NULL, gdim, ldim,
		 0, NULL, NULL));
		 
	return;
		
}

void multV(int m,int n, int vecnum, double*restrict A_mod, double*restrict X,double*restrict V){
    /*****************************************
     * Author: Steven Delong
     *
     * Description: creates a right singular vector V from the reflectors stored in A after
     * bidiagonalization and from the singular vectors found in X after running
     * CalcRightSingularVectors
     *
     * inputs:
     *       m - row dimension of the original matrix A, output will be m x m
     *       mn - minimum dimension, min(m,n)
     *       vecnum - number of the vector to compute.
     *       A_mod - stores reflectors, it will be the TRANSPOSE of the overwritten A created
     *           by the bidiagonalization function (e.g. bidiag_seq)
     *      X - right singular vectors for the reduced bidiagonal matrix, output from
     *           CalcRightSingularVectors
     *
     * Outputs:
     *        V - the vecnum'th right singular vector of A
     ****************************************************/

    int mn, num_refs,rest;
    double inner_prod;
    if (m<n){
        mn = m;
        num_refs = mn;
    }
    else{
        mn = n;
        num_refs = mn-1;
    }
    

    // first set V to vecnumth vector of X, padded with zeros
    for(int i = 0; i < num_refs + 1; ++i){
        V[i] = X[vecnum*mn + i];
    }
    if(num_refs + 1 < n){
        for(int i = num_refs + 1; i < n; ++i){
            V[i] = 0.0;
        }
    }

    for (int j=num_refs + 1; j >= 0; j--){
        inner_prod = dot_prod(n-j-1,A_mod+j+1+j*m,V+j+1);                
        for (int k=j+1; k<n; k++){
            V[k] -= 2*A_mod[j*m+ k]*inner_prod;
	}
	

    }


}


void multU(int m,int n, int vecnum, double*restrict A_mod, double*restrict Y,double*restrict U){
    /*****************************************
     * Author: Steven Delong
     *
     * Description: creates a left singular vector U from the reflectors stored in A after
     * bidiagonalization and from the singular vectors found in Y after running
     * RighttoLeftSingularVecrors.
     *
     * inputs:
     *       m - row dimension of the original matrix A, output will be m x m
     *       mn - minimum dimension, min(m,n)
     *       vecnum - number of the vector to compute.
     *       A_mod - stores reflectors, it will be the overwritten A created
     *           by the bidiagonalization function (e.g. bidiag_seq)
     *       Y - left singular vectors for the reduced bidiagonal matrix, output from
     *           RighttoLeftsingularvectors
     *
     * Outputs:
     *        U - the vecnum'th left singular vector of A
     ****************************************************/

    int mn, num_refs;
    if (m<n){
        mn = m;
        num_refs = mn;
    }
    else{
        mn = n;
        num_refs = mn-1;
    }
    
    double inner_prod;
    // first set U to vecnumth vector of Y, padded with zeros
    for(int i = 0; i < mn; ++i){
        U[i] = Y[vecnum*mn + i];
    }
    if(mn  < m){
        for(int i = mn; i < m; ++i){ // shouldn't need to do this.
            U[i] = 0.0;
        }
    }

    // apply reflectors to U until we have the singular vector we want
    for (int j=num_refs; j >= 0; j--){
        inner_prod = dot_prod(m-j,U+j,A_mod+j+j*m);
        for (int k=j; k<m; k++){
            U[k] -= 2*A_mod[k+j*m]*inner_prod;
        }
    }
}
