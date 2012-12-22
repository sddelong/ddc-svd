/**********************************************************************
 * 
 * Copyright (C) 2012 Travis Askham, Steven Delong
 * 
 * Permission is hereby granted, free of charge, to any person obtaining 
 * a copy of this software and associated documentation files (the 
 * "Software"), to deal in the Software without restriction, including 
 * without limitation the rights to use, copy, modify, merge, publish, 
 * distribute, sublicense, and/or sell copies of the Software, and to 
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 * 
**********************************************************************/

#ifndef BIDIAG_PAR
#define BIDIAG_PAR

void bidiag_par(int m, int n, double *restrict A, double *restrict alpha,
			double *restrict beta);
							
void form_u_par(int m, int n, const double*restrict A_mod, double*restrict U);
void form_v_par(int m, int n, const double*restrict A_mod, double*restrict V);


void find_reflector_and_scale_col(int m, int n, int mn, int col, 
				int work_per_item_red, int wgdim_red, int work_per_item_upd, int wgdim_upd,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_numsq_matcol,cl_kernel knl_update_scale_matcol,
				cl_kernel knl_sum,
				cl_mem buf_A, cl_mem buf_alpha,
				cl_mem buf_scratch, cl_mem buf_scratch2, cl_mem buf_temp);
				
void find_reflector_and_scale_row(int m, int n, int mn, int row, 
				int work_per_item_red, int wgdim_red, int work_per_item_upd, int wgdim_upd,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_normsq_matrow,cl_kernel knl_update_scale_matrow, 
				cl_kernel knl_sum,
				cl_mem buf_A, cl_mem buf_beta,
				cl_mem buf_scratch, cl_mem buf_scratch2, cl_mem buf_temp);
				
void no_reflect( int m, int n, int mat_loc, int vec_loc,
				int work_per_item_nr, int wgdim_nr,
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_sanders,
				cl_mem buf_mat, cl_mem buf_vec);
				
void left_householder_par_2d(int m, int n, int mn, int offset,
				size_t ldim_dp[], size_t ldim_update_mat[],
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_left_dotprods, cl_kernel knl_update_left_hh,
				cl_mem buf_A, cl_mem buf_scratch);
				
void right_householder_par_2d(int m, int n, int mn, int offset,
				size_t ldim_dp[], size_t ldim_update_mat[],
				cl_context ctx, cl_command_queue queue,
				cl_kernel knl_right_dotprods, cl_kernel knl_right_update_mat,
				cl_mem buf_A, cl_mem buf_scratch);
				

void multU(int m,int n, int vecnum, double*restrict A_mod, double*restrict Y,double*restrict U);
void multV(int m,int n, int vecnum, double*restrict A_mod, double*restrict X,double*restrict V);
							
#endif
