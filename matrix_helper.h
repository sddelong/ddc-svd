/**********************************************************************
 * 
 * Copyright (C) 2012 Travis Askham
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

#ifndef MATRIX_HELPER
#define MATRIX_HELPER
		
void print_matrix(const double * A, long m, long n, char* message);		

double l2_normv(int l, const double*restrict v);
void scale_vector(int l, double*restrict v, double scale);
double dot_prod(int l, const double* a, const double* b);

double l2_norm_mat_row(int m, int n, int l, const double*restrict row);
void scale_mat_row(int m, int n, int l, double*restrict row, double scale);
double dot_prod_mat_rows(int m, int n, int l, const double* a, const double* b);
double dot_prod_mat_row_with_vec(int m, int n, int l, const double* row, const double* vec);

void set_vec_to_zero(int l, double*restrict v);
void dgemm_simple( const int M, const int N, const int L, const double *A, const double *B, double *C);

void form_bidiag( const int M, const int N, const double *alpha, const double *beta, double * mat);
void transpose( const int M, const int N, const double * A, double * AT);
#endif
