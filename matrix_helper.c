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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_helper.h"

/******************************************************************
 * 
 * The following are straightforward routines which allow for simple 
 * operations on matrices (stored in column-major format) and vectors. 
 * They are not optimized for speed in any way.
 * 
 ******************************************************************/

// prints a matrix (stored as an array in column-major order) 
void print_matrix(const double * A, long m, long n, char * message){
	
    printf("%s \n",message);
    for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			printf("%11.3E",A[i+m*j]);
		}
		printf("\n");
	}
	return;
}
 
 // Returns the l2 norm of the array of length l, starting at v
 double l2_normv(int l, const double*restrict v){
	 double temp = 0;
	 for (int i=0; i<l; i++){
		 temp+= v[i]*v[i];
	 }
	 return sqrt(temp);
 }
 
 // scales the vector v of length l by scale scale
 
void scale_vector(int l, double*restrict v, double scale){
	for (int i=0; i<l; i++){
		v[i] *= scale;
	}
	return;
}

// computes the dot product of the given vectors of length l
double dot_prod(int l, const double* a, const double* b){
	double temp = 0;
	for (int i=0; i<l; i++){
		temp+= a[i]*b[i];
	}
	return temp;
}
 
// Returns the l2 norm of the partial row (length l) of a mxn matrix 
// starting at row
 double l2_norm_mat_row(int m, int n, int l, const double*restrict row){
	 double temp = 0;
	 for (int i=0; i<l; i++){
		 temp+= row[i*m]*row[i*m];
	 }
	 return sqrt(temp);
 }
 
 // scales the partial row (length l) of a mxn matrix 
// starting at row
 
void scale_mat_row(int m, int n, int l, double*restrict row, double scale){
	for (int i=0; i<l; i++){
		row[i*m] *= scale;
	}
	return;
}	 

// computes the dot product of the given partial rows of length l of an mxn matrix
double dot_prod_mat_rows(int m, int n, int l, const double* a, const double* b){
	double temp = 0;
	for (int i=0; i<l; i++){
		temp+= a[i*m]*b[i*m];
	}
	return temp;
}

double dot_prod_mat_row_with_vec(int m, int n, int l, const double* row, const double* vec){
	double temp = 0;
	for (int i=0; i<l; i++){
		temp+= row[i*m]*vec[i];
	}
	return temp;
}

void set_vec_to_zero(int l, double*restrict v){
	for (int i=0; i<l; i++){
		v[i] = 0;
	}
	return;
}


// Performs a naive matrix-matrix multiply
// C is M x N, A is M x L, B is L x N
void dgemm_simple( const int M, const int N, const int L, const double *A, const double *B, double *C){
	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			for (int k=0; k<L; k++){
				 C[i+j*M] = C[i+j*M] + A[i+k*M]*B[k+j*L];
			}
		}
	}
	return;
}

// fills the M x N matrix 'mat' with zeros
// and alpha as the main diagonal and beta as the super diagonal
// if M >= N then alpha is length N, beta is length N-1
// if N > M then alpha is length M, beta is length M
void form_bidiag( const int M, const int N, const double *alpha, const double *beta, double * mat){
	for (int j=0; j < N; j++){
		for (int i=0; i < M; i++){
			mat[i+j*M] = 0;
		}
	}
	
	if (N>M){
		for (int i=0; i < M; i++){
			mat[i+i*M] = alpha[i];
			mat[i+(i+1)*M] = beta[i];
		}
	}
	else{
		for (int i=0; i < N-1; i++){
			mat[i+i*M] = alpha[i];
			mat[i+(i+1)*M] = beta[i];
		}
		mat[N-1 + (N-1)*M] = alpha[N-1];
	}
	
	return;
}

void transpose( const int M, const int N, const double * A, double * AT){
	for (int i=0; i < N; i++){
		for (int j=0; j < M; j++){
			AT[i+j*N] = A[j + i*M];
		}
	}
	
	return;
}
	
	
