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
#include "bidiag.h"

void bidiag_seq(int m, int n, double*restrict A, double*restrict alpha, double*restrict beta){
			
/**********************************************************
 * Function: bidiag_seq
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Description: reduces the input matrix to bidiagonal form
 * using a sequential algorithm. The reduction is accomplished via
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
 
	int mn;
	int sign;
	double temp_norm;
	double x1;
	
	if (m<n){
		mn = m;
	}
	else{
		mn = n;
	}
	
	for (int i=0; i< mn-1; i++){
		
		// do the left Householder reflection 
		x1 = A[i+i*m];
		if ( x1 < 0){
			sign = -1;
		}
		else{
			sign = 1;
		}
		
		temp_norm = l2_normv(m-i,A+i+i*m); // norm of the partial column
											// starting at A[i,i]							
		
		alpha[i] = -sign*temp_norm;
		
		A[i+i*m] += sign*temp_norm;			// the unscaled reflector is 
											// in the partial column 
											// starting at A[i,i]
		// norm of reflector
		temp_norm = sqrt(2.0)*sqrt(temp_norm*temp_norm + fabs(temp_norm*x1)); 
		scale_vector(m-i,A+i+i*m,1.0/temp_norm);		// scale the reflector
		
		left_householder(m,n,m-i,n-i-1,A+i+i*m,A+i+(i+1)*m); //apply reflection
															 // to the remainder of the matrix
		
		if ( i < n-2){
			// do the right Householder reflection
			x1 = A[i+(i+1)*m];
			if ( x1 < 0){
				sign = -1;
			}
			else{
				sign = 1;
			}
			
			temp_norm = l2_norm_mat_row(m,n,n-i-1,A+i+(i+1)*m); // norm of the partial row
												// starting at A[i,i+1]							
			
			beta[i] = -sign*temp_norm;
			
			A[i+(i+1)*m] += sign*temp_norm;			// the unscaled reflector is 
												// in the partial row 
												// starting at A[i,i+1]
			
			// norm of reflector
			temp_norm = sqrt(2.0)*sqrt(temp_norm*temp_norm + fabs(temp_norm*x1)); 
			scale_mat_row(m,n,n-i-1,A+i+(i+1)*m,1.0/temp_norm);			// scale the reflector
			
			right_householder(m,n,m-i-1,n-i-1,A+i+(i+1)*m,A+i+1+(i+1)*m); //apply reflection
																 // to the remainder of the matrix
		}
		else{
			// no reflection on right
			beta[i] = A[i+(i+1)*m];
			A[i+(i+1)*m] = 0;
		}
	
	}
	
	if ( n >= m+1 ){
		// do the right Householder reflection
		x1 = A[mn-1+mn*m];
		if ( x1 < 0){
			sign = -1;
		}
		else{
			sign = 1;
		}
		
		temp_norm = l2_norm_mat_row(m,n,n-mn,A+mn-1+mn*m); // norm of the partial row
											// starting at A[i,i+1]							
		
		beta[mn-1] = -sign*temp_norm;
		
		A[mn-1+mn*m] += sign*temp_norm;			// the unscaled reflector is 
											// in the partial row 
											// starting at A[i,i+1]
		
		// norm of reflector
		temp_norm = sqrt(2.0)*sqrt(temp_norm*temp_norm + fabs(temp_norm*x1)); 
		scale_mat_row(m,n,n-mn,A+mn-1+mn*m,1.0/temp_norm);			// scale the reflector
		
		// no reflection on left
		alpha[mn-1] = A[mn-1+(mn-1)*m]; 
		A[mn-1 + (mn-1)*m] = 0;
		
	}
	else {
		// do the left Householder reflection 
		x1 = A[mn-1+(mn-1)*m];
		if ( x1 < 0){
			sign = -1;
		}
		else{
			sign = 1;
		}
		
		temp_norm = l2_normv(m-mn+1,A+mn-1+(mn-1)*m); // norm of the partial column
											// starting at A[i,i]							
		
		alpha[mn-1] = -sign*temp_norm;
		
		A[mn-1+(mn-1)*m] += sign*temp_norm;			// the unscaled reflector is 
											// in the partial column 
											// starting at A[i,i]
		// norm of reflector
		temp_norm = sqrt(2.0)*sqrt(temp_norm*temp_norm + fabs(temp_norm*x1)); 
		scale_vector(m-mn+1,A+mn-1+(mn-1)*m,1.0/temp_norm);		// scale the reflector
		
	}	
	
	return;
}

void left_householder(int m, int n, int l, int k, double*restrict v, double*restrict A){
								
/************************************************************
 * Function: left_householder
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	l - number of rows in submatrix
 * 	k - number of columns in submatrix
 *  v - reflection vector of length l
 * 	A - pointer to top left of submatrix
 * 
 * 
 ************************************************************/

	double inner_prod;
	
	// step through columns
	for (int j=0; j<k; j++){
		inner_prod = dot_prod(l,v,A+j*m);
		for (int i=0; i<l; i++){
			A[i+j*m] -= 2*v[i]*inner_prod;
		}
	}
	
	return;	

}

void right_householder(int m, int n, int l, int k, double*restrict v, double*restrict A){
								
/************************************************************
 * Function: right_householder
 * 
 * Original Author: Travis Askham (12/20/2012)
 * 
 * Input:
 * 	m - number of rows in matrix
 * 	n - number of columns in matrix
 * 	l - number of rows in submatrix
 * 	k - number of columns in submatrix
 *  v - reflection row of length l (embedded in mxn matrix)
 * 	A - pointer to top left of submatrix
 * 
 * 
 ************************************************************/

	double inner_prod;
	
	// step through columns
	for (int i=0; i<l; i++){
		inner_prod = dot_prod_mat_rows(m,n,k,v,A+i);
		for (int j=0; j<k; j++){
			A[i+j*m] -= 2*v[j*m]*inner_prod;
		}
	}
	
	return;	

}

void form_u(int m, int n, const double*restrict A_mod, double*restrict U){

/************************************************************
 * Function: form_u
 * 
 * Original Author: Travis Askham (12/20/2012)
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

void form_v(int m, int n, const double*restrict A_mod, double*restrict V){

/************************************************************
 * Function: form_v
 * 
 * Original Author: Travis Askham (12/20/2012)
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
