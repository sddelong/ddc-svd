/*********************************************************************
 * File: bidiag_dr.c
 * Author: Travis Askham, created 11/11/2012
 * Description: This is the testing driver for the bidiagonalization
 * routines in bidiag_par.c
 * 
 * Compile: make bidiag_dr
 * Run: ./bidiag_dr m n 
 * 
 * Input: 
 * 		m - the number of rows in the matrix
 * 		n - the number of columns in the matrix
 * 
 * Output:
 * 		the program prints result info to the console 
 * 
 * Details:
 * 		- The matrices are stored in column major order, in double 
 * 		  precision.
 * 		- The random numbers are drawn from [a,b), where a and b
 * 		  are parameters in the code.
 * 		- The timing is inaccurate if the device for the parallel routine
 *        is chosen "INTERACTIVE"ly. See bidiag_par in bidiag_par.c
 *  
 ********************************************************************/
#include "timing.h"
#include "cl-helper.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "bidiag_par.h"
#include "bidiag.h"
#include "matrix_helper.h"

/**********************
*
* FUNCTION DECLARATIONS
*
***********************/

// random doubles
double rand_d(double a, double b);


/***************************************
 * 
 * Random number stuff
 * 
 * *************************************/
 
 // returns a random double in the interval [a,b) 
double rand_d(double a, double b){
	double c;	
	//c = rand()/(1.0*RAND_MAX); // in [a,b]
	c = rand()/(1.0*RAND_MAX+1); // in [a,b)
	c = a + c*(b-a);	
	return c;
}

/***************************************
 *** DRIVER ****************************
 ***************************************/

int main(int argc, char ** argv){

	// check input
	if (argc != 3)
	{
	  fprintf(stderr, "in main: need two arguments!\n");
	  abort();
	}

	// seed the random number generator
	//srand( (int) time(0));
	srand( (int) 4);

	// parameters
	const long m = atol(argv[1]);  
	const long n = atol(argv[2]);  
	long mn = 0;	// min of m,n
	long len_beta = 0;
	if ( m < n){
		mn = m;
		len_beta = mn;
	}
	else{
		mn = n;
		len_beta = mn-1;
	}
	double a = 1;
	double b = 2;
	double tol = 1.0e-9;

	// big matrix storage
	double *A = (double *) malloc(sizeof(double) *m*n);
	if(!A) { fprintf(stderr,"in main: failed to allocate A\n"); abort();}
	double *A2 = (double *) malloc(sizeof(double) *m*n);
	if(!A2) { fprintf(stderr,"in main: failed to allocate A2\n"); abort();}
	double *B = (double *) malloc(sizeof(double) *m*n);
	if(!B) { fprintf(stderr,"in main: failed to allocate B\n"); abort();}
	double *A_Copy = (double *) malloc(sizeof(double) *m*n);
	if(!A_Copy) { fprintf(stderr,"in main: failed to allocate A_Copy\n"); abort();}
	double *A_Result = (double *) malloc(sizeof(double) *m*n);
	if(!A_Result) { fprintf(stderr,"in main: failed to allocate A_Result\n"); abort();}
	double *temp = (double *) malloc(sizeof(double) *m*n);
	if(!temp) { fprintf(stderr,"in main: failed to allocate temp\n"); abort();}
	double *temp2 = (double *) malloc(sizeof(double) *m*m);
	if(!temp2) { fprintf(stderr,"in main: failed to allocate temp2\n"); abort();}
	double *temp3 = (double *) malloc(sizeof(double) *n*n);
	if(!temp3) { fprintf(stderr,"in main: failed to allocate temp3\n"); abort();}
	double *U = (double *) malloc(sizeof(double) *m*m);
	if(!U) { fprintf(stderr,"in main: failed to allocate U\n"); abort();}
	double *UT = (double *) malloc(sizeof(double) *m*m);
	if(!UT) { fprintf(stderr,"in main: failed to allocate UT\n"); abort();}
	double *V = (double *) malloc(sizeof(double) *n*n);
	if(!V) { fprintf(stderr,"in main: failed to allocate V\n"); abort();}
	double *VT = (double *) malloc(sizeof(double) *n*n);
	if(!VT) { fprintf(stderr,"in main: failed to allocate VT\n"); abort();}
	
	// diagonal component storage
	double *alpha = (double *) malloc(sizeof(double) *mn);
	if(!alpha) { fprintf(stderr,"in main: failed to allocate alpha\n"); abort();}
	double *beta = (double *) malloc(sizeof(double) *len_beta);
	if(!beta) { fprintf(stderr,"in main: failed to allocate beta\n"); abort();}
	double *alpha2 = (double *) malloc(sizeof(double) *mn);
	if(!alpha) { fprintf(stderr,"in main: failed to allocate alpha2\n"); abort();}
	double *beta2 = (double *) malloc(sizeof(double) *len_beta);
	if(!beta) { fprintf(stderr,"in main: failed to allocate beta2\n"); abort();}
	
	// fill A, A_Copy
	for (int i=0; i<m*n; i++){
		A[i] = rand_d(a,b);
		A_Copy[i] = A[i];
		A2[i] = A[i];
	}
	
	timestamp_type time1, time2;
	
	// compute the bidiagonal form
	get_timestamp(&time1);
	bidiag_par(m,n,A,alpha,beta);
	get_timestamp(&time2);
	double elapsed_par = timestamp_diff_in_seconds(time1,time2);
	printf("time_par = %g\n",elapsed_par);
	get_timestamp(&time1);
	bidiag_seq(m,n,A2,alpha,beta);
	get_timestamp(&time2);
	double elapsed_seq = timestamp_diff_in_seconds(time1,time2);
	printf("time_seq = %g\n",elapsed_seq);
	// form the orthogonal matrices
	//form_u_par(m,n,A,U);
	//form_v_par(m,n,A,V);
	//form_bidiag(m,n,alpha,beta,B);
	//transpose(n,n,V,VT);
	//transpose(m,m,U,UT);
	
	
	// check the result of A_Result = U * B * V^T
	//dgemm_simple(m,n,n,B,VT,temp);
	//dgemm_simple(m,n,m,U,temp,A_Result);
	//dgemm_simple(n,n,n,VT,V,temp3);
	//dgemm_simple(m,m,m,UT,U,temp2);
	
	int errors = 0;
	for (int i=0; i < m*n; i++){
		if ( fabs(A[i]-A2[i]) > tol ){
			errors++;
		}
	}
	printf("ERRORS = %d\n",errors);
	
	
	//print_matrix(A,m,n,"A = ");
	//print_matrix(A2,m,n,"A2 = ");
	//print_matrix(A_Copy,m,n,"A_Copy = ");
	//print_matrix(A_Result,m,n,"A_Result = ");
	//print_matrix(B,m,n,"B = ");
	//print_matrix(U,m,m,"U = ");
	//print_matrix(V,n,n,"V = ");
	//print_matrix(temp2,m,m,"temp2 = ");
	//print_matrix(temp3,n,n,"temp3 = ");
		
	free(A);
	free(A2);
	free(A_Copy);
	free(A_Result);
	free(B);
	free(temp);
	free(temp2);
	free(temp3);
	free(U);
	free(UT);
	free(V);
	free(VT);
	free(alpha);
	free(alpha2);
	free(beta);
	free(beta2);
	

	return 0;
	
}
