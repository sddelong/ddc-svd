#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "svd_gpu.h"
#include "matrix_helper.h"
#include <stdbool.h>
#include "cl-helper.h"

// random doubles
double rand_d(double a, double b);

static bool IsNaN(double d)
{
    return (d != d);
}


double rand_d(double a, double b){
	double c;	
	//c = rand()/(1.0*RAND_MAX); // in [a,b]
	c = rand()/(1.0*RAND_MAX+1); // in [a,b)
	c = a + c*(b-a);	
	return c;
}




int main(int argc, char** argv){


    if(argc != 3){
        perror("Need 2 arguments!\n");
        abort();
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[1]);
    int mn = (m > n) ? n : m;
    int len_beta = (m >= n) ? n-1 : m;
    int NaNcount = 0;
    int errcount = 0;

    double* A = (double *) malloc(sizeof(double)*m*n);
    if(!A) { fprintf(stderr,"in main: failed to allocate A\n"); abort();}
    double *A_Copy = (double *) malloc(sizeof(double) *m*n);
    if(!A_Copy) { fprintf(stderr,"in main: failed to allocate A_Copy\n"); abort();}
    double *A_Result = (double *) malloc(sizeof(double) *m*n);
    if(!A_Result) { fprintf(stderr,"in main: failed to allocate A_Result\n"); abort();}
    double *temp = (double *) malloc(sizeof(double) *m*n);
    if(!temp) { fprintf(stderr,"in main: failed to allocate temp\n"); abort();}

    double *B = (double *) malloc(sizeof(double) *m*n);
    if(!B) { fprintf(stderr,"in main: failed to allocate B\n"); abort();}


    double *sigma = (double *) malloc(sizeof(double) *mn);
    if(!sigma) { fprintf(stderr,"in main: failed to allocate sigma\n"); abort();}
    double *sigmamat = (double *) malloc(sizeof(double) *mn*(len_beta + 1));
    if(!sigmamat) { fprintf(stderr,"in main: failed to allocate sigmamat\n"); abort();}
    double *U = (double *) malloc(sizeof(double) *m*m);
    if(!U) { fprintf(stderr,"in main: failed to allocate U\n"); abort();}
    double *V = (double *) malloc(sizeof(double) *n*n);
    if(!V) { fprintf(stderr,"in main: failed to allocate V\n"); abort();}
    double *VT = (double *) malloc(sizeof(double) *n*n);
    if(!VT) { fprintf(stderr,"in main: failed to allocate VT\n"); abort();}
    

    for(int i = 0; i < m*n; ++i){
        A[i] = rand_d(1.,4.);
        A_Copy[i] = A[i];
        temp[i] = 0.0;
    }

    svd_gpu(m,n,A,sigma,U,V);
    
    for(int i = 0; i < m*n; ++i){
        A[i] = 0.0;
    }

#if 0
    transpose(n,mn,V,VT);

    form_bidiag(mn,mn,sigma,temp,B);
    
    dgemm_simple(m,mn,mn,U,B,temp);
    dgemm_simple(m,n,mn,temp,VT,A);

    for (int i=0; i < m*n; i++){
      B[i] = fabs(A[i] - A_Copy[i]);
    }

    printf("Frobenius error in SVD = %2.15f \n",l2_norm_mat(m,n,B));
    printf("relative error in Frobenius norm = %2.15f \n",(l2_norm_mat(m,n,B)/l2_norm_mat(m,n,A_Copy)));

#endif
    
    
}
