
/************************************
 *    Description:  Library of routines to find singular vectors given singular values.
                    Done using OpenMP parallelization

      Author:   Steven Delong - sdd255@nyu.edu

************************************/

/*****************************************
 * INCLUDES
 *****************************************/


#include "parallel-twisted.h"
// include sequential header to use those functions as backup if need be.
//#include "sequential-twisted.h" 
#include "matrix_helper.h"
#include<stdlib.h>
#include<stdio.h>
#include "cl-helper.h"
#include <omp.h>
#include <math.h>
#include "timing.h"



/****************************************
 * FUNCTIONS FOR TWISTED FACTORIZATION
****************************************/


void BidiagMatVec(int n, int m, double* A, double* B, double* x,double * y){
  /*
     Description:  simple bidiagonal matrix vector multiply.
     
     Inputs:
           n - Size of diagonal and superdiagonal (n x n+1 matrix)
	   A - Main Diagonal, length n
	   B - Superdiagonal, length n, 
	   x - input vector, length n+1
	   
    Outputs:
          y - output vector, length n

  *********************************/

  // first n-1 entries are the same, whether we're square or not
  #pragma omp parallel for
    for(int i = 0; i < n-1; ++i){
        y[i] = A[i]*x[i] + B[i]*x[i+1];
        }

    if(n == m){ // handle square case
      y[n-1] = A[n-1]*x[n-1];
    }
    else{ // handle non-square case
      y[n-1] = A[n-1]*x[n-1] + B[n-1]*x[n];
    }
    
} // BidiagMatVec

void NormalizeVectors(int n, int m, double* X){
  /* Description: Normalizes a group of n, length m, vectors so that the L2 norm is 1.
     
     inputs:
             n - how many vectors
	     m - length of vectors
	     X - vectors, jth component of ith vector is X[i*m + j]
	     
    output:
            X - the vectors now normalized
	    
  ******************************************/


    double * Xnorms = (double *) malloc(sizeof(double) * n);
    if (!Xnorms) { perror("alloc Xnorms"); abort();}    


#pragma omp parallel for
    for(int i = 0; i < n; ++i){
        Xnorms[i] = 0.0;
        for(int j = 0; j < m; ++j){
            Xnorms[i] = Xnorms[i] + X[i*m + j]*X[i*m + j];
        }
        Xnorms[i] = sqrt(Xnorms[i]);
    }

#pragma omp parallel for    
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m; ++j){
            X[i*m + j] = X[i*m + j]/Xnorms[i];
        }
    }

    free(Xnorms);
} //NormalizeVectors


void CalcGamma(int n,int m, double* A,double* B,double* D1,double* D2,double* sigma,double * out){
    // Function that calculates gamma = D1 + D2  - (A^2 + B^2 - sigma^2).
    // use gpu since all of this is perfectly parallelizeable

    cl_context ctx;
    cl_command_queue queue;
    cl_int status;
    create_context_on("Advanced Micro Devices, Inc.", NULL, 0, &ctx, &queue, 0);
    //create_context_on("Intel", NULL, 0, &ctx, &queue, 0);  

    char *knl_text = read_file("CalcGamma.cl");
    cl_kernel knl = kernel_from_string(ctx, knl_text, "CalcGamma", NULL);
    free(knl_text);


    // create device memory
    cl_mem buf_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                    sizeof(double) * n, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");
    
    cl_mem buf_B = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				  sizeof(double) * (m-1), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");
    
    cl_mem buf_D1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                     sizeof(double) * n * m, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");
    
    cl_mem buf_D2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                     sizeof(double) * n * m, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    cl_mem buf_sigma = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                     sizeof(double) * n, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                     sizeof(double) * n * m, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");


  // --------------------------------------------------------------------------
  // transfer to device
  // --------------------------------------------------------------------------
    

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_A, /*blocking*/ CL_TRUE, /*offset*/ 0,
        n * sizeof(double), A,
        0, NULL, NULL));


    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_B, /*blocking*/ CL_TRUE, /*offset*/ 0,
        (m-1) * sizeof(double), B,
        0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
                        queue, buf_D1, /*blocking*/ CL_TRUE, /*offset*/ 0,
                        n * m * sizeof(double), D1,
                        0, NULL, NULL));
    
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
                        queue, buf_D2, /*blocking*/ CL_TRUE, /*offset*/ 0,
                        n * m * sizeof(double), D2,
                        0, NULL, NULL));
    
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
                        queue, buf_sigma, /*blocking*/ CL_TRUE, /*offset*/ 0,
                        n * sizeof(double), sigma,
                        0, NULL, NULL));


    CALL_CL_GUARDED(clFinish, (queue));
    
    SET_8_KERNEL_ARGS(knl, buf_A, buf_B, buf_D1,buf_D2,buf_sigma,buf_out,n,m);
    size_t ldim[] = { 1 };
    size_t gdim[] = { n * m };
    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		    (queue, knl, 
		     /*dimensions*/ 1, NULL, gdim, ldim,
		     0, NULL, NULL));    

    CALL_CL_GUARDED(clFinish, (queue));




    CALL_CL_GUARDED(clEnqueueReadBuffer, (
					  queue, buf_out, /*blocking*/ CL_TRUE, /*offset*/ 0,
					  n * m * sizeof(double), out,
					  0, NULL, NULL));
    CALL_CL_GUARDED(clFinish, (queue));

#pragma omp parallel for // End cases "by hand" for now.  Probably move this to GPU
    for(int i = 0;i < n; ++i){
      out[i*m] = D1[i*m] + D2[i*m] - (A[0]*A[0] - sigma[i]*sigma[i]);
      if(m > n){
	out[i*m + m-1] = D1[i*m + m-1] + D2[i*m + m-1] - (B[n-1]*B[n-1] - sigma[i]*sigma[i]);
      }
      else{
	out[i*m + m-1] = D1[i*m + m-1] + D2[i*m + m-1] - (A[m-1]*A[m-1] + B[m-2]*B[m-2] - sigma[i]*sigma[i]);
      }
    }


}// CalcGamma 



minindex which_min_gamma(double * gamma, int start, int end,int level){

//    figure out the index of minimum and minimum of
//    gamma[start] -> gamma[end]
    
  int tid, nthreads;
  minindex result, min1, min2;
  if(start  == end){
    result.value = gamma[start];
    result.index = start;
  }
  else{
    /*****************************************
   /* removed parallelism here, faster to do it outside on the loop
    ****************************************/
    /*    if(level < 2)
      {
#pragma omp parallel  private(tid,nthreads)
	{
	  tid = omp_get_thread_num();
#pragma omp sections
	  {
#pragma omp section
	    {
	      min1 = which_min_gamma(gamma, start, start + (end - start)/2,level + 1);
	    }
#pragma omp section
	    {
	      min2 = which_min_gamma(gamma, start + 1 + (end-start)/2,end, level + 1);
	    }
	  }
	}
      }
      else{*/
      min1 = which_min_gamma(gamma, start, start + (end - start)/2,level + 1);
      min2 = which_min_gamma(gamma, start + 1 + (end-start)/2,end, level + 1);
      //    }
    if(fabs(min1.value) < fabs(min2.value)){
          result.value = min1.value;
          result.index = min1.index;
      }
      else{
          result.value = min2.value;
          result.index = min2.index;
      }
  }
  return(result);

}//which_min_gamma

void SquareB(int n, int m, double * ina, double * inb, double * outa, double* outb){
  /*  calculate B^t B, straightforward way.  B is n x m bidiagonal
      B^t B is therefore m x m and tridiagonal
      
      inputs:
            n - length of diagonal ina 
	    m - length of superdiagonal inb
	    ina - diagonal of B
	    inb - superdiagonal of B
     outputs:
            outa - diagonal of B^t B, length n + 1
	    outb - offdiagonals of B^t B, length is n
  */
  
    outa[0] = ina[0]*ina[0];

#pragma omp parallel for
    for(int i = 0;i < n-1; ++i){
        outb[i] = ina[i]*inb[i];
        outa[i + 1] = ina[i+1]*ina[i+1] + inb[i]*inb[i];
    }
    if(m > n){
      outb[n-1] = ina[n-1]*inb[n-1];
      outa[n] = inb[n-1]*inb[n-1];
    }
    
}//SquareB



void CholFactorization(int n,int m, double* sigma, double* ina, double* inb,double* outp, double* outd1,double * outq, double *outd2){
  /* Description: does P(D1)P^t and Q(D2)Q^t factorizations of tridiagonal matrix
     P will be lower bidiagonal with 1s on diagonal
     Q will be upper bidiagonal with 1s on diagonal
     
     inputs:
             n - length of sigma
	     m - length of ina
	     sigma - vector of singular values, length n
	     ina - diagonal of matrix, length m
	     inb - offdiagonal of matrix, length m-1
	     
     outputs:
            outp - lower diag of L for each sigma, length (m-1)*n
	    outd1 - diag of D1, length m*(n)
	    outq - upper diag of U, for each sigma, length (m-1)*n
	    outd2 - diag of D2, length m*(n)
  **********************************************************/

#pragma omp parallel for
  for(int k = 0;k < n;++k){ // iterate through sigmas
    // do first element of d1, last of d2
        outd1[k*m] = ina[0] - sigma[k]*sigma[k];
        outd2[k*m + m-1] = ina[m-1] - sigma[k]*sigma[k];

	// iterate forward for D1, P, backward for D2, Q
        for(int i = 0;i < m-1; ++i){
	  
	  // Forward factorization pieces (lower, diag, upper)
	  outp[k*m + i] = inb[i]/outd1[k*m + i];
	  
	  outd1[k*m + i + 1] = (ina[i + 1] - sigma[k]*sigma[k] - outp[k*m + i]*outp[k*m + i]*outd1[k*m + i]);

	  // backward factorization pieces (Upper, Diag, Lower)
	  outq[k*m + m-2 - i] = inb[m - 2 - i]/outd2[k*m + m-1 - i];

	  outd2[k*m + m-2 - i] = ina[m - 2 - i] - sigma[k]*sigma[k] - outd2[k*m + m-1 - i]*outq[k*m + m-2 - i]*outq[k*m + m-2 - i];

        }

    }

}//CholFactorization


void backsolve(int n, int m, double* P,double* Q, double* D1,double* D2, double* sigma, double * gamma, double* x){
  /****************************************
      Description:  Backsolve the N D N^t X^(2)  = X^(1) to try to get more accuracy.
                    P, D1, Q, D2 contain a vector for each sigma.

       inputs:
             n - matrix rows
             m - matrix columns
             P - lower part of LDL, length m*n
             D1 - diagonal of LDL, length m*n
             Q - upper part of UDU, length m*n
             D2 - diagonal part of UDU, length m*n
             sigma - singular values, length n
             gamma - vector of gamma values, calculated in TwistedFactorization
             x - guess at right singular vectors, x1, length m*n
                 will be overwritten with more accurate estimate

      outputs:
              x - (hopefully) more accurate estimate of right singular values.
  **************************************************/
  
  
  double * xtemp = (double * ) malloc(sizeof(double)*n*m);
  if (!xtemp) { perror("alloc xtemp"); abort();}        


  // backsolve
#pragma omp parallel for
    for(int i = 0;i < n; ++i){


      // first solve N_k Xtemp = X^1
      // start from the ends and work inwards
      xtemp[i*m] = x[i*m];
      xtemp[i*m + m-1] = x[i*m + m-1];

      for(int j = 1; j < m/2; ++j){
	xtemp[i*m + j] = x[i*m + j] - P[i*m + j-1]*xtemp[i*m + j-1];
      }
      for(int j = m-2;j > (m/2); --j){
	xtemp[i*m + j] = x[i*m + j] - Q[i*m + j]*xtemp[i*m + j + 1];
      }

      xtemp[i*m + m/2] = x[i*m + m/2] - Q[i*m + m/2]*xtemp[i*m + m/2 + 1] - P[i*m + m/2 - 1]*xtemp[i*m + m/2 - 1];
      
      // now solve D_k N_k^t X^2 = Xtemp 
      // start with m/2 entry, can backsolve in both directions from there;
      x[i*m + m/2] = xtemp[i*m + m/2]/gamma[i*m + m/2];

      // first direction
      for(int j = m/2+1; j < m; ++j){
	x[i*m + j] = (xtemp[i*m + j] - D2[i*m + j]*Q[i*m + j-1]*x[i*m + j -1])/D2[i*m + j];
      }
      
      // other direction
      for(int j = m/2-1; j >= 0; --j){
	x[i*m + j] = (xtemp[i*m + j] - D1[i*m + j]*P[i*m + j]*x[i*m + j + 1])/D1[i*m + j];
      }

    }

    free(xtemp);
}



void TwistedFactorization(int n, int m, double* P,double* Q, double* D1,double* D2,double* ina, double* inb, double* sigma, double* x)
{

  /* Description:   Backsolves the twisted factorization N_k x_k = e_k

     inputs:
            n - number of rows in original B
	    m - number of columns in original B (n or n+1)
	    P - offdiag from P, lower bidiagonal
	    D1 - diagonal of D1   P(D1)P^t = B^t B - sigma^2 I
	    Q - offdiag from Q, upper bidiagonal
	    D2 -  diagonal of D2   Q(D2)Q^t = B^t B - sigma^2 I
	    ina - diagonal of B, length n
	    inb - diagonal of B, length m-1
	    sigma - singular values of B, length n
	    
    Outputs:
           X - singular vectors of B, m x n when stored row-major.

   */

    // find min of D1 + D2 + (a^2 + b^2 - sigma^2);
    double * gamma = (double * ) malloc(sizeof(double)*n*m);
    if (!gamma) { perror("alloc gamma"); abort();}

    timestamp_type time1, time2;
    get_timestamp(&time1);
#pragma omp parallel for
    for(int i = 0;i < n; ++i){
      gamma[i*m] = D1[i*m] + D2[i*m] - (ina[0]*ina[0] - sigma[i]*sigma[i]);
      for(int j = 1;j< n; ++j){
	gamma[i*m + j] = D1[i*m + j] + D2[i*m + j] - (ina[j]*ina[j] + inb[j-1]*inb[j-1] - sigma[i]*sigma[i]);
	}
      if(m > n){
	gamma[i*m + m-1] = D1[i*m + m-1] + D2[i*m + m-1] - (inb[n-1]*inb[n-1] - sigma[i]*sigma[i]);
      }
    }

    //    CalcGamma(n,m,ina,inb,D1,D2,sigma,gamma);
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1,time2);
//    printf("parallel Calc Gamma took %f s \n", elapsed);




    int * kvec = (int * ) malloc(sizeof(int)*n);
    if (!kvec) { perror("alloc kvec"); abort();}

    
    // fill Kvec with minimum indices

    get_timestamp(&time1);    
#pragma omp parallel for
    for(int i = 0;i < n;++i)
    {
      kvec[i] = which_min_gamma(gamma,i*m,i*m + m - 1,0).index - i*m;
    }
    get_timestamp(&time2);
    elapsed = timestamp_diff_in_seconds(time1,time2);
//    printf("parallel which min Gamma took %f s \n", elapsed);

    // backsolve 
#pragma omp parallel for
    for(int i = 0;i < n; ++i){
      // start with kth entry, can backsolve in both directions from there;
      x[i*m + kvec[i]] = 1.0;
      //#pragma omp parallel
      //      {

      //#pragma omp sections   
      //	{
      //#pragma omp section
      //	  {					
	    if(kvec[i] < m-1){
	      for(int j = kvec[i]+1; j < m; ++j){
		x[i*m + j] = -1.0*Q[i*m + j-1]*x[i*m + j -1];
	      }
	    }
	      //	  }
	      //#pragma omp section                             
	      //	  {
	    
	    // other direction
	    if(kvec[i] > 0){
	       for(int j = kvec[i]-1; j >= 0; --j){
		 x[i*m + j] = -1.0*P[i*m + j]*x[i*m + j + 1];
	       }
	    }
	    //	  }
	}
    //      }
    //    }
    backsolve(n,m,P,Q,D1,D2,sigma,gamma,x);
    
    free(gamma);
    
}//TwistedFactorization

void RighttoLeftSingularVectors(int n, int m, double* A, double * B, double * sigma, double * X, double * Y){
  /* Description:  Calculate left singular vectors given singular values and right vectors
     
     inputs :
            n - Size of bidiagonal matrix (A and B)
	    A - diagonal of matrix, length n
	    B - superdiagonal of matrix, length n
	    sigma - singular values
	    X - right singular vectors
	    
     outputs:
           Y - left singular vectors
  */

  
  for(int i = 0; i < n;++i){ // iterate through n X vectors, each of length m
    BidiagMatVec(n,m,A,B,&(X[i*m]),&(Y[i*n]));
      for(int j = 0; j < n; ++j){
	Y[i*n + j] = Y[i*n + j]/sigma[i];
      }
    }
}//RighttoLeftSingularVectors
        

void CalcRightSingularVectors(int n, int m, double* A, double* B,double* sigma, double* X){

  /*   Description:  Calculate right singular vectors of bidiagonal n x m matrix
               with diagonal A, superdiagonal B, and singular values sigma
	       m must be either n or n+1
	       
       inputs:
              n - size of diagonal
	      m - size of superdiagonal
	      A - diagonal, length n
	      B - superdiagonal, length m
	      sigma - singular values, length n

       outputs:
                X - Right Singular Values, m x m
  */

  // test out openMP stuff
//  omp_set_num_threads(6);
  int nthreads,tid;

  double * squaredA = (double *) malloc(sizeof(double) * m);
  if (!squaredA) { perror("alloc squaredA"); abort();}
  
  double * squaredB = (double *) malloc(sizeof(double) * m); // really only need n
  if (!squaredB) { perror("alloc squaredB"); abort();}

  double * D1 = (double *) malloc(sizeof(double) * n*m); 
  if (!D1) { perror("alloc D1"); abort();}
  
  double * D2 = (double *) malloc(sizeof(double) *n*m); 
  if (!D2) { perror("alloc D2"); abort();}

  double * Q = (double *) malloc(sizeof(double) * n*m);
  if (!Q) { perror("alloc Q"); abort();}

  double * P = (double *) malloc(sizeof(double) * n*m); // really only need n * (n+1)
  if (!P) { perror("alloc P"); abort();}


  // calculate B^t B
  timestamp_type time1, time2;
  get_timestamp(&time1);
  SquareB(n,m,A,B,squaredA,squaredB);
  get_timestamp(&time2);

  double elapsed = timestamp_diff_in_seconds(time1,time2);
//  printf("Squaring took %f s \n", elapsed);

  
  // now we have a tridiag 
  // square m x m matrix with diagonal squaredA, offdiagonals squaredB

  // do chol factorization both ways
  get_timestamp(&time1);
  CholFactorization(n,m,sigma,squaredA,squaredB,P,D1,Q,D2);
  get_timestamp(&time2);
  elapsed = timestamp_diff_in_seconds(time1,time2);
//  printf("parallel Cholfactorization took %f s \n", elapsed);

  // do twisted factorization solve to get singular vectors
  get_timestamp(&time1);
  TwistedFactorization(n,m,P,Q,D1,D2,A,B,sigma,X);
  get_timestamp(&time2);
  elapsed = timestamp_diff_in_seconds(time1,time2);
//  printf("parallel TF + Backsolve took %f s \n", elapsed);


  // now X has unnormalized singular vectors
  // they are stored in column major order
  // this works with our left-to-right singular vector approach, 
  // and this X can be passed directly to RighttoLeft SingularVectors

  // normalize vectors
  NormalizeVectors(n,m,X);

  free(squaredA);
  free(squaredB);
  free(D1);
  free(D2);
  free(P);
  free(Q);

}//CalcRightSingularVectors
