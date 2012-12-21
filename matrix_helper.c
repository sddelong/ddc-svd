#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_helper.h"

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


 // Returns the l2 norm of the array of length l, starting at v
double l2_norm_mat(int m, int n,const double*restrict M){
	 double temp = 0;
	 for (int i=0; i<m*n; i++){
		 temp+= M[i]*M[i];
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

double dot_prod2(int l, const double* a, const double* b){
	double temp = 0;
        int repeat = l/8;
        int rest = l%8;
        if(repeat > 0){
            for (int i=0; i<l-7; i+=8){
		temp+= a[i]*b[i];
                temp+= a[i+1]*b[i+1];
                temp+= a[i+2]*b[i+2];
                temp+= a[i+3]*b[i+3];
                temp+= a[i+4]*b[i+4];
		temp+= a[i+5]*b[i+5];
		temp+= a[i+6]*b[i+6];
		temp+= a[i+7]*b[i+7];                

            }
        }
        if(rest != 0){
            for(int i = repeat*8;i < l; ++i){
                temp+= a[i]*b[i];
            }
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
			mat[i+j*M] = 0.0;
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
	
	
void padwithzeros(const int M, const int N, const int L, const int K,double *smallmat, double *mat){
  /* description: takes a small matrix smallmat and pads it with zeros to be a larger matrix, mat
     
     inputs:
               M - row size of smallmat
	       N - col size of smallmat
	       L - row size of mat
	       K - col size of mat
	       smallmat - m x n matrix that will have zeros added
	       
       outputs:
             mat - L x K matrix with upper left block = SmallMat, the rest zeros;

    **************************************/

  for(int i =0; i < L*K;++i){
    mat[i] = 0.0;
  }

  for(int i =0 ; i < M; ++i){
    for(int j =0 ; j < N; ++j){
      mat[L*j + i] = smallmat[M*j + i];

    }
  }

  return;
}// padwithzeros

void checkldl(int n, int m,double * D,double * P,double * sigma, double* A, double * B){

  double * LDLdiag = (double *) malloc(sizeof(double)*n*m);
  double * LDLsup = (double *) malloc(sizeof(double)*n*m);
  double * BBsigdiag = (double *) malloc(sizeof(double)*n*m);
  double * BBsigsup = (double *) malloc(sizeof(double)*n*m);

  int errcount = 0;
  double TOL = 10e-6;
    for(int i = 0;i < n;++i){
      LDLdiag[i*m] = D[i*m];
      BBsigdiag[i*m] = (A[0]*A[0] - sigma[i]*sigma[i]);
      if(fabs(D[i*m] - (A[0]*A[0] - sigma[i]*sigma[i])) > TOL){
	errcount += 1;
      }
      for(int j = 1; j < m; ++j){
	LDLsup[i*m + j - 1] = D[i*m + j-1]*P[i*m + j-1];
	BBsigsup[i*m + j - 1] = (A[j-1]*B[j-1]);
	if(fabs(D[i*m + j-1]*P[i*m + j-1] - (A[j-1]*B[j-1])) > TOL){
	  errcount +=1;
	}
	LDLdiag[i*m + j] = (D[i*m + j - 1]*P[i*m + j-1]*P[i*m + j-1] + D[i*m + j]);
	BBsigdiag[i*m + j] = (A[j]*A[j] + B[j-1]*B[j-1]- sigma[i]*sigma[i]);
	if(fabs((D[i*m + j - 1]*P[i*m + j-1]*P[i*m + j-1] + D[i*m + j]) - (A[j]*A[j] + B[j-1]*B[j-1]- sigma[i]*sigma[i])) > TOL){
	  errcount += 1;
	}
      }
    }
    /*
    print_matrix(sigma,1,n,"sigma = ");
    print_matrix(LDLdiag,m,n,"LDLdiag = ");
    print_matrix(BBsigdiag,m,n,"BBsigdiag = ");
    print_matrix(LDLsup,m,n,"LDLsup = ");
    print_matrix(BBsigsup,m,n,"BBsigsup = ");*/
    printf("Number of errors in ldl is %d \n",errcount);


}//checkldl

/* Based on code found at at http://www.cprogramming.com/tutorial/computersciencetheory/merge.html */

/* left is the index of the leftmost element of the subarray; right is one
 * past the index of the rightmost element */
void merge_helper(double *input, int left, int right, double *scratch)
{
    /* base case: one element */
    if(right == left + 1)
    {
        return;
    }
    else
    {
        int i = 0;
        int length = right - left;
        int midpoint_distance = length/2;
        /* l and r are to the positions in the left and right subarrays */
        int l = left, r = left + midpoint_distance;

        /* sort each subarray */
        merge_helper(input, left, left + midpoint_distance, scratch);
        merge_helper(input, left + midpoint_distance, right, scratch);

        /* merge the arrays together using scratch for temporary storage */ 
        for(i = 0; i < length; i++)
        {
            /* Check to see if any elements remain in the left array; if so,
             * we check if there are any elements left in the right array; if
             * so, we compare them.  Otherwise, we know that the merge must
             * use take the element from the left array */
            if(l < left + midpoint_distance && 
                    (r == right || ( input[l] > input[r] ? input[l] : input[r] ) == input[l]))
            {
                scratch[i] = input[l];
                l++;
            }
            else
            {
                scratch[i] = input[r];
                r++;
            }
        }
        /* Copy the sorted subarray back to the input */
        for(i = left; i < right; i++)
        {
            input[i] = scratch[i - left];
        }
    }
} //merge_helper

/* mergesort returns true on success.  Note that in C++, you could also
 * replace malloc with new and if memory allocation fails, an exception will
 * be thrown.  If we don't allocate a scratch array here, what happens? 
 *
 * Elements are sorted in reverse order -- greatest to least */

int mergesort(double *input, int size)
{
    double *scratch = (double *)malloc(size * sizeof(double));
    if(scratch != NULL)
    {
        merge_helper(input, 0, size, scratch);
        free(scratch);
        return 1;
    }
    else
    {
        return 0;
    }
}// mergesort

int construct_right_SV(int n,int mn,int lb,double* XT, double* VT,double* Vout){
/*construct the appropriate V^t from
 * the V given by form_V and the X given by CalcRightSingularVectors
 *  X is mn x lb
 *  V is n x n
 *
 * ******************************/
    for(int i = 0; i < mn; ++i){
        for(int j = 0; j < n; ++j){
            Vout[i + mn*j] = 0.0;
            for(int k = 0; k < lb;++k){
                Vout[i + mn*j] += XT[i + mn*k]*VT[k + j*n];
            }
        }
    }
    return(1);
}

int construct_left_SV(int m,int mn,double* Y, double* U,double* Uout){
/*construct the appropriate U from
 * the U given by form_u and the Y given by RighttoLeftSingularVectors
 *  Y is mn x mn
 *  U is m x m
 *
 *  get out a Uout that is m x mn
 *
 * ******************************/

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < mn; ++j){
            Uout[i + m*j] = 0.0;
            for(int k = 0; k < mn;++k){
                Uout[i + m*j] += U[i + k*m]*Y[k + mn*j];
            }
        }
    }
    return(1);
}

    
