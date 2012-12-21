#include<stdlib.h>
#include <stdio.h>
#include "Calculations-Parallel.h"
#include <math.h> // has sqrt function
#include <omp.h>  // OpenMP


// Remove these after you're done debugging


/*
* Author             : Michael Lewis
* Last Date Modified : 12 / 16 / 2012
* Email Address      : mjlewis@cims.nyu.edu
* Filename           : Calculations-Parallel.cpp
*/

// POSSIBLE ISSUES WITH THIS CODE
// 1) I really hope we never get two equal singular values
// 2) I really hope z_j never equal 0
// THIS ASSUMES THE SUB MATRICES WERE K x K + 1 and N - K - 1 x N - K, instead of in the paper, they were K - 1 x K and N - K x N - K + 1
void SolveSecularEquation_Parallel( int K , int N , double d[] , double z[] , double sigma[] )
{
  // Find zeros of the secular equation derived from these values of d and z in M
  // N is the size of the vectors d and z
  // d are the values on the diagonal. Note, they are NOT assumed to be sorted yet, but they are assumed to be pseudo sorted,
  // i.e. d[0] = 0 by assumption, the next K values are sorted in increasing order, and the last N - K - 1 values are also sorted in increasing order

  // The output, an array of length N of singular values, will also be sorted in increasing order. This will be the array sigma.
  // No assumptions are made on the N values in z

  // first, sort the d as necessary
  int i, j, k;
  double temp[ N ]; // This will hold the ordered values of d_j^2
  double c[ N ];    // This will hold the "ordered" values of z_j^2, ordered alongside d
  double d2[ N ];   // This will hold the values of temp - temp[i] for the ith singular value search

  temp[0] = 0.0;        // we know this is in d
  c[0]    = z[0]*z[0];
  i = 1;
  j = K + 1;
  k = 1;

  while( (i <= K) && ( j < N ) )
  {
     if( d[i] <= d[j] )
     {
        temp[k] = d[i]*d[i];
        c[k]    = z[i]*z[i];
        i++;
     } else {
        temp[k] = d[j]*d[j];
        c[k]    = z[j]*z[j];
        j++;
     }
     k++;
  }
  if( i <= K )
  {
     while( i<= K )
     {
        temp[k] = d[i]*d[i];
        c[k]    = z[i]*z[i];
        i++;
        k++;
     }
  }
  if( j < N )
  {
     while( j < N )
     {
        temp[k] = d[j]*d[j];
        c[k]    = z[j]*z[j];
        j++;
        k++;
     }
  }


double delta, a1, a2_start, a2, a3, gamma, g, gp, bp, hold, gamma_min, g_min, gamma_max, g_max;
double prev, result, eps, iter, maxiter;


#pragma omp parallel for default(none)                             \
  shared(N, temp, sigma, c)                                     \
  private(i, j, k, d2, delta, a1, a2_start, a2, a3, gamma, g, gp, bp, hold, gamma_min, g_min, gamma_max, g_max, prev, result, eps, iter, maxiter )
  for(i = 0; i < N; ++i) // for each singular value
  {
//       double delta, a1, a2_start, a2, a3, gamma, g, gp, bp, hold, gamma_min, g_min, gamma_max, g_max;

       for(j = 0; j < N; ++j)
       {
           d2[j] = temp[j] - temp[i];
       }
       prev   = 9999.0;
       result = 0.0;
       eps    = 1e-12;
       iter      = 0;
       maxiter   = 30;

       // Now, the secular equation for sigma_i is f( t ) = 1 + sum_j=1^n c_j / (d2_j - t) = 0 for t = sigma_i^2
       // For now, let's assume i does not equal N - 1 (the last index)
       // notice we let delta = d2_{i+1}
       // we now consider gamma = 1 / t and
       // g( gamma ) = ( gamma - 1 / delta ) f( 1 / gamma )
       // and we want to find the zeros for this function
       // Specifically, we want to find the zero between 1 / delta and +infinity
       // This function has the nice property that it behaves well at gamma = 1 / delta
       // Specifically, we know g( 1 / delta ) = C_{i+1} / delta^2
       // Also, g looks "kind of" like a quadratic function, as g( gamma ) = a gamma^2 + b( gamma ) gamma + c
       // where a = - C_i , c = - 1 / delta, and
       // b( gamma ) = 1 + [ C_i + C_{i+1} ] / delta + sum_{j ne (i,i+1)} [ c_j / d2_j ] * [ ( gamma - 1 / delta ) / ( gamma - 1 / d2_j ) ]
       // We see g( 1 / delta ) = - C_i / delta^2 + [ 1 + [ C_i + C_{i+1} ] / delta ] / delta - 1 / delta = C_{i+1} / delta^2
       // Furthermore, if we solve this quadratic formula for this particular b( gamma ) = 1 + [ C_i + C_{i+1} ] / delta
       // we obtain a root gamma_0. This tends to get in the ballpark, so we do a Newton's method from here.

       if( c[i] < 1e-20 ) { // this c[i] is getting very small
          sigma[i] = sqrt( temp[i] ) + 1e-14;
       } else if( i < N - 1 ) // not working on the last index
       {
          delta = d2[i+1]; // This clearly doesn't work for i = N - 1

          // Here, we find the zero for g( gamma ) = a1 * gamma^2 + a2(gamma) * gamma + a3
          a1       = - c[i];
          a2_start = 1.0 + ( c[i] + c[i+1] ) / delta;
          a2       = a2_start;
          a3       =  - 1.0 / delta;

          // Initial guess to put it near the zero
          gamma = ( - a2 - sqrt( a2*a2 - 4.0 * a1 * a3 ) ) / (2.0 * a1);
          g  = 0.0;
          gp = 1.0; // derivative of g
          bp = 0.0; // derivative of b( gamma )
          hold = 0.0;

          // Newton's Method
          while ( ( (gp > 0.0) || ( fabs( g ) > 1e-5 ) || ( fabs( prev - result ) / ( fabs( prev ) + fabs( result ) ) > eps ) ) && ( iter < maxiter ) )
          {
             prev = result;
             a2 = a2_start;
             bp = 0.0;
             for(k = 0; k < N; ++k) // length of c
             {
                if( (k != i) && (k != (i+1)) )
                {
                   a2 += c[k] * ( gamma + a3 ) / (d2[k] * gamma - 1.0  );
                   bp += - c[k] * ( d2[k] * a3 + 1.0 ) / ( ( d2[k] * gamma - 1.0 ) * ( d2[k] * gamma - 1.0 ) ) ;
                }
             }
             g  = a3 + a2 * gamma + a1 * gamma * gamma;
             gp = 2.0 * a1 * gamma + a2 + gamma * bp;
             hold = gamma - (g / gp);
             if( hold < 1.0 / delta )
             {
                gamma = gamma / 2.0 + (1.0 / (2.0 * delta ));
             } else if ( (g > 0.0) && (gp > 0) ) {
                if( iter < 5 ) {
                   gamma = ( (gamma > 1.0) ? gamma : 1.0 ) * 10.0; // just throw it out there
                } else {
                   gamma = ( - a2 - sqrt( a2*a2 - 4.0 * a1 * a3 ) ) / ( 2.0 * a1 );
                }
             } else if( (g > 0.0) && (gamma > 10e24) ) { // gamma is damn near inf => sigma_i near d_i, just stop the search
                prev = result;
                gp = 0.0;
                g = 0.0;
             } else {
                gamma = hold;
             }
             iter++;
             result = gamma;
          }

          // In case it doesn't converge, I use bisection method
          if( iter == maxiter )
          {
            gamma_min = 1 / delta;
            g_min     = c[i+1] / ( delta * delta );
            gamma_max = -1.0;
            g_max     = 0.0;

            gamma = gamma_min;
            while( gamma_max < 0 )
            {
	      gamma *= 10.0;
               a2 = a2_start;
               for(k = 0; k < N; ++k) // length of c
               {
                  if( (k != i) && (k != (i+1)) )
                  {
                    a2 += c[k] * ( gamma + a3 ) / (d2[k] * gamma - 1.0 );
                  }
               }
               g  = a3 + a2 * gamma + a1 * gamma * gamma;
               if( g < 0.0 )
               {
                  gamma_max = gamma;
                  g_max = g;
               } else {
                  gamma_min = gamma;
                  g_min = g;
               }
            }


            gamma = (gamma_min + gamma_max) / 2.0;

            while ( fabs( gamma_min - gamma_max )/(fabs(gamma_min) + fabs(gamma_max)) > eps)
            {
               a2 = a2_start;
               for(k = 0; k < N; ++k) // length of c
               {
                  if( (k != i) && (k != (i+1)) )
                  {
                     a2 += c[k] * ( gamma + a3 ) / (d2[k] * gamma - 1.0 );
                  }
               }
               g  = a3 + a2 * gamma + a1 * gamma * gamma;
               if( g > 0.0 )
               {
                  gamma_min = gamma;
                  g_min = g;
               } else {
                  gamma_max = gamma;
                  g_max = g;
               }
               gamma = (gamma_min + gamma_max) / 2.0;
            }
          }
          sigma[i] = sqrt( temp[i] + 1.0 / gamma );
       } else {
          // We are working on the last index
          // i = N - 1
          // Here, we find the zero for g( gamma ) = 1 - c_n * gamma + sum_{k = 1}^(n-1) c_k * gamma / (d2_k * gamma - 1 )
          gamma = (c[i] > 1e-14 ? 1.0 / c[i] : 1);
          g  = 0.0;
          gp = 1.0;
          hold = 0.0;

          // Newton's method
          while ( ((gp > 0) || (fabs( g ) > 1e-5) || (fabs( prev - result ) / ( fabs( prev ) + fabs( result ) ) > eps)) && (iter < maxiter))
          {
             prev = result;
             g  = 1 - c[i] * gamma;
             gp =    -c[i];
             for(k = 0; k < N - 1; ++k)
             {
                g  += c[k] * gamma / (d2[k] * gamma - 1.0);
                gp -= c[k] / ( (d2[k] * gamma - 1.0) * (d2[k] * gamma - 1.0) );
             }
             hold = gamma - g / gp;
             if( hold < 0.0 )
             {
                gamma = gamma / 2.0;
             } else if( (g > 0.0) && (gamma > 10e24) ) { // gamma is damn near inf => sigma_i near d_i, just stop the search
                prev = result;
                gp = 0.0;
                g = 0.0;
             } else {
                gamma = hold;
             }
             result = gamma;
             iter++;
          }

          // In case it doesn't converge, I use bisection method
          if( iter == maxiter )
          {
             gamma_min = 0.0;
             g_min     = 1.0;
             gamma_max = -1.0;
             g_max     = 0.0;

             gamma = 0.1;
             while( gamma_max < 0.0 )
             {
                gamma *= 10.0;
                g  = 1 - c[i] * gamma;
                for(k = 0; k < N - 1; ++k)
                {
                   g += c[k] * gamma / (d2[k] * gamma - 1);
                }

                if( g < 0.0 )
                {
                   gamma_max = gamma;
                   g_max = g;
                } else {
                   gamma_min = gamma;
                   g_min = g;
                }

             }

             gamma = (gamma_min + gamma_max) / 2.0;
 
//             while ( fabs( gamma_min - gamma_max ) > eps )
             while ( fabs( gamma_min - gamma_max )/(fabs(gamma_min) + fabs(gamma_max)) > eps)
             {
                g  = 1 - c[i] * gamma;
                for(k = 0; k < N - 1; ++k)
                {
                  g += c[k] * gamma / (d2[k] * gamma - 1);
                }

                if( g > 0.0 )
                {
                   gamma_min = gamma;
                   g_min = g;
                } else {
                   gamma_max = gamma;
                   g_max = g;
                }

                gamma = (gamma_min + gamma_max) / 2.0;
             }
          }
          sigma[i] = sqrt( temp[i] + 1.0 / gamma );
       }
  }
  /* end of parallel for construct */

  return;
}


void GetTopAndBottomRows_Parallel( int K , int N , double d[] , double sigma[] , double z[], double first_row[] , double last_row[] , int parent_needs )
{
  double temp[ N ];       // This will hold the ordered values of d_j^2
  double z_ordered[ N ];  // This will hold the ordered values of z_j
  double sig_square[ N ]; // This will hold the sigma_i^2
  double hold_first[ N ]; // This will hold the first_row data, and we'll just modify the first row data in place
  double hold_last[ N ];  // This will hold the last_row data, and we'll just modify the last row data in place
  double z_hat_sign[ N ]; // This will hold the sign (+1 or -1) for the z_hat. This will ensure the result is consistent with the SVD of the matrix M
  // Technically, the hold vectors will permuted versions of the first and last rows, as it's just easier for the calculations
  // parent_needs is a flag = 0, 1, 2, 3.
  //              : 0  = parent doesn't need the first row or the last row (only happens if parent is top level)
  //              : 1  = parent needs the first row as an output
  //              : 2  = parent needs the last row as an output
  //              : 3  = parent needs the last row AND the first row as an output
 

  int i, j, k;
  int first_row_needed = ( (parent_needs == 1 || parent_needs == 3) ? 1 : 0);
  int last_row_needed  = ( (parent_needs == 2 || parent_needs == 3) ? 1 : 0);

  // we know the first row doesn't change, and that d[0] = 0
  temp[0]       = 0.0;
  z_ordered[0]  = z[ 0 ];
  if( first_row_needed == 1 )
  {
     hold_first[0] = first_row[0];
  }
  if( last_row_needed == 1 )
  {
     hold_last[0]  = last_row[0];
  }

  i = 1;
  j = K + 1;
  k = 1;

  while( (i <= K) && ( j < N ) )
  {
     if( d[i] <= d[j] )
     {
        temp[k] = d[i]*d[i];
        z_ordered[k] = z[i];
        if( first_row_needed == 1 )
        {
           hold_first[k] = first_row[i];
        }
        if( last_row_needed == 1 )
        {
           hold_last[k] = last_row[i];
        }
        i++;
     } else {
        temp[k] = d[j]*d[j];
        z_ordered[k] = z[j];
        if( first_row_needed == 1 )
        {
           hold_first[k] = first_row[j];
        }
        if( last_row_needed == 1 )
        {
           hold_last[k] = last_row[j];
        }
        j++;
     }
     k++;
  }
  if( i <= K )
  {
     while( i<= K )
     {
        temp[k] = d[i]*d[i];
        z_ordered[k] = z[i];
        if( first_row_needed == 1 )
        {
           hold_first[k] = first_row[i];
        }
        if( last_row_needed == 1 )
        {
           hold_last[k] = last_row[i];
        }
        i++;
        k++;
     }
  }
  if( j < N )
  {
     while( j < N )
     {
        temp[k] = d[j]*d[j];
        z_ordered[k] = z[j];
        if( first_row_needed == 1 )
        {
           hold_first[k] = first_row[j];
        }
        if( last_row_needed == 1 )
        {
           hold_last[k] = last_row[j];
        }
        j++;
        k++;
     }
  }

  // Calculate our sigma_i^2
  for(i = 0; i < N; ++i)
  {
     sig_square[i] = sigma[ i ] * sigma[ i ];
  }

  // First, calculate our z_hat, well, technically log( z_hat ), I'm thinking it's more stable
  double z_hat[ N ];
#pragma omp parallel for  default(none)             \
  shared(N, temp, sig_square, z_hat)  \
  private(i, j)
  for(i = 0; i < N; ++i)
  {
     if( (fabs( temp[ i ] - sig_square[ i ] ) < 1e-14) || (i > 0 && (fabs( temp[ i ] - sig_square[ i-1 ] ) < 1e-14)) ) {
        z_hat[i] = 0.0; // technically -inf for log z_hat, but I will use this later to denote the eigenvectors
     } else {
        z_hat[i] = log( sig_square[ N - 1 ] - temp[ i ] );
        for(j = 0; j < i; ++j)
        {
           z_hat[i] += log( temp[ i ] - sig_square[ j ] ) - log( temp[ i ] - temp[ j ] );
        }
        for(j = i; j < N - 1; ++j)
        {
           z_hat[i] += log( sig_square[ j ] - temp[ i ] ) - log( temp[ j + 1 ] - temp[ i ] );
        }
        z_hat[i] /= 2; // Don't need the square root because z_hat is now log( z_hat )
     }
  }
  /* end of parallel for construct */

  // Need to calculate the signs for the z_hat now
  double norm_v[ N ];   // used for normalizing the vector v
  double norm_u[ N ];   // used for normalizing the vector u
  double term;          // for holding the term
  // First, calculate the norms of the singular vectors
#pragma omp parallel for default(none)                  \
  shared(N, z_hat, norm_v, norm_u, temp, sig_square)    \
  private(i, j, term)
  for(i = 0; i < N; ++i)
  {
     if( z_hat[i] == 0.0 ) {
        // This singular value is VERY close to d_i or d_i+1
        norm_v[i] = 1.0;
        norm_u[i] = 1.0;
     } else {
        norm_v[i] = 0.0;
        norm_u[i] = 1.0;
        for(j = 0; j < N; ++j)
        {
           term = exp( z_hat[ j ] -  log( fabs( temp[ j ] - sig_square[ i ] ) ) );
           term = term * term;
           norm_v[i] += term;
           norm_u[i] += term * temp[ j ];
        }
        norm_v[i] = sqrt( norm_v[i] );
        norm_u[i] = sqrt( norm_u[i] );
     }
  }
  /* end of parallel for construct */

  // Now, calculate the signs of the z_hat
#pragma omp parallel for default(none)       \
  shared(N, temp, sig_square, z_hat, z_hat_sign, sigma, norm_v, norm_u, z_ordered)  \
  private(i, term, k)
  for(i = 0; i < N; ++i)
  {
     if( z_hat[ i ] == 0.0 ) {
        if( fabs( temp[ i ] - sig_square[ i ] ) < 1e-14 ) // case where sigma_i very close to d_i
        {
           z_hat_sign[ i ] = ( z_ordered[ i ] > 0 ? 1 : -1 );
        } else { // case where sigma_i very close to d_i+1
           z_hat_sign[ i ] = ( z_ordered[ i ] > 0 ? -1 : 1 );
        }
     } else {
        term = 0.0;
        for(k = 0; k < N; ++k)
        {
           term += sigma[ k ] / norm_v[k] / norm_u[k] / ( temp[ i ] - sig_square[ k ] );
        }
        // we _should_ have z[ i ] = - z_hat[ i ] * term => sign( z_i ) = -1 * sign( z_hat_i ) * sign( term ) => sign( z_hat_i ) = -1 * sign( z_i ) * sign( term )
        z_hat_sign[ i ] = -1 * ( z_ordered[ i ] > 0 ? 1 : -1 ) * ( term > 0 ? 1 : -1 );
     }
  }
  /* end of parallel for construct */

  // Now calculate our eigenvectors
  double v[ N ]; // our eigenvector
#pragma omp parallel for  default(none)             \
  shared(N, first_row_needed, last_row_needed, first_row, last_row, z_hat, temp, sig_square, z_hat_sign, norm_v, hold_first, hold_last)  \
  private(i, j, v)
  for(i = 0; i < N; ++i)
  {
     if( first_row_needed == 1 )
     {
        first_row[i] = 0.0;
     }
     if( last_row_needed == 1 )
     {
        last_row[i]  = 0.0;
     }

     // Calculating the ith eigenvector, and then doing the dot product necessary on the first and last rows
     for(j = 0; j < N; ++j)
     {
        if( z_hat[ i ] == 0.0 ) { // d_i VERY close to sigma_i
              v[ j ] = (j == i ? 1.0 : 0.0 );
        } else {
           if( z_hat[ j ] == 0.0 ) {
               v[ j ] = 0.0;
           } else {
              v[ j ] = exp( z_hat[ j ] -  log( ( j > i ? temp[ j ] - sig_square[ i ] : sig_square[ i ] - temp[ j ] ) ) );
              // if j > i, then d_j^2 > sigma_i^2, else sigma_i^2 > d_j^2. Just keeping the argument of the log positive
              v[ j ] *= ( j > i ? 1 : -1 );
           }
        }
        v[ j ] *= z_hat_sign[ j ];
     }

     for(j = 0; j < N; ++j)
     {
        v[ j ] /= norm_v[i];
        if( first_row_needed == 1 )
        {
           first_row[i] += v[ j ] * hold_first[j];
        }
        if( last_row_needed == 1 )
        {
           last_row[i]  += v[ j ] * hold_last[j];
        }
     }
  }
  /* end of parallel for construct */

  return;
}

void SolveSmallMatrices_Parallel(int N , double b1[] , double b2[] , double sigma[] , double first_line[] , double last_line[] , double * phi , double * psi , int parent_needs ) {
// I ASSUME THIS MATRIX IS REAL
// I WILL ALSO ASSUME THAT NONE OF B1 or B2 EQUAL ZERO
// N , b1, and be are inputs
// parent_needs is a flag = 0, 1, 2, 3.
//              : 0  = parent doesn't need the first row or the last row (only happens if parent is top level)
//              : 1  = parent needs the first row as an output
//              : 2  = parent needs the last row as an output
//              : 3  = parent needs the last row AND the first row as an output
// All the rest are outputs

   if( N == 1 )
   {
       // Dealing with a 1 x 2 matrix
       // In this case, B = [ b1 b2 ]
       // In this case, sigma = norm( B )
       // V = [ b1/sigma b2/sigma ; b2/sigma -b1/sigma ]
       sigma[0]      = sqrt( b1[0]*b1[0] + b2[0] * b2[0] );
       if( parent_needs == 1 || parent_needs == 3 ) // if parent needs the first row
       {
          first_line[0] = b1[0] / sigma[0];
       }
       if( parent_needs == 2 || parent_needs == 3 ) // if parent needs the last row
       {
          last_line[0]  = b2[0] / sigma[0];
       }
       phi[0]           =  b2[0] / sigma[0];
       psi[0]           = -b1[0] / sigma[0];
   } else if( N == 2) {
       // Dealing with a 2 x 3 matrix
       // In this case, B = [ b1 b2 0 ; 0 b3 b4 ]
       // In this case, sigma12 =  sqrt( sqrt( n1 \pm sqrt( n1*n1 - n2*n2 )))
       // where n1 = z_0*z_0 + e , n2 = z_0*z_0 - e
       // e = z_1^2 + z_2^2 + z_3^2 , a = b1^2 + b2^2 , d = b3^2 + b4^2
       // z_0 = (a + d ) / 2 , z_1 = 0 , z_2 = (a - d) / 2 , z_3 = b2 * b3
       // NOTE: we get z_1 = 0, because I'm assuming this matrix is real and NOT complex
       double a = b1[0]*b1[0] + b2[0]*b2[0]; // b1^2 + b2^2
       double d = b1[1]*b1[1] + b2[1]*b2[1]; // b3^2 + b4^2
       double z_0 = (a + d)/2.0;
       double z_1 = 0.0;
       double z_2 = (a - d)/2.0;
       double z_3 = b2[0] * b1[1]; // b2 * b3
       double e = z_1*z_1 + z_2*z_2 + z_3*z_3;
       double n1 = z_0*z_0 + e;
       double n2 = z_0*z_0 - e;

       // NOTE: We want sigma in increasing order
       sigma[0] = sqrt( sqrt( n1 - sqrt( n1*n1 - n2*n2 )));
       sigma[1] = sqrt( sqrt( n1 + sqrt( n1*n1 - n2*n2 )));

       // ASSUMING NONE OF B1 or B2 EQUAL ZERO
       // the vector for the null space v = [ -(b2 / b1) 1 -(b3 / b4) ], properly normalized
       // This goes into the values for phi and psi

       /* b4 = 0 => v = ( 0 0 1 )^T , b1 = 0 => v = ( 1 0 0 )^T , if both = 0, we're screwed */
       phi[0] = ( b2[1] == 0.0 ? 0.0 : ( b1[0] == 0.0 ? 1.0 : -b2[0] / b1[0] ) );
       psi[0] = ( b2[1] == 0.0 ? 1.0 : ( b1[0] == 0.0 ? 0.0 : -b1[1] / b2[1] ) );
       double norm = ( ((b2[1] == 0.0) || (b1[0] == 0.0)) ? 1.0 : sqrt(1 + phi[0]*phi[0] + psi[0]*psi[0]) );
       phi[0] /= norm;
       psi[0] /= norm;
/*
       if( b2[1] == 0.0 ) {
           phi[0] = 0.0;
           psi[0] = 1.0;
       } else if ( b1[0] == 0.0 ) {
           phi[0] = 1.0;
           psi[0] = 0.0;
       } else {
          phi[0] = -b2[0] / b1[0];
          psi[0] = -b1[1] / b2[1];
          norm = sqrt(1 + phi[0]*phi[0] + psi[0]*psi[0]);
          phi[0] /= norm;
          psi[0] /= norm;
       }
*/

       double v1, v3;
       // Similar analysis can be done by finding the null space for A^T A - sigma^2 I
       v1   = -b2[0] * b1[0] / ( b1[0]*b1[0] - sigma[0]*sigma[0] ); // -(b2 * b1) / (b1^2 - sig_0^2)
       v3   = -b2[1] * b1[1] / ( b2[1]*b2[1] - sigma[0]*sigma[0] ); // -(b4 * b3) / (b4^2 - sig_0^2)
       norm = sqrt(1 + v1 * v1 + v3 * v3);
       v1  /= norm;
       v3  /= norm;
       if( parent_needs == 1 || parent_needs == 3 ) // if parent needs first row
       {
          first_line[0] = v1;
       }
       if( parent_needs == 2 || parent_needs == 3 ) // if parent needs last row
       {
          last_line[0]  = v3;
       }

       v1   = -b2[0] * b1[0] / ( b1[0]*b1[0] - sigma[1]*sigma[1] ); // -(b2 * b1) / (b1^2 - sig_1^2)
       v3   = -b2[1] * b1[1] / ( b2[1]*b2[1] - sigma[1]*sigma[1] ); // -(b4 * b3) / (b4^2 - sig_1^2)
       norm = sqrt(1 + v1 * v1 + v3 * v3);
       v1  /= norm;
       v3  /= norm;
       if( parent_needs == 1 || parent_needs == 3 ) // if parent needs first row
       {
          first_line[1] = v1;
       }
       if( parent_needs == 2 || parent_needs == 3 ) // if parent needs last row
       {
          last_line[1]  = v3;
       }

   } else {
       // ERROR : Too large to be in here
       perror("Error: Trying to use SolveSmallMatrices on a matrix of size > 2");
       abort();
   }

   return;
}

// THIS ASSUMES THE SUB MATRICES WERE K x K + 1 and N - K - 1 x N - K, instead of in the paper, they were K - 1 x K and N - K x N - K + 1
void DivideAndConquer_Parallel( int N , double b1[] , double b2[] , double sigma[] , double scratch[] , double first_row[] , double last_row[] , double * phi , double * psi , int parent_needs )
{
    // The N, b1, b2 are inputs.
    // The overall matrix is N x N + 1, and B1 is the main diagonal, and B2 is the super diagonal, each of length N
    // The outputs are sigma , scratch (a.k.a. d) , first_row, last_row , phi, and psi
    // Len(Sigma) = Len(scratch) = Len(first_row) = Len(last_row) = N
    // phi and psi are scalar
    // parent_needs is a flag = 0, 1, 2, 3.
    //              : 0  = parent doesn't need the first row or the last row (only happens if parent is top level)
    //              : 1  = parent needs the first row as an output
    //              : 2  = parent needs the last row as an output
    //              : 3  = parent needs the last row AND the first row as an output
    // NOTE: Current level always needs L1 and F2 to make the Z values to get the sigma values
    //       Current level also needs F1 if parent needs front row
    //       Current level also needs L2 if parent needs last row
    //       STILL NEED TO FULLY IMPLEMENT!!!!!!!!!!!!!!!!!

    if( N <= 2 )
    {
        // This is a small enough matrix
        SolveSmallMatrices_Parallel( N , b1 , b2 , sigma , first_row , last_row , phi , psi , parent_needs );
    } else {
        // This is not a small enough matrix, so solve smaller matrices
        // For this algorithm, take elements 0 through K - 1 in b1, b2; that's the first matrix, thus its size is K x K + 1
        // Take elements K + 1 to N - 1 in b1, b2; that's the second matrix, thus its size is N - K - 1 x N - K

        int K = N / 2;
        double psi_prev[ 2 ];
        double phi_prev[ 2 ];
        double * b1_child1      , * b1_child2,
               * b2_child1      , * b2_child2,
               * scratch_child1 , * scratch_child2,
               * sigma_child1   , * sigma_child2;   // pointers to the appropriate locations in b1, b2, scratch (a.k.a. d), and sigma
        double * first_row_1    , * last_row_2;     // These are useless, unless the parents needs the first / last row
        double last_row_1[ K ];                     // These area always useful for creating the z to get the sigma values
        double first_row_2[ N - K - 1];             // These area always useful for creating the z to get the sigma values

        b1_child1 = &b1[0];
        b2_child1 = &b2[0];
        b1_child2 = &b1[K + 1];
        b2_child2 = &b2[K + 1];
        scratch[0]      = 0;
        sigma_child1    = &sigma[1];
        sigma_child2    = &sigma[K + 1];
        scratch_child1  = &scratch[1];
        scratch_child2  = &scratch[K + 1];
        if( parent_needs == 1 || parent_needs == 3 ) // parent needs first row
        {
           first_row_1  = &first_row[1];
        }
        if( parent_needs == 2 || parent_needs == 3 ) // parent needs last row
        {
           last_row_2   = &last_row[K + 1];
        }

        // Solve for submatrix 1
        // NOTE: our child's sigma's are our scratch, and vice versa. This is done for space saving and memory efficiency
        DivideAndConquer_Parallel( K         , b1_child1 , b2_child1 , scratch_child1 , sigma_child1 , first_row_1, last_row_1 , phi_prev     , psi_prev     , ( (parent_needs == 1 || parent_needs == 3) ? 3 : 2) );

        // Solve for submatrix 2
        // NOTE: our child's sigma's are our scratch, and vice versa. This is done for space saving and memory efficiency
        DivideAndConquer_Parallel( N - K - 1 , b1_child2 , b2_child2 , scratch_child2 , sigma_child2 , first_row_2, last_row_2 , phi_prev + 1 , psi_prev + 1 , ( (parent_needs == 2 || parent_needs == 3) ? 3 : 1) );

        // Now start patching things together at this level
        double r_0 = sqrt( ( b1[ K ] * psi_prev[0] ) * ( b1[ K ] * psi_prev[0] ) + ( b2[ K ] * phi_prev[1] ) * ( b2[ K ] * phi_prev[1] ) );
        double c_0 = b1[ K ] * psi_prev[0] / r_0;
        double s_0 = b2[ K ] * phi_prev[1] / r_0;

        double z[ N ];  // Holds the top row of the matrix M
        z[0] = r_0;
        for(size_t l = 1; l <= K; ++l)
        {
             z[l] = b1[ K ] * last_row_1[l - 1];
        }
        for(size_t l = K + 1; l < N; ++l)
        {
             z[l] = b2[ K ] * first_row_2[l - K - 1];
        }

        SolveSecularEquation_Parallel( K , N , scratch , z , sigma ); // Find zeros of the secular equation derived from these values of d and z in M

        // Now we want to calculate the top and bottom rows for the next level of the algorithm
        if( parent_needs == 1 || parent_needs == 3 ) // parent needs first row
        {
           first_row[0] = c_0 * phi_prev[0];
           for(size_t l = K + 1; l < N; ++l)
           {
              first_row[l] = 0.0;
           }
        }
        if( parent_needs == 2 || parent_needs == 3 ) // parent needs last row
        {
           last_row[0]  = s_0 * psi_prev[1];
           for(size_t l = 1; l <= K; ++l)
           {
              last_row[l]  = 0.0;
           }
        }

        if( parent_needs != 0 ) {
           GetTopAndBottomRows_Parallel( K , N , scratch , sigma , z , first_row , last_row , parent_needs );   // Get rows for the next level to do their calculations
           // Finally, calculate those other elements that will be passed to the next level
           phi[0] = -s_0 * phi_prev[0];
           psi[0] =  c_0 * psi_prev[1];
        }

        /* normalize the output vectors */
        /* Do I want to do an error check to make sure the norm is close to 1? */

        if( parent_needs == 1 || parent_needs == 3 ) // parent needs first row )
        {
           double norm = 0.0;
           for(size_t l = 0; l < N; ++l)
           {
              norm += first_row[l] * first_row[l];
           }
           norm += phi[0] * phi[0];
           norm = sqrt(norm);
           for(size_t l = 0; l < N; ++l)
           {
              first_row[l] /= norm;
           }
           phi[0] /= norm; // NOTE: Phi will NOT be normalized UNLESS the first row is needed. Hopefully numerical error doesn't build up too much
//           printf("norm = %2.15f\n",norm); // This values should be very close to 1
        }

        if( parent_needs == 2 || parent_needs == 3 ) // parent needs last row )
        {
           double norm = 0.0;
           for(size_t l = 0; l < N; ++l)
           {
              norm += last_row[l] * last_row[l];
           }
           norm += psi[0] * psi[0];
           norm = sqrt(norm);
           for(size_t l = 0; l < N; ++l)
           {
              last_row[l] /= norm;
           }
           psi[0] /= norm; // NOTE: Psi will NOT be normalized UNLESS the last row is needed. Hopefully numerical error doesn't build up too much
//           printf("norm = %2.15f\n",norm); // This values should be very close to 1

        }

        // free up pointers now



        // Should these be freed up? They are connected to data that will be used at higher levels
/*      free(b1_child1);
        free(b1_child2);
        free(b2_child1);
        free(b2_child2);
        free(scratch_child1);
        free(scratch_child2);
        free(sigma_child1);
        free(sigma_child2);
        free(first_row_1);
*/
    }

    return;
}

void GetSingularValues_Parallel( int N , double b1[] , double b2[] , double sigma[] )
{
    // GetSingularValues is just a staging function
    // N, b1, and b2 are inputs, sigma is an output
    // b1, b2 are assumed to be of length N
    // sigma is of length N

    // The underlying variables needed for good memory allocation will be done here

    double *scratch = (double *) malloc(sizeof(double) *N);
    if(!scratch) { fprintf(stderr,"in main: failed to allocate scratch\n"); abort();}
//    double *first_row = (double *) malloc(sizeof(double) *N);
//    if(!first_row) { fprintf(stderr,"in main: failed to allocate first_row\n"); abort();}
//    double *last_row = (double *) malloc(sizeof(double) *N);
//    if(!last_row) { fprintf(stderr,"in main: failed to allocate last_row\n"); abort();}
    double *first_row, *last_row; // these don't need space, because we don't need these values

    double phi, psi;
    int lines_needed = 0;

    DivideAndConquer_Parallel( N , b1 , b2 , sigma, scratch, first_row, last_row , &phi , &psi , lines_needed );

    // Now free up the memory
    free(scratch);
//    free(first_row);
//    free(last_row);

    return;
}
