#include <stdlib.h>
#include <math.h>

/*
* Author             : Michael Lewis
* Last Date Modified : 12 / 16 / 2012
* Email Address      : mjlewis@cims.nyu.edu
* Filename           : Calculations-Parallel.h
*/

void SolveSecularEquation_Parallel( int K , int N , double d[] , double z[] , double sigma[]);
void GetTopAndBottomRows_Parallel( int K , int N , double d[] , double sigma[] , double z[] , double first_row[] , double last_row[] , int parent_needs );
void SolveSmallMatrices_Parallel( int N , double b1[] , double b2[] , double sigma[] , double first_line[] , double last_line[] , double * phi , double * psi , int parent_needs );
void DivideAndConquer_Parallel( int N , double b1[] , double b2[] , double sigma[] , double scratch[], double first_row[] , double last_row[] , double * phi , double * psi , int parent_needs );
void GetSingularValues_Parallel( int N , double b1[] , double b2[] , double sigma[] );
