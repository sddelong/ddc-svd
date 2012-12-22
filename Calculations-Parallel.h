#include <stdlib.h>
#include <math.h>

/*
* Author             : Michael Lewis
* Last Date Modified : 12 / 16 / 2012
* Email Address      : mjlewis@cims.nyu.edu
* Filename           : Calculations-Parallel.h
*/

/**********************************************************************
 * 
 * Copyright (C) 2012 Michael Lewis
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

void SolveSecularEquation_Parallel( int K , int N , double d[] , double z[] , double sigma[]);
void GetTopAndBottomRows_Parallel( int K , int N , double d[] , double sigma[] , double z[] , double first_row[] , double last_row[] , int parent_needs );
void SolveSmallMatrices_Parallel( int N , double b1[] , double b2[] , double sigma[] , double first_line[] , double last_line[] , double * phi , double * psi , int parent_needs );
void DivideAndConquer_Parallel( int N , double b1[] , double b2[] , double sigma[] , double scratch[], double first_row[] , double last_row[] , double * phi , double * psi , int parent_needs );
void GetSingularValues_Parallel( int N , double b1[] , double b2[] , double sigma[] );
