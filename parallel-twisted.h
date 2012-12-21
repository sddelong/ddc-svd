#ifndef PARALLELTWISTED
#define PARALLELTWISTED


typedef struct {
    int index;
    double value;
} minindex;

void BidiagMatVec(int n,int m, double* A, double* B, double* x,double * y);
void NormalizeVectors(int n, int m, double* X);
minindex which_min_gamma(double * gamma, int start, int end,int level);
void SquareB(int n, int m, double * ina, double * inb, double * outa, double* outb);
void CholFactorization(int n,int m, double* sigma, double* ina, double* inb,double* outp, double* outd1,double * outq, double *outd2);
void CalcGamma(int n,int m, double* A,double* B,double* D1,double* D2,double* sigma,double* gamma);
void backsolve(int n, int m, double* P,double* Q, double* D1,double* D2, double* sigma, double * gamma, double* x);
void TwistedFactorization(int n, int m, double* P,double* Q, double* D1,double* D2,double* ina, double* inb, double* sigma, double* x);
void RighttoLeftSingularVectors(int n, int m, double* A, double * B, double * sigma, double * X, double * Y);
void CalcRightSingularVectors(int n, int m, double* A, double* B,double* sigma, double* X);

#endif
