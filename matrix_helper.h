#ifndef MATRIX_HELPER
#define MATRIX_HELPER

//putting the dsvd prototype here

int dsvd(float **a, int m, int n, double *w, float **v);
void print_matrix(const double * A, long m, long n, char* message);		
double l2_normv(int l, const double*restrict v);
double l2_norm_mat(int m, int n,const double*restrict M);
void scale_vector(int l, double*restrict v, double scale);
double dot_prod(int l, const double* a, const double* b);

double l2_norm_mat_row(int m, int n, int l, const double*restrict row);
void scale_mat_row(int m, int n, int l, double*restrict row, double scale);
double dot_prod_mat_rows(int m, int n, int l, const double* a, const double* b);
double dot_prod_mat_row_with_vec(int m, int n, int l, const double* row, const double* vec);

void set_vec_to_zero(int l, double*restrict v);
void dgemm_simple( const int M, const int N, const int L, const double *A, const double *B, double *C);

void form_bidiag( const int M, const int N, const double *alpha, const double *beta, double * mat);
void transpose( const int M, const int N, const double * A, double * AT);
void padwithzeros(const int M, const int N, const int L, const int K,double *smallmat, double *mat);
void checkldl(int n, int m,double * D,double * P,double * sigma, double* A, double * B);
int mergesort(double *input, int size);
void merge_helper(double *input, int left, int right, double *scratch);
int construct_right_SV(int n,int mn,int lb,double* XT, double* VT,double* Vout);
int construct_left_SV(int m,int mn,double* Y, double* U,double* Uout);
#endif
