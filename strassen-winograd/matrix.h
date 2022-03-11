#ifndef strassen_winograd_matrix_h
#define strassen_winograd_matrix_h

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

typedef struct {

    double* matrix;
    unsigned int rowNum;
    unsigned int colNum;
    unsigned int offset;
    unsigned int strand;
    unsigned int skip;

} matrix;

void zero_matrix(unsigned int rows, unsigned int cols, matrix* m);
void random_matrix(unsigned int rows, unsigned int cols, matrix* m);
void eye(unsigned int size, matrix* m);
void free_matrix(matrix* m);
void print_matrix(matrix* m);
void add_matrices(matrix* m1, matrix* m2, matrix* res);
void subtract_matrices(matrix* m1, matrix* m2, matrix* res);
double element(matrix* m, unsigned int i, unsigned int j);
void set_element(matrix* m, unsigned int i, unsigned int j, double element);
void add_to_element(matrix* m, unsigned int i, unsigned int j, double element);
void submatrix(matrix* m, unsigned int rowNum, unsigned int colNum, unsigned int offset, unsigned int strand, unsigned int skip, matrix* res);
void copy_matrix(matrix* src, matrix* dest);
matrix* alloc_matrix();

void strassen_mult(matrix* m1, matrix* m2, matrix* res, unsigned int maxDepth, unsigned int naiveSize);
void winograd_mult(matrix* m1, matrix* m2, matrix* res);
void naive_mult(matrix* m1, matrix* m2, matrix* res);

#endif

//#define __DEBUG__

#ifdef __DEBUG__
#define DEBUG(str) printf(ANSI_COLOR_RED "%s\n" ANSI_COLOR_RESET, str);
#else
#define DEBUG(...)
#endif
