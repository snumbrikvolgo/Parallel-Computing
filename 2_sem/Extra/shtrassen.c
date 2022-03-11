#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

#define PRINT_ERROR(str) do {perror(str); return EXIT_FAILURE;} while(0);

int matrix64_mult(double* A, double* B, double* C);
int matrix_mult(double* A, double* B, double* C, size_t deg);
int matrix_mult_omp(double* A, double* B, double* C, size_t deg);
static void matrix_sum(double* A, double* B, double* C, size_t size);
static void matrix_sub(double* A, double* B, double* C, size_t size);

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr,
                "Bad number of args = %d. Try ./a.out matrix_size 2^n\n", argc);
        return EXIT_FAILURE;
    }

    errno = 0;
    long int deg = strtol(argv[1], NULL, 10);
    if (errno < 0)
        PRINT_ERROR("Bad matrix size\n");

    size_t matrix_size = ((size_t)1) << deg;

    errno = 0;
    double* A = (double*) malloc(sizeof(*A) * matrix_size * matrix_size);
    if (A == NULL)
        PRINT_ERROR("[main] Alloc A returned error\n");

    for (size_t row = 0; row < matrix_size; row++)
        for(size_t col = 0; col < matrix_size; col++)
            A[row * matrix_size + col] = (double)col - 1.1 * (double)row;

    errno = 0;
    double* B = (double*) malloc(sizeof(*B) * matrix_size * matrix_size);
    if (B == NULL)
        PRINT_ERROR("[main] Alloc B returned error\n");

    for (size_t row = 0; row < matrix_size; row++)
        for(size_t col = 0; col < matrix_size; col++)
            B[row * matrix_size + col] = (double)col - 1.07 * (double)row;

    errno = 0;
    double* C = (double*) malloc(sizeof(*A) * matrix_size * matrix_size);
    if (C == NULL)
        PRINT_ERROR("[main] Alloc B returned error\n");

    double time1 = omp_get_wtime();
    int err = matrix_mult_omp(A, B, C, deg);
    if (err != 0)
        PRINT_ERROR("[main] matrix multiplication retuned error\n");
    double time2 = omp_get_wtime();
    printf("Time = %lg\n", time2 - time1);

    return 0;
}

int matrix_mult_omp(double* A, double* B, double* C, size_t deg)
{
    if (A == NULL || B == NULL || C == NULL)
        PRINT_ERROR("[matrix_mult] Bad input matrix\n");

    //printf("recursive matrix multiplication with degree %lu\n", deg);

    if (deg <= 6) // 64
        return matrix64_mult(A, B, C);

    size_t next_matrix_size = ((size_t)1) << (deg - 1);

    double* A_halfs[4];
    for (size_t i = 0; i < 4; i++)
    {
        errno = 0;
        A_halfs[i] = (double*) malloc(sizeof(double) * next_matrix_size * next_matrix_size);
        if (A_halfs[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc half of A returned error\n");
    }

    for (size_t row = 0; row < next_matrix_size; row++)
    {
        for (size_t col = 0; col < next_matrix_size; col++)
        {
            A_halfs[0][row * next_matrix_size + col] =
                     A[row * next_matrix_size + col];
            A_halfs[1][row * next_matrix_size + col] =
                     A[row * next_matrix_size + (next_matrix_size + col)];
            A_halfs[2][row * next_matrix_size + col] =
                     A[(next_matrix_size + row) * next_matrix_size + col];
            A_halfs[3][row * next_matrix_size + col] =
                     A[(next_matrix_size + row) * next_matrix_size + (next_matrix_size + col)];
        }
    }

    double* B_halfs[4];
    for (size_t i = 0; i < 4; i++)
    {
        errno = 0;
        B_halfs[i] = (double*) malloc(sizeof(double) * next_matrix_size * next_matrix_size);
        if (B_halfs[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc half of B returned error\n");
    }

    for (size_t row = 0; row < next_matrix_size; row++)
    {
        for (size_t col = 0; col < next_matrix_size; col++)
        {
            B_halfs[0][row * next_matrix_size + col] =
                     B[row * next_matrix_size + col];
            B_halfs[1][row * next_matrix_size + col] =
                     B[row * next_matrix_size + (next_matrix_size + col)];
            B_halfs[2][row * next_matrix_size + col] =
                     B[(next_matrix_size + row) * next_matrix_size + col];
            B_halfs[3][row * next_matrix_size + col] =
                     B[(next_matrix_size + row) * next_matrix_size + (next_matrix_size + col)];
        }
    }

    double* C_halfs[4];
    for (size_t i = 0; i < 4; i++)
    {
        errno = 0;
        C_halfs[i] = (double*) malloc(sizeof(double) * next_matrix_size * next_matrix_size);
        if (C_halfs[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc half of C returned error\n");
    }

    double* S[8];
    for (size_t i = 0; i < 8; i++)
    {
        errno = 0;
        S[i] = (double*) malloc(sizeof(*S[i]) * next_matrix_size * next_matrix_size);
        if (S[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc S returned error\n");
    }

    double* P[7];
    for (size_t i = 0; i < 7; i++)
    {
        errno = 0;
        P[i] = (double*) malloc(sizeof(*P[i]) * next_matrix_size * next_matrix_size);
        if (P[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc P returned error\n");
    }

    double* T[2];
    for (size_t i = 0; i < 2; i++)
    {
        errno = 0;
        T[i] = (double*) malloc(sizeof(*T[i]) * next_matrix_size * next_matrix_size);
        if (T[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc T returned error\n");
    }

    //printf("All mallocs were finished\n");

    matrix_sum(A_halfs[2], A_halfs[3], S[0], next_matrix_size); // S1
    matrix_sub(S[0], A_halfs[0], S[1], next_matrix_size); // S2
    matrix_sub(A_halfs[0], A_halfs[2], S[2], next_matrix_size); //S3
    matrix_sub(A_halfs[1], S[1], S[3], next_matrix_size); //S4
    matrix_sub(B_halfs[1], B_halfs[0], S[4], next_matrix_size); // S5
    matrix_sub(B_halfs[3], S[4], S[5], next_matrix_size); // S6
    matrix_sub(B_halfs[3], B_halfs[1], S[6], next_matrix_size); // S7
    matrix_sub(S[5], B_halfs[2], S[7], next_matrix_size); // S8

    //printf("All S were computed\n");

    #pragma omp parallel shared(A_halfs, B_halfs, P, S) num_threads(8)
    {
        #pragma omp single
        {
            int err = 0;
            size_t next_deg = deg - 1;
            #pragma omp task
            {
                matrix_mult_omp(S[1], S[5], P[0], next_deg);
            }

            #pragma omp task
            {
                matrix_mult_omp(A_halfs[0], B_halfs[0], P[1], next_deg);
            }

            #pragma omp task
            {
                matrix_mult_omp(A_halfs[1], B_halfs[2], P[2], next_deg);
            }

            #pragma omp task
            {
                matrix_mult_omp(S[2], S[6], P[3], next_deg);
            }

            #pragma omp task
            {
                matrix_mult_omp(S[0], S[4], P[4], next_deg);
            }

            #pragma omp task
            {
                matrix_mult_omp(S[3], B_halfs[3], P[5], next_deg);
            }

            #pragma omp task
            {
                matrix_mult_omp(A_halfs[3], S[7], P[6], next_deg);
            }

            #pragma omp taskwait
        }
    }

    matrix_sum(P[0], P[1], T[0], next_matrix_size);
    matrix_sum(T[0], P[3], T[1], next_matrix_size);

    matrix_sum(P[1], P[2], C_halfs[0], next_matrix_size);
    matrix_sum(T[0], P[4], C_halfs[1], next_matrix_size);
    matrix_sum(C_halfs[1], P[6], C_halfs[1], next_matrix_size);
    matrix_sub(T[1], P[6], C_halfs[2], next_matrix_size);
    matrix_sum(T[1], P[4], C_halfs[3], next_matrix_size);

    for (size_t row = 0; row < next_matrix_size; row++)
        for (size_t col = 0; col < next_matrix_size; col++)
        {
            C[row * next_matrix_size + col] =
            C_halfs[0][row * next_matrix_size + col];
            C[row * next_matrix_size + (next_matrix_size + col)] =
            C_halfs[1][row * next_matrix_size + col];
            C[(next_matrix_size + row) * next_matrix_size + col] =
            C_halfs[2][row * next_matrix_size + col];
            C[(next_matrix_size + row) * next_matrix_size + (next_matrix_size + col)] =
            C_halfs[3][row * next_matrix_size + col];
        }

    for (int i = 0; i < 4; i++)
    {
        free(A_halfs[i]);
        free(B_halfs[i]);
        free(C_halfs[i]);
    }

    for (int i = 0; i < 8; i++)
        free(S[i]);

    for (int i = 0; i < 7; i++)
        free(P[i]);

    for (int i = 0; i < 2; i++)
        free(T[i]);

    return 0;
}

int matrix_mult(double* A, double* B, double* C, size_t deg)
{
    if (A == NULL || B == NULL || C == NULL)
        PRINT_ERROR("[matrix_mult] Bad input matrix\n");

    //printf("recursive matrix multiplication with degree %lu\n", deg);

    if (deg <= 6) // 64
        return matrix64_mult(A, B, C);

    size_t next_matrix_size = ((size_t)1) << (deg - 1);

    double* A_halfs[4];
    for (size_t i = 0; i < 4; i++)
    {
        errno = 0;
        A_halfs[i] = (double*) malloc(sizeof(double) * next_matrix_size * next_matrix_size);
        if (A_halfs[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc half of A returned error\n");
    }

    for (size_t row = 0; row < next_matrix_size; row++)
    {
        for (size_t col = 0; col < next_matrix_size; col++)
        {
            A_halfs[0][row * next_matrix_size + col] =
                     A[row * next_matrix_size + col];
            A_halfs[1][row * next_matrix_size + col] =
                     A[row * next_matrix_size + (next_matrix_size + col)];
            A_halfs[2][row * next_matrix_size + col] =
                     A[(next_matrix_size + row) * next_matrix_size + col];
            A_halfs[3][row * next_matrix_size + col] =
                     A[(next_matrix_size + row) * next_matrix_size + (next_matrix_size + col)];
        }
    }

    double* B_halfs[4];
    for (size_t i = 0; i < 4; i++)
    {
        errno = 0;
        B_halfs[i] = (double*) malloc(sizeof(double) * next_matrix_size * next_matrix_size);
        if (B_halfs[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc half of B returned error\n");
    }

    for (size_t row = 0; row < next_matrix_size; row++)
    {
        for (size_t col = 0; col < next_matrix_size; col++)
        {
            B_halfs[0][row * next_matrix_size + col] =
                     B[row * next_matrix_size + col];
            B_halfs[1][row * next_matrix_size + col] =
                     B[row * next_matrix_size + (next_matrix_size + col)];
            B_halfs[2][row * next_matrix_size + col] =
                     B[(next_matrix_size + row) * next_matrix_size + col];
            B_halfs[3][row * next_matrix_size + col] =
                     B[(next_matrix_size + row) * next_matrix_size + (next_matrix_size + col)];
        }
    }

    double* C_halfs[4];
    for (size_t i = 0; i < 4; i++)
    {
        errno = 0;
        C_halfs[i] = (double*) malloc(sizeof(double) * next_matrix_size * next_matrix_size);
        if (C_halfs[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc half of C returned error\n");
    }

    double* S[8];
    for (size_t i = 0; i < 8; i++)
    {
        errno = 0;
        S[i] = (double*) malloc(sizeof(*S[i]) * next_matrix_size * next_matrix_size);
        if (S[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc S returned error\n");
    }

    double* P[7];
    for (size_t i = 0; i < 7; i++)
    {
        errno = 0;
        P[i] = (double*) malloc(sizeof(*P[i]) * next_matrix_size * next_matrix_size);
        if (P[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc P returned error\n");
    }

    double* T[2];
    for (size_t i = 0; i < 2; i++)
    {
        errno = 0;
        T[i] = (double*) malloc(sizeof(*T[i]) * next_matrix_size * next_matrix_size);
        if (T[i] == NULL)
            PRINT_ERROR("[matrix_mult] Alloc T returned error\n");
    }

    //printf("All mallocs were finished\n");

    matrix_sum(A_halfs[2], A_halfs[3], S[0], next_matrix_size); // S1
    matrix_sub(S[0], A_halfs[0], S[1], next_matrix_size); // S2
    matrix_sub(A_halfs[0], A_halfs[2], S[2], next_matrix_size); //S3
    matrix_sub(A_halfs[1], S[1], S[3], next_matrix_size); //S4
    matrix_sub(B_halfs[1], B_halfs[0], S[4], next_matrix_size); // S5
    matrix_sub(B_halfs[3], S[4], S[5], next_matrix_size); // S6
    matrix_sub(B_halfs[3], B_halfs[1], S[6], next_matrix_size); // S7
    matrix_sub(S[5], B_halfs[2], S[7], next_matrix_size); // S8

    //printf("All S were computed\n");

    int err = 0;
    size_t next_deg = deg - 1;

    err = matrix_mult(S[1], S[5], P[0], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    err = matrix_mult(A_halfs[0], B_halfs[0], P[1], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    err = matrix_mult(A_halfs[1], B_halfs[2], P[2], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    err = matrix_mult(S[2], S[6], P[3], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    err = matrix_mult(S[0], S[4], P[4], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    err = matrix_mult(S[3], B_halfs[3], P[5], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    err = matrix_mult(A_halfs[3], S[7], P[6], next_deg);
    if (err != 0)
        PRINT_ERROR("[matrix_mult] recursive multiplication returned error\n");

    matrix_sum(P[0], P[1], T[0], next_matrix_size);
    matrix_sum(T[0], P[3], T[1], next_matrix_size);

    matrix_sum(P[1], P[2], C_halfs[0], next_matrix_size);
    matrix_sum(T[0], P[4], C_halfs[1], next_matrix_size);
    matrix_sum(C_halfs[1], P[6], C_halfs[1], next_matrix_size);
    matrix_sub(T[1], P[6], C_halfs[2], next_matrix_size);
    matrix_sum(T[1], P[4], C_halfs[3], next_matrix_size);

    for (size_t row = 0; row < next_matrix_size; row++)
        for (size_t col = 0; col < next_matrix_size; col++)
        {
            C[row * next_matrix_size + col] =
            C_halfs[0][row * next_matrix_size + col];
            C[row * next_matrix_size + (next_matrix_size + col)] =
            C_halfs[1][row * next_matrix_size + col];
            C[(next_matrix_size + row) * next_matrix_size + col] =
            C_halfs[2][row * next_matrix_size + col];
            C[(next_matrix_size + row) * next_matrix_size + (next_matrix_size + col)] =
            C_halfs[3][row * next_matrix_size + col];
        }

    for (int i = 0; i < 4; i++)
    {
        free(A_halfs[i]);
        free(B_halfs[i]);
        free(C_halfs[i]);
    }

    for (int i = 0; i < 8; i++)
        free(S[i]);

    for (int i = 0; i < 7; i++)
        free(P[i]);

    for (int i = 0; i < 2; i++)
        free(T[i]);

    return 0;
}

static void matrix_sum(double* A, double* B, double* C, size_t size)
{
    for (size_t row = 0; row < size; row++)
        for (size_t col = 0; col < size; col++)
        {
            size_t pos = row * size + col;
            C[pos] = A[pos] + B[pos];
        }
}

static void matrix_sub(double* A, double* B, double* C, size_t size)
{
    for (size_t row = 0; row < size; row++)
        for (size_t col = 0; col < size; col++)
        {
            size_t pos = row * size + col;
            C[pos] = A[pos] - B[pos];
        }
}

int matrix64_mult(double* A, double* B, double* C)
{
    if (A == NULL || B == NULL || C == NULL)
        PRINT_ERROR("[matrix64_mult] Bad input matrix");

    //printf("64*64 matrix multiplication\n");

    double transp[64 * 64]; // (4 * 8)kB isn't too mush for stack
    for (int row = 0; row < 64; row++)
        for (int col = 0; col < 64; col++)
        {
            transp[row * 64 + col] = B[col * 64 + row];
            C[row * 64 + col] = 0.0;
        }

    for (int row = 0; row < 64; row++)
        #pragma omp simd
        for (int col = 0; col < 64; col++)
        {
            int pos = row * 64 + col;
            for (int k = 0; k < 64; k++)
                C[pos] += A[row * 64 + k] * transp[row * 64 + k];
        }

    return 0;
}
