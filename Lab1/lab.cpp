#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

#define K 40 //nodes in
#define M 20

#define A 0.5

#define T 1 // t in [0; T]
#define X 1 // x in [0; X]

using namespace std;

double U_EXACT_DUMB(const int &k,
                    const int &m);

double F_EXACT_DUMB(const int &k,
                    const int &m);

double U_EXPLICIT_DUMB(const int &m,
                       double *DATA_LINE,
                       double *fooo_LINE);

double U_EXACT(const double &k,
               const double &m);

double F_EXACT(const double &kek,
              const double &mem);

double U_EXPLICIT(const int &m,
                  double *DATA_LINE,
                  double *fooo_LINE);

void Print(double **DATA_EXACT,
           double **DATA);

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    if(argc != 1) {
        cout << "No argument is needed!\n";
        MPI_Finalize();
        exit(1);
    }

    MPI_Request REQUEST;

    int SIZE = 0;
    int RANK = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    double **DATA_EXACT = new double *[K + 1]; // Exact values of U(t, x) on (K, M) grid
    double **DATA = new double *[K + 1]; // U(t, x) calculated by our method on (K, M) grid

    double **fooo = new double *[K + 1]; // F(t, x) on (K, M) grid

    if(RANK == 0) {
        for(long i = 0; i < K + 1; i++) {
            DATA_EXACT[i] = new double[M + 1];
            DATA[i] = new double[M + 1];

            fooo[i] = new double[M + 1];
        }
    }


    long P = (M + 1) / SIZE; // Distribution of x-points for each process on every time-layer (all points)

    double *SUBDATA_EXACT = new double[P];
    double *SUBDATA = new double[P];

    // Initial conditions: U(0, x) = PHI(x)

    MPI_Barrier(MPI_COMM_WORLD);

    for(long p = 0; p < P; p++) {
        SUBDATA[p] = U_EXACT(0, P * RANK + p);
    }

    if(RANK == 0) {
        for(long p = (M + 1) - (M + 1) % SIZE; p <= M; p++) {
            DATA[0][p] = U_EXACT(0, p);
        }
    }

    MPI_Gather(&SUBDATA[0], P, MPI_DOUBLE, &DATA[0][0], P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    // Initial conditions: U(t, 0) = PSI(t), U(t, X) = PSI'(t), F(t, x)

    for(long k = 0; k <= K; k++) {
        MPI_Barrier(MPI_COMM_WORLD);

        for(long p = 0; p < P; p++) {
            SUBDATA_EXACT[p] = U_EXACT(k, P * RANK + p);
            SUBDATA[p] = F_EXACT(k+0.5, P * RANK + p + 0.5); // F(t, x) -- the majority
        }

        if(RANK == 0) {
            for(long p = (M + 1) - (M + 1) % SIZE; p <= M; p++) {
                DATA_EXACT[k][p] = U_EXACT(k, p);
                fooo[k][p] = F_EXACT(k + 0.5, p+0.5); // F(t, x) -- the rest
            }

            DATA[k][0] = U_EXACT(k, 0); // Initial conditions: U(t, 0) = PSI(t)
            DATA[k][M] = U_EXACT(k, M); // Initial conditions: U(t, X) = PSI'(t)
        }

        MPI_Gather(&SUBDATA_EXACT[0], P, MPI_DOUBLE, &DATA_EXACT[k][0], P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&SUBDATA[0], P, MPI_DOUBLE, &fooo[k][0], P, MPI_DOUBLE, 0, MPI_COMM_WORLD); // gathering F(t, x) in root process
    }

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    MPI_Barrier(MPI_COMM_WORLD);

    // Explicit method: calculating U(t, x)

    delete[] SUBDATA;

    P = (M - 1) / SIZE; // Distribution of x-points for each process on every time-layer (all points except for boundary points)

    SUBDATA = new double[P];

    double *DATA_LINE = new double[P + 2]; // Preceding line of DATA
    double *fooo_LINE = new double[P + 2]; // Preceding line of fooo

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    double TIME_BEGIN = 0;
    double TIME_END = 0;

    if(RANK == 0) {
        TIME_BEGIN = MPI_Wtime();
    }

    // Initialize DATA_LINE at (k = 1)

    MPI_Scatter(&DATA[0][1], P, MPI_DOUBLE, &DATA_LINE[1], P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(RANK == 0) {
        for(long i = 0; i < SIZE; i++) {
            MPI_Isend(&DATA[0][P * i], 1, MPI_DOUBLE, i, 10, MPI_COMM_WORLD, &REQUEST);
            MPI_Isend(&DATA[0][P * (i + 1) + 1], 1, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, &REQUEST);
        }
    }

    {
        MPI_Irecv(&DATA_LINE[0], 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &REQUEST);
        MPI_Irecv(&DATA_LINE[P + 1], 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, &REQUEST);
    }

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    MPI_Barrier(MPI_COMM_WORLD);

    for(int k = 1; k <= K; k++) {

        // Initialize fooo_LINE at each k

        MPI_Scatter(&fooo[k - 1][1], P, MPI_DOUBLE, &fooo_LINE[1], P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if(RANK == 0) {
            for(int i = 0; i < SIZE; i++) {
                MPI_Isend(&fooo[k - 1][P * i], 1, MPI_DOUBLE, i, 10, MPI_COMM_WORLD, &REQUEST);
                MPI_Isend(&fooo[k - 1][P * (i + 1) + 1], 1, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, &REQUEST);
            }
        }

        {
            MPI_Irecv(&fooo_LINE[0], 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &REQUEST);
            MPI_Irecv(&fooo_LINE[P + 1], 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, &REQUEST);
        }

        //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        MPI_Barrier(MPI_COMM_WORLD);

        //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        for(int p = 0; p < P; p++) {
            SUBDATA[p] = U_EXPLICIT(1 + p, &DATA_LINE[0], &fooo_LINE[0]);
        }

        //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        // Start init pre line DATA

        for(int p = 0; p < P; p++) {
            DATA_LINE[p + 1] = SUBDATA[p];
        }

        //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        if(RANK == 0) {
            for(int p = (M - 1) - (M - 1) % SIZE + 1; p < M; p++) {
                DATA[k][p] = U_EXPLICIT(p, &DATA[k - 1][0], &fooo[k - 1][0]);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Gather(&SUBDATA[0], P, MPI_DOUBLE, &DATA[k][1], P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        // Finish init pre line DATA

        MPI_Barrier(MPI_COMM_WORLD);

        //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        if(RANK == 0) {
            for(int i = 0; i < SIZE; i++) {
                MPI_Isend(&DATA[k][P * i], 1, MPI_DOUBLE, i, 10, MPI_COMM_WORLD, &REQUEST);
                MPI_Isend(&DATA[k][P * (i + 1) + 1], 1, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, &REQUEST);
            }
        }

        {
            MPI_Irecv(&DATA_LINE[0], 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &REQUEST);
            MPI_Irecv(&DATA_LINE[P + 1], 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, &REQUEST);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    // Print all results

    if(RANK == 0) {
        Print(DATA_EXACT, DATA);
    }

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    delete[] SUBDATA_EXACT;
    delete[] SUBDATA;

    delete[] DATA_LINE;
    delete[] fooo_LINE;

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    if(RANK == 0) {
        for(long i = 0; i < K + 1; i++) {
            delete[] DATA_EXACT[i];
            delete[] DATA[i];

            delete[] fooo[i];
        }
        delete[] DATA_EXACT;
        delete[] DATA;

        delete[] fooo;

        TIME_END = MPI_Wtime();

        int a = (int) (TIME_END - TIME_BEGIN);
        int b = (int) ((TIME_END - TIME_BEGIN - a) * 10000000);

        cout << a << ',' << setfill('0') << setw(7) << b << endl;

        // cout << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";
        // cout << "░░░░░░░░░░░░░░░░░░░░TIME SPENT: " << fixed << setprecision(7) << TIME_END - TIME_BEGIN << endl;
        // cout << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";
    }

    //----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    // Finalizing parallel program

    MPI_Finalize();

    return 0;
}

//----- ----- ----- ----- ----- ----- ----- ----- ----- -----

double U_EXACT(const  double &k,
               const double &m) {
    return sin(M_PI * (0.5 + k * T / ((double) K))) *
           sin(M_PI * m * X / ((double) M));
}

double F_EXACT(const double &kek,
               const double &mem) {
    double k = kek - 0.5;
    double m = mem - 0.5;

    return (U_EXACT(k + 1, m - 1) + U_EXACT(k + 1, m)-  U_EXACT(k, m - 1)-  U_EXACT(k, m)) * K / ((double) T) / 2+
           (U_EXACT(k + 1, m) - U_EXACT(k + 1 , m - 1) - U_EXACT(k , m - 1) + U_EXACT(k, m)) * A * M / ((double) X) / 2;
}

double U_EXPLICIT(const  double &mem,
                  double *DATA_LINE,
                  double *fooo_LINE) {
    double m = mem - 0.5;
    return DATA_LINE[m] +
           (fooo_LINE[mem] - (DATA_LINE[m] - DATA_LINE[m - 1]) * A * M / ((double) X)) * T / ((double) K);
}

//----- ----- ----- ----- ----- ----- ----- ----- ----- -----

// double U_EXACT_DUMB(const int &k,
//                     const int &m) {
//     return -1 * pow((k * T / ((double) K) - 2), 2) +
//            pow((m * X / ((double) M - 1)), 2);
// }
//
// double F_EXACT_DUMB(const int &k,
//                     const int &m) {
//     return (U_EXACT(k + 1, m) - U_EXACT(k, m)) * K / ((double) T) +
//            (U_EXACT(k, m) - U_EXACT(k, m - 1)) * A * M / ((double) X);
// }
//
// double U_EXPLICIT_DUMB(const int &m,
//                        double *DATA_LINE,
//                        double *fooo_LINE) {
//     return DATA_LINE[m] +
//            (fooo_LINE[m] - (DATA_LINE[m] - DATA_LINE[m - 1]) * A * M / ((double) X)) * T / ((double) K);
// }

//----- ----- ----- ----- ----- ----- ----- ----- ----- -----

void Print(double **DATA_EXACT,
           double **DATA) {
    ofstream out("Result.txt");

    long flag = out.precision();
    out << fixed << setprecision(7);

    out << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";
    out << "░░░░░░░░░░░░░░░░░░░░░░░░░DATA_EXACT░░░░░░░░░░░░░░░░░░░░░░░░░\n";
    out << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";

    for(int k = 0; k <= K; k++) {
        for(int m = 0; m <= M; m++) {
            out << k << ' ' << m << ' ' << DATA_EXACT[k][m] << endl;
        }
        out << endl;
    }

    out << "\n\n\n\n\n";

    out << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";
    out << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░DATA░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";
    out << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n";

    for(int k = 0; k <= K; k++) {
        for(int m = 0; m <= M; m++) {
            out << k << ' ' << m << ' ' << DATA[k][m] << endl;
        }
        out << endl;
    }

    out.precision(flag);
    out << resetiosflags(ios::fixed);

    out.close();
}
