#include <iostream>
#include <mpi.h>

long long fact(int x)
{
    long long p = 1;
    for (int i = x; i > 0; i--)
    {
        p *= i;
    }
    return p;
}
using namespace std;
int main(int argc, char*argv[]) {

     int rank, size, n, i;
     double e = 2;

     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Status status;

     if (rank == 0)
     {
         double sum = 0;
         for (int i = 2; i < 7; i += 1)
         {
             sum += static_cast<double> (1.0/fact(i));
         }
         e += sum;
         MPI_Recv(&sum, 1, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD, &status);
         e += sum;
         MPI_Recv(&sum, 1, MPI_DOUBLE, 2, 5, MPI_COMM_WORLD, &status);
         e += sum;
         MPI_Recv(&sum, 1, MPI_DOUBLE, 3, 5, MPI_COMM_WORLD, &status);
         e += sum;
     }
     else if (rank == 1)
     {
         double sum = 0;
         for (int i = 7; i < 12; i+= 1)
         {
             sum += static_cast<double> (1.0/fact(i));
         }
         MPI_Send(&sum, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
     }
     else if (rank == 2)
     {
         double sum = 0;
         for (int i = 12; i < 17; i+= 1)
         {
             sum += static_cast<double> (1.0/fact(i));
         }
         MPI_Send(&sum, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
     }
     else {
         double sum = 0;
         for (int i = 12; i < 17; i+= 1)
         {
             sum += static_cast<double> (1.0/fact(i));
         }
         MPI_Send(&sum, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
     }

    MPI_Finalize();
    if (rank == 0)
    {
    std::cout << e << "\n";
    }

     return 0;
 }
