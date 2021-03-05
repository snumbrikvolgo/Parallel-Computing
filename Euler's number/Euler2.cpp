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
     double e = 0;

     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Status status;
     int mult = 10;

     if (rank == 0)
     {
         double sum = 0;
         for (int i = rank * mult; i < (rank + 1) * mult; i += 1)
         {
             sum += static_cast<double> (1.0/fact(i));
         }
         e += sum;
         for (int i = 1; i < size; i ++){
           MPI_Recv(&sum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
           e += sum;
         }
     }
     else
     {
         std::cout << "my rank " << rank << std::endl;
         double sum = 0;
         for (int i = rank * mult; i < (rank + 1) * mult; i += 1)
         {
             sum += static_cast<double> (1.0/fact(i));
         }
         MPI_Send(&sum, 1, MPI_DOUBLE , 0, 0, MPI_COMM_WORLD);
     }

    MPI_Finalize();
    if (rank == 0)
    {
    std::cout << e << "\n";
    }

     return 0;
 }
