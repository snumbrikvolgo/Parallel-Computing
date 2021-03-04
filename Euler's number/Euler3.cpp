#include <iostream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <iomanip>
#include <mpi.h>
using namespace boost::multiprecision ;

int128_t fact(int128_t x)
{
    if (x == 0)
      return 1;
    int128_t p = 1;
    for (int128_t i = x; i > 0; i--)
    {
        p *= i;
    }

    return p;
}

int main(int argc, char*argv[])
{

    using float100 = cpp_dec_float_100 ;
    int rank, size, n, i;
    float100 e = 0; //initial value
    float100 e_true = 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320030599218174135966290435729003342952605956307381;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    int mult = 20;
    if (rank == 0)
    {
        float100 sum = 0;

        for (int128_t i = rank * mult; i < (rank + 1) * mult; i += 1)
        {
            sum += static_cast<float100>(1.0)/static_cast<float100>(fact(i));
        }
        e += sum;
        MPI_Recv(&sum, sizeof(cpp_dec_float_100), MPI_CHAR , MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, &status);
        e += sum;
        MPI_Recv(&sum, sizeof(cpp_dec_float_100),  MPI_CHAR , MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, &status);
        e += sum;
        MPI_Recv(&sum, sizeof(cpp_dec_float_100),  MPI_CHAR , MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, &status);
        e += sum;
    }
    else
    {
        float100 sum = 0;
        for (int128_t i = rank * mult; i < (rank + 1) * mult; i += 1)
        {
            sum += static_cast<float100>(1.0)/static_cast<float100>(fact(i));
        }
        MPI_Send(&sum, sizeof(cpp_dec_float_100), MPI_CHAR , 0, 5, MPI_COMM_WORLD);
    }


   MPI_Finalize();

   if (rank == 0)
   {
   std::cout << "computated " << std::fixed << std::setprecision(100) << e << "\n";
   std::cout << "true       " << std::fixed << std::setprecision(100) << e_true << "\n";
   std::cout << "diff       " << std::fixed << std::setprecision(100) << e_true - e << "\n";
   }

    return 0;
}
