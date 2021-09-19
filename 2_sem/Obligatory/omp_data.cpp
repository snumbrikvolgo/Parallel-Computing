#include <iostream>
#include <sstream>
#include <string>
#include <omp.h>

int main(int argc, char** argv)
{
    int i;
    int my_id;
    int threads = atoi(argv[1]);
    std::string data = "data:";

    omp_set_num_threads(threads);

    #pragma omp parallel for shared(data) private(i, my_id) ordered
    for (i = 0; i < threads; i++)
    {
        my_id = omp_get_thread_num();
        #pragma omp ordered
        data += " " + std::to_string(my_id);
    }

    std::cout << data << std::endl;

    return 0;
}
