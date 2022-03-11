#include <iostream>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <mutex>

void PrintSudoku(int LAYER, std::vector<std::vector<int>>& game_field);
void ReadMatrix(std::string file_name, int& LAYER, std::vector<std::vector<int>>& game_field);
int SolveSudoku(int LAYER, std::vector<std::vector<int>>& game_field);

int main(int argc, char** argv) {
    int LAYER;                                              
    if (argc <= 1){
        std::cout << "Enter the filename" << std::endl;
        return -1;
    }

    std::vector<std::vector<int>> game_field;
    std::string file_name = std::string(argv[1]);

    ReadMatrix(file_name, LAYER, game_field);

    double time_start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        if (SolveSudoku(LAYER, game_field) == 1) { 
            std::cout << "No solution" << std::endl;
            exit(-1);
        }
    }
    double time_end = omp_get_wtime();
    double time_d = time_end - time_start;
    PrintSudoku(LAYER, game_field);

    std::cout << std::endl << "Total Time: " << time_d << " seconds" << std::endl;
}


