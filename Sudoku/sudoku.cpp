#include <iostream>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <mutex>

std::mutex MUT;

void PrintSudoku(int LAYER, std::vector<std::vector<int>>& game_field) {
    int LAYER2 = LAYER * LAYER;
    for (int x = 0; x < LAYER2; ++x) {
        for (int y = 0; y < LAYER2; ++y) {
            if(game_field[x][y] > 9) {
                std::cout << game_field[x][y] << " ";
            }

            else { 
                std::cout << game_field[x][y] << "  "; 
            }

            if ((y + 1) % LAYER == 0 && y != LAYER2 - 1){ 
                std::cout << "|  "; 
            }
        }
        std::cout << std::endl;

        if ((x + 1) % LAYER == 0 && x != LAYER2 - 1) { 
            for (int i = 0; i < LAYER2 + LAYER - 1; ++i) {
                 std::cout << "_ _";
             }
            std::cout << std::endl << std::endl;
        }
    }
}

void ReadMatrix(std::string file_name, int& LAYER, std::vector<std::vector<int>>& game_field) {
    std::ifstream fin;
    std::string buf_str;
    int buf_int;

    fin.open(file_name);

    getline(fin, buf_str, ',');
    std::istringstream(buf_str) >> buf_int;

    LAYER = buf_int;
    int LAYER2 = LAYER * LAYER;

    game_field.resize(LAYER2);
    for (int i = 0; i < LAYER2; ++i) { 
        game_field[i].resize(LAYER2); 
    }
    

    int i = 0;
    while (getline(fin, buf_str, ',')) {
        std::istringstream(buf_str) >> buf_int;
        game_field[i / LAYER2][i % LAYER2] = buf_int;
        ++i;
    }

}

void FindPossibleValues(int LAYER, std::vector<std::vector<int>>& game_field, std::vector<std::vector<bool>>& is_known,
    std::vector<bool>& variants, int cell) {

    int LAYER2 = LAYER * LAYER;

    int cell_x = cell / LAYER2;
    int cell_y = cell % LAYER2;

    for (int iy = 0; iy < LAYER2; ++iy) {
        if (is_known[cell_x][iy] == true) {
            variants[game_field[cell_x][iy]] = true;
        }
    }

    for (int ix = 0; ix < LAYER2; ++ix) {
        if (is_known[ix][cell_y] == true) {
            variants[game_field[ix][cell_y]] = true;
        }
    }

    int start_x = (cell_x / LAYER) * LAYER;
    int start_y = (cell_y / LAYER) * LAYER;

    for (int ix = 0; ix < LAYER; ++ix) {
        for (int iy = 0; iy < LAYER; ++iy) {
            if (is_known[start_x + ix][start_y + iy] == true) {
                variants[game_field[start_x + ix][start_y + iy]] = true;
            }
        }
    }
}


int SolveSudoku(int LAYER, std::vector<std::vector<int>>& game_field) {

    bool DONE = false;

    int LAYER2;
    int LAYER4;

    LAYER2 = LAYER * LAYER;
    LAYER4 = LAYER2 * LAYER2;


    std::vector<int> tasks;                             
    std::vector<int> next_tasks;                  

    std::vector<std::vector<bool>> is_known(LAYER2);   
    for (int i = 0; i < LAYER2; ++i) {
        is_known[i].resize(LAYER2); 
    }

    for (int ix = 0; ix < LAYER2; ++ix){
        for (int iy = 0; iy < LAYER2; ++iy){
            if (game_field[ix][iy] != 0) {
                is_known[ix][iy] = true;
            }
            else {
                is_known[ix][iy] = false;
                tasks.push_back(ix * LAYER2 + iy); //unknown numbers are pushed to tasks
            }
        }
    }

    while (DONE == false) {
        std::vector<int> potential_vector(LAYER4, LAYER2 + 1);
        
        for (int cell : tasks) {
            int cell_x = cell / LAYER2;
            int cell_y = cell % LAYER2;
    
            std::vector<bool> variants(LAYER2 + 1, false);
            //if variants are false then it is possible to use this value
            FindPossibleValues(LAYER, game_field, is_known, variants, cell);

            int first_false = 0;
            int false_counter = 0;
            //count variants and define whether its possible to use single one
            for (int i = 1; i < LAYER2 + 1; ++i) {
                if (variants[i] == false) {
                    false_counter++;
                    if (false_counter == 1) {
                         first_false = i; 
                    }
                }
            }
            //insert number of possible values into the vector
            potential_vector[cell] = false_counter;

            if (false_counter == 0) { 
                return 1;
            }
            else if (false_counter == 1) {
                is_known[cell_x][cell_y] = true;
                game_field[cell_x][cell_y] = first_false;
            }
            else {
                next_tasks.push_back(cell);
            }

        }

        if (tasks.empty() == true) { 
            DONE = true; 
        }
        else if (tasks.size() == next_tasks.size()) { //all of them are unknown in the same way
            //choosing easieset cells to start
            std::pair<int, int> min_cell = std::pair<int, int>(-1, LAYER2 + 1); //pair.second = number of potential values
            for (int i = 0; i < LAYER4; ++i) {
                if (potential_vector[i] < min_cell.second) { 
                    min_cell.first = i;
                    min_cell.second = potential_vector[i];
                }
            }

            std::vector<bool> variants(LAYER2 + 1, false);
            FindPossibleValues(LAYER, game_field, is_known, variants, min_cell.first);

            bool result = true;

            for (int i = 1; i < LAYER2 + 1; ++i) {
                if (variants[i] == false) {

                    #pragma omp task firstprivate(i, min_cell) shared(DONE, game_field, result)
                    {
                        std::vector<std::vector<int>> game_field_copy(LAYER2);
                        for (int i = 0; i < LAYER2; ++i) {
                            game_field_copy[i].resize(LAYER2); 
                        }
                        game_field_copy = game_field;
                        game_field_copy[min_cell.first / LAYER2][min_cell.first % LAYER2] = i;
                        
                        if(DONE == false) {
                            if (SolveSudoku(LAYER, game_field_copy) == 0) {
                                MUT.lock();
                                DONE = true;
                                game_field = game_field_copy;
                                result = false;
                                MUT.unlock();
                            }
                        }
                    }
                }

            }
            #pragma omp taskwait
                if(result == false) { 
                    return 0; 
                }
            return 1;
        }
        tasks = next_tasks;
        next_tasks.clear();
    }
    return 0;
}


