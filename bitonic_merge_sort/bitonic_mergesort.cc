#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <valarray>

//#include <iostream>
//#include <stack>
//#include <tuple>

#include <cfloat>
#define ASCENDING 1
#define DESCENDING 0

/*
void bitonicMerge(int size, int st, float *arr, int dir){
    if(size > 1){
        int subsize = size / 2;
        for(int i = st; i < st + subsize; i++) if((arr[i] > arr[i + subsize]) == dir) std::swap(arr[i], arr[i + subsize]);
        bitonicMerge(subsize, st, arr, dir);
        bitonicMerge(subsize, st + subsize, arr, dir);
    }
}
*/

void bitonicMerge(int size, int st, float *arr, int dir){
    std::stack<std::tuple<int, int, float*, int>> stack;
    stack.push(std::make_tuple(size, st, arr, dir));
    
    while(!stack.empty()){
        auto [currentSize, currentSt, currentArr, currentDir] = stack.top();
        stack.pop();
        
        if(currentSize > 1) {
            int subsize = currentSize / 2;
            
            #pragma omp simd
            for(int i = currentSt; i < currentSt + subsize; i++){
                if((currentArr[i] > currentArr[i + subsize]) == currentDir){
                    std::swap(currentArr[i], currentArr[i + subsize]);
                }
            }
            
            stack.push(std::make_tuple(subsize, currentSt, currentArr, currentDir));
            stack.push(std::make_tuple(subsize, currentSt + subsize, currentArr, currentDir));
        }
    }
}

void bitonicSort(int size, int st, float *arr, int dir){

    if(size > 1){
        int subsize = size / 2;
        //std::sort(arr, arr + subsize);
        //std::sort(arr + subsize, arr + subsize * 2, std::greater<float>());
        bitonicSort(subsize, st, arr, ASCENDING);
 
        bitonicSort(subsize, st + subsize, arr, DESCENDING);
 
        bitonicMerge(size, st, arr, dir);
    }
}

/*
TODO: 
1. Optimize passing, reverse the array so we don't need to sort again
2. Bypass SENDRECV if don't need to sort
3. Data Layout improvement
*/

int main(int argc, char *argv[]){

    int rank, size;
    int ARRAY_SIZE = atoi(argv[1]);
    char *input_filename = argv[2], *output_filename = argv[3];
    MPI_File input_file, output_file;
    float start_time, end_time;
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if(size & (size - 1)){
        if(rank == 0) printf("Please use a process count that is a power of 2.\n");
        MPI_Finalize(); 
        return 0; 
    } 

    int power = 1;
    while(power < ARRAY_SIZE) power *= 2;

    int padding = (power - ARRAY_SIZE) / size + ((power - ARRAY_SIZE) % size > 0); 
    int chunksize = (ARRAY_SIZE / size) + padding;
    
    float *chunk = (float *)malloc(chunksize * 2 * sizeof(float));

    // read a chunk of input to each process
    int org_padding = (size - (ARRAY_SIZE % size)) % size;
    int org_chunksize = org_padding > 0 ? (ARRAY_SIZE / size) + 1 : (ARRAY_SIZE / size);
    int display = rank < org_padding ? rank * org_chunksize - rank : rank * org_chunksize - org_padding;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * display, chunk, org_chunksize - (rank < org_padding), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    
    for(int i = 0; i < padding - (rank >= org_padding && org_padding); i++){
        chunk[chunksize - i - 1] = FLT_MAX;
    }

    // start bitomic merge sort
    start_time = MPI_Wtime();

    int direct, pair, order, pair_order;
    float *send = chunk, *recv = chunk + chunksize;

    
    if(rank % 2) {bitonicSort(chunksize, 0, chunk, DESCENDING); order = DESCENDING;}
    else {bitonicSort(chunksize, 0, chunk, ASCENDING); order = ASCENDING;}
    /*
    if(rank % 2) {std::sort(chunk, chunk + chunksize, std::greater<float>()); order = DESCENDING;}
    else {std::sort(chunk, chunk + chunksize); order = ASCENDING;}
    */
    for(int stage = 1; stage < size; stage *= 2){
        direct = rank / (2 * stage) % 2;
        for(int step = stage; step >= 1; step /= 2){
            pair = rank ^ step;

            MPI_Sendrecv(&order, 1, MPI_INT,
                pair, 0, &pair_order, chunksize,
                MPI_INT, pair, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(send, chunksize, MPI_FLOAT,
                pair, 0, recv, chunksize,
                MPI_FLOAT, pair, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if(pair_order == order) std::reverse(recv, recv + chunksize);

            bitonicMerge(chunksize * 2, 0, chunk, !direct);
            order = !direct;

            if(rank < pair){
                send = chunk;
                recv = chunk + chunksize;
            }
            else if(rank > pair){
                send = chunk + chunksize;
                recv = chunk;
            }
        }
    }

    end_time = MPI_Wtime();

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    display = rank * chunksize;

    if(size == 1) MPI_File_write_at(output_file, sizeof(float) * display, send, ARRAY_SIZE, MPI_FLOAT, MPI_STATUS_IGNORE);
    else if(0 < ARRAY_SIZE - display && ARRAY_SIZE - display < chunksize) MPI_File_write_at(output_file, sizeof(float) * display, send, ARRAY_SIZE - display, MPI_FLOAT, MPI_STATUS_IGNORE);
    else if(display < ARRAY_SIZE) MPI_File_write_at(output_file, sizeof(float) * display, send, chunksize, MPI_FLOAT, MPI_STATUS_IGNORE);

    if(rank == 0) printf("Time: %f\n", end_time - start_time);

    MPI_File_close(&output_file); 

    free(chunk);

    MPI_Finalize(); 
    return 0; 
}
