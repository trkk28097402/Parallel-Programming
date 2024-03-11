#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <valarray>
//#include <boost/sort/sort.hpp>
#define INF 1.79769313486231570e+308d

void printresult(double *arr, int size, double elapsed_time){
    
    printf("Result array :\n");
    for(int i = 0; i < size; i++){
        printf("%lf ", arr[i]);
    }
    printf("\n");

    printf("Elapsed time : %lf\n", elapsed_time);
}

void merge(int size, double *local, double *recv, double *result, int dir, int arrange){
    // dir 升或降冪
    // 升冪且小 拿小的
    // 升冪且大 拿大的
    // 降冪且小 拿大的
    // 降冪且大 拿小的

    if((!dir && !arrange) || (dir && arrange)){ // 小的
        int i = 0, j = 0;
        for(int k = 0; k < size; k++){
            if(local[i] < recv[j]){
                result[k] = local[i];
                i++;
            }
            else{
                result[k] = recv[j];
                j++;
            }
        }
    }
    else{ // 大的
        int i = size - 1, j = size - 1;
        for(int k = size - 1; k >= 0; k--){
            if(local[i] > recv[j]){
                result[k] = local[i];
                i--;
            }
            else{
                result[k] = recv[j];
                j--;
            }
        }
    }
}

void quickSort(double *arr, int low, int high){
    if(low < high){
        double pivot = arr[high];
        int i = low - 1;

        for(int j = low; j <= high - 1; j++){
            if(arr[j] < pivot){
                i++;
                std::swap(arr[i], arr[j]);
            }
        }

        std::swap(arr[i + 1], arr[high]);

        quickSort(arr, low, i);
        quickSort(arr, i + 2, high);
    }
}

// 如果我是降冪，我又是小的process，我就要拿比較大的那塊，如果我是大的process，我就要拿比較小的那塊
int main(int argc, char *argv[]){

    int rank, size;
    int ARRAY_SIZE = atoi(argv[1]);
    char *input_filename = argv[2], *output_filename = argv[3];
    MPI_File input_file, output_file;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if(size & (size - 1)){
        if(rank == 0) printf("Please use a process count that is a power of 2.\n");
        MPI_Finalize(); 
        return 0; 
    } 

    int padding = (size - (ARRAY_SIZE % size)) % size;
    int chunksize = padding > 0 ? (ARRAY_SIZE / size) + 1 : (ARRAY_SIZE / size);
    int display = rank < padding ? rank * chunksize - rank : rank * chunksize - padding;

    double *chunk = (double *)malloc(chunksize * sizeof(double));
    double *chunk_received = (double*)malloc(chunksize * sizeof(double));
    double *chunk_merge = (double*)malloc(chunksize * sizeof(double));

    // read a chunk of input to each process
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(double) * display, chunk, chunksize - (rank < padding), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    
    if(rank < padding) chunk[chunksize - 1] = INF; 

    // start bitomic merge sort
    start_time = MPI_Wtime();

    quickSort(chunk, 0, chunksize - 1);
    //boost::sort::spreadsort::spreadsort(chunk, chunk + chunksize);

    int direct; // 0 升冪, 1 降冪
    int pair;
    double *tmp;
    
    for(int stage = 1; stage < size; stage *= 2){
        direct = rank / (2 * stage) % 2;
        for(int step = stage; step >= 1; step /= 2){
            pair = rank ^ step;
            MPI_Sendrecv(chunk, chunksize, MPI_DOUBLE,
                pair, 0, chunk_received, chunksize,
                MPI_DOUBLE, rank ^ step, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            merge(chunksize, chunk, chunk_received, chunk_merge, direct, pair < rank);
            
            tmp = chunk;
            chunk = chunk_merge;
            chunk_merge = tmp;
        }
    }

    end_time = MPI_Wtime();

    //if(rank == size - 1) printresult(chunk, chunksize, end_time - start_time);

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    display = rank * chunksize;
    int count = 1;
    while(chunksize < padding){
        padding -= chunksize;
        count++;
    }
    
    if(rank == size - count) MPI_File_write_at(output_file, sizeof(double) * display, chunk, chunksize - padding, MPI_DOUBLE, MPI_STATUS_IGNORE);
    else if(rank < size - count) MPI_File_write_at(output_file, sizeof(double) * display, chunk, chunksize, MPI_DOUBLE, MPI_STATUS_IGNORE);
    
    MPI_File_close(&output_file); 

    free(chunk);
    free(chunk_received);
    free(chunk_merge);

    MPI_Finalize(); 
    return 0; 
}
