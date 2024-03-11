#include <stdio.h>
#include <stdlib.h>
#include <valarray>
#include <mpi.h>
#include <omp.h>

void printresult(double *arr, int size, double elapsed_time){
    
    /*
    printf("Result array :\n");
    for(int i = 0; i < size; i++){
        printf("%lf ", arr[i]);
    }
    printf("\n");
    */

    printf("Elapsed time : %lf\n", elapsed_time);
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

double *merge(double *local, int localsize, double *recv, int recvsize){
    
    double* result = (double*)malloc((localsize + recvsize) * sizeof(double));
    int i = 0, j = 0, k = 0;

    for(k = 0; k < localsize + recvsize; k++){
        if(i >= localsize){
            result[k] = recv[j];
            j++;
        }
        else if(j >= recvsize){
            result[k] = local[i];
            i++;
        }
        else if(local[i] < recv[j]){
            result[k] = local[i];
            i++;
        }
        else{
            result[k] = recv[j];
            j++;
        }
    }

    return result;
}

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

    int chunksize = (rank < (ARRAY_SIZE % size)) ? (ARRAY_SIZE / size) + 1 : ARRAY_SIZE / size;
    int display = (rank < (ARRAY_SIZE % size)) ? (ARRAY_SIZE / size) * rank + rank : (ARRAY_SIZE / size) * rank + (ARRAY_SIZE % size);

    double *chunk = (double *)malloc(chunksize * sizeof(double));

    // read a chunk of input to each process
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(double) * display, chunk, chunksize, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // start parallel merge sort
    start_time = MPI_Wtime();
    // first, sort the local array before iteration
    quickSort(chunk, 0, chunksize - 1);

    int recv_chunksize;
    double *result;
    // iterate for log2(size) times
    for(int step = 1; step < size; step *= 2){
        if(rank % (2 * step) != 0){
            MPI_Send(&chunksize, 1, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            MPI_Send(chunk, chunksize, MPI_DOUBLE, rank - step, 0, MPI_COMM_WORLD);
            break;
        }

        MPI_Recv(&recv_chunksize, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double *chunk_received = (double*)malloc(recv_chunksize * sizeof(double));
        MPI_Recv(chunk_received, recv_chunksize,  MPI_DOUBLE, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        result = merge(chunk, chunksize, chunk_received, recv_chunksize);

        chunksize += recv_chunksize;
        free(chunk);
        free(chunk_received);
        chunk = result;
    }

    end_time = MPI_Wtime();

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if(rank == 0){
        MPI_File_write_at(output_file, 0, chunk, chunksize, MPI_DOUBLE, MPI_STATUS_IGNORE);
        
        //printf("%d\n", chunksize);
        printresult(chunk, chunksize, end_time - start_time);
    } 

    MPI_File_close(&output_file); 

    MPI_Finalize(); 
    return 0; 
}
