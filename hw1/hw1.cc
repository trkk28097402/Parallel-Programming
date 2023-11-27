#include <stdio.h>
#include <mpi.h>
#include <boost/sort/sort.hpp>

void merge_up(float *local, float *recv, float *tmp, int local_n, int recv_n){ // 跟下一個process
    int a_id = 0, b_id = 0, j = 0, k = 0;
    while(b_id < recv_n && j < local_n){
        if(local[a_id] < recv[b_id]) tmp[j++] = local[a_id++];  
        else tmp[j++] = recv[b_id++]; 
    }

    if(j < local_n) tmp[j] = local[a_id];
}

void merge_down(float *local, float *recv, float *tmp, int local_n, int recv_n){ // 跟上一個process
    int a_id = local_n - 1, b_id = recv_n - 1, j = local_n - 1, k = 0;
    while(a_id >= 0 && j >= 0){
        if(local[a_id] > recv[b_id]) tmp[j--] = local[a_id--];  
        else tmp[j--] = recv[b_id--]; 
    }

    if(j == 0) tmp[j] = recv[b_id];
}

int main(int argc, char** argv){

    int my_rank, comm_size;
    int i;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int state = my_rank % 2;
    
    int n = atoi(argv[1]);
    char *in_name = argv[2], *out_name = argv[3];

    MPI_File in_file, out_file;
    int base_data_n = n / comm_size;
    int left = n % comm_size;

    int last_ct = left > (my_rank - 1) ? base_data_n + 1 : base_data_n;
    int my_ct = left > my_rank ? base_data_n + 1 : base_data_n;
    int next_ct = left > (my_rank + 1) ? base_data_n + 1 : base_data_n;
    int display = my_rank > left ? base_data_n * my_rank + left : base_data_n * my_rank + my_rank;

    float *local_data = (float *)malloc(sizeof(float) * my_ct * 2);
    float *recv_data = (float *)malloc(sizeof(float) * last_ct);

    MPI_File_open(MPI_COMM_WORLD, in_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);
    MPI_File_read_at(in_file, sizeof(float) * display, local_data, my_ct, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&in_file);

    boost::sort::spreadsort::spreadsort(local_data, local_data + my_ct);

    int o1 = 0, o2 = my_ct; // offset

    int flipflop = 1;
    int iter = left != 0 ? comm_size + 1 : comm_size;

    for(i = 0; i < iter; i++){
        if(flipflop == 0){ //even phase
            if(state == 0){
                if(my_rank < comm_size - 1){ // 小 -> 大
                    MPI_Sendrecv(&local_data[my_ct - 1 + o1], 1, MPI_FLOAT,
                    my_rank + 1, 0, recv_data, 1,
                    MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
                    
                    // 如果對方最小的比我最大的大就不做，反之進入if
                    if(recv_data[0] < local_data[my_ct - 1 + o1]){
                        MPI_Sendrecv(local_data + o1, my_ct, MPI_FLOAT,
                        my_rank + 1, 0, recv_data, next_ct,
                        MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                        merge_up(local_data + o1, recv_data, local_data + o2, my_ct, next_ct);
                        std::swap(o1, o2);
                    }
                }
            }
            else{
                if(my_rank > 0){ // 大 -> 小
                    MPI_Sendrecv(local_data + o1, 1, MPI_FLOAT,
                    my_rank - 1, 0, recv_data, 1,
                    MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);                    

                    // 如果對方最大的比我最小的小就不做，反之進入if
                    if(recv_data[0] > local_data[o1]){
                        MPI_Sendrecv(local_data + o1, my_ct, MPI_FLOAT,
                        my_rank - 1, 0, recv_data, last_ct,
                        MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                        merge_down(local_data + o1, recv_data, local_data + o2, my_ct, last_ct);
                        std::swap(o1, o2);
                    }
                }
            }

            flipflop = 1;
        }
        else{ //odd phase
            if(state == 0){
                if(my_rank > 0){ // 大 -> 小
                    MPI_Sendrecv(local_data + o1, 1, MPI_FLOAT,
                    my_rank - 1, 0, recv_data, 1,
                    MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);                    

                    // 如果對方最大的比我最小的小就不做，反之進入if
                    if(recv_data[0] > local_data[o1]){
                        MPI_Sendrecv(local_data + o1, my_ct, MPI_FLOAT,
                        my_rank - 1, 0, recv_data, last_ct,
                        MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                        merge_down(local_data + o1, recv_data, local_data + o2, my_ct, last_ct);
                        std::swap(o1, o2);
                    }
                }
            }
            else{
                if(my_rank < comm_size - 1){ // 小 -> 大
                    MPI_Sendrecv(&local_data[my_ct - 1 + o1], 1, MPI_FLOAT,
                    my_rank + 1, 0, recv_data, 1,
                    MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
                    
                    // 如果對方最小的比我最大的大就不做，反之進入if
                    if(recv_data[0] < local_data[my_ct - 1 + o1]){
                        MPI_Sendrecv(local_data + o1, my_ct, MPI_FLOAT,
                        my_rank + 1, 0, recv_data, next_ct,
                        MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                        merge_up(local_data + o1, recv_data, local_data + o2, my_ct, next_ct);
                        std::swap(o1, o2);
                    }                  
                }
            }

            flipflop = 0;
        }
    }    

    MPI_File_open(MPI_COMM_WORLD, out_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file);
    MPI_File_write_at(out_file, sizeof(float) * display, local_data + o1, my_ct, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&out_file); 
    MPI_Finalize();

    free(local_data);
    free(recv_data);

    return 0;
}
