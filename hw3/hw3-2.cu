#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define DEV_NO 0

/*
// 不要用半精度，gtx沒完全support
// from cuda_runtime.h
#define make_int4(a, b, c, d) ((int4){.x = a, .y = b, .z = c, .w = d})

// from helper_math.h
inline __host__ __device__ int4 operator+(int4 a, int4 b){
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}
*/

const int INF = ((1 << 30) - 1);
const int B = 64;
const int n_thread = 32;
const int n_per_iter = 4;
const int n_sm_size = 4096;
const int shift = 6;

void input(char* infile);
void output(char* outFileName);
void block_FW();

__global__ void cal1(int *D, int r, int n);
__global__ void cal2(int *D, int r, int n);
__global__ void cal3(int *D, int r, int n);

int n, m, padding_n;
int *Dist, *dev_Dist;
size_t Dist_size;

int main(int argc, char* argv[]){

    input(argv[1]);

    block_FW();
    output(argv[2]);
    return 0;
}

void input(char* infile){
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    padding_n = n + B -((n % B + B - 1) % B + 1);
    Dist_size = padding_n * padding_n * sizeof(int);
    cudaMallocHost(&Dist, Dist_size);
    for(int i = 0; i < padding_n; i++){
        for(int j = 0; j < padding_n; j++){
            if((i == j) && (i < n)) {
                Dist[i * padding_n + j] = 0;
            }else{
                Dist[i * padding_n + j] = INF;
            }
        }
    }

    int pair[3];
    for(int i = 0; i < m; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * padding_n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName){
    FILE* outfile = fopen(outFileName, "w");
    for(int i = 0; i < n; i++){
        fwrite(&Dist[i * padding_n], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void block_FW(){
    int round = (padding_n + B - 1) / B;
    dim3 block(n_thread, n_thread), grid2(2, round - 1), grid3(round - 1, round - 1);
    cudaMalloc(&dev_Dist, Dist_size);
    cudaMemcpy(dev_Dist, Dist, Dist_size, cudaMemcpyHostToDevice);

    for(int r = 0; r < round; r++){
        cal1 <<<1, block>>> (dev_Dist, r, padding_n);
        cal2 <<<grid2, block>>> (dev_Dist, r, padding_n);
        cal3 <<<grid3, block>>> (dev_Dist, r, padding_n);
    }
    
    cudaMemcpy(Dist, dev_Dist, Dist_size, cudaMemcpyDeviceToHost);
    //cudaFree(dev_Dist);
}

__global__ void cal1(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; // 64 * 64

    // local
    int b_i, b_j;
    int i, j;
    int ij, ik, kj;
    
    // 只有pivot
    b_i = b_j = r << shift; 
    i = threadIdx.y, j = threadIdx.x; // < n_thread
    ij = (threadIdx.y * n_thread + threadIdx.x) * n_per_iter;

    // memory放一起
    // 先放ij
    sm[ij] = D[(b_i + i) * n + b_j + j]; 
    sm[ij + 1] = D[(b_i + i) * n + b_j + j + n_thread];
    sm[ij + 2] = D[(b_i + i + n_thread) * n + b_j + j];
    sm[ij + 3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    // 再做min(ij, ik + kj)
    #pragma unroll n_thread
    for(int k = 0; k < n_thread; k++){
        __syncthreads(); //牽扯到share memory都要同步化
        ik = (i * n_thread + k) * 4;
        kj = (k * n_thread + j) * 4;
        sm[ij] = min(sm[ij], sm[ik] + sm[kj]);
        sm[ij + 1] = min(sm[ij + 1], sm[ik] + sm[kj + 1]);
        sm[ij + 2] = min(sm[ij + 2], sm[ik + 2] + sm[kj]);
        sm[ij + 3] = min(sm[ij + 3], sm[ik + 2] + sm[kj + 1]);

        sm[ij] = min(sm[ij], sm[ik + 1] + sm[kj + 2]);
        sm[ij + 1] = min(sm[ij + 1], sm[ik + 1] + sm[kj + 3]);
        sm[ij + 2] = min(sm[ij + 2], sm[ik + 3] + sm[kj + 2]);
        sm[ij + 3] = min(sm[ij + 3], sm[ik + 3] + sm[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j] = sm[ij]; 
    D[(b_i + i) * n + b_j + j + n_thread] = sm[ij + 1]; 
    D[(b_i + i + n_thread) * n + b_j + j] = sm[ij + 2]; 
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = sm[ij + 3]; 
};

__global__ void cal2(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; // 64 * 64
    __shared__ int cp[n_sm_size]; // 兩個都用sm會讓access亂掉
    
    int b_i, b_j, b_k;
    int i, j;
    volatile int ik, kj;
    int tmp[4];
    
    // 不算pivot，在邊上
    b_i = (blockIdx.x * r + (!blockIdx.x) * (blockIdx.y + (blockIdx.y >= r))) << shift;
    b_j = (blockIdx.x * (blockIdx.y + (blockIdx.y >= r)) + (!blockIdx.x) * r) << shift;
    b_k = r << shift;
    i = threadIdx.y, j = threadIdx.x;
    ik = kj = (i * n_thread + j) * n_per_iter;

    // local
    tmp[0] = D[(b_i + i) * n + b_j + j]; 
    tmp[1] = D[(b_i + i) * n + b_j + j + n_thread];
    tmp[2] = D[(b_i + i + n_thread) * n + b_j + j];
    tmp[3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    // ij
    sm[ik] = D[(b_i + i) * n + b_k + j]; 
    sm[ik + 1] = D[(b_i + i) * n + b_k + j + n_thread];
    sm[ik + 2] = D[(b_i + i + n_thread) * n + b_k + j];
    sm[ik + 3] = D[(b_i + i + n_thread) * n + b_k + j + n_thread];

    // kj
    cp[kj] = D[(b_k + i) * n + b_j + j]; 
    cp[kj + 1] = D[(b_k + i) * n + b_j + j + n_thread];
    cp[kj + 2] = D[(b_k + i + n_thread) * n + b_j + j];
    cp[kj + 3] = D[(b_k + i + n_thread) * n + b_j + j + n_thread];

    __syncthreads();

    #pragma unroll n_thread
    for(int k = 0; k < n_thread; k++){
        ik = (i * n_thread + k) * n_per_iter;
        kj = (k * n_thread + j) * n_per_iter;
        tmp[0] = min(min(tmp[0], sm[ik] + cp[kj]), sm[ik + 1] + cp[kj + 2]);
        tmp[1] = min(min(tmp[1], sm[ik] + cp[kj + 1]), sm[ik + 1] + cp[kj + 3]);
        tmp[2] = min(min(tmp[2], sm[ik + 2] + cp[kj]), sm[ik + 3] + cp[kj + 2]);
        tmp[3] = min(min(tmp[3], sm[ik + 2] + cp[kj + 1]), sm[ik + 3] + cp[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j] = tmp[0]; 
    D[(b_i + i) * n + b_j + j + n_thread] = tmp[1]; 
    D[(b_i + i + n_thread) * n + b_j + j] = tmp[2]; 
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = tmp[3]; 
};

__global__ void cal3(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; 
    __shared__ int cp[n_sm_size]; 
    
    int b_i, b_j, b_k;
    int i, j;
    volatile int ik, kj;
    int tmp[4];

    // 剩下一大塊
    b_i = (blockIdx.x + (blockIdx.x >= r)) << shift;
    b_j = (blockIdx.y + (blockIdx.y >= r)) << shift;
    b_k = r << shift;
    i = threadIdx.y, j = threadIdx.x;
    ik = kj = (i * n_thread + j) * n_per_iter;

    tmp[0] = D[(b_i + i) * n + b_j + j]; 
    tmp[1] = D[(b_i + i) * n + b_j + j + n_thread];
    tmp[2] = D[(b_i + i + n_thread) * n + b_j + j];
    tmp[3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    sm[ik] = D[(b_i + i) * n + b_k + j]; 
    sm[ik + 1] = D[(b_i + i) * n + b_k + j + n_thread];
    sm[ik + 2] = D[(b_i + i + n_thread) * n + b_k + j];
    sm[ik + 3] = D[(b_i + i + n_thread) * n + b_k + j + n_thread];

    cp[kj] = D[(b_k + i) * n + b_j + j]; 
    cp[kj + 1] = D[(b_k + i) * n + b_j + j + n_thread];
    cp[kj + 2] = D[(b_k + i + n_thread) * n + b_j + j];
    cp[kj + 3] = D[(b_k + i + n_thread) * n + b_j + j + n_thread];

    __syncthreads();

    #pragma unroll n_thread
    for(int k = 0; k < n_thread; k++){
        ik = (i * n_thread + k) * n_per_iter;
        kj = (k * n_thread + j) * n_per_iter;
        tmp[0] = min(min(tmp[0], sm[ik] + cp[kj]), sm[ik + 1] + cp[kj + 2]);
        tmp[1] = min(min(tmp[1], sm[ik] + cp[kj + 1]), sm[ik + 1] + cp[kj + 3]);
        tmp[2] = min(min(tmp[2], sm[ik + 2] + cp[kj]), sm[ik + 3] + cp[kj + 2]);
        tmp[3] = min(min(tmp[3], sm[ik + 2] + cp[kj + 1]), sm[ik + 3] + cp[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j] = tmp[0];    
    D[(b_i + i) * n + b_j + j + n_thread] = tmp[1];   
    D[(b_i + i + n_thread) * n + b_j + j] = tmp[2];   
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = tmp[3];    
};