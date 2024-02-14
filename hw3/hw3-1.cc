#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sched.h>
#include <math.h>
#include <smmintrin.h>
#define min(a,b) (a < b ? a : b)

const int INF = ((1 << 30) - 1);
const int V = 6000;
const int B = 64;
void input(char* inFileName);
void output(char* outFileName);
void block_FW();
void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height);

static int n, m;
static int ncpus;
static int Dist[V][V];

int main(int argc, char* argv[]){
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    input(argv[1]);

    // caculate the optimized block factor
    block_FW();
    output(argv[2]);
    return 0;
}

void input(char* infile){
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for(int i = 0; i < m; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName){
    FILE* outfile = fopen(outFileName, "w");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void block_FW() {
    int round = (n + B - 1) / B;

    #pragma omp parallel num_threads(ncpus)
    {
        for(int r = 0; r < round; r++){
            /* Phase 1*/
            cal(r, r, r, 1, 1);

            /* Phase 2*/
            cal(r, r, 0, r, 1);
            cal(r, r, r + 1, round - r - 1, 1);
            cal(r, 0, r, 1, r);
            cal(r, r + 1, r, 1, round - r - 1);

            /* Phase 3*/
            cal(r, 0, 0, r, r);
            cal(r, 0, r + 1, round - r - 1, r);
            cal(r, r + 1, 0, r, round - r - 1);
            cal(r, r + 1, r + 1, round - r - 1, round - r - 1);
        }
    }
}

void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height){
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    int k_start = Round * B, k_limit = min(k_start + B, n);
    __m128i ik, kj, ij;

    #pragma omp for collapse(2) schedule(auto) 
    for(int b_i = block_start_x; b_i < block_end_x; b_i++){
        for(int b_j = block_start_y; b_j < block_end_y; b_j++){
            int block_internal_start_x = b_i * B;
            int block_internal_end_x = min(block_internal_start_x + B, n);
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = min(block_internal_start_y + B, n);
            
            for(int k = k_start; k < k_limit; k++){
                for(int i = block_internal_start_x; i < block_internal_end_x; i++){
                    ik = _mm_set1_epi32(Dist[i][k]);
                    for(int j = block_internal_start_y; j < block_internal_end_y; j += 4){
                        ij = _mm_loadu_si128((__m128i*)&Dist[i][j]);
                        kj = _mm_loadu_si128((__m128i*)&Dist[k][j]);

                        _mm_storeu_si128((__m128i*)&Dist[i][j], _mm_min_epi32(_mm_add_epi32(ik, kj), ij));
                    }
                }
            }
        }
    }
}
