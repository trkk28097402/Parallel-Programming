#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <emmintrin.h>
#define CHUNK 100
// MPI_Wtime()

const double initzero = 0, inittwo = 2;
int *image = NULL, *totalimage = NULL;

int rank, size;
double offsety, offsetx;

int ct, ncpus, cts;
omp_lock_t lock;

union Vectorize{
    alignas(16) double org[2];
    __m128d vec;
};

// 沒事不要load或set，很吃指令
int main(int argc, char **argv){
    setenv("OMP_DYNAMIC", "false", 1);
    setenv("OMP_STACKSIZE", "1G", 1);

    MPI_Init(&argc, &argv);
    //int provide;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provide);
    //printf("%d", provide);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //MPI_Request *request;
    omp_init_lock(&lock);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    
    ct = ceil((double)height / size);
    cts = width * ct;

    if(rank == 0) image = (int *)malloc(width * ct * size * sizeof(int));
    else image = (int *)malloc(width * ct * sizeof(int));

    offsety = (upper - lower) / height;
    offsetx = (right - left) / width;
    int xid = 0, yid = rank, iterid = 0;

    #pragma omp parallel num_threads(ncpus)
    {
        int i, j, lct, lxid, jpartition;
        int repeats[2], pos[2];
        double initx0[2], inity0;
        Vectorize x0, y0, x, y, lengthSquared, two;

        two.vec = _mm_load1_pd(&inittwo);
        while (true){
            omp_set_lock(&lock);
            if(yid >= height){
                omp_unset_lock(&lock);
                break;
            }
            i = xid, j = yid;
            jpartition = j / size;

            if(xid + CHUNK > width){
                lct = width - xid;
                xid = 0;
                yid += size;
            }
            else{
                lct = CHUNK;
                xid += CHUNK;
            }
            omp_unset_lock(&lock);
            lxid = i + 2;

            x.vec = y.vec = lengthSquared.vec = _mm_setzero_pd();

            inity0 = j * offsety + lower;
            y0.vec = _mm_load_pd1(&inity0);

            pos[0] = i, pos[1] = i + 1;
            initx0[0] = pos[0] * offsetx + left;
            initx0[1] = pos[1] * offsetx + left;
            x0.vec = _mm_load_pd(initx0);

            repeats[0] = 0, repeats[1] = 0;

            while(pos[0] < i + lct && pos[1] < i + lct){
                while(true){
                    __m128d xsq = _mm_mul_pd(x.vec, x.vec), ysq = _mm_mul_pd(y.vec, y.vec);
                    
                    lengthSquared.vec = _mm_add_pd(xsq, ysq);

                    if(lengthSquared.org[0] > 4 || lengthSquared.org[1] > 4) break;
                    y.vec = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x.vec, y.vec), two.vec), y0.vec);
                    x.vec = _mm_add_pd(_mm_sub_pd(xsq, ysq), x0.vec);
                    repeats[0]++, repeats[1]++;
                    if(repeats[0] >= iters || repeats[1] >= iters) break;
                }
            
                if(repeats[0] >= iters || lengthSquared.org[0] >= 4){
                    image[jpartition * width + pos[0]] = repeats[0];

                    pos[0] = lxid++; // 如果結束就直接進入下個pixel，並將標記+1
                    repeats[0] = 0;
                    initx0[0] = pos[0] * offsetx + left;

                    x.vec = _mm_loadl_pd(x.vec, &initzero), y.vec = _mm_loadl_pd(y.vec, &initzero);
                    lengthSquared.vec = _mm_loadl_pd(lengthSquared.vec, &initzero);
                }

                if(repeats[1] >= iters || lengthSquared.org[1] >= 4){
                    image[jpartition * width + pos[1]] = repeats[1];

                    pos[1] = lxid++; // 同上部分
                    repeats[1] = 0;
                    initx0[1] = pos[1] * offsetx + left;

                    x.vec = _mm_loadh_pd(x.vec, &initzero), y.vec = _mm_loadh_pd(y.vec, &initzero);
                    lengthSquared.vec = _mm_loadh_pd(lengthSquared.vec, &initzero);
                }

                x0.vec = _mm_load_pd(initx0);
            }

            if(pos[0] < i + lct){
                while(repeats[0] < iters && lengthSquared.org[0] < 4){
                    double temp = x.org[0] * x.org[0] - y.org[0] * y.org[0] + x0.org[0];
                    y.org[0] = 2 * x.org[0] * y.org[0] + y0.org[0];
                    x.org[0] = temp;
                    lengthSquared.org[0] = x.org[0] * x.org[0] + y.org[0] * y.org[0];
                    repeats[0]++;
                }
                image[jpartition * width + pos[0]] = repeats[0];
            }
            if(pos[1] < i + lct){
                while(repeats[1] < iters && lengthSquared.org[1] < 4){
                    double temp = x.org[1] * x.org[1] - y.org[1] * y.org[1] + x0.org[1];
                    y.org[1] = 2 * x.org[1] * y.org[1] + y0.org[1];
                    x.org[1] = temp;
                    lengthSquared.org[1] = x.org[1] * x.org[1] + y.org[1] * y.org[1];
                    repeats[1]++;
                }
                image[jpartition * width + pos[1]] = repeats[1];
            }
        }
    }

    
    if(rank == 0){
        int add = cts;
        #pragma omp for nowait
        for(int i = 1; i < size; i++){
            MPI_Recv(image + add, cts, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            add += cts;
        }        
    }
    else{
        MPI_Send(image, cts, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if(rank != 0) return 0;
    
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    int tmp;
    for (int y = 0; y < height; y++) {
        memset(row, 0, row_size);
        tmp = height - 1 - y;
        tmp = (tmp % size * ct + tmp / size) * width; 
        #pragma omp for nowait
        for (int x = 0; x < width; x++) {
            int p = image[tmp + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    png_write_end(png_ptr, NULL);
    fclose(fp);
}
    

