#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if(rank == 0) fprintf(stderr, "must provide exactly 2 arguments!\n");
        
        MPI_Finalize();
        return 1;
    }
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
    
    unsigned long long total_pixels;
    if(rank == 0) total_pixels = 0;

    for (unsigned long long x = rank; x < r; x+=size) {
        unsigned long long y = ceil(sqrtl(r*r - x*x));
        pixels += y;
        pixels %= k;
    }

    MPI_Allreduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if(rank == 0) {
	
	printf("%llu\n", (4 * total_pixels) % k);
	}

    MPI_Finalize();
}
