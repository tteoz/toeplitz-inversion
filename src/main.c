/*
 * Copyright (c) 2015-16 Matteo Sartori mttsrt@gmail.com
 *                       Diego Di Carlo
 *                       Tommaso Padoan
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include <complex.h>
#include <fftw3-mpi.h>
#include <stdlib.h>
#include <time.h>

#include "toeplitz-inversion.h"
#include "mem-util.h"


double get_time_log();


int isPowerOfTwo (unsigned int x) {
    
    return ((x != 0) && !(x & (x - 1)));
}



int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    fftw_mpi_init();


    //test the number of processors
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    if(comm_size % 3 || !isPowerOfTwo(comm_size / 3))
        if(!rank) {
            puts("\n*** The number of processes must be 3 * 2 ^ k *** \n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }


    int cmd_input_size = 8;
    if(argc > 1) cmd_input_size = atoi(argv[1]); 

    //input forming/reading
    const size_t input_size = 1 << cmd_input_size;

    if(!rank) printf("%d\n", input_size);

    if(!rank) printf("%d\n", comm_size);

    fftw_complex *input_data = mu_alloc_complex(input_size);

    for(int i=0; i < input_size; i++)
        input_data[i] =  (i % 10) * 0.1 + 0.1 + I * 0.1;


    clock_t start = clock();

    //call the algorithm
    toeplitz_inversion(input_size, input_data);

    clock_t stop = clock();

    double global_time = 1000.0 * (stop - start) / CLOCKS_PER_SEC; 

    printf("gt %f mt %f\n", global_time, get_time_log());

    fftw_free(input_data);
    fftw_mpi_cleanup();
    MPI_Finalize();
    return 0;
}
