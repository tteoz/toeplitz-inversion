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
#include <fftw3.h>
#include <mpi.h>
#include <string.h>

#include "fftw-convolution.h"
#include "logic.h"
#include "mem-util.h"


typedef unsigned int uint;


void serial_matrix_inversion(size_t size, const fftw_complex *input, fftw_complex *result);



static void print_complex_vector(size_t size, fftw_complex *vector) {
    return;
    for(int i=0; i < size; i++)
        printf("( %f + i %f )\n", creal(vector[i]), cimag(vector[i]));
}


/*
 * Simply reverse a vector.
 */
static void reverse_vector(size_t size, fftw_complex *vec) {

    for(int i=0; i < size / 2; i++) {
        fftw_complex temp = vec[i];
        vec[i] = vec[size - i - 1];
        vec[size - i - 1] = temp;
    }
}


/*
 * Set vector's memory to zero.
 */
static void zero_memory(size_t size, fftw_complex *vec) {

    while(size--) *vec++ = 0.0 + I*0.0;
}


/*
 * Util to access communicator info.
 */
static void get_comm_info(MPI_Comm comm, int *rank, int *size) {
    MPI_Comm_rank(comm, rank);
    MPI_Comm_size(comm, size);
}


/*
 * Read input for the corresponding process, at the initial phase, prior to run
 * the parallel algorithm.
 */
static fftw_complex *first_input_setup(size_t size, MPI_Comm comm,
                                       const fftw_complex *input) {

    int rank, comm_size;
    get_comm_info(MPI_COMM_WORLD, &rank, &comm_size);

    size_t sub_size = size * (comm_size * 2 / 3);

    fftw_complex *output = mu_alloc_complex(size);
    zero_memory(size, output);

    if(rank < comm_size * 2 / 3) {

        int start_index = sub_size - (rank*size) - 1;

        //read specific input in reverse order
        for(int i=0; i < size; i++) {

            if(rank == comm_size * 2 / 3 - 1 && i == size - 1) {
                //the last element of this process must be set to zero
                output[i] = 0.0 + I*0.0;
            }
            else {
                output[i] = input[start_index - i];
            }
        }
    }

    return output;
}


/*
 * At each recursion step, some processes have to read and prepare a newer part
 * of the input for the next step.
 */
static void input_setup(size_t size, MPI_Comm comm, size_t input_size,
                        const fftw_complex *input, fftw_complex *ret) {

    int rank, comm_size;
    get_comm_info(comm, &rank, &comm_size);


    if(rank >= 2 * comm_size / 3)
        zero_memory(size, ret);

    else if(rank < comm_size / 3) {

        const size_t start_index = size * 2 * comm_size / 3 - size;

        if(start_index + size > input_size)
            zero_memory(size, ret);

        else {
            memcpy(ret, input + start_index - rank * size, size * sizeof(fftw_complex));
            reverse_vector(size, ret);
        }
    }
}


/*
 * Set up memory for the second convolution step. In particular, some regions
 * must be set to zero.
 */
static void input_setup_2nd_conv(size_t size, MPI_Comm comm, fftw_complex *vec) {

    int rank, comm_size;
    get_comm_info(comm, &rank, &comm_size);

    if(rank == comm_size / 3 - 1)
        zero_memory(size - 1, vec);
        //we keep the last element because it's part of the next stage input

    else if(rank == 2 * comm_size / 3 - 1)
        vec[size - 1] = 0.0 + I*0.0;
        //instead, we have to null this

    else if(rank < comm_size / 3 || rank >= 2 * comm_size / 3)
        zero_memory(size, vec);
}


/*
 * This calculates the smallest integer sizes that permits the input to be
 * equally divided among all the processes participating in the communicator.
 */
static uint check_how_much_serial(MPI_Comm comm) {
    ptrdiff_t ni, no;
    ptrdiff_t i_start, o_start;

    int num_proc;
    MPI_Comm_size(comm, &num_proc);

    size_t size = 3;

    do {
        size *= 2;
        conv_local_size(size, comm, &ni, &no, &i_start, &o_start);
    } while ( ni * num_proc != size );

    return size;
}


/*
 * This is used to solve the base case of the recursive algorithm, in serial
 * mode for now, prior to go for parallel computation. This is necessary
 * because the way fftw_mpi needs its input distributed among processors.
 *
 * We compute the inversion for the smallest size permitted by fftw, then
 * distribute the result among 1/3 of the processors so everything
 * is prepared for the parallel algorithm to start.
 */
static fftw_complex *serial_compute(size_t size, MPI_Comm comm, const fftw_complex *input) {

    int rank, comm_size;
    get_comm_info(comm, &rank, &comm_size);

    //only the first process has to execute the serial part
    //then it'll send result to the others
    const int output_size = size / (comm_size / 3);

    if(rank == 0) {

        fftw_complex *result = mu_alloc_complex(size);

        //serial computation of toeplitz matrix inversion
        serial_matrix_inversion(size, input, result);


        //reverse the vector because the order we maintain convolution input
        reverse_vector(size, result);

        //now send to other processes
        //useful result to the first comm_size / 3 processes
        for(int i=1; i < comm_size / 3; i++)
            MPI_Send(result + i * output_size, output_size, MPI_C_DOUBLE_COMPLEX, i, 0, comm);


        //zero vectors to the others
        fftw_complex *zero_res = mu_alloc_complex(output_size);
        zero_memory(output_size, zero_res);

        for(int i=comm_size / 3; i < comm_size; i++)
            MPI_Send(zero_res, output_size, MPI_C_DOUBLE_COMPLEX, i, 0, comm);


        fftw_complex *output = mu_alloc_complex(output_size);
        memcpy(output, result, output_size * sizeof(fftw_complex));

        fftw_free(result);
        fftw_free(zero_res);
        return output;

    } else {

        fftw_complex *output = mu_alloc_complex(output_size);

        MPI_Recv(output, output_size, MPI_C_DOUBLE_COMPLEX, 0, 0, comm, MPI_STATUS_IGNORE);
        return output;
    }
}


/*
 * Perform inversion of a Toeplitz matrix whose coefficients are represented
 * by the vector 'input' of size 'input_size'.
 * This follows the recursive algorithm which incrementally invert a sub part
 * of the input matrix and doubles its size at each step.
 *
 * The convolution needed to compose new inverted part of the matrix is performed
 * in parallel with fttw_mpi.
 *
 * IMP - Because of the way fftw_mpi requires the input distributed across
 * processes it is necessary to perform a preliminary step in which the inverted
 * coefficients are computed in serial mode, distributed among the processes, to
 * be ready for the parallel machinery to start on.
 */
void toeplitz_inversion(const size_t input_size, const fftw_complex *input) {

    int rank, comm_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    initComm(&comm, &rank, &comm_size);


    if(input_size & (input_size - 1))
        //we have input not power of two
        if(!rank) {
            puts("Input not power of two");
            MPI_Abort(comm, 0);
        }


    size_t start_size = check_how_much_serial(comm) / 3;
        //divided by 3 because for the convolution we have input * 3
    // if(! rank) printf("start_size %zu\n", start_size);

    if(input_size < start_size)
        if(!rank) {
            puts("\n*** Not sufficient input size to start parallel execution ***");
            printf("start_size (%zu) > input_size (%zu)\n", start_size, input_size);
            MPI_Abort(comm, 0);
        }


    fftw_complex *current_inverted = serial_compute(start_size, comm, input);

    size_t local_size = start_size / (comm_size / 3);


    if(input_size < local_size * 2 * comm_size / 3)
        return;


    //input reading and memory setup
    fftw_complex *current_input = first_input_setup(local_size, comm, input);


    //recursion cycle
    while(local_size * 2 * comm_size / 3 <= input_size) {

        //call first convolution
        fftw_complex *conv_output = mu_alloc_complex(local_size);
        convolution(local_size, local_size * comm_size, comm,
                    current_inverted, current_input, conv_output);


        //prepare memory for the 2nd convolution
        input_setup_2nd_conv(local_size, comm, conv_output);


        //2nd convolution
        fftw_complex *conv_2n_out = mu_alloc_complex(local_size);
        convolution(local_size, local_size * comm_size, comm,
                    current_inverted, conv_output, conv_2n_out);


        //shift results to right
        shift2right(conv_2n_out, local_size, comm, rank, comm_size);

        //and complement sign of the output
        if(rank >= 2 * comm_size / 3) {
            for (int i = 0; i < local_size; i++)
                conv_2n_out[i] = - conv_2n_out[i];
        }


        //merge results and form a new communicator
        fftw_complex *temp = mu_alloc_complex(local_size * 2);
        memcpy(temp, current_inverted, local_size * sizeof(fftw_complex));
        fftw_free(current_inverted);
        current_inverted = temp;

        temp = mu_alloc_complex(local_size * 2);
        memcpy(temp, current_input, local_size * sizeof(fftw_complex));
        fftw_free(current_input);
        current_input = temp;

        mergeData(current_inverted, current_input, conv_2n_out, local_size,
                  comm, rank, comm_size);
        rearrangeComm(&comm, &rank, comm_size);


        //from now we double the size of the data
        local_size *= 2;


        //after the comm split we set to zero proper memory area
        if(rank >= comm_size / 3)
            zero_memory(local_size, current_inverted);


        //setup input for next stage
        input_setup(local_size, comm, input_size, input, current_input);


        // if(! rank) printf("recursion step, size %zu\n", local_size * comm_size / 3);
        print_complex_vector(local_size, current_inverted);

        //puts("current_input");
        print_complex_vector(local_size, current_input);


        //clean
        fftw_free(conv_output);
        fftw_free(conv_2n_out);
    }
}
