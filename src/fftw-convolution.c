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

#include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <assert.h>
#include <string.h>

#include "mem-util.h"


/*
 * Set vector's memory to zero.
 */
static void zero_memory(size_t size, fftw_complex *vec) {

    while(size--) *vec++ = 0.0 + I*0.0;
}


/*
 * Return the size of the local data needed by fftw_mpi algorithm.
 * This very particular and strict. Further information on fftw.org under
 * 'Data distribution'.
 * It is used to allocate the correct amount of buffer space to perform convolution.
 */
size_t conv_local_size(size_t global_size, MPI_Comm comm,
                       ptrdiff_t *ni, ptrdiff_t *no,
                       ptrdiff_t *i_start, ptrdiff_t *o_start) {

    return fftw_mpi_local_size_1d(global_size, comm,
                                        FFTW_FORWARD, //sign
                                        FFTW_ESTIMATE | FFTW_MPI_SCRAMBLED_OUT,
                                        ni, i_start,
                                        no, o_start);
}


/*
 * This function returns a copy buffer
 * augmented just a little to fit the fftw_mpi local memory constraints. This
 * memory layout is decided by conv_local_size().
 */
static fftw_complex *conv_alloc(size_t size, size_t global_size, MPI_Comm comm) {
    ptrdiff_t ni, no;
    ptrdiff_t i_start, o_start;
    size_t alloc_size;
    int rank;

    MPI_Comm_rank(comm, &rank);

    alloc_size = conv_local_size(global_size, comm, &ni, &no, &i_start, &o_start);

    //check if we have the correct distribution set up
    assert(size == ni);
    assert(ni == no);
    assert(size * rank == i_start);
    assert(i_start == o_start);

    fftw_complex *ret = mu_alloc_complex(alloc_size);
    zero_memory(alloc_size, ret);
    return ret;
}


/*
 * As conv_alloc() but also copy the input buffer in the new one.
 */
static fftw_complex *conv_input_alloc(size_t size, size_t global_size, MPI_Comm comm,
                                      const fftw_complex *data) {

    fftw_complex *local_data = conv_alloc(size, global_size, comm);

    //copy input data in the new memory buffer
    memcpy(local_data, data, size * sizeof(fftw_complex));

    return local_data;
}



/*
 * Multiply element-by-element (C-field product) the two signals. The output
 * signal is returned in the first input buffer.
 */
static void product_signal(size_t size, fftw_complex *xss, fftw_complex *yss) {

    while(size--) { *xss++ *= *yss++; }
}



/*
 * Perform convolution with fftw_mpi.
 * First allocates two slightly bigger buffer, as specifed by fftw_mpi_local_size,
 * then transform and multiply the input as the convolution wants.
 */
void convolution(size_t size, size_t global_size, MPI_Comm comm,
                 const fftw_complex *datax, const fftw_complex *datay, fftw_complex *ret) {

    fftw_complex *local_datax, *local_datay;
    ptrdiff_t ni, no;
    ptrdiff_t i_start, o_start;
    size_t alloc_size = conv_local_size(global_size, comm,
                                        &ni, &no, &i_start, &o_start);

    // we need to allocate a local copy of input buffers because the input
    // sizes needed by fftw_mpi may be a little bigger than 'size'.
    local_datax = conv_input_alloc(size, global_size, comm, datax);
    local_datay = conv_input_alloc(size, global_size, comm, datay);


    // FORWARD, ESTIMATE and SCRAMBLED_OUT
    fftw_plan forwd_plan;
    forwd_plan = fftw_mpi_plan_dft_1d(global_size, local_datax, local_datax, comm,
                                      FFTW_FORWARD,
                                      FFTW_ESTIMATE | FFTW_MPI_SCRAMBLED_OUT);

    fftw_execute(forwd_plan);
    // we can recycle the plan (I HOPE)
    fftw_mpi_execute_dft(forwd_plan, local_datay, local_datay);


    // per-element multiplication
    product_signal(alloc_size, local_datax, local_datay);

    // BACKWARD, ESTIMATE and SCRAMBLED_IN
    fftw_plan backwd_plan;
    backwd_plan = fftw_mpi_plan_dft_1d(global_size, local_datax, local_datax, comm,
                                       FFTW_BACKWARD,
                                       FFTW_ESTIMATE | FFTW_MPI_SCRAMBLED_IN);

    fftw_execute(backwd_plan);

    // copy only the useful part of the output
    memcpy(ret, local_datax, size * sizeof(fftw_complex));


    //scale the output signal
    for(int i = 0; i < size; i++)
        ret[i] /= global_size;


    // clean
    fftw_destroy_plan(forwd_plan);
    fftw_destroy_plan(backwd_plan);
    fftw_free(local_datax);
    fftw_free(local_datay);
}
