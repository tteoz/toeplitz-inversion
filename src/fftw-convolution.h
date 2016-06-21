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

#ifndef POLYDIV_FFTW_CONVOLUTION_H
#define POLYDIV_FFTW_CONVOLUTION_H

#include <mpi.h>
#include <complex.h>
#include <fftw3.h>


/*
 * Return the size of the local data needed by fftw_mpi algorithm.
 * This very particular and strict. Further information on fftw.org under
 * 'Data distribution'.
 * It is used to allocate the correct amount of buffer space to perform convolution.
 */
size_t conv_local_size(size_t global_size, MPI_Comm comm,
                       ptrdiff_t *ni, ptrdiff_t *no,
                       ptrdiff_t *i_start, ptrdiff_t *o_start);


/*
 * Perform convolution with fftw_mpi.
 * First allocates two slightly bigger buffer, as specifed by fftw_mpi_local_size,
 * then transform and multiply the input as the convolution wants.
 */
void convolution(size_t size, size_t global_size, MPI_Comm comm,
                 const fftw_complex *datax, const fftw_complex *datay, fftw_complex *ret);


#endif //POLYDIV_FFTW_CONVOLUTION_H
