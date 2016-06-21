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


/* Wrapper around fftw_alloc_complex(). Just to ease exception handling in allocation */
fftw_complex *mu_alloc_complex(size_t size) {

	fftw_complex *ret = fftw_alloc_complex(size);
	
	if(!ret) {
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		printf("process %d Not possible to allocate memory\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	return ret;
}
