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

#ifndef POLYDIV_LOGIC_H
#define POLYDIV_LOGIC_H

void initComm (MPI_Comm* comm, int* rank, int* size);

void shift2right (fftw_complex *r, const int dataSize, MPI_Comm comm, int rank, int size);

void mergeData (fftw_complex *t1, fftw_complex *t2, fftw_complex *r, const int dataSize, MPI_Comm comm, int rank, int size);

void rearrangeComm (MPI_Comm* comm, int* rank, int size);

void mergeSingleDataWithNewProc (fftw_complex *t1, fftw_complex *t2, fftw_complex *r, MPI_Comm* comm, int* rank, int usedSize);

#endif //POLYDIV_LOGIC_H

