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
#include <fftw3.h>

#include "mem-util.h"

void initComm (MPI_Comm* comm, int* rank, int* size) {
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);

	MPI_Comm_split(MPI_COMM_WORLD, 0, *rank, comm);

	MPI_Comm_rank(*comm, rank);
}

void shift2right (fftw_complex *r, const int dataSize, MPI_Comm comm, int rank, int size) {
	if (dataSize > 1) {
		fftw_complex newR[2];

		if (rank == 2*size/3-1) {
			MPI_Request request;

			MPI_Isend(&(r[dataSize-2]), 2, MPI_C_DOUBLE_COMPLEX, rank+1, 0, comm, &(request));

			MPI_Wait(&request, MPI_STATUS_IGNORE);
		}

		if (rank > 2*size/3-1 && rank < size-1) {
			MPI_Request requests[2];

			MPI_Isend(&(r[dataSize-2]), 2, MPI_C_DOUBLE_COMPLEX, rank+1, 0, comm, &(requests[0]));
			MPI_Irecv(newR, 2, MPI_C_DOUBLE_COMPLEX, rank-1, MPI_ANY_TAG, comm, &(requests[1]));

			MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

			int i = dataSize - 1;
			while (i > 1) {
				r[i] = r[i-2];
				i --;
			}

			while (i > -1) {
				r[i] = newR[i];
				i --;
			}
		}

		if (rank == size-1) {
			MPI_Request request;

			MPI_Irecv(newR, 2, MPI_C_DOUBLE_COMPLEX, rank-1, MPI_ANY_TAG, comm, &(request));

			MPI_Wait(&request, MPI_STATUS_IGNORE);

			int i = dataSize - 1;
			while (i > 1) {
				r[i] = r[i-2];
				i --;
			}

			while (i > -1) {
				r[i] = newR[i];
				i --;
			}
		}
	} else {
		fftw_complex newR[1];

		if (rank == 2*size/3-2 || rank == 2*size/3-1) {
			MPI_Request request;

			MPI_Isend(r, 1, MPI_C_DOUBLE_COMPLEX, rank+2, 0, comm, &(request));

			MPI_Wait(&request, MPI_STATUS_IGNORE);
		}

		if (rank > 2*size/3-1 && rank < size-2) {
			MPI_Request requests[2];

			MPI_Isend(r, 1, MPI_C_DOUBLE_COMPLEX, rank+2, 0, comm, &(requests[0]));
			MPI_Irecv(newR, 1, MPI_C_DOUBLE_COMPLEX, rank-2, MPI_ANY_TAG, comm, &(requests[1]));

			MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

			r[0] = newR[0];
		}

		if (rank > size-3 && rank < size) {
			MPI_Request request;

			MPI_Irecv(newR, 1, MPI_C_DOUBLE_COMPLEX, rank-2, MPI_ANY_TAG, comm, &(request));

			MPI_Wait(&request, MPI_STATUS_IGNORE);

			r[0] = newR[0];
		}
	}
}

void mergeData (fftw_complex *t1, fftw_complex *t2, fftw_complex *r, const int dataSize, MPI_Comm comm, int rank, int size) {
	
	//allocate memory on data seg
	fftw_complex *newT1 = mu_alloc_complex(dataSize);
	fftw_complex *newT2 = mu_alloc_complex(dataSize);
	fftw_complex *newR = mu_alloc_complex(dataSize);
	
	int dest, i = 0;

	//take care of the case we have only 3 processes
	if(size == 3) {
		if(rank == 0) {
			 MPI_Request requests[2];

 	  		 MPI_Isend(t2, dataSize, MPI_C_DOUBLE_COMPLEX, 1, 0, comm, &(requests[0]));
   	 		 MPI_Irecv(newT1, dataSize, MPI_C_DOUBLE_COMPLEX, 2, MPI_ANY_TAG, comm, &(requests[1]));

   			 MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

			 int i = 0;
    			 while (i < dataSize) {
       				t1[dataSize+i] = t1[i];
       				t1[i] = newT1[i];
       				i ++;
    			 }
		}
		else if(rank == 1) {
			MPI_Request request;

 		   	MPI_Irecv(newT2, dataSize, MPI_C_DOUBLE_COMPLEX, 0, MPI_ANY_TAG, comm, &request);

 			MPI_Wait(&request, MPI_STATUS_IGNORE);

				int i = 0;
    			while (i < dataSize) {
       				t2[dataSize+i] = t2[i];
       				t2[i] = newT2[i];
       				i ++;
    			}
		}
		else {
			MPI_Request request;

    			MPI_Isend(t1, dataSize, MPI_C_DOUBLE_COMPLEX, 0, 0, comm, &request);

    			MPI_Wait(&request, MPI_STATUS_IGNORE);
		}
	
		return;
	}

	if (rank < size/3) {
		if (rank % 2 == 0) {
			MPI_Request requests[2];
			dest = rank + 1;

			MPI_Isend(t1, dataSize, MPI_C_DOUBLE_COMPLEX, dest, 0, comm, &(requests[0]));
			MPI_Irecv(newT2, dataSize, MPI_C_DOUBLE_COMPLEX, dest, MPI_ANY_TAG, comm, &(requests[1]));

			MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

			while (i < dataSize) {
				t2[dataSize+i] = newT2[i];
				i ++;
			}
		} else {
			MPI_Request requests[2];
			dest = rank - 1;

			MPI_Isend(t2, dataSize, MPI_C_DOUBLE_COMPLEX, dest, 0, comm, &(requests[0]));
			MPI_Irecv(newT1, dataSize, MPI_C_DOUBLE_COMPLEX, dest, MPI_ANY_TAG, comm, &(requests[1]));

			MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

			while (i < dataSize) {
				t1[dataSize+i] = t1[i];
				t1[i] = newT1[i];
				i ++;
			}
		}
	} else {
		if (rank < 2*size/3) {
			if (rank % 2 == 0) {
				MPI_Request request;
				dest = rank + 1;

				MPI_Isend(t2, dataSize, MPI_C_DOUBLE_COMPLEX, dest, 0, comm, &request);

				MPI_Wait(&request, MPI_STATUS_IGNORE);
			} else {
				MPI_Request request;
				dest = rank - 1;

				MPI_Irecv(newT2, dataSize, MPI_C_DOUBLE_COMPLEX, dest, MPI_ANY_TAG, comm, &request);

				MPI_Wait(&request, MPI_STATUS_IGNORE);

				while (i < dataSize) {
					t2[dataSize+i] = t2[i];
					t2[i] = newT2[i];
					i ++;
				}
			}
		} else {
			if (rank % 2 == 0) {
				MPI_Request request;
				dest = rank + 1;

				MPI_Irecv(newR, dataSize, MPI_C_DOUBLE_COMPLEX, dest, MPI_ANY_TAG, comm, &request);

				MPI_Wait(&request, MPI_STATUS_IGNORE);

				while (i < dataSize) {
					t1[i] = r[i];
					t1[dataSize+i] = newR[i];
					i ++;
				}
			} else {
				MPI_Request request;
				dest = rank - 1;

				MPI_Isend(r, dataSize, MPI_C_DOUBLE_COMPLEX, dest, 0, comm, &request);

				MPI_Wait(&request, MPI_STATUS_IGNORE);
			}
		}
	}

	fftw_free(newT1);
	fftw_free(newT2);
	fftw_free(newR);
}

void rearrangeComm (MPI_Comm* comm, int* rank, int size) {
	
	//take care of the case we have 3 processes
	if(size == 3) return; //we do not modify comm

	int newRank;
	MPI_Comm newComm;

	if (*rank % 2 == 0)
		newRank = ((*rank) + size/3 - ((*rank) % (size/3))/2) % size;
	else
		newRank = ((*rank) + size/6 - (((*rank) % (size/3)) + 1)/2) % size;

	MPI_Comm_split(*comm, 0, newRank, &newComm);
	MPI_Comm_free(comm);

	*comm = newComm;
	MPI_Comm_rank(*comm, rank);
}

void mergeSingleDataWithNewProc (fftw_complex *t1, fftw_complex *t2, fftw_complex *r, MPI_Comm* comm, int* rank, int usedSize) {
	int newRank;
	MPI_Comm newComm;

	if (*rank < usedSize/3) {
		MPI_Request request;
		MPI_Isend(t2, 1, MPI_DOUBLE, (*rank) + usedSize, 0, *comm, &request);
		MPI_Wait(&request, MPI_STATUS_IGNORE);

		newRank = (*rank) + usedSize/3;
	}

	if (*rank >= usedSize/3 && *rank < 2*usedSize/3) {
		newRank = (*rank) + 2*usedSize/3;
	}

	if (*rank >= 2*usedSize/3 && *rank < usedSize) {
		t1[0] = r[0];

		newRank = (*rank) - 2*usedSize/3;
	}

	if (*rank >= usedSize && *rank < usedSize + usedSize/3) {
		double newT2[1];
		MPI_Request request;
		MPI_Irecv(newT2, 1, MPI_DOUBLE, (*rank) - usedSize, MPI_ANY_TAG, *comm, &request);
		MPI_Wait(&request, MPI_STATUS_IGNORE);

		t2[0] = newT2[0];

		newRank = (*rank) - usedSize/3;
	}

	if (*rank >= usedSize + usedSize/3) {
		newRank = *rank;
	}

	MPI_Comm_split(*comm, 0, newRank, &newComm);
	MPI_Comm_free(comm);

	*comm = newComm;
	MPI_Comm_rank(*comm, rank);
}
