#include <mpi.h>
#include <time.h>


/* Used to account execution time in a given variable */
#define TIME(proc, time, ret) \
	{ \
	clock_t start = clock(); \
	ret = proc; \
	clock_t stop = clock(); \
	time += 1000.0 * (stop - start) / CLOCKS_PER_SEC; \
	}


/* Global to account time spent in MPI calls */
static double time_log;

/* Access to time_log */
double get_time_log() { return time_log; }



/* These are MPI calls to be intercepted following MPI standard mechanism
   'nameshift profiling' 
*/

int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {

  int ret;
  TIME(PMPI_Send(buf, count, datatype, dest, tag, comm), time_log, ret)
  return ret;
}


int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {

  int ret;
  TIME(PMPI_Recv(buf, count, datatype, source, tag, comm, status), time_log, ret)
  return ret;
}


int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
    int tag, MPI_Comm comm, MPI_Request *request) {

  int ret;
  TIME(PMPI_Isend(buf, count, datatype, dest, tag, comm, request), time_log, ret)
  return ret;
}


int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request *request) {

  int ret;
  TIME(PMPI_Irecv(buf, count, datatype, source, tag, comm, request), time_log, ret)
  return ret;
}


int MPI_Wait(MPI_Request *request, MPI_Status *status) {

  int ret;
  TIME(PMPI_Wait(request, status), time_log, ret)
  return ret;
}


int MPI_Waitall(int count, MPI_Request array_of_requests[],
    MPI_Status *array_of_statuses) {
  
  int ret; 
  TIME(PMPI_Waitall(count, array_of_requests, array_of_statuses), time_log, ret)
  return ret;
}


int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
    int dest, int sendtag, void *recvbuf, int recvcount,
    MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status *status) {

  int ret;
  TIME(PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
		recvtag, comm, status), time_log, ret)
  return ret;
}


int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
 
  int ret;
  TIME(PMPI_Bcast(buffer, count, datatype, root, comm), time_log, ret)
  return ret;
}


int MPI_Alltoall(void *sendbuf, int sendcount,
    MPI_Datatype sendtype, void *recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm) {

  int ret; 
  TIME(PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm), time_log, ret)
  return ret;
}


int MPI_Alltoallv(void *sendbuf, int sendcounts[],
    int sdispls[], MPI_Datatype sendtype,
    void *recvbuf, int recvcounts[],
    int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {

  int ret;
  TIME(PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm), time_log, ret)
  return ret;
}


int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

  int ret; 
  TIME(PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm), time_log, ret)
  return ret;
}

