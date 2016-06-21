

# polydiv executable
SRC_POLYDIV= main.c fftw-convolution.c toeplitz-inversion.c toeplitz-serial.c logic.c mem-util.c
OBJ_POLYDIV=$(SRC_POLYDIV:.c=.o)
EXEC_POLYDIV=polydiv

# serial executable
SRC_SERIAL= main-serial.c toeplitz-serial.c
OBJ_SERIAL=$(SRC_SERIAL:.c=.o)
EXEC_SERIAL=serial


CC=c99
CFLAGS?=
ALL_FLAGS=-bmaxdata:0x80000000
MPIFLAGS=-I/usr/lpp/ppe.poe/include -binitfini:poe_remote_main -lmpi_r -lvtd_r -I/opt/rsct/lapi/include -llapi_r -lpthreads -L/usr/lpp/ppe.poe/lib/threads -L/usr/lpp/ppe.poe/lib

FFTW_BUILD_DIR=../fftw-build
FFTW_INCLUDE_DIR=$(FFTW_BUILD_DIR)/include
FFTW_LIB_DIR=$(FFTW_BUILD_DIR)/lib

MPI_INCLUDE_DIR=/usr/lpp/ppe.poe/include
MPI_LIB_DIR=/usr/lpp/ppe.poe/lib


all: 
	cd mpiprof && $(MAKE)
	$(MAKE) $(EXEC_POLYDIV) 
	$(MAKE) $(EXEC_SERIAL)


.c.o:
	$(CC) $(ALL_FLAGS) $(CFLAGS) -c -I$(FFTW_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR) $< -o $@

$(EXEC_SERIAL): $(OBJ_SERIAL)
	$(CC) $(ALL_FLAGS) -L$(FFTW_LIB_DIR) -lm -lfftw3 $(OBJ_SERIAL) -o $(EXEC_SERIAL)

$(EXEC_POLYDIV): $(OBJ_POLYDIV)
	$(CC) $(ALL_FLAGS) -L$(FFTW_LIB_DIR) -lfftw3 -lfftw3_mpi -lm -L./mpiprof -lmpiprof $(MPIFLAGS) $(OBJ_POLYDIV) -o $(EXEC_POLYDIV)


clean:
	cd mpiprof && $(MAKE) clean
	rm -f *.o
	rm -f $(EXEC_POLYDIV)
	rm -f $(EXEC_SERIAL)

