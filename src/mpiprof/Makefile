
all: libmpiprof.a

libmpiprof.a: mpiprof.o
	ar r libmpiprof.a mpiprof.o
	ranlib libmpiprof.a

mpiprof.o: mpiprof.c
	mpcc -bmaxdata:0x80000000 -g -c mpiprof.c -o mpiprof.o

clean:
	rm -f mpiprof.o
	rm -f libmpiprof.a

