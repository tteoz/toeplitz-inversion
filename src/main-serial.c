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
#include <time.h>
#include <stdlib.h>

void serial_matrix_inversion(size_t size, const fftw_complex *input, fftw_complex *result);

fftw_complex *alloc_complex(size_t size);


static void print_complex_vector(size_t size, fftw_complex *vector) {
    for(int i=0; i < size; i++)
        printf("( %f + i %f )\n", creal(vector[i]), cimag(vector[i]));
}


int main(int argc, char *argv[]) {

    int cmd_input_size = 8;
    if(argc > 1) cmd_input_size = atoi(argv[1]);
    
    const size_t input_size = 1 << cmd_input_size;
    
    printf("input_size %d\n", input_size);
    
    fftw_complex *input = alloc_complex(input_size);
    fftw_complex *output = alloc_complex(input_size);

    for(int i=0; i < input_size; i++)
        input[i] = (i % 10) * 0.1 +  0.1 + I*0.1;


    clock_t start = clock();

    serial_matrix_inversion(input_size, input, output);
    
    clock_t stop = clock();

    double global_time = 1000.0 * (stop - start) / CLOCKS_PER_SEC;
    
    printf("time %f\n", global_time);

    return 0;
}
