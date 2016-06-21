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
#include <string.h>
#include <fftw3.h>
#include <stdlib.h>


fftw_complex *alloc_complex(size_t size) {

  fftw_complex *ret = fftw_alloc_complex(size);

  if(!ret) abort();

  return ret;
}


/*
 * Execute scalar product between the two vectors.
 */
static fftw_complex scalar_product(size_t size, fftw_complex *xvec, fftw_complex *yvec) {

    fftw_complex ret = 0.0 + I*0.0;

    while(size--) ret += *(xvec++) * *(yvec++);

    return ret;
}


/*
 * Multiply a toeplitz matrix with a vector (convolution).
 * IMP - The t_components have to be given in reverse order and the output is in
 * reverse order.
 */
static void toeplitz_mult(size_t size, fftw_complex *t_components, fftw_complex *vec,
                          fftw_complex *result) {
    size_t dsize = size;

    while(dsize--) *result++ = scalar_product(size, t_components++, vec);
}


/*
 * Set vector's memory to zero.
 */
static void zero_memory(size_t size, fftw_complex *vec) {

    while(size--) *vec++ = 0.0 + I*0.0;
}


/*
 * Read vec in a newly allocated buffer in reverse order.
 */
static fftw_complex *alloc_reverse_from_vec(size_t size, const fftw_complex *vec) {

    fftw_complex *ret = alloc_complex(size);
    fftw_complex *tmp = ret;
    zero_memory(size, ret);

    while(size--) *tmp++ = vec[size + 1];

    return ret;
}


/*
 * Simply reverse a vector.
 */
static void reverse_vec(size_t size, fftw_complex *vec) {

    for(int i=0; i < size / 2; i++) {
        fftw_complex temp = vec[i];
        vec[i] = vec[size - i - 1];
        vec[size - i - 1] = temp;
    }
}


/*
 * Invert a toeplitz matrix. Input is the first column of the matrix and also
 * is the output.
 * IMP - size must be a power of two.
 */
void serial_matrix_inversion(size_t size, const fftw_complex *input, fftw_complex *result) {

    if(size < 1) return;

    //base case
    result[0] = 1 / input[0];
    if(size < 2) return;
    result[1] = - result[0] * input[1] * result[0];


    unsigned int current_size = 2;

    while(current_size < size) {

        fftw_complex *reversed_input = alloc_reverse_from_vec(current_size * 2 - 1, input);

        fftw_complex *temporary = alloc_complex(current_size);


        //first convolution
        toeplitz_mult(current_size, reversed_input, result, temporary);


        fftw_complex *reversed_inverted_pad = alloc_complex(current_size * 2 - 1);
        zero_memory(current_size * 2 - 1, reversed_inverted_pad);

        memcpy(reversed_inverted_pad + (current_size-1), result,
               current_size * sizeof(fftw_complex));

        reverse_vec(current_size * 2 - 1, reversed_inverted_pad);


        //second convolution
        reverse_vec(current_size, temporary);
        toeplitz_mult(current_size, reversed_inverted_pad, temporary, reversed_input);

        //complement sign
        for(int i=0; i < current_size; i++)
            reversed_input[i] = - reversed_input[i];

        reverse_vec(current_size, reversed_input);

        //finally copy in result vec
        memcpy(result + current_size, reversed_input, current_size * sizeof(fftw_complex));


        fftw_free(temporary);
        fftw_free(reversed_input);
        fftw_free(reversed_inverted_pad);
        current_size *= 2;
    }
}
