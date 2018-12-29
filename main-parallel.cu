/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <time.h>
#include <iostream>
#define WS 8 // word size
#define CN 62 // characters number
#define DICT_SIZE 285465 // dictionary size
#define THREADS_NUM 512 // number of threads for each block


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "c_utils.h"
#include "des.h"
#include "des_utils.h"
#include "bit_utils.h"
#include "des_consts.h"
#include "des_kernel.h"
#include "cuda_utils.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *,
    cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char *file, unsigned line,
    const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
        << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

// print an error message and terminate the program
void error(char *message){
	puts(message);
	exit(1);
}

// convert a string of 8 chars into an uint64_t
__device__ __host__ uint64_t strtouint64(char *string){
	uint64_t uword = 0;
	for(int i = 0; i < 8; i++){
		uint8_t uchar = (uint8_t) (int) string[i];
		uword += uchar;
		if(i < 7){
			uword<<=8;
		}
	}
	return uword;
}

__constant__ uint64_t c_target; // constant memory copy of target
__constant__ char c_charset[CN]; // constant memory copy of charset

// dictionary kernel
__global__ void dict_kernel(uint64_t *dictionary, uint64_t *result){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// check if the thread has some work to do
	if(index < DICT_SIZE){
		uint64_t word = dictionary[index];
		uint64_t hash_word = full_des_encode_block(word, word);
		if(hash_word == c_target){ // the thread found the solution
			*result = word;
			return;
		}
	}
}

__global__ void matrix_kernel(uint64_t *result, int offset){
	// thread coordinates
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx.x = 0 ix defines the column in the matrix
	int iy = (blockIdx.y * blockDim.y + threadIdx.y) + (offset * 16); // blockIdx.y = 0 iy defines the row in the matrix

	// shared control flag
	__shared__ int found;
	found = 0;

	__syncthreads();

	// check if the thread has some work to do
	if(ix < CN && iy < CN){
		// generated word
		char word[WS];
		word[0] = c_charset[iy]; // stays in registers
		word[1] = c_charset[ix]; // stays in registers

		for(int a = 0; a < CN; a++){
			for(int b = 0; b < CN; b++){
				for(int c = 0; c < CN; c++){
					for(int d = 0; d < CN; d++){
						for(int e = 0; e < CN; e++){
							for(int f = 0; f < CN; f++){
								if(found == 1){
									return;
								}

								word[2] = c_charset[a];
								word[3] = c_charset[b];
								word[4] = c_charset[c];
								word[5] = c_charset[d];
								word[6] = c_charset[e];
								word[7] = c_charset[f];

								// conversion and encryption
								uint64_t uword = strtouint64(word);
								uint64_t hash_word = full_des_encode_block(uword, uword);

								if(hash_word == c_target){
									*result = uword;
									found = 1;
									return;
								}
							}
						}
					}
				}
			}
		}
	}
	return;
}

int main(int argc, char **argv) {
	char *password = (char*)malloc(WS * sizeof(char)); // contains the target
	uint64_t upassword; // uint64_t version of the target
	uint64_t crypted_target; // uint64_t version of the crypted target

	char *curr_word = (char*)malloc(WS * sizeof(char)); // readed from dictionary or generated
	uint64_t uint_curr_word; // uint64_t version of curr_word

	FILE *dictionary; // dictionary
	uint64_t *h_dictionary = (uint64_t *) malloc(DICT_SIZE * sizeof(uint64_t)); // host copy of dictionary
	uint64_t *d_dictionary; // device copy of dictionary

	// charset
	char characters[CN] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
						   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
						   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

	// decryption result
	uint64_t *result = (uint64_t *)malloc(sizeof(uint64_t));
	memset(result, 0, sizeof(uint64_t));
	uint64_t *d_result;

	// password to find
	password = "aaaaaazb";
	// verify if the user inserted eight characters password
	if((int)strlen(password) != 8){
		error("error: insert an eight characters password");
	}
	printf("target:%s\n", password);

	// conversion and encryption
	upassword = strtouint64(password);
	crypted_target = full_des_encode_block(upassword, upassword);
	printf("crypted target:");
	bits_print_grouped(crypted_target, 8, 64);

	// start counting clock cycles
	clock_t start_t = clock();

	puts("\nPhase 1: Try with dictionary");
	// open dictionary file
	puts("opening dictionary...");
	if((dictionary = fopen("/home/mauroliuzzo/cuda-workspace/des-brute-force-sequential/src/dictionary.txt", "r")) == NULL){
		error("error: dictionary not found");
	}
	puts("dictionary opened...");
	puts("");

	// dictionary import and converting
	puts("dictionary import and converting...");
	int i = 0;
	while(!feof(dictionary)){
		// import
		fscanf(dictionary, "%8s", curr_word);
		// conversion
		uint_curr_word = strtouint64(curr_word);
		// insert into the array
		h_dictionary[i] = uint_curr_word;
		i++;
	}
	//closing the file
	fclose(dictionary);
	puts("import/conversion done...");
	puts("");

	// gpu malloc and memset
	puts("gpu malloc and memset...");
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_dictionary, DICT_SIZE * sizeof(uint64_t)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_result, sizeof(uint64_t)));
	CUDA_CHECK_RETURN(cudaMemset(d_result, 0, sizeof(uint64_t)));
	puts("malloc and memset done...");
	puts("");

	//gpu memcpy
	puts("gpu memcpy...");
	CUDA_CHECK_RETURN(cudaMemcpy(d_dictionary, h_dictionary, DICT_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_target, &crypted_target, sizeof(uint64_t)));
	puts("gpu memcpy done...");
	puts("");

	// dictionary kernel launch
	puts("dictionary kernel launch...");
	int block_size = DICT_SIZE/THREADS_NUM + 1;
	dict_kernel<<<block_size, THREADS_NUM>>>(d_dictionary, d_result);

	// copying result
	CUDA_CHECK_RETURN(cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	// check the result
	if(*result != 0x0000000000000000){
		printf("password found:");
		bits_print_grouped(*result, 8, 64);

		// gpu memory deallocation
		CUDA_CHECK_RETURN(cudaFree(d_dictionary));
		CUDA_CHECK_RETURN(cudaFree(d_result));

		// stop counting clock cycles and calculate elapsed time
		clock_t end_t = clock();
		clock_t total_t = (end_t - start_t);
		printf("Elapsed Time:%.3f seconds\n", (double)total_t/((double)CLOCKS_PER_SEC));

		return 0;
	}else{
		puts("password not in dictionary...");
	}

	// gpu memory deallocation
	CUDA_CHECK_RETURN(cudaFree(d_dictionary));

	// Phase 2
	puts("\nPhase2: password generation");

	// copy the charset on device constant memory
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_charset, characters, CN*sizeof(char)));

	// organizing threads
	dim3 block(64, 16);

	// matrix kernel launches
	for(int offset = 0; offset < 4; offset++){
		printf("%d° matrix kernel launch...\n", offset + 1);
		matrix_kernel<<<1, block>>>(d_result, offset);

		// copying result
		CUDA_CHECK_RETURN(cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));

		// check the result
		if(*result != 0x0000000000000000){
			printf("password found:");
			bits_print_grouped(*result, 8, 64);

			// gpu memory deallocation
			CUDA_CHECK_RETURN(cudaFree(d_result));

			// stop counting clock cycles and calculate elapsed time
			clock_t end_t = clock();
			clock_t total_t = (end_t - start_t);
			printf("Elapsed Time:%.3f seconds\n", (double)total_t/((double)CLOCKS_PER_SEC));

			return 0;
		}else{
			printf("password not found in %d matrix kernel...\n", offset + 1);
		}

	}
	return 0;
}
