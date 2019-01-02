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
#define WS 8 // word size
#define CN 62 // characters number

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "c_utils.h"
#include "des.h"
#include "des_utils.h"
#include "bit_utils.h"
#include "des_consts.h"
#include "des_kernel.h"
#include "cuda_utils.h"

// print an error message and terminate the program
void error(char *message){
	puts(message);
	exit(1);
}

// convert a string of 8 chars into an uint64_t
uint64_t strtouint64(char *string){
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

int main(int argc, char *argv[]) {
	char *password = (char*)malloc(WS * sizeof(char)); // contains the target
	uint64_t upassword; // uint64_t version of the target
	uint64_t crypted_target; // uint64_t version of the crypted target

	char *curr_word = (char*)malloc(WS * sizeof(char)); // readed from dictionary or generated
	uint64_t uint_curr_word; // uint64_t version of curr_word
	uint64_t hash_word; // hash of curr_word

	FILE *dictionary;

	int a, b, c, d, e, f, g, h;
	char characters[CN] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
						   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
						   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

	password = "aaaabaaa";

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

	// open dictionary file
	puts("opening dictionary...");
	if((dictionary = fopen("/home/mauroliuzzo/cuda-workspace/des-brute-force-sequential/src/dictionary.txt", "r")) == NULL){
		error("error: dictionary not found");
	}
	puts("dictionary opened...");

	// start counting clock cycles
	clock_t start_t = clock();

	// try if a word of dictionary works
	puts("trying with dictionary words. This may takes a while...");
	while(!feof(dictionary)){
		// read a word from dictionary
		fscanf(dictionary, "%8s", curr_word);
		// DES hash
		uint_curr_word = strtouint64(curr_word);
		hash_word = full_des_encode_block(uint_curr_word,uint_curr_word);

		if(hash_word == crypted_target){ // if the password is found
			printf("password found: %s\n", curr_word);
			// close the dictionary
			fclose(dictionary);

			// stop counting clock cycles and calculate elapsed time
			clock_t end_t = clock();
			clock_t total_t = (end_t - start_t);
			printf("Time taken by CPU:%.3f seconds\n", (double)total_t/((double)CLOCKS_PER_SEC));

			return 0;
		}
	}
	// close the dictionary
	fclose(dictionary);

	printf("password not present in our dictionary\n");
	printf("try with brute force. This may takes a long time...\n");

	// brute force: try every combination
	for(a = 0; a < CN; a++){
		for(b = 0; b < CN; b++){
			for(c = 0; c < CN; c++){
				for(d = 0; d < CN; d++){
					for(e = 0; e < CN; e++){
						for(f = 0; f < CN; f++){
							for(g = 0; g < CN; g++){
								for(h = 0; h < CN; h++){
									curr_word[0] = characters[a];
									curr_word[1] = characters[b];
									curr_word[2] = characters[c];
									curr_word[3] = characters[d];
									curr_word[4] = characters[e];
									curr_word[5] = characters[f];
									curr_word[6] = characters[g];
									curr_word[7] = characters[h];

									// DES hash
									uint_curr_word = strtouint64(curr_word);
									hash_word = full_des_encode_block(uint_curr_word,uint_curr_word);

									if(hash_word == crypted_target){ // if the password is found
										printf("password found: %s\n", curr_word);

									// stop counting clock cycles and calculate elapsed time
									clock_t end_t = clock();
									clock_t total_t = (end_t - start_t);
									printf("Time taken by CPU:%.3f seconds\n", (double)total_t/((double)CLOCKS_PER_SEC));

									return 0;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return 0;
}
