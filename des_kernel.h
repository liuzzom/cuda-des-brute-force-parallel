#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "cuda_utils.h"
#include "bit_utils.h"
#include "c_utils.h"
#include "des.h"

__global__ void cuda_des_encode_block(uint64_t block, uint64_t key,
		uint64_t *encoded);

void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result);


__global__ void cuda_des_encode_block(uint64_t block, uint64_t key,
		uint64_t *encoded) {
	uint64_t keys[16];
	des_create_subkeys(key, keys);
	uint64_t result = des_encode_block(block, keys);
	*encoded = result;
}

void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result) {
	uint64_t *dev_result;
	_cudaMalloc((void**) &dev_result, sizeof(uint64_t));

	cuda_des_encode_block<<<1, 1>>>(block, key, dev_result);
	_cudaDeviceSynchronize("cuda_des_encode_block");

	_cudaMemcpy(result, dev_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(dev_result);
}
