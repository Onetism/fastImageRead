/*
 * @Author: your name
 * @Date: 2022-03-31 23:36:23
 * @LastEditTime: 2022-04-05 09:32:55
 * @LastEditors: Onetism_SU
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \DiffImages\diffcuda.h
 */
#ifndef __DIFFCUDA_H__
#define __DIFFCUDA_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

void predictor7_tiles_GPU(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth);
void predictor1_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void predictor2_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void predictor3_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void predictor4_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void predictor5_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void predictor6_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void predictor7_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream);
void unpredictor7_tiles(const int8_t* in, uint8_t* out, int width, int height, int depth);

void symbolize_GPU(uint8_t* dpSymbols, const int8_t* dpData, uint32_t width, uint32_t height, uint32_t depth);
void symbolize_GPU_Stream(uint8_t* dpSymbols, const int8_t* dpData, uint32_t width, uint32_t height, uint32_t depth, cudaStream_t* stream);
void unsymbolize_GPU(int8_t* dpData, const uint8_t* dpSymbols, uint32_t width, uint32_t height, uint32_t depth);
#endif