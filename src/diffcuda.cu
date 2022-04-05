#include "diffcuda.h"
#include <math.h>
#include <stdio.h>
__device__ inline int16_t getNegativeSign(int16_t val)
{
    return (val >> 15);
}

__device__ inline uint16_t symbolize(int16_t value)
{
    // map >= 0 to even, < 0 to odd
    int16_t absValue = value > 0 ? value : (-1*value);
    return 2 * absValue + getNegativeSign(value);
    // return 2 * abs(value);
}

__device__ inline int16_t unsymbolize(uint16_t symbol)
{
   int16_t negative = symbol % 2;
    // map even to >= 0, odd to < 0
    return (1 - 2 * negative) * ((symbol + negative) / 2);
}

__global__ void _predictor1_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
   }
}

__global__ void _predictor2_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x];
       }
   }
}

__global__ void _predictor3_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x - 1] ;
       }
   }
}

__global__ void _predictor4_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x]
                                       - ((int16_t)in[z * offset + p * y + x - 1] + (int16_t)in[z * offset + p * (y - 1) + x]
                                       -  (int16_t)in[z * offset + p * (y - 1) + x - 1]);
       }
   }
}

__global__ void _predictor5_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x]
                                       - ((int16_t)in[z * offset + p * y + x - 1] + (((int16_t)in[z * offset + p * (y - 1) + x]
                                       - (int16_t)in[z * offset + p * (y - 1) + x - 1])>>1));
       }
   }
}

__global__ void _predictor6_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x]
                                        - ((int16_t)in[z * offset + p * (y - 1) + x] + (((int16_t)in[z * offset + p * y + x - 1]
                                        - (int16_t)in[z * offset + p * (y - 1) + x - 1])>>1));
       }
   }
}

__global__ void _predictor7_tiles(const uint8_t* __restrict__ in,
    int8_t* __restrict__ out, uint64_t width, uint64_t height, uint64_t depth)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint64_t p = width;
    uint64_t offset = width * height;

    if ( x < width && y < height && z < depth)
    {
       
       if (x == 0) 
       {
           if (y > 0) 
           {
               out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * (y - 1) + x]);
           }
           else 
           {
               out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
           }
       }
       else if (y == 0) 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] - (int16_t)in[z * offset + p * y + x - 1];
       }
       else 
       {
           out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x]
                                       - (((int16_t)in[z * offset + p * y + x - 1] + (int16_t)in[z * offset + p * (y - 1) + x])>>1);
       }
   }
}

void predictor7_tiles_GPU(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor7_tiles << <dimGrid, dimBlock >> > (in, out, width, height, depth);

    return;
}

void predictor1_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor1_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

void predictor2_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor2_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

void predictor3_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor3_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

void predictor4_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor4_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

void predictor5_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor5_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

void predictor6_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor6_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

void predictor7_tiles_GPU_Stream(const uint8_t* in, int8_t* out, uint64_t width, uint64_t height, uint64_t depth,cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4);

    uint64_t tilesX = (width + 16 - 1) / 16;
    uint64_t tilesY = (height + 16 - 1) / 16;

    dim3 dimGrid(tilesX, tilesY, (depth + dimBlock.z - 1) / dimBlock.z);

    _predictor7_tiles << <dimGrid, dimBlock, 0, *stream>> > (in, out, width, height, depth);

    return;
}

__global__ void symbolizeKernel(
   uint8_t* __restrict__ pSymbols, const int8_t* __restrict__ pData,
   uint32_t width, uint32_t height, uint32_t depth)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

   if (x >= width || y >= height || z >= depth) return;

   uint32_t indexSrc = x + y * width + z * width * height;
   uint32_t indexDst = x + y * width + z * width * height;

   pSymbols[indexDst] = symbolize(pData[indexSrc]);
//    if(z== 0 && x == 0 && y ==0)
//         printf("%d ", pSymbols[indexDst] );
}

void symbolize_GPU(uint8_t* dpSymbols, const int8_t* dpData, uint32_t width, uint32_t height, uint32_t depth)
{
    dim3 dimBlock(16, 16, 4 );
    uint32_t blockCountX = (width + dimBlock.x - 1) / dimBlock.x;
    uint32_t blockCountY = (height + dimBlock.y - 1) / dimBlock.y;
    uint32_t blockCountZ = (depth + dimBlock.z - 1) / dimBlock.z;
    dim3 dimGrid(blockCountX, blockCountY, blockCountZ);

    symbolizeKernel << <dimGrid, dimBlock >> > (dpSymbols, dpData, width, height, depth);
}

void symbolize_GPU_Stream(uint8_t* dpSymbols, const int8_t* dpData, uint32_t width, uint32_t height, uint32_t depth, cudaStream_t* stream)
{
    dim3 dimBlock(16, 16, 4 );
    uint32_t blockCountX = (width + dimBlock.x - 1) / dimBlock.x;
    uint32_t blockCountY = (height + dimBlock.y - 1) / dimBlock.y;
    uint32_t blockCountZ = (depth + dimBlock.z - 1) / dimBlock.z;
    dim3 dimGrid(blockCountX, blockCountY, blockCountZ);

    symbolizeKernel << <dimGrid, dimBlock, 0, *stream>> > (dpSymbols, dpData, width, height, depth);
}

__global__ void unsymbolizeKernel(
   int8_t* __restrict__ pData, const uint8_t* __restrict__ pSymbols,
   uint32_t width, uint32_t height, uint32_t depth)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

   if (x >= width || y >= height || z >= depth) return;

   uint32_t indexSrc = x + y * width + z * width * height;
   uint32_t indexDst = x + y * width + z * width * height;

   pData[indexDst] = unsymbolize(pSymbols[indexSrc]);
}


void unsymbolize_GPU(int8_t* dpData, const uint8_t* dpSymbols, uint32_t width, uint32_t height, uint32_t depth)
{
    dim3 dimBlock(32, 32, 1);
    uint32_t blockCountX = (width + dimBlock.x - 1) / dimBlock.x;
    uint32_t blockCountY = (height + dimBlock.y - 1) / dimBlock.y;
    uint32_t blockCountZ = (depth + dimBlock.z - 1) / dimBlock.z;
    dim3 dimGrid(blockCountX, blockCountY, blockCountZ);

    unsymbolizeKernel << <dimGrid, dimBlock >> > (dpData, dpSymbols, width, height, depth);
}

void unpredictor7_tiles(const int8_t* in, uint8_t* out, int width, int height, int depth)
{
   int p = width;
   int offset = width * height;
   for (int z = 0; z < depth; z++)
   {
       for (int x = 0; x < width; x++)
       {
           for (int y = 0; y < height; y++)
           {
               if (x == 0) 
               {
                   if (y > 0) 
                   {
                       out[z * offset + p * y + x] = ((int16_t)in[z * offset + p * y + x] + (int16_t)out[z * offset + p * (y - 1) + x]);
                   }
                   else 
                   {
                       out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] ;
                   }
               }
               else if (y == 0) 
               {
                   out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x] + (int16_t)out[z * offset + p * y + x - 1];
               }
               else 
               {
                   out[z * offset + p * y + x] = (int16_t)in[z * offset + p * y + x]
                                               + (((int16_t)out[z * offset + p * y + x - 1] + (int16_t)out[z * offset + p * (y - 1) + x])>>1);
               }
           }
       }
   }
}
