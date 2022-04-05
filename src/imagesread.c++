/*
 * @LastEditors: Onetism_SU
 * @Author: Onetism_su
 * @LastEditTime: 2022-04-05 18:43:39
 */
// Copyright 2022 test
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "imagesread.h"
#include "diffcuda.h"
#include "opencv2/opencv.hpp"
// #include "omp.h"
#include <regex>
#include <thread>
#include <dirent.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>



/**
 * @description: 构造函数
 * @param {string} fielpath：图片所在路径
 * @return {string} suffix: 图片的后缀(例如 ".jpg", ".png", ......)
 */
images_read::images_read(const char* fielpath, const char* suffix, IMAGE_PREDICTORS_TYPE pType)
{

    filePath = fielpath;
    suffix = suffix;
    predictorType = pType;
    imageFileNames = getAllImagesVector();
    cv::Mat image = cv::imread(imageFileNames.at(0), cv::IMREAD_UNCHANGED);
    width = image.cols;
    height = image.rows;
    depth = image.channels();
    filesNum = imageFileNames.size();
    if (image.type() == CV_8UC3)
    {
        imageType = IMAGE_DATA_TYPE::UINT8_TYPE;
    }
    else if (image.type() == CV_16U)
    {
        imageType = IMAGE_DATA_TYPE::UINT16_TYPE;
    }
    std::cout << "The file need to reading is : " << imageFileNames.size() << std::endl;
}
/**
 * @description: 构造函数，Cython必要
 * @param {*}
 * @return {*}
 */
images_read::images_read()
{

}

/**
 * @description: 析构函数，可有可无
 * @param {*}
 * @return {*}
 */
images_read::~images_read()
{

}

/**
 * @description: 获取图像类型
 * @param {*}
 * @return {IMAGE_DATA_TYPE} ： 图像类型
 */
IMAGE_DATA_TYPE images_read::getImageType()
{
    return imageType;
}

uint64_t  images_read::getWidth()
{
    return width;
}
uint64_t  images_read::getHeight()
{
    return height;
}
uint64_t  images_read::getDepth()
{
    return depth;
}
uint64_t  images_read::getFilesNum()
{
    return filesNum;
}

/**
 * @description: 获取路径下的对应suffix后缀文件的所有绝对路径名称集合
 * @param {*}
 * @return : 绝对路径名称集合
 */
std::vector<std::string> images_read::getAllImagesVector(void)
{
    // std::string string_suffix(suffix);
    // std::string string_filePath(filePath);
    std::vector<std::string> files;
    std::regex regObj(suffix, std::regex::icase); //正则化项

    std::vector<std::string> paths;
    paths.push_back(filePath);

    for(int i = 0; i < paths.size(); i++)
    {
        std::string currPath = paths[i];
        DIR *dp;
        struct dirent *dirp;
        if((dp = opendir(currPath.c_str())) == NULL)
        {
            std::cout << "can not open this file." << std::endl;
            continue;
        }
        while((dirp = readdir(dp)) != NULL)
        {
            if(dirp->d_type == 4)
            {
                if((dirp->d_name)[0] == '.')//排除目录下的 "."和".."
                    continue;
                std::string tmpPath = currPath + dirp->d_name;
                paths.push_back(tmpPath);
            }
            else if(dirp->d_type == 8)
            {
                if(std::regex_search(dirp->d_name, regObj))//名称匹配到后缀则输出
                {
                    std::string fullPath = currPath + dirp->d_name;
                    files.push_back(fullPath);
                }
            }
        }
        closedir(dp);
    }
    return files;
}

/**
 * @description: 读取图像文件的线程
 * @param {void*} imageData：图像数据
 * @param {atomic<uint64_t>} *blockId：线程读取同步值
 * @return {*}
 */
void images_read::imagesReadThread(void* imageData, std::atomic<uint64_t> *blockId)
{    
    std::uint64_t blockId_t;
    cv::Mat image;
    while(true)
    {
        blockId_t = atomic_fetch_add(blockId, (uint64_t)1);
        if(blockId_t >= filesNum)
			break;
        image = cv::imread(imageFileNames.at(blockId_t), cv::IMREAD_UNCHANGED);
        if (image.empty())
        {
            printf("\n %s read failed! \n", imageFileNames.at(blockId_t).c_str());
            return;
        }

        std::uint64_t offset = blockId_t * width * height *3;
        if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
        {
            uint8_t* tempImage = (uint8_t*)imageData + offset;
            for(int i= 0;  i < height; i++)
            {
                for(int j = 0; j < width; j++)
                {
                    tempImage[i * width + j ] = image.at<cv::Vec3b>(i,j)[0];
                    tempImage[i * width + j + width * height] = image.at<cv::Vec3b>(i,j)[1];
                    tempImage[i * width + j + width * height * 2] = image.at<cv::Vec3b>(i,j)[2];
                }
            }
        }
        else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
        {
            uint16_t* tempImage = (uint16_t*)imageData + offset;
            for(int i= 0;  i < height; i++)
            {
                for(int j = 0; j < width; j++)
                {
                    tempImage[i * width + j ] = image.at<cv::Vec3b>(i,j)[0];
                    tempImage[i * width + j + width * height] = image.at<cv::Vec3b>(i,j)[1];
                    tempImage[i * width + j + width * height * 2] = image.at<cv::Vec3b>(i,j)[2];
                }
            }
        }
    }
    return;
}

/**
 * @description: 多线程读取图像文件
 * @param {void*} imageData：图像数据存放处
 * @param {uint16_t} numThreads：使用线程数
 * @return {*}
 */
int images_read::multiThreadReadImages(void* imageData, uint16_t numThreads)
{
    if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();

    std::atomic<uint64_t> blockId; 
    atomic_store(&blockId, (uint64_t)0);

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++)
    {
        threads.push_back(std::thread(&images_read::imagesReadThread, this, imageData, &blockId));
    }
    for (auto& t : threads)
    {
        t.join();
    }
    return 0;
}


void images_read::diffImagesThread(void* imageData, int totalimages, uint64_t gpuSupportNums, std::atomic<uint64_t> *blockId, int gpuNum)
{
    cudaSetDevice(gpuNum);
    int gpu_id = -1;
    cudaGetDevice(&gpu_id);
    printf("CPU thread uses CUDA device %d\n", gpu_id);
    std::uint64_t blockId_t;
    uint8_t* dpImage = nullptr;
	int8_t* dpBuffer = nullptr;
	uint8_t* dpSymbols = nullptr;

    cudaMalloc((void **)&dpImage, (uint64_t)(width * height *  gpuSupportNums * sizeof(uint8_t)));
    cudaMalloc((void **)&dpBuffer, (uint64_t)(width * height *  gpuSupportNums * sizeof(int8_t)));
    cudaMalloc((void **)&dpSymbols, (uint64_t)(width * height *  gpuSupportNums * sizeof(uint8_t)));
    //难道只有这么恶心的办法才能让出线程吗？实际使用发现不主动让出线程的结果就是会出现最后只有一个线程占据CPU的情况。
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const int nStreams = 8;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) 
    {
        cudaStreamCreate(&streams[i]);
    }
    while(true)
    {
        uint64_t copyToDeviceNums = gpuSupportNums;
        blockId_t = atomic_fetch_add(blockId, (uint64_t) 1);
        if(totalimages < blockId_t * gpuSupportNums)
			break;
        else if(totalimages - blockId_t * gpuSupportNums < gpuSupportNums)
            copyToDeviceNums = totalimages - blockId_t * gpuSupportNums;

         std::thread::id this_id = std::this_thread::get_id();
        
        // g_diffImages_mutex.lock();
        std::cout << "thread " << this_id << " working...\n";
        // g_diffImages_mutex.unlock();
        
        std::uint64_t offset = blockId_t * gpuSupportNums * width * height;

        void* tempImage = NULL;
        if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
            tempImage = (uint8_t*)imageData + offset;
        else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
            tempImage = (uint16_t*)imageData + offset;

        uint64_t perStreamSize = (copyToDeviceNums + nStreams - 1) / nStreams;
        for(int i = 0; i < nStreams; i++)
        {
            uint64_t perStreamCopySize = perStreamSize;
            if(gpuSupportNums - i * perStreamSize < perStreamSize)
                perStreamCopySize = gpuSupportNums - i * perStreamSize;

            std::uint64_t streamOffset = i * perStreamSize * width * height;
            void* streamTempImage = NULL;
            if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
                streamTempImage = (uint8_t*)tempImage + streamOffset;
            else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
                streamTempImage = (uint16_t*)tempImage + streamOffset;            
            // uint8_t* streamTempImage = tempImage + streamOffset;
            cudaMemcpyAsync(dpImage, streamTempImage, (uint64_t)(width * height * perStreamSize * sizeof(uint8_t)), cudaMemcpyHostToDevice, streams[i]);
            predictor7_tiles_GPU_Stream(dpImage, dpBuffer, width, height, copyToDeviceNums, &streams[i]);
            symbolize_GPU_Stream(dpSymbols, dpBuffer, width, height, perStreamSize, &streams[i]);
            cudaMemcpyAsync(streamTempImage, dpSymbols, (uint64_t)(width * height * perStreamSize * sizeof(uint8_t)), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        // cudaMemcpy(dpImage, tempImage, (uint64_t)(width * height * copyToDeviceNums * sizeof(uint8_t)), cudaMemcpyHostToDevice);
        // predictor7_tiles_GPU(dpImage, dpBuffer, width, height, copyToDeviceNums);
        // symbolize_GPU(dpSymbols, dpBuffer, width, height, copyToDeviceNums);
        // cudaMemcpy(tempImage, dpSymbols, (uint64_t)(width * height * copyToDeviceNums * sizeof(uint8_t)), cudaMemcpyDeviceToHost);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cudaFree(dpImage);
    cudaFree(dpBuffer);
    cudaFree(dpSymbols);
}

/**
 * @description: 根据GPU个数执行多线程差值计算
 * @param {void*} imageData: 图像数据存储
 * @param {uint16_t} numThreads：读取文件时使用线程数
 * @return {*}
 */
int images_read::getDiffImages(void* imageData, uint16_t numThreads)
{
    if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();
    images_read::multiThreadReadImages(imageData, numThreads);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) 
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int globalMem = deviceProp.totalGlobalMem>>20;
    int imagePerMem = ((width * height * sizeof(uint8_t))>>20)+1;
    uint64_t gpuSupportImages = globalMem / imagePerMem / 3;

    std::atomic<uint64_t> diffBlockId; 
    atomic_store(&diffBlockId, (uint64_t)0);

    std::vector<std::thread> diffThreads;
    for (int i = 0; i < deviceCount; i++)
    {
        diffThreads.push_back(std::thread(&images_read::diffImagesThread, this, imageData , depth * filesNum, gpuSupportImages, &diffBlockId, i));
    }
    for (auto& t : diffThreads)
    {
        t.join();
    }
}

void images_read::imagesWriteThread(void* imageData, const char* path, const char* suffix, std::atomic<uint64_t>* blockId)
{
    std::uint64_t blockId_t;
    
    cv::Mat image;
    if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
        image = cv::Mat(height, width, CV_8UC3);
    else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
        image = cv::Mat(height, width, CV_16UC1);
    while(true)
    {
        blockId_t = atomic_fetch_add(blockId, (uint64_t)1);
        if(blockId_t >= filesNum)
            break;

        std::uint64_t offset = blockId_t * width * height *3;

        if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
        {
            uint8_t* tempImage = (uint8_t*)imageData + offset;
            for(int i= 0;  i < height; i++)
            {
                for(int j = 0; j < width; j++)
                {
                    image.at<cv::Vec3b>(i,j)[0] = tempImage[i * width +j + 0 * width * height];
                    image.at<cv::Vec3b>(i,j)[1] = tempImage[i * width +j + 1 * width * height];
                    image.at<cv::Vec3b>(i,j)[2] = tempImage[i * width +j + 2 * width * height];
                }
            }
        }

        else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
        {
            uint16_t* tempImage = (uint16_t*)imageData + offset;
            for(int i= 0;  i < height; i++)
            {
                for(int j = 0; j < width; j++)
                {
                    image.at<cv::uint16_t>(i,j) = tempImage[i * width +j];
                }
            }
        }

        std::string pathName = imageFileNames.at(blockId_t);
        std::string outPath = pathName.substr(0, pathName.rfind("/")).append("_imagesReadOut");
        if(path != NULL)
            outPath = path;
        std::string fileSuffix = ".png";
        if(suffix != NULL)
            fileSuffix = suffix;
        std::string fileName = pathName.substr(pathName.rfind("/"), pathName.rfind(".") - pathName.rfind("/")).append(fileSuffix);
        DIR *dir;
        if((dir=opendir(outPath.c_str())) == NULL)
        {
            std::string command = "mkdir -p " + outPath;  
            system(command.c_str());
        }
        bool result = cv::imwrite(outPath.append(fileName), image);
        if (result == false)
            std::cout << outPath.append(fileName) << " write failed!" << std::endl;
    }
   
}

void images_read::multiThreadImagesWirte(void* imageData, const char* path, const char* suffix, uint16_t numThreads)
{
    if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();
    std::atomic<uint64_t> wirteBlockId; 
    atomic_store(&wirteBlockId, (uint64_t)0);

    std::vector<std::thread> wirteThreads;
    for (int i = 0; i < numThreads; i++)
    {
        wirteThreads.push_back(std::thread(&images_read::imagesWriteThread, this, imageData , path, suffix, &wirteBlockId));
    }
    for (auto& t : wirteThreads)
    {
        t.join();
    }
    // omp_set_num_threads(numThreads);  // create as many CPU threads as there are CUDA devices
    // #pragma omp parallel
    // {
    //     unsigned int cpu_thread_id = omp_get_thread_num();
    //     unsigned int num_cpu_threads = omp_get_num_threads();

    //     cv::Mat image;
    //     if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
    //         image = cv::Mat(height, width, CV_8UC3);
    //     else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
    //         image = cv::Mat(height, width, CV_16UC1);
    //     while(true)
    //     {
    //         uint32_t blockId_t = atomic_fetch_add(&wirteBlockId, (uint64_t)1);
    //         if(blockId_t >= filesNum)
    //             break;

    //         std::uint64_t offset = blockId_t * width * height *3;

    //         if (imageType == IMAGE_DATA_TYPE::UINT8_TYPE)
    //         {
    //             uint8_t* tempImage = (uint8_t*)imageData + offset;
    //             for(int i= 0;  i < height; i++)
    //             {
    //                 for(int j = 0; j < width; j++)
    //                 {
    //                     image.at<cv::Vec3b>(i,j)[0] = tempImage[i * width +j + 0 * width * height];
    //                     image.at<cv::Vec3b>(i,j)[1] = tempImage[i * width +j + 1 * width * height];
    //                     image.at<cv::Vec3b>(i,j)[2] = tempImage[i * width +j + 2 * width * height];
    //                 }
    //             }
    //         }

    //         else if(imageType == IMAGE_DATA_TYPE::UINT16_TYPE)
    //         {
    //             uint16_t* tempImage = (uint16_t*)imageData + offset;
    //             for(int i= 0;  i < height; i++)
    //             {
    //                 for(int j = 0; j < width; j++)
    //                 {
    //                     image.at<cv::uint16_t>(i,j) = tempImage[i * width +j];
    //                 }
    //             }
    //         }

    //         std::string pathName = imageFileNames.at(blockId_t);
    //         std::string outPath = pathName.substr(0, pathName.rfind("/")).append("_imagesReadOut");
    //         if(path != NULL)
    //             outPath = path;
    //         std::string fileSuffix = ".png";
    //         if(suffix != NULL)
    //             fileSuffix = suffix;
    //         std::string fileName = pathName.substr(pathName.rfind("/"), pathName.rfind(".") - pathName.rfind("/")).append(fileSuffix);
    //         DIR *dir;
    //         if((dir=opendir(outPath.c_str())) == NULL)
    //         {
    //             std::string command = "mkdir -p " + outPath;  
    //             system(command.c_str());
    //         }
    //         bool result = cv::imwrite(outPath.append(fileName), image);
    //         if (result == false)
    //             std::cout << outPath.append(fileName) << " write failed!" << std::endl;
    //     }
    // }
}