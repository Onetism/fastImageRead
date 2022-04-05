/*
 * @LastEditors: Onetism_SU
 * @Author: Onetism_su
 * @LastEditTime: 2022-04-04 20:03:38
 */
#include "diffcuda.h"
#include <string>
#include <regex>
#include <vector>
#include <dirent.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "opencv2/opencv.hpp"

std::mutex g_cudamalloc_mutex;


/**
 * @description: 获取路径下的对应sffix后缀文件的所有绝对路径名称集合
 * @param {string} path : 文件所在目录
 * @param {string} suffix : 获取文件的后缀名(例如 jpg, png)
 * @return {vector} 所有绝对路径名称集合
 */
std::vector<std::string> get_all_files(std::string path, std::string suffix)
{
    std::vector<std::string> files;
    std::regex reg_obj(suffix, std::regex::icase);

    std::vector<std::string> paths;
    paths.push_back(path);

    for(int i = 0; i < paths.size(); i++)
    {
        std::string curr_path = paths[i];
        DIR *dp;
        struct dirent *dirp;
        if((dp = opendir(curr_path.c_str())) == NULL)
        {
            std::cout << "can not open this file." << std::endl;
            continue;
        }
        while((dirp = readdir(dp)) != NULL)
        {
            if(dirp->d_type == 4)
            {
                if((dirp->d_name)[0] == '.')
                    continue;
                std::string tmp_path = curr_path + dirp->d_name;
                paths.push_back(tmp_path);
            }
            else if(dirp->d_type == 8)
            {
                if(std::regex_search(dirp->d_name, reg_obj))
                {
                    std::string full_path = curr_path + dirp->d_name;
                    files.push_back(full_path);
                }
            }
        }
        closedir(dp);
    }
    return files;
}

template<typename ImageType>
int readFilesThread(std::vector<std::string> filepaths, ImageType * imagesOut, int width, int height, std::atomic<uint64_t> *blockId, int fileNums)
{
    std::uint64_t blockId_t;
    cv::Mat image;
    while(true)
    {
        blockId_t = atomic_fetch_add(blockId, (uint64_t) 1);
        if(blockId_t >= fileNums)
			break;
        image = cv::imread(filepaths.at(blockId_t), cv::IMREAD_UNCHANGED);
        if (image.empty())
        {
            printf("\n %s read failed! \n", filepaths.at(blockId_t).c_str());
            return 1;
        }

        std::uint64_t offset = blockId_t * width * height *3;
        ImageType* tempImage = imagesOut + offset;
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
    return 0;
}


// uint8_t* readImagesMultiThread(std::string path, std::string suffix, int numThreads)
// {
//     if (numThreads <= 0)//use maximum available
// 		numThreads = std::thread::hardware_concurrency();
//     std::vector<std::string> files =  get_all_files(path, suffix);

//     cv::Mat image = cv::imread(files.at(0), cv::IMREAD_UNCHANGED);
//     int width = image.cols;
//     int height = image.rows;
//     std::atomic<uint64_t> blockId; 
//     atomic_store(&blockId, (uint64_t)0);
//     uint8_t* images = new uint8_t[width * height * 3 * files.size()];
//     // std::vector<uint8_t> images(width * heigth * 3 * files.size());
//     std::cout << "file size : " << files.size() << std::endl;

//     std::vector<std::thread> threads;
//     for (int i = 0; i < numThreads; i++)
//     {
//         threads.push_back(std::thread(&multRead, files, images, width, height, &blockId, files.size()));
//     }
//     for (auto& t : threads)
//     {
//         t.join();
//     }
//     return images;    
// }
template<typename  ImageType>
int readImageThread(std::string path, std::string suffix, ImageType* im, int numThreads)
{
    if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();
    std::vector<std::string> files =  get_all_files(path, suffix);

    cv::Mat image = cv::imread(files.at(0), cv::IMREAD_UNCHANGED);
    int width = image.cols;
    int height = image.rows;

    std::atomic<uint64_t> blockId; 
    atomic_store(&blockId, (uint64_t)0);

    std::cout << "The file need to readind is : " << files.size() << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++)
    {
        threads.push_back(std::thread(&readFilesThread, files, im, width, height, &blockId, files.size()));
    }
    for (auto& t : threads)
    {
        t.join();
    }
    return 0;    
}

int readImages(const char* path, const char* suffix, void* im, int numThreads)
{
    std::string filePathOIn(path);
    std::string fileSuffix(suffix);
    readImageThread(filePathOIn, fileSuffix, im, numThreads);
    return 0;
}

int mulThreadDiff(uint8_t* imagesIn, int totalimages, uint64_t gpuSupportNums, uint64_t width, uint64_t height, std::atomic<uint64_t> *blockId, int gpuNum)
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
        
        g_cudamalloc_mutex.lock();
        std::cout << "thread " << this_id << " working...\n";
        g_cudamalloc_mutex.unlock();
        
        std::uint64_t offset = blockId_t * gpuSupportNums * width * height;
        uint8_t* tempImage = imagesIn + offset;

        uint64_t perStreamSize = (copyToDeviceNums + nStreams - 1) / nStreams;
        for(int i = 0; i < nStreams; i++)
        {
            uint64_t perStreamCopySize = perStreamSize;
            if(gpuSupportNums - i * perStreamSize < perStreamSize)
                perStreamCopySize = gpuSupportNums - i * perStreamSize;

            std::uint64_t streamOffset = i * perStreamSize * width * height;
            uint8_t* streamTempImage = tempImage + streamOffset;
            cudaMemcpyAsync(dpImage, streamTempImage, (uint64_t)(width * height * perStreamSize * sizeof(uint8_t)), cudaMemcpyHostToDevice, streams[i]);
            predictor7_tiles_GPU_Stream(dpImage, dpBuffer, width, height, copyToDeviceNums, streams[i]);
            symbolize_GPU_Stream(dpSymbols, dpBuffer, width, height, perStreamSize, streams[i]);
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
    return 0;
}

uint8_t* diffImages(std::string path, std::string suffix, int numThreads)
{
    if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();
    std::vector<std::string> files =  get_all_files(path, suffix);

    cv::Mat image = cv::imread(files.at(0), cv::IMREAD_UNCHANGED);
    uint64_t width = image.cols;
    uint64_t height = image.rows;
    uint64_t totalimages = files.size() * 3;
    std::atomic<uint64_t> blockId; 
    atomic_store(&blockId, (uint64_t)0);
    uint8_t* images = new uint8_t[width * height * 3 * files.size()];
    // std::vector<uint8_t> images(width * heigth * 3 * files.size());

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++)
    {
        threads.push_back(std::thread(&multRead, files, images, width, height, &blockId, files.size()));
    }
    for (auto& t : threads)
    {
        t.join();
    }

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
        diffThreads.push_back(std::thread(&mulThreadDiff, images, 3 * files.size() , gpuSupportImages, width, height, &diffBlockId, i));
    }
    for (auto& t : diffThreads)
    {
        t.join();
    }

    //openmp also working
//*******************************************************************************************************//
    // omp_set_num_threads(deviceCount);  // create as many CPU threads as there are CUDA devices
    // #pragma omp parallel 
    // {
    //     unsigned int cpu_thread_id = omp_get_thread_num();
    //     unsigned int num_cpu_threads = omp_get_num_threads();
    //     cudaSetDevice(cpu_thread_id);

    //     int gpu_id = -1;
    //     cudaGetDevice(&gpu_id);
    //     printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

    //     uint8_t* dpImage = nullptr;
    //     int8_t* dpBuffer = nullptr;
    //     uint8_t* dpSymbols = nullptr;

    //     cudaMalloc((void **)&dpImage, (uint64_t)(width * height *  gpuSupportImages * sizeof(uint8_t)));
    //     cudaMalloc((void **)&dpBuffer, (uint64_t)(width * height *  gpuSupportImages * sizeof(int8_t)));
    //     cudaMalloc((void **)&dpSymbols, (uint64_t)(width * height *  gpuSupportImages * sizeof(uint8_t)));
        

    //     const int nStreams = 8;
    //     cudaStream_t streams[nStreams];
    //     for (int i = 0; i < nStreams; i++) 
    //     {
    //         cudaStreamCreate(&streams[i]);
    //     }
    //     while(true)
    //     {
    //         uint64_t copyToDeviceNums = gpuSupportImages;
    //         uint64_t blockId_t = atomic_fetch_add(&diffBlockId, (uint64_t) 1);
    //         if(totalimages < blockId_t * gpuSupportImages)
    //             break;
    //         else if(totalimages - blockId_t * gpuSupportImages < gpuSupportImages)
    //             copyToDeviceNums = totalimages - blockId_t * gpuSupportImages;

    //         std::uint64_t offset = blockId_t * gpuSupportImages * width * height;
    //         uint8_t* tempImage = images + offset;

    //         unsigned int cpu_thread_id = omp_get_thread_num();
    //         printf("The working thread is  %d\n", omp_get_thread_num());

    //         uint64_t perStreamSize = (copyToDeviceNums + nStreams - 1) / nStreams;
    //         for(int i = 0; i < nStreams; i++)
    //         {
    //             uint64_t perStreamCopySize = perStreamSize;
    //             if(copyToDeviceNums - i * perStreamSize < perStreamSize)
    //                 perStreamCopySize = copyToDeviceNums - i * perStreamSize;

    //             std::uint64_t streamOffset = i * perStreamSize * width * height;
    //             uint8_t* streamTempImage = tempImage + streamOffset;
    //             cudaMemcpyAsync(dpImage, streamTempImage, (uint64_t)(width * height * perStreamSize * sizeof(uint8_t)), cudaMemcpyHostToDevice, streams[i]);
    //             predictor7_tiles_GPU_Stream(dpImage, dpBuffer, width, height, copyToDeviceNums, streams[i]);
    //             symbolize_GPU_Stream(dpSymbols, dpBuffer, width, height, perStreamSize, streams[i]);
    //             cudaMemcpyAsync(streamTempImage, dpSymbols, (uint64_t)(width * height * perStreamSize * sizeof(uint8_t)), cudaMemcpyDeviceToHost);
    //         }
    //         cudaDeviceSynchronize();

    //         // cudaMemcpy(dpImage, tempImage, (uint64_t)(width * height * copyToDeviceNums * sizeof(uint8_t)), cudaMemcpyHostToDevice);
    //         // predictor7_tiles_GPU(dpImage, dpBuffer, width, height, copyToDeviceNums);
    //         // symbolize_GPU(dpSymbols, dpBuffer, width, height, copyToDeviceNums);
    //         // cudaMemcpy(tempImage, dpSymbols, (uint64_t)(width * height * copyToDeviceNums * sizeof(uint8_t)), cudaMemcpyDeviceToHost);
    //     }
    //     cudaFree(dpImage);
    //     cudaFree(dpBuffer);
    //     cudaFree(dpSymbols);
    // }
//*******************************************************************************************************//
    std::atomic<uint64_t> wirteBlockId; 
    atomic_store(&wirteBlockId, (uint64_t)0);

    omp_set_num_threads(numThreads);  // create as many CPU threads as there are CUDA devices
    #pragma omp parallel 
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        cv::Mat image(height, width, CV_8UC3);
        while(true)
        {
            uint32_t blockId_t = atomic_fetch_add(&wirteBlockId, (uint64_t) 1);
            if(blockId_t >= files.size())
                break;
            image = cv::imread(files.at(blockId_t), cv::IMREAD_UNCHANGED);

            std::uint64_t offset = blockId_t * width * height *3;
            uint8_t* tempImage = images + offset;
            for(int i= 0;  i < height; i++)
            {
                for(int j = 0; j < width; j++)
                {
                    image.at<cv::Vec3b>(i,j)[0] = tempImage[i * width +j + 0 * width * height];
                    image.at<cv::Vec3b>(i,j)[1] = tempImage[i * width +j + 1 * width * height];
                    image.at<cv::Vec3b>(i,j)[2] = tempImage[i * width +j + 2 * width * height];
                }
            }
            std::string pathName = files.at(blockId_t);
            std::string outPath = pathName.substr(0, pathName.rfind("/")).append("_diffresult");
            std::string fileName = pathName.substr(pathName.rfind("/"), pathName.rfind(".") - pathName.rfind("/")).append(".png");
            DIR *dir;
            if((dir=opendir(outPath.c_str())) == NULL)
            {
                std::string command = "mkdir -p " + outPath;  
                system(command.c_str());
            }
      
            bool test = cv::imwrite(outPath.append(fileName), image);
        }
    }

    uint64_t offset = width * height * 3 * 1000;
    uint8_t* temp = images + offset;
    cv::Mat bgr(height, width, CV_8UC3);
    for (int i = 0 ; i < height ; i++)
    {
        for (int j = 0; j < width; j++)
        {
            bgr.at<cv::Vec3b>(i,j)[0] = temp[i * width +j + 0 * width * height];
            bgr.at<cv::Vec3b>(i,j)[1] = temp[i * width +j + 1 * width * height];
            bgr.at<cv::Vec3b>(i,j)[2] = temp[i * width +j + 2 * width * height];
        }
    }
    cv::imwrite("../temp.png",bgr);
    return images;
}


int main()
{
//    std::vector<std::string> files =  get_all_files("/data/liutianqiang/hilloc/csfy/valid/", ".jpg");
    uint8_t* images = diffImages("/data/liutianqiang/hilloc/csfy/train/", ".jpg", 20);

   return 0;
}


// int main()
// {
//     cv::Mat matSrcImage = cv::imread("../test.jpg", cv::IMREAD_UNCHANGED);
//     int width = matSrcImage.cols;
//     int height = matSrcImage.rows;
//     int depth = matSrcImage.channels();

//     uint8_t* u8SrcImage = matSrcImage.data;
//     std::vector<uint8_t> srcImage(width * height * depth);
//     for(int i= 0;  i < height; i++)
//     {
//         for(int j = 0; j < width; j++)
//         {
//             srcImage.at(i * width + j) = matSrcImage.at<cv::Vec3b>(i,j)[0];
//             srcImage.at(i * width + j + width * height) = matSrcImage.at<cv::Vec3b>(i,j)[1];
//             srcImage.at(i * width + j + width * height * 2) = matSrcImage.at<cv::Vec3b>(i,j)[2];
//         }
//     }

//     std::vector<int8_t> symbols(width * height * depth);
//     // std::memcpy(symbols.data(), u8SrcImage, width * height * depth * sizeof(uint8_t));

//     std::vector<uint8_t> recoveryImage(width * height * depth);
//     std::memcpy(recoveryImage.data(), u8SrcImage, width * height * depth * sizeof(uint8_t));

//     uint8_t* dpImage = nullptr;
// 	int8_t* dpBuffer = nullptr;
// 	uint8_t* dpSymbols = nullptr;

// 	cudaMalloc(&dpImage, (long)(width * height * depth * sizeof(uint8_t)));
// 	cudaMalloc(&dpBuffer, (long)(width * height * depth * sizeof(int8_t)));
// 	cudaMalloc(&dpSymbols, (long)(width * height * depth * sizeof(uint8_t)));

//     cudaMemcpy(dpImage, srcImage.data(), (long)(width * height * depth * sizeof(uint8_t)), cudaMemcpyHostToDevice);

//     predictor7_tiles_GPU(dpImage, dpBuffer, width, height, depth);
//     symbolize_GPU(dpSymbols, dpBuffer, width, height, depth);

//     cudaMemcpy(recoveryImage.data(), dpSymbols, (long)(width * height * depth * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

//     cv::Mat ch1(height, width, CV_8UC1);
//     for (int i = 0 ; i < height ; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             ch1.at<uchar>(i,j) = recoveryImage.at(i * width +j);
//         }
//     }
//     cv::imwrite("../diffch1.png",ch1);

//     cv::Mat ch2(height, width, CV_8UC1);
//     for (int i = 0 ; i < height ; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             ch2.at<uchar>(i,j) = recoveryImage.at(i * width +j + width * height);
//         }
//     }
//     cv::imwrite("../diffch2.png",ch2);

//     cv::Mat ch3(height, width, CV_8UC1);
//     for (int i = 0 ; i < height ; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             ch3.at<uchar>(i,j) = recoveryImage.at(i * width +j + 2 * width * height);
//         }
//     }
//     cv::imwrite("../diffch3.png",ch3);

//     cv::Mat bgr(height, width, CV_8UC3);
//     for (int i = 0 ; i < height ; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             bgr.at<cv::Vec3b>(i,j)[0] = recoveryImage.at(i * width +j + 0 * width * height);
//             bgr.at<cv::Vec3b>(i,j)[1] = recoveryImage.at(i * width +j + 1 * width * height);
//             bgr.at<cv::Vec3b>(i,j)[2] = recoveryImage.at(i * width +j + 2 * width * height);
//         }
//     }
//     cv::imwrite("../diffbgr.png",bgr);

//     unsymbolize_GPU(dpBuffer, dpSymbols, width, height, depth);
//     cudaMemcpy(symbols.data(), dpBuffer, (long)(width * height * depth * sizeof(uint8_t)), cudaMemcpyDeviceToHost);
//     unpredictor7_tiles(symbols.data(), recoveryImage.data(), width, height, depth);

//     for(int i= 0; i < recoveryImage.size(); i++)
//     {
//         if (srcImage.at(i) != recoveryImage.at(i))
//             printf("\nRecover Image is Wrong!");
//     }
//     printf("\nComplete!");
//     return 0;
// }