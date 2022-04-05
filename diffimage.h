/*
 * @LastEditors: Onetism_SU
 * @Author: Onetism_su
 * @LastEditTime: 2022-04-04 20:45:36
 */

#ifndef __DIFFIMAGE_H__
#define __DIFFIMAGE_H__

#include<string>
#include<vector>
#include <atomic>

template <class ImageType>
class images_read{
    
    private:
        std::string filePath;
        std::string suffix;
        uint16_t width;
        uint16_t height;
        uint16_t depth;
        uint16_t filesNum;

    public:
        images_read(std::string fielpath, std::string suffix);   
        std::vector<std::string> getAllImagesVector(void);
        void imagesReadThread(ImageType* outimage, std::atomic<uint64_t> *blockId);
        void multiThreadReadImages(ImageType* outimage);

}

// int readImages(const char* path, const char* suffix, void* im, int numThreads);


#endif