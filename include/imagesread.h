/*
 * @LastEditors: Onetism_SU
 * @Author: Onetism_su
 * @LastEditTime: 2022-04-05 18:35:39
 */
#ifndef __IMAGESREAD_H__
#define __IMAGESREAD_H__

#include<string>
#include<vector>
#include<atomic>
#include<mutex>

enum IMAGE_DATA_TYPE
{
	UINT8_TYPE = 0,
	UINT16_TYPE = 1,
	UINT32_TYPE = 2,
	UINT64_TYPE = 3,
	INT8_TYPE = 4,
	INT16_TYPE = 5,
	INT32_TYPE = 6,
	INT64_TYPE = 7,
	FLOAT32_TYPE = 8,
	FLOAT64_TYPE = 9
};

enum IMAGE_PREDICTORS_TYPE
{
	NO_PREIDICTORS = 0,
	PREIDCTORS_A = 1,
	PREIDCTORS_B = 2,
	PREIDCTORS_C = 3,
	PREIDCTORS_APB_DC = 4,
	PREIDCTORS_A_BDC_Div2 = 5,
	PREIDCTORS_B_ADC_Div2 = 6,
	PREIDCTORS_APB_Div2 = 7,
	PREIDCTORS_APB_Div2_Exten = 8
};


class images_read{
    
    private:
        std::string filePath;                       //图像文件所在目录
        std::string suffix;                         //图像后缀
        std::vector<std::string> imageFileNames;    //所有图像绝对路径集合
        uint64_t width;                             //图像宽度
        uint64_t height;                            //图像高度
        uint64_t depth;                             //图像通道数
        uint64_t filesNum;                          //图像文件个数
        IMAGE_DATA_TYPE imageType;                  //图像类型
        IMAGE_PREDICTORS_TYPE predictorType;        //图像预测方式

    public:

        images_read();
        images_read(const char* fielpath, const char* suffix, IMAGE_PREDICTORS_TYPE pType);
        ~images_read();
        uint64_t getWidth();
        uint64_t getHeight();
        uint64_t getDepth();
        uint64_t getFilesNum();
        IMAGE_DATA_TYPE getImageType();
        std::vector<std::string> getAllImagesVector(void);
        void imagesReadThread(void* imageData, std::atomic<uint64_t>* blockId);
        int multiThreadReadImages(void* imageData, uint16_t numThreads);
        int getDiffImages(void* imageData, uint16_t numThreads);
        void diffImagesThread(void* imageData, int totalimages, uint64_t gpuSupportNums, std::atomic<uint64_t> *blockId, int gpuNum);
        void imagesWriteThread(void* imageData, const char* path, const char* suffix, std::atomic<uint64_t>* blockId);
        void multiThreadImagesWirte(void* imageData, const char* path, const char* suffix,uint16_t numThreads);
};

#endif
