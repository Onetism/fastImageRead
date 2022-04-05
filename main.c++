/*
 * @LastEditors: Onetism_SU
 * @Author: Onetism_su
 * @LastEditTime: 2022-04-05 18:42:42
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
#include "opencv2/opencv.hpp"

int main()
{
    const char* path = "/data/liutianqiang/hilloc/csfy/train/";
    const char* suffix = "jpg";
    images_read test = images_read(path, suffix, IMAGE_PREDICTORS_TYPE::PREIDCTORS_APB_Div2);

    std::vector<uint8_t> images((uint64_t)3000*3*1936*1216);
    test.getDiffImages(images.data(), 20);
    const char* outpath = "/data/liutianqiang/hilloc/csfy/temp/";
    const char* outsuffix = ".png";    
    test.multiThreadImagesWirte(images.data(), outpath, outsuffix,20);
    uint64_t offset = (uint64_t)1936 * 1216 * 3 * 1000;
    uint64_t width = 1936;
    uint64_t height = 1216;
    uint8_t* temp = images.data() + offset;
    cv::Mat bgr(height, width, CV_8UC3);
    for (int i = 0 ; i < 1216 ; i++)
    {
        for (int j = 0; j < 1936; j++)
        {
            bgr.at<cv::Vec3b>(i,j)[0] = temp[i * width +j + 0 * width * height];
            bgr.at<cv::Vec3b>(i,j)[1] = temp[i * width +j + 1 * width * height];
            bgr.at<cv::Vec3b>(i,j)[2] = temp[i * width +j + 2 * width * height];
        }
    }
    cv::imwrite("../temp.png",bgr);
    int a = 0;
    return 0;
}