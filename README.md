<!--
 * @LastEditors: Onetism_SU
-->
# fastImageRead
    使用线程来加快python中数据的读写速度，并且使用多GPU来加速获取差值图像，可以一次性将文件目录下所需格式的图片全部读入，并且计算差值图，存出。

## 安装依赖项
- NumPy > 1.20.0
- Cython
- cudatoookit
- cudatoolkit-dev
- opencv

```sh
  
  conda install cudatoolkit cudatoolkit-dev opencv -c conda-forge
  可以分别安装，注意cudatoolkit和cudatoolkit-dev的版本要是一致的
```

## pip安装
```sh
  pip install git+https://github.com/Onetism/fastImageRead.git@skbuild
```

## 预测方式
```
    enum IMAGE_PREDICTORS_TYPE:
        NO_PREIDICTORS = 0,
        PREIDCTORS_A = 1,
        PREIDCTORS_B = 2,
        PREIDCTORS_C = 3,
        PREIDCTORS_APB_DC = 4,
        PREIDCTORS_A_BDC_Div2 = 5,
        PREIDCTORS_B_ADC_Div2 = 6,
        PREIDCTORS_APB_Div2 = 7,
        PREIDCTORS_APB_Div2_Exten = 8
```

## 使用方式
```
  from imreadfast import pyImagesRead

  //初始化，输入参数{imagespath(图片所在路径), suffix(图片后缀), ptype=7(预测方式，默认为7)}
  //imageData格式[图片个数, 通道数, 高度, 宽度]
  a = pyImagesRead('/data/liutianqiang/hilloc/csfy/train/', '.jpg')

  //获取差值图像，输入参数{numThreads(使用线程数)}
  h = a.getDiffImage(20)

  //获取原始图像，输入参数{numThreads(使用线程数)}
  g = a.getImageData(20)

  //存出图像，输入参数{imageData(存出图像array), path(输出路径), suffix(图片格式后缀),numThreads(使用线程数，默认多)}
  //imageData格式[图片个数, 通道数, 高度, 宽度]
  a.writeImageData(h,"/data/liutianqiang/temp/",".png")
```