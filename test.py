'''
Author: your name
Date: 2022-04-04 12:25:45
LastEditTime: 2022-04-05 09:42:29
LastEditors: Onetism_SU
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /python_cuda/test.py
'''
'''
Author: your name
Date: 2022-04-04 12:25:44
LastEditTime: 2022-04-04 12:25:45
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /python_cuda/test.py
'''
# Copyright 2022 liutianqiang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from imagesread import pyImagesRead
import cv2


a = pyImagesRead('/data/liutianqiang/hilloc/csfy/train/', '.jpg')
h = a.getDiffImage(20)

a.writeImageData(h,"/data/liutianqiang/temp/",".png")

cv2.imwrite('./test.png',h[1000,:,:,:].transpose(1,2,0))

b = a[:,:,0:1216,0:1920]

c = b.reshape(3000,3,19,30,64,64)


# cv2.imwrite('./1.png',a[1000,:,0,0,:,:].transpose(1,2,0))

temp = np.empty((5,), np.uint32)

