# Gaofen Challenge

The fifth winning solution (5/152) in the Object Recognition in  Sar Images and third place in the final, 2021 Gaofen Challenge on Automated High-Resolution Earth Observation Image Interpretation. 

![image-20230327142923716](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230327142923716.png#pic_center)

![image-20230327152616143](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230327152616143.png)

## Members

[Caihua kong](https://github.com/kongyan66),  [Longgang Dai](https://github.com/dailonggang), [Zhentao Fan](https://github.com/zt-fan), [Xiang Chen](https://github.com/cschenxiang).

## Solution

* **On-line date augmentation**  
  We use random combination of affine transformation, flip, scaling, optical distortion for data augmentation.
* **copy-paste off-line data augmention**
* **Multi-scale training and testing**    
  The training images are resized into sizes of 600, 800, and 1024 for training and testing.


* **WBF**
* **PPYOLOE**

## Experiment

|   版本号   | val@mAP |  test@mAP   | val@time(s)/TTA | test@time(s) |
| :--------: | :-----: | :---------: | :-------------: | :----------: |
|  V1.0.13   |    -    |   70.3977   |        -        |     764      |
|   v2.2.2   |  71.30  |   67.3279   |      67.32      |     175      |
|   v2.2.4   |  76.31  |   67.9190   |       82        |     125      |
|   v2.2.5   |  78.60  |      -      |       33        |     None     |
|   v2.2.6   |    -    |   67.5479   |       80        |  121(x1.5)   |
|   v2.2.7   |  80.7   |   68.4315   |     40(68s)     |     127      |
|   v2.2.8   |   86    |   69.7269   |     31(74s)     |  146(x1.9)   |
| **v2.2.9** |  83.7   | **67.6951** |     29(83)      |      57      |

For detailed records, please refer to the `experiment.md`

## How to use

1. Prepare  [sar dataset](https://drive.google.com/drive/folders/1VI-9jHOkcnOFf_Cw5feA5GCi3ndf3Ogf?usp=sharing)

   Please execute the script under `dataset_converter` to convert the data format from xml format to coco format.

   Note: the official only gave the training set, and the verification set is mainly defined by itself (8:2).

2. Prepare your environment

   Our code is implemented based on mmyolo, see [mmyolo](https://github.com/open-mmlab/mmyolo) for environment installation.

3. Train and Test

   Please see `command.md`.

4. Test for submit

   Run `python run.py /input_path /out_path`.
   
5. Docker submit

   Please see [my_blog](https://blog.csdn.net/qq_41719643/article/details/120730411?spm=1001.2014.3001.5501)

## Detections

![image-20230327152202202](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230327152202202.png)
