# Gaofen Challenge

Golinning solution (3/152) in the Object Recognition in  Sar Images, 2021 Gaofen Challenge on Automated High-Resolution Earth Observation Image Interpretation. 

![image-20230327142923716](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230327142923716.png#pic_center)



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

![image-20230327143233229](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230327143233229.png)