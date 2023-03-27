

# 高分大赛实验记录-决赛-2023/3/10

## **当前最佳模型排名：**

> 1. **yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco mAP@89.40**    ❌
> 2. **yolov8_x_mask-refine_syncbn_fast_8xb16 mAP@88.1**    ✔️
> 3. **rtmdet_x_syncbn_fast_8xb32-300e_sar_ms mAP@78.3**     ✔️
> 4. **ppyoloe_plus_s_fast_8xb8-80e_coco mAP@71.3**  ✔️
> 5. **YOLOx_l_8x8_100e mAP@66.9** ❌
> 6. **Cascade_rcnn_swin_fpn_ms_4x_sar  mAP@65.4** ❌
> 7. **YOLOx_l_8x8_200e  mAP@61.8** ❌
>



**目标成绩： mAP@70  time@120s**

|   版本号   |            模型            | val@mAP |    test@mAP(差值)     | val@time(s)/TTA | test@time(s) |
| :--------: | :------------------------: | :-----: | :-------------------: | :-------------: | :----------: |
|   初赛版   |           carcnn           |    -    |        70.3977        |        -        |     764      |
|   v2.2.2   |    yolov8_x_mask_refine    |  88.1   |  67.3279 ( -20.7721)  |      67.32      |     175      |
|   v2.2.4   |  rtmdet_x_syncbn_fast_300  |  78.3   |   67.9190 (-10.381)   |       82        |     125      |
|   v2.2.5   |    yolov8_m_mask-refine    |  89.40  |           -           |       33        |     None     |
|   v2.2.6   | yolov8_m_mask-refine (TTA) |    -    |        67.5479        |       80        |  121(x1.5)   |
|   v2.2.7   |    ppyoloe_plus_s (TTA)    |  71.3   |    68.4315(-2.87)     |     40(68s)     |     127      |
|   v2.2.8   |  ppyoloe_plus_s_170e(TTA)  |  80.7   |   69.7269(-10.9787)   |     31(74s)     |  146(x1.9)   |
| **v2.2.9** | ppyoloe_plus_s_no_val_220e |   86    | **67.6951**(-18.3049) |     29(83)      |      57      |



|   版本号   | val@mAP |  test@mAP   | val@time(s)/TTA | test@time(s) |
| :--------: | :-----: | :---------: | :-------------: | :----------: |
|   初赛版   |    -    |   70.3977   |        -        |     764      |
|   v2.2.2   |  71.30  |   67.3279   |      67.32      |     175      |
|   v2.2.4   |  76.31  |   67.9190   |       82        |     125      |
|   v2.2.5   |  78.60  |      -      |       33        |     None     |
|   v2.2.6   |    -    |   67.5479   |       80        |  121(x1.5)   |
|   v2.2.7   |  80.7   |   68.4315   |     40(68s)     |     127      |
|   v2.2.8   |   86    |   69.7269   |     31(74s)     |  146(x1.9)   |
| **v2.2.9** |  83.7   | **67.6951** |     29(83)      |      57      |


> 1. **yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco mAP@68.3**    ❌
> 2. **yolov8_x_mask-refine_syncbn_fast_8xb16 mAP@69.4**    ✔️
> 3. **rtmdet_x_syncbn_fast_8xb32-300e_sar_ms mAP@70.2**   ✔️
> 4. **ppyoloe_plus_s_fast_8xb8-80e_coco mAP@71.3**  ✔️
> 5. **YOLOx_l_8x8_100e mAP@66.9** ❌
> 6. **Cascade_rcnn_swin_fpn_ms_4x_sar  mAP@65.4** ❌
> 7. **YOLOx_l_8x8_200e  mAP@61.8** ❌

## 改进点

思路

- 无val训练   测试
- TTA               有效
- 模型集成      弃用
- 锚框优化     弃用
- 数据增强     参考yolov8

```
dict(type='Mosaic', img_scale=img_scale, pad_val=114.0), 
dict( 
    type='RandomAffine', 
    scaling_ratio_range=(0.1, 2), 
    border=(-img_scale[0] // 2, -img_scale[1] // 2)), 
dict( 
    type='MixUp', 
    img_scale=img_scale, 
     ratio_range=(0.8, 1.6), 
     pad_val=114.0), 
```





## 数据分析

### Train

![image-20230313145424365](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230313145424365.png)

**area**

![image-20230313151045899](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230313151045899.png)

**box_num**

![YOLOv5CocoDataset_bbox_num](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/YOLOv5CocoDataset_bbox_num.jpg)

**box_ratio**

![YOLOv5CocoDataset_bbox_ratio](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/YOLOv5CocoDataset_bbox_ratio.jpg)

**wh**

![YOLOv5CocoDataset_bbox_wh](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/YOLOv5CocoDataset_bbox_wh.jpg)





## 2023/3/6

**今日任务：**

> - [x] yolox 200epoch
>
> - [x] cascade_rcnn_swin  12epoch
>

- **YOLOx_l_8x8_100e**

**实验结果：**

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.669
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.962
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.784
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.622
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.667
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.685
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.677
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.739
```

注：还不是最优结构，可继续训练

- **Cascade_rcnn_swin_fpn_2x**

**实验结果：**

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.870
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.577
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.551
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.608
```

注：欠拟合，需要继续训练



## 2023/3/7

**今日任务：**

> - [x] yolox 加训100epoch
> - [x] cascade_rcnn_swin 加训12epoch
> - [x] 安装mmyolo环境及**移植(None)**
> - [x] 测试多尺度训练及初赛模型

- **YOLOx_l_8x8_200e**

反而精度下降了，建议重新训练一下。

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.618
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.936
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.722
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.616
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.676
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.704
```

- **Cascade_rcnn_swin_fpn_ms_4x_sar(封装测试-3/10)**

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.654
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.955
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.780
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.565
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.654
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.715
```

-  **Cascade_rcnn_swin_fpn_3x_sar**

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.905
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.674
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.524
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.595
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.673
```

## 2023/3/8

**今日任务：**

> - [ ] yolox 多尺度训练（设置多尺度就报错）
> - [x] mmyolo测试与移植
> - [x] yolov8训练
> - [x] yolox 在mmyolo上300epoch训练

- **yolov8_x_mask-refine_syncbn_fast_8xb16-300e**

最好结果在300epoch

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.881
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.977
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.927
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.824
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.869
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.893
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.917
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.918
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.885
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.905
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.937
```

**mAP曲线**

![output_mAP](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/output_mAP.jpg)

**loss曲线**

![output_loss](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/output_loss.jpg)

- **yolox_x_fast_8xb8-300e**

比`YOLOx_l_8x8_200e`低太多了，不正常

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.493
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.852
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.632
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676
```



## 2023/3/9  rtmdet

**今日任务：**

> - [ ] 搭建mmyolo容器环境并测试
> - [ ] 学习WBF模型融合方法
> - [x] RTMDet测试
> - [ ] 测试下mmyolo TTA

- **rtmdet_x_syncbn_fast_8xb32-300e_sar_ms**

采用多尺度训练：

```
# 多尺度训练
pad_size_divisor=32,
batch_augments=[
dict(
type='YOLOXBatchSyncRandomResize',
# 多尺度范围是 480~800
random_size_range=(640, 1024),
# 输出尺度需要被 32 整除
size_divisor=32,
# 每隔 1 个迭代改变一次输出输出
interval=1)
```

实验结果：epoch@270

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.783
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.977
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.917
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.745
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.780
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.791
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.835
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.754
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.830
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.836
```

![rtmdet_loss](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/rtmdet_loss.jpg)

![rtmdet_bbox_mAP](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/rtmdet_bbox_mAP.jpg)



- **ppyoloe_plus_s_fast_8xb8-80e_coco**

160 epoch 

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.713
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.968
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.854
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.662
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.708
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.742
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.778
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.784
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.677
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.779
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.807
```

**mAP**

![ppyoloe_plus_mAP](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/ppyoloe_plus_mAP.jpg)

**Loss**

![ppyoloe_plus_loss](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/ppyoloe_plus_loss.jpg)

## 2023/3/10

**今日任务：**

> - [x] 搭建mmyolo容器环境并测试
> - [ ] 学习WBF模型融合方法
> - [x] 测试下mmyolo TTA
> - [x] 完成答辩PPT





## 2023/3/11

**今日任务：**

> - [x] yolov8-m训练
>
> - [ ] 全训练集训练
>
> - [ ] ppyoloe-m训练
>
>   

- **yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco**

500 epoch有点过拟合了 ，

```
2023/03/11 18:51:04 - mmengine - INFO - Epoch(val) [450][7/7]  coco/bbox_mAP: 0.8940  coco/bbox_mAP_50: 0.9780  coco/bbox_mAP_75: 0.9410  coco/bbox_mAP_s: 0.8740  coco/bbox_mAP_m: 0.8850  coco/bbox_mAP_l: 0.9160
```





## 2023/3/12

**今日任务：**

> - [x] 提交ppyoloe(TTA)
>
> - [x] 提交yolov8_m(TTA)
>



## 2023/3/13  ppyoloe

**今日任务：**

> - [x] ppyoloe+170 epoch + 多尺度 （晚饭后完成）
>- [x] ppyoloe+240 epoch + 无val+多尺度   
> - [ ] rtmdet + 300epoch+多尺度   **训练失败**

**ppyoloe_plus_s_fast_8xb8-240e_ms_sar**  **epoch 170**

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.807
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.915
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.761
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.804
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.827
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.848
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.851
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.842
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.874
```

![ppyoloe_plus_lo](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/ppyoloe_plus_lo.jpg)

![ppyoloe_plus_bbox_mAP](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/ppyoloe_plus_bbox_mAP.jpg)

**ppyoloe+240 epoch + 无val+多尺度**  选用220e

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.865
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.993
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.964
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.848
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.857
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.890
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.891
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.893
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.885
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.883
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.922
```

![ppyoloe_loss](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/ppyoloe_loss.jpg)

![ppyoloe_bbox_mAP](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/ppyoloe_bbox_mAP.jpg)

## 2023/3/14

> - [ ] ppyoloe+240 epoch + 无val+多尺度  + TTA



## 动态排名

**3.10**

初赛版本

![image-20230312171803174](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230312171803174.png)

**3.12**

v2.2.2

![image-20230312164106901](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230312164106901.png)

v2.2.4

![image-20230312171558402](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230312171558402.png)

2023/3/13

v2.2.7

![image-20230313120103899](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230313120103899.png)



![image-20230313205727437](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230313205727437.png)



3.15

![image-20230315173440370](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230315173440370.png)

![image-20230315203414842](https://raw.githubusercontent.com/kongyan66/Img-for-md/master/img/image-20230315203414842.png)

## 参考资料

[第十七届全国大学生智能汽车竞赛：智慧交通组创意赛线上资格赛-冠军方案](https://aistudio.baidu.com/aistudio/projectdetail/4217664)

