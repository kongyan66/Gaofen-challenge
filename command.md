# Train
## YOLOV8
### 单卡
python tools/train.py configs/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py 

### 多卡
CUDA_VISIBLE_DEVICES=2,3 PORT=29501 ./tools/dist_train.sh work_dirs/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py 2

## YOLOX
### 多尺度
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh configs/yolox/yolox_x_fast_8xb8-300e_coco.py  2

## RTMDet
### 单卡
python tools/train.py configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py
### 多卡
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py

- 400epoch

#### 多尺度
- 300 
CUDA_VISIBLE_DEVICES=2,3 PORT=29501 ./tools/dist_train.sh configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_sar_ms.py 2
- 400 epoch 
CUDA_VISIBLE_DEVICES=2,3 PORT=29501 ./tools/dist_train.sh configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-400e_sar_ms.py 2

## PPYOLOE
### 单卡
python tools/train.py configs/ppyoloe/ppyoloe_x_fast_8xb16-300e_coco.py (太大跑不动)
python tools/train.py configs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py

python tools/train.py configs/ppyoloe/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar.py

### 多卡
- base
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh configs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py 2
- 
- epoch_240+无val+多尺度
CUDA_VISIBLE_DEVICES=0,1   PORT=29500 ./tools/dist_train.sh configs/ppyoloe/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar.py  2
## YOLOV8-m
CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=29500 ./tools/dist_train.sh configs/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py  4

# Test

## yolov8
python tools/test.py work_dirs/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py  work_dirs/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/epoch_300.pth --tta --json-prefix "outputs/yolov8.json"

python tools/test.py configs/yolox/yolox_x_fast_8xb8-300e_coco.py work_dirs/yolox_x_fast_8xb8-300e_coco/epoch_280.pth

# 可视化
mim run mmdet analyze_logs plot_curve \
    work_dirs/ppyoloe_plus_s_fast_8xb8-240e_ms_sar/20230313_173341/vis_data/20230313_173341.json \
    --keys loss \
    --legend ppyoloe\
    --eval-interval 10 \
    --out ppyoloe_plus_loss.jpg

python tools/test.py work_dirs/ppyoloe_plus_s_fast_8xb8-240e_ms_sar/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar.py  work_dirs/ppyoloe_plus_s_fast_8xb8-240e_ms_sar/epoch_170.pth  --show-dir "work_dirs/vis"


