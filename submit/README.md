# 本脚本适用于mmdet生成json格式结果文件转换为voc格式进行提交
1. 生成结果文件
>python tools/test.py  
>work_dirs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py  # config文件路径 
>work_dirs/faster_rcnn/epoch_12.pth\  # weights文件路径
>--format-only\
> --eval-options "jsonfile_prefix=output"

2.使用 mmdet2xml.py 将生成的output.json 转换为xml格式
