import os
import subprocess as sp
import time
# from mmcv import Config
# from mmdet.datasets import build_dataset

from summit_tools.img2coco import create_test_json
from summit_tools.GF_sar_submission_v2 import creat_submission

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='summit fianal result ')
    parser.add_argument('input', default='data/summit_test/images')
    parser.add_argument('--result_json', default='output.json.bbox.json')
    parser.add_argument('output', default='work_dirs/submission')

    args = parser.parse_args()
    return args

def run_cmd(cmd):
    cp = sp.run(cmd,shell=True, capture_output=True, encoding = "utf-8")
    if cp.returncode != 0:
        error = f"""Something wrong has happend when running command [{cmd}]:{cp.stderr}"""
        raise Exception(error)
    return cp.stdout, cp.stderr

def main():
    args = parse_args()
    classnames = ('A220', 'A330', 'A320/321', 'Boeing737-800', 'Boeing787', 'ARJ21', 'other')
    start_time = time.time()
    # get test_images annotation .json file
    try:
        create_test_json(args.input)
    except:
        print("fail to get test_images annotation .json file")
        raise
    # get detection result: .json file
    print('begining to test')
    try:
        run_cmd('python tools/test.py models/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar.py models/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar/epoch_220.pth  --tta  --json-prefix "output.json"')
        # run_cmd('python tools/test.py models/rtmdet_x_syncbn_fast_8xb32-300e_sar_ms/rtmdet_x_syncbn_fast_8xb32-300e_sar_ms.py models/rtmdet_x_syncbn_fast_8xb32-300e_sar_ms/epoch_270.pth --json-prefix "output.json"')
    except Exception:
        print('*********************test failed************************')
        raise
        
    # creat xml annotations
    creat_submission(args.result_json, args.output, classnames)

    run_cmd('rm test.csv && rm test.json && rm output.json.bbox.json')

    print('finshed the submission!!!')
    print('total time:', time.time() - start_time)

if __name__=="__main__":
    main()
