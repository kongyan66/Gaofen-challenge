# 'https://github.com/ming71/toolbox/blob/master/converter/toVOC/gaofen2020_submission.py'
# get summission files by test image path
import os
import json
import shutil
import argparse
import os.path as osp

from tqdm import tqdm
from lxml.etree import Element,SubElement,tostring
from xml.dom.minidom import parseString

# from mmdet.datasets import build_dataset
# from mmcv import Config

def creat_submission(json_result, out_path, classnames):
    '''
    Args:
        data_test: test1 image path
        dets: bbox (np.array)
        dstpath: save to path
        classnames: class names

    Returns: xml of output
    '''
    # get result.json
    #print(json_result)
    f = open(json_result)
    anns = json.load(f)

    det_folder = osp.join(out_path)
   # if osp.exists(det_folder):
   #     shutil.rmtree(det_folder)
   # os.mkdir(det_folder)
    res = {}
    for i, ann in enumerate(anns):
        res[ann['image_id']] = []
    for i, ann in enumerate(anns):
        res[ann['image_id']].append(ann)
    pbar = tqdm(res)
    for i, img_id in enumerate(pbar):
        pbar.set_description("create_submission:")
        im_name = img_id
        ## root note "annotation"
        node_root=Element('annotation')
        # sub-node "source"
        node_source = SubElement(node_root,'source')
        SubElement(node_source,'filename').text = "%s" % im_name + '.tif'
        SubElement(node_source,'origin').text = 'GF3'
        # sub-node "research"
        node_research = SubElement(node_root,'research')
        SubElement(node_research,'version').text = '1.0'
        SubElement(node_research,'provider').text = 'Company/School of team'
        SubElement(node_research,'author').text = '1+3>4'
        SubElement(node_research,'pluginname').text = "Airplane Detection and Recognition in Optical Images"
        SubElement(node_research,'pluginclass').text = 'Detection'
        SubElement(node_research,'time').text = '2021-10-7'
        # sub-node "objects"

        node_objects = SubElement(node_root, 'objects')
        for j, ann in enumerate(res[img_id]):
            # import ipdb;ipdb.set_trace()

            conf = ann['score']
            x1, y1, x2, y2, x3, y3, x4, y4 = rbox2poly_single(ann['bbox'])
            node_object = SubElement(node_objects, 'object')
            SubElement(node_object,'coordinate').text = 'pixel'
            SubElement(node_object,'type').text = 'rectangle'
            SubElement(node_object,'description').text = 'None'
            node_possibleresult = SubElement(node_object,'possibleresult')
            SubElement(node_possibleresult,'name').text = "%s" % classnames[ann['category_id']-1]
            SubElement(node_possibleresult,'probability').text = "%s" % format(conf,'.3f')
            # two-sub-node "points"
            node_points = SubElement(node_object,'points')
            SubElement(node_points,'point').text = "%s" % ','.join([format(x1,'.1f'),format(y1,'.1f')])
            SubElement(node_points,'point').text = "%s" % ','.join([format(x2,'.1f'),format(y2,'.1f')])
            SubElement(node_points,'point').text = "%s" % ','.join([format(x3,'.1f'),format(y3,'.1f')])
            SubElement(node_points,'point').text = "%s" % ','.join([format(x4,'.1f'),format(y4,'.1f')])
            SubElement(node_points,'point').text = "%s" % ','.join([format(x1,'.1f'),format(y1,'.1f')])

        xml = tostring(node_root,encoding='utf-8', method="xml",xml_declaration=True,pretty_print=True)
        dom = parseString(xml)
        with open(osp.join(det_folder, str(im_name) + '.xml'), 'wb') as f:
            f.write(xml)
    # os.system('cd {} && zip -r -q  {} {} '.format(dstpath, 'test.zip', 'test'))
    # delete thr files
    #shutil.rmtree(det_folder)

def rbox2poly_single(obj):
    x1, y1, w, h= obj[0],obj[1],obj[2],obj[3]
    bbox_new = [x1,y1,x1+w,y1,x1+w,y1+h,x1,y1+h]
    return bbox_new

def parse_args():
    parser = argparse.ArgumentParser(description='summit result ')
    parser.add_argument('--config', default='work_dirs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py')
    parser.add_argument('--input', default='output.bbox.json')
    parser.add_argument('--output', default='work_dirs/submission')
    args = parser.parse_args()
    return args



def submit():
    args = parse_args()

    # config_file = args.config
    # cfg = Config.fromfile(config_file)
    # data_test = cfg.data['test']
    # dataset = build_dataset(data_test)
    classnames = ('A220', 'A330', 'A320/321', 'Boeing737-800', 'Boeing787', 'ARJ21', 'other')
    creat_submission(args.input, args.output, classnames)

if __name__ == '__main__':
   submit()
