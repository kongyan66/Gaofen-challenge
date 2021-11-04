"""
created for isprs （“中科星图杯国际高分遥感大赛”）
2021.9.27
xml to csv(中间格式）
"""
import csv
import os
import xml.dom.minidom as md

from tqdm import tqdm

xmls = '/home/z/code/s2anet/data/summit_test/gt/'
res = csv.writer(open('/home/z/code/s2anet/data/summit_test/test.csv','w', newline=''))
unique_cat = []

for xml in tqdm(os.listdir(xmls)):

    DOMTree = md.parse(xmls + xml)
    #print(type(DOMTree))
    collection = DOMTree.documentElement
    # print(collection.toxml())                    #返回xml的文档内容
    # print(collection.firstChild)
    categories = collection.getElementsByTagName('name')
    boxes = collection.getElementsByTagName("point")
    category_list = []
    box_list = []

    for cat in categories:
        category_list.append(cat.childNodes[0].data)
        if cat.childNodes[0].data not in unique_cat:
            unique_cat.append(cat.childNodes[0].data)

    for box in boxes:
        box_list.append(box.childNodes[0].data.split(','))

    n =0
    step = 5
    for category in category_list:
        res.writerow([xml.split('.')[0],box_list[n][0],box_list[n][1],box_list[n+2][0],box_list[n+2][1],category])
        n += step
print('done!')
