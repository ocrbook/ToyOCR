
import cv2
import json


gt_path="../datasets/icdar/val.json"
pred_path="../exp_results/final/inference/text_results.json"

with open(gt_path) as fp:
    gt_data=json.load(fp)

with open(pred_path) as fp:
    pred_data=json.load(fp)

gt_images = gt_data["images"]
gt_annos = gt_data["annotations"]
gt_image_maps = dict()

for image in gt_images:
    gt_image_maps[image["id"]]=image["file_name"]

gt_image_anno_maps=dict()

pred_image_anno_maps=dict()

for anno in gt_annos:
    img_id=anno["image_id"]
    if img_id in gt_image_anno_maps:
        gt_image_anno_maps[img_id].append(anno)
    else:
        anno_list=[anno]
        gt_image_anno_maps[img_id] = anno_list

for anno in pred_data:
    img_id=anno["image_id"]
    if img_id in pred_image_anno_maps:
        pred_image_anno_maps[img_id].append(anno)
    else:
        anno_list=[anno]
        pred_image_anno_maps[img_id] = anno_list

import cv2
import os 
import numpy as np
image_root = "../datasets/icdar/images/"
first =True

cnt = 1 
for img_id,file_name in gt_image_maps.items():
    gt_annos=gt_image_anno_maps[img_id]
    if img_id not in pred_image_anno_maps:
        print("bad img_id:",img_id)
        continue
    pred_annos=pred_image_anno_maps[img_id]
    
    image_path=os.path.join(image_root,file_name)
    if first:
        print(image_path)
        first =False
    img=cv2.imread(image_path)
    for pred_anno in pred_annos:
        pred_polys=pred_anno["polys"]     
        pred_polys=np.array(pred_polys,np.int)
        img=cv2.polylines(img, [pred_polys], True, color=(255, 255, 155), thickness=2)
        
    for gt_anno in gt_annos:
        print(gt_anno)
        gt_polys=gt_anno["text"]["points"]
        gt_polys=np.array(gt_polys,np.int)
        img= cv2.polylines(img, [gt_polys], True, color=(0, 0, 255), thickness=2)
        
    cnt +=1
    print(img.shape)
    cv2.imwrite(os.path.join("results",file_name),img)
    if cnt>300:
        break
