# -*- coding: utf-8 -*-

import argparse
import glob
import os.path as osp
from functools import partial
from fvcore.common.file_io import PathHandler, PathManager
import sys
import numpy as np
from shapely.geometry import Polygon, geo
from pycocotools.coco import COCO
import json
import os
from evaluation import geometry
import numpy as np

sys.path.insert(0, '.')


def convert_coco_to_icdar(img_info, anno_infos, out_dir):
    file_name = img_info["id"]
    txt_file_name = os.path.join(out_dir, str(file_name) + ".txt")

    for anno in anno_infos:
        points = anno["text"]["points"]
        if len(points) > 4:
            print(img_info["file_name"])
            return

    with open(txt_file_name, "w", encoding='utf-8') as fp:
        for anno in anno_infos:

            text = anno["text"]
            label = text["label"]
            points = text["points"]

            #points = order_points_clockwise(points,reverse=True)
            str_points = [str(p[0])+","+str(p[1]) for p in points]
            str_out = ",".join(str_points)+',####'+label

            fp.writelines(str_out+"\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO format to Icdar2015 or Icdar2017 annotations format'
    )
    parser.add_argument('coco_path', help='coco root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('-d', '--dataset', help='coco')
    parser.add_argument(
        '--split-list',
        nargs='+',
        help='a list of splits. e.g., "--split-list training validation test"')

    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    out_dir = args.out_dir
    PathManager.mkdirs(out_dir)

    coco = COCO(coco_path)

    img_ids = coco.imgs.keys()

    for img_id in img_ids:

        img_infos = coco.loadImgs([img_id])

        annIds = coco.getAnnIds(imgIds=img_infos[0]['id'])
        anno_infos = coco.loadAnns(annIds)
        convert_coco_to_icdar(img_infos[0], anno_infos, out_dir)
    print("Done!!!")


if __name__ == '__main__':
    main()
