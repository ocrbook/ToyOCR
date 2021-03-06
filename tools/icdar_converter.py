from utils import convert_annotations, drop_orientation, is_not_png, parallel_task, draw_mask
import argparse
import glob
import os.path as osp
from functools import partial
import cv2
import sys
import numpy as np
from shapely.geometry import Polygon
import os


sys.path.insert(0, '.')


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory

    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    files = []
    for img_file in imgs_list:
        gt_file = gt_dir + '/gt_' + osp.splitext(
            osp.basename(img_file))[0] + '.txt'
        files.append((img_file, gt_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, dataset, out_segm_path, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        dataset(str): The dataset name, icdar2015 or icdar2017
        nproc(int): The number of process to collect annotations

    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(dataset, str)
    assert dataset
    assert isinstance(nproc, int)

    load_img_info_with_dataset = partial(
        load_img_info, dataset=dataset, out_segm_path=out_segm_path)

    images = parallel_task(
        load_img_info_with_dataset, files, nproc=nproc)

    return images


def load_img_info(files, dataset, out_segm_path):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
        dataset(str): Dataset name, icdar2015 or icdar2017

    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)
    assert isinstance(dataset, str)

    img_file, gt_file = files
    
    # read imgs with ignoring orientations
    img = cv2.imread(img_file)
    height, width = img.shape[0:2]

    if dataset == 'icdar2017':
        with open(gt_file) as f:
            gt_list = f.readlines()
    elif dataset == 'icdar2015':
        with open(gt_file, mode='r', encoding='utf-8-sig') as f:
            gt_list = f.readlines()
    else:
        raise NotImplementedError(f'Not support {dataset}')


    
    anno_info = []
    polys = []
    for line in gt_list:
        # each line has one ploygen (4 vetices), and others.
        # e.g., 695,885,866,888,867,1146,696,1143,Latin,9
        line = line.strip()
        strs = line.split(',')
        category_id = 1
        xy = [int(x) for x in strs[0:8]]
        coordinates = np.array(xy).reshape(-1, 2)
        polys.append(coordinates)

        polygon = Polygon(coordinates)
        iscrowd = 0
        # set iscrowd to 1 to ignore 1.
        if (dataset == 'icdar2015'
                and strs[8] == '###') or (dataset == 'icdar2017'
                                          and strs[9] == '###'):
            iscrowd = 1

        area = polygon.area

        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[xy])
        anno_info.append(anno)

    mask = draw_mask(polys=polys, height=height, width=width)

    cv2.imwrite(osp.join(out_segm_path, osp.basename(img_file)), mask)

    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.basename(img_file),
        height=img.shape[0],
        width=img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(osp.basename(out_segm_path),osp.basename(img_file)))
    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Icdar2015 or Icdar2017 annotations to COCO format'
    )
    parser.add_argument('icdar_path', help='icdar root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('-d', '--dataset', help='icdar2017 or icdar2015')

    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    icdar_path = args.icdar_path
    out_dir = args.out_dir if args.out_dir else icdar_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    

    for split in ["train", "test"]:
        segm_path = os.path.join(out_dir, split+"_segms")
        if not os.path.exists(segm_path):
            os.makedirs(segm_path)
        
        print(f"{split} phase")
        img_dir = osp.join(icdar_path, split+'_images')
        gt_dir = osp.join(icdar_path, split+'_gts')

        json_name = split+".json"

        files = collect_files(img_dir, gt_dir)
        image_infos = collect_annotations(
            files, args.dataset, segm_path, nproc=args.nproc)
        convert_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
