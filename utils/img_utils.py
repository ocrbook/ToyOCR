import os
import cv2
import json
import numpy as np
import pyclipper
import traceback


def drop_orientation(img_file):
    """Check if the image has orientation information. If yes, ignore it by
    converting the image format to png, and return new filename, otherwise
    return the original filename.

    Args:
        img_file(str): The image path

    Returns:
        The converted image filename with proper postfix
    """
    assert isinstance(img_file, str)
    assert img_file

    # read imgs with ignoring orientations

    target_file = os.path.splitext(img_file)[0] + '.png'
    # read img with ignoring orientation information
    img = cv2.imread(img_file)
    cv2.imwrite(img, target_file)
    os.remove(img_file)
    print(f'{img_file} has orientation info. Ignore it by converting to png')
    return target_file


def perimeter(poly):
    try:
        p = 0
        nums = poly.shape[0]
        for i in range(nums):
            p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
        # logger.debug('perimeter:{}'.format(p))
        return p
    except Exception as e:
        traceback.print_exc()
        raise e


def shrink_poly(poly, r=0.92):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    try:
        area_poly = abs(pyclipper.Area(poly))
        perimeter_poly = perimeter(poly)
        poly_s = []
        pco = pyclipper.PyclipperOffset()
        if perimeter_poly:
            d = area_poly * (1 - r * r) / perimeter_poly
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            poly_s = pco.Execute(-d)
        shrinked_poly = poly_s[0]
        if len(shrinked_poly) < 4:
            return poly
        return shrinked_poly

    except Exception as e:
        return poly


def draw_mask(polys, height, width):
    """Draw tower of the mask

    Args:
        poly: numpy.array
        height: int
        width: int 

    Returns:
        The mask.
    """

    mask = np.zeros((height, width), np.float32)
    grad_list = [1.0, 0.9, 0.8, 0.7]
    score_list = [0.5, 0.7, 0.9, 1.0]
    for poly in polys:
        for grad, score in zip(grad_list, score_list):
            if grad < 1:
                shrinked_poly = shrink_poly(poly, grad)
                cv2.fillPoly(
                    mask, [np.array(shrinked_poly, np.int32)], 255. * score)
            else:
                cv2.fillPoly(mask, [np.array(poly, np.int32)], 255. * score)

    return mask


def is_not_png(img_file):
    """Check img_file is not png image.

    Args:
        img_file(str): The input image file name

    Returns:
        The bool flag indicating whether it is not png
    """
    assert isinstance(img_file, str)
    assert img_file

    suffix = os.path.splitext(img_file)[1]

    return suffix not in ['.PNG', '.png']


def convert_annotations(image_infos, out_json_name):
    """Convert the annotation into coco style.

    Args:
        image_infos(list): The list of image information dicts
        out_json_name(str): The output json filename

    Returns:
        out_json(dict): The coco style dict
    """
    assert isinstance(image_infos, list)
    assert isinstance(out_json_name, str)
    assert out_json_name

    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    cat = dict(id=1, name='text')
    out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    with open(out_json_name, "w") as fp:
        json.dump(out_json, fp)

    return out_json
