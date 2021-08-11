import os
import cv2


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
    img = cv2.imread(img_file, 'unchanged')
    # read imgs with orientations as dataloader does when training and testing
    img_color = cv2.imread(img_file, 'color')
    # make sure imgs have no orientation info, or annotation gt is wrong.
    if img.shape[:2] == img_color.shape[:2]:
        return img_file

    target_file = os.path.splitext(img_file)[0] + '.png'
    # read img with ignoring orientation information
    img = cv2.imread(img_file, 'unchanged')
    cv2.imwrite(img, target_file)
    os.remove(img_file)
    print(f'{img_file} has orientation info. Ignore it by converting to png')
    return target_file


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



import json 

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
    
    with open(out_json_name,"w") as fp:
        json.dump(out_json,fp)

    return out_json
