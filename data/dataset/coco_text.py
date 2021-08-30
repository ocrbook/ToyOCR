from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
import os

__all__ = ["load_coco_text_instances", "register_text_class"]

# fmt: off
CLASS_NAMES = ("text",)

# fmt: on


def load_coco_text_instances(json_file, image_root, class_names):
    """
    Load  detection annotations to Detectron2 format.
    
    Args:
        json_file(str): json file path
        image_root(str): image files root
        class_names(list): class name list
    
    Returns:
        dict: all annotations
    """
    # Needs to read many small annotation files. Makes sense at local

    coco = COCO(json_file)
    catIds = coco.getCatIds(catNms=class_names)
    coco.loadCats()
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(ids=imgIds)


    dicts = []
    for img in imgs:
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        w,h = img['width'],img['height']

        r = {
            "file_name": os.path.join(image_root, img['file_name']),
            "image_id": img['id'],
            "height": img['height'],
            "width": img['width'],
        }
        instances = []
        if "train_images" in r["file_name"]:
            segm_file_name=r["file_name"].replace("train_images","segms")
        else:
            segm_file_name=r["file_name"].replace("images","segms")
            
        if os.path.exists(segm_file_name):
            r["segm_file"]=segm_file_name

        for obj in anns:
            cls = coco.loadCats(obj['category_id'])[0]['name']
            assert cls in class_names
            bbox = obj['bbox']
            bbox = [float(x) for x in bbox]
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            bbox[0] = max(bbox[0], 0.0)
            bbox[1] = max(bbox[1], 0.0)
            bbox[2] = min(bbox[2], float(w))
            bbox[3] = min(bbox[3], float(h))
            if bbox[2] - bbox[0] > 1.0 and bbox[3] - bbox[1] > 1.0:
                instances.append(
                    {"category_id": class_names.index(cls),
                     "bbox": bbox,
                     "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances
        if len(instances) > 0:
            dicts.append(r)

    return dicts


def register_text_class(name, json_file, image_root, class_names):
    DatasetCatalog.register(name, lambda: load_coco_text_instances(json_file, image_root, list(class_names)))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names)
    )
