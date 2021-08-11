from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
import os



__all__ = ["load_layout_instances", "register_layout"]

CLASS_NAMES=('text','title','list','table','figure')


def load_layout_instances(json_file, image_root, class_names):
    """
    Load crowdhuman detection annotations to Detectron2 format.
    """
    coco = COCO(json_file)
    print(class_names)
    catIds = coco.getCatIds(catNms=class_names)

    coco.loadCats()
    imgIds = coco.getImgIds()
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


def register_layout(name, json_file, image_root, class_names):
    DatasetCatalog.register(name, lambda: load_layout_instances(json_file, image_root, list(class_names)))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names)
    )
