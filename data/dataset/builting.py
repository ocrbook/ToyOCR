from .coco_text import register_text_class
from .layout_analysis import register_layout
from detectron2.data import MetadataCatalog
import os


def register_layout_analysis(root):
    SPLITS=[(
        "layout_train",
        "datasets/layout/train_layout.json",
        "datasets/layout/images/",
        ['text','title','list','table','figure',]),
        (
        "layout_val",
        "datasets/layout/val_layout.json",
        "datasets/layout/images/",
        ['text','title','list','table','figure',])
        ]
    for name, json_file, image_root, class_names in SPLITS:
        register_layout(name, json_file, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "coco_class"

def register_text_detection(root):
    SPLITS=[(
        "icdar_train",
        "datasets/icdar/train.json",
        "datasets/icdar/images/",
        ['text',]),
        (
        "icdar_val",
        "datasets/icdar/val_filter_curve.json",
        "datasets/icdar/images/",
        ['text',])
        ]
    for name, json_file, image_root, class_names in SPLITS:
        register_text_class(name, json_file, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "text"

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_text_detection(_root)
register_layout_analysis(_root)

