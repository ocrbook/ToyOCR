from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from inference.centernet import build_model
from detectron2.config import get_cfg
from configs import add_centernet_config
import cv2

import sys
import os
from tqdm import tqdm
from utils import create_text_labels
import numpy as np

sys.path.insert(0, '.')

if __name__ == "__main__":
    # cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(
        "yamls/layout_analysis/centernet_res50_layout_analysis.yaml")

    # model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(
        "exp_results/layout_exp_R50/model_final.pth")
    model.eval()

    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    class_names = ('text', 'title', 'list', 'table', 'figure')
    # txt
    txt = open('results.txt', 'w')

    out_dir = 'exp_results/layout_exp_R50/results/'
    os.mkdir(out_dir)
    thickness = 1
    bbox_color = (0, 255, 0)
    text_color = bbox_color
    font_scale = 0.5
    # images
    #lines = open('datasets/bjz_multicameras_20200615/txts/bjz_val_multicameras_20200615.txt').readlines()
    root = 'datasets/layout/images/'
    images = [root + i for i in sorted(os.listdir(root))]
    images = images[0:200]
    bs = 8
    for i in tqdm(range(0, len(images), 8)):
        images_rgb = [cv2.imread(j)[:, :, ::-1] for j in images[i:i + 8]]
        img_names = [os.path.basename(j) for j in images[i:i + 8]]
        results = model.inference_on_images(images_rgb, K=100, max_size=640)
        for k, result in enumerate(results):
            cls = result['cls'].cpu().numpy()
            boxes = result['bbox'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            H, W, C = images_rgb[k].shape
            img = images_rgb[k][:, :, ::-1]
            img_name = img_names[k]
            for c, (x1, y1, x2, y2), s in zip(cls, boxes, scores):
                if c != 0.0 or s < 0.35:
                    continue
                x1 = str(max(0, int(x1)))
                y1 = str(max(0, int(y1)))
                x2 = str(min(W, int(x2)))
                y2 = str(min(H, int(y2)))
                s = str(round(float(s), 3))
                line = ','.join([img_name, s, x1, y1, x2, y2])
                cv2.rectangle(img, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)
                txt.write(line+'\n')
            classes = cls.tolist()
            labels = [class_names[int(c)] for c in classes]
            for bbox, label, score in zip(boxes, labels, scores):
                if score < 0.35:
                    continue
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                cv2.rectangle(
                    img, left_top, right_bottom, bbox_color, thickness=thickness)
                if len(bbox) > 4:
                    label_text += f'|{bbox[-1]:.02f}'
                cv2.putText(img, label, (bbox_int[0], bbox_int[1] - 2),
                            cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

            cv2.imwrite(os.path.join(out_dir, img_name), img)
