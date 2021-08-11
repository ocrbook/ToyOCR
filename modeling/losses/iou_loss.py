import math

import torch
import torch.nn as nn

from .utils import weighted_loss


@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IOU loss.

    Computing the Iou loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IOU.
    Args
    :param pred:(torch.Tensor) Predicted bboxes of format (x1,y1,x2,y2)
    :param target: (torch.Tensor): Correspongding gt bboxes,shape (n,4).
    :param eps: (float):Eps to avoid log(0).
    :return:
    """

    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)

    return ious


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1(Tensor): shape(m,4) int (x1,y1,x2,y2) format.
        bboxes2(Tensor): shape(n,4) in (x1,y1,x2,y2) format.
        mode(str): "iou" (intersection over union) or iof(intersection over foreground).
        is_aligned:

    Returns:
        ious(Tensor): shape(m,n) if is aligned= False else

    """

    assert mode in ["iou", "iof"]

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows,2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows,2]

        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == "iou":
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)

        else:
            ious = overlap / area1

    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])

        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)

            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious
