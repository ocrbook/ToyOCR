from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
from enum import Enum
import numpy as np
import cv2

def batch_padding(batch_images,
                  div=32,
                  pad_value: float = 0.0):
    max_size = (
        # In tracing mode, x.shape[i] is Tensor, and should not be converted
        # to int: this will cause the traced graph to have hard-coded shapes.
        # Instead we should make max_size a Tensor that depends on these tensors.
        # Using torch.stack twice seems to be the best way to convert
        # list[list[ScalarTensor]] to a Tensor
        torch.stack(
            [
                torch.stack([torch.as_tensor(dim) for dim in size])
                for size in [tuple(img.shape) for img in batch_images]
            ]
        )
        .max(0)
        .values
    )

    if div > 1:
        stride = div
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = torch.cat(
            [max_size[:-2], (max_size[-2:] + (stride - 1)) // stride * stride])

    image_sizes = [tuple(im.shape[-2:]) for im in batch_images]

    if len(batch_images) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [0, max_size[-1] - image_size[1],
                        0, max_size[-2] - image_size[0]]
        if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
            batched_imgs = batch_images[0].unsqueeze(0)
        else:
            padded = F.pad(batch_images[0], padding_size, value=pad_value)
            batched_imgs = padded.unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore use tuple()
        batch_shape = (len(batch_images),) + tuple(max_size)
        batched_imgs = batch_images[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(batch_images, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs


def create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):
    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100)
                      for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "")
                  for l, crowd in zip(labels, is_crowd)]
    return labels


def mask_up_dim(img, ratio=4):
    x, y = img.shape[0:2]
    img = cv2.resize(img, (y*ratio, x*ratio))
    new_img = np.zeros((y*ratio, x*ratio, 3))
    new_img[:, :, 0] = img
    return new_img


class Color(Enum):
    """An enum that defines common colors.
    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
