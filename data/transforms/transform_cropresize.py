# -*- coding: utf-8 -*-
# Copyright (c) shuchun  and its affiliates.
import cv2
import numpy as np

"""
Thanks the https://github.com/MhLiao/DB & https://github.com/argman/EAST

"""


__all__ = [
    "RandomCropTransform"
]


class RandomCropTransform:
    """
    Extracts a subregion from the source image and scales it to the outpu size.

    In the same time, we also extract the same area mask and gts.
    """

    def __init__(self, crop_size: tuple = (512, 512), max_tries: int = 50, min_crop_side_ratio: float = 0.1):
        """
        Args:
            crop_size(h,w): dst crop size
            max_tries(int): max try times.
            min_crop_size_ratio(float): min ratio of the croped size of the origin image

        """
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.size = crop_size

    def __call__(self, img: np.ndarray, polys: list, segm: np.ndarray, mask: np.ndarray):
        """
        Args:
            img(ndarray): origin image
            polys(ndarray): the polys
            segm(ndarray): the gt segm
            mask(ndarray): the mask area

        Returns:
            ndarray: the cropped image(s) after applying affine transform. 
            ndarray: the cropped segm
            ndarray: the cropped mask
            meta: the scale 

        """

        assert img.shape[0:2] == segm.shape
        assert img.shape[0:2] == mask.shape

        all_care_polys = polys
        crop_x, crop_y, crop_w, crop_h = self._crop_area(img, all_care_polys)
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)

        h = int(crop_h * scale)
        w = int(crop_w * scale)

        rets = []
        for im in [img, segm, mask]:

            cropped_size = self.size
            if im.ndim > 2:
                cropped_size = (self.size[1], self.size[0], im.shape[-1])
            cropped_img = np.zeros(cropped_size, im.dtype)
            cropped_img[:h, :w] = cv2.resize(
                im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))

            rets.append(cropped_img)

        return rets

    def _is_poly_in_rect(self, poly, x, y, w, h):
        """
        Check if the poly in the rect.
        """
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def _is_poly_outside_rect(self, poly, x, y, w, h):
        """
        Check if the poly out side the rect.
        """
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def _split_regions(self, axis):
        """
        split the regions 
        """
        regions = []
        min_axis = 0

        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def _random_select(self, axis, max_size):
        """
        Random select the min and max value
        """
        xx = np.random.choice(axis, size=2)

        xmin = np.min(xx)
        xmax = np.max(xx)

        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)

        return xmin, xmax

    def _region_wise_random_select(self, regions, max_size):
        """
        Random select the region
        """
        selected_index = list(np.random.choice(len(regions), 2))

        selected_values = []

        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)

        return min(selected_values), max(selected_values)

    def _crop_area(self, img, polys):
        """
        Random crop the valid area with the polys info
        """
        h, w, _ = img.shape

        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)

        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1

        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self._split_regions(h_axis)
        w_regions = self._split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self._region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self._random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self._region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self._random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self._is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h
