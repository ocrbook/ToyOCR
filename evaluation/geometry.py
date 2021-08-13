import math
import numpy as np


def order_points_clockwise(pts, reverse=False):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    is_list = False

    if isinstance(pts, list):
        is_list = True
        pts = np.array(pts)

    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    ret = [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]

    if reverse:
        ret = ret[::-1]

    if not is_list:
        ret = np.array(ret)

    return ret
