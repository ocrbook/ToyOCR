
import pyclipper
import numpy as np



def perimeter(poly):

    p = 0
    nums = poly.shape[0]
    for i in range(nums):
        p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
    # logger.debug('perimeter:{}'.format(p))
    return p



def shrink_poly(poly, r=0.92):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''

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




def draw_mask(poly, height, width):
    mask = np.zeros((height, width), np.float32)
    grad_list = [1.0, 0.95, 0.9, 0.85]
    score_list = [0.8, 0.9, 0.95, 1.0]
    for grad, score in zip(grad_list, score_list):
        if grad < 1:
            shrinked_poly = shrink_poly(poly, grad)
            cv2.fillPoly(
                mask, [np.array(shrinked_poly, np.int32)], 255. * score)
        else:
            cv2.fillPoly(mask, [np.array(poly, np.int32)], 255. * score)

    mask = mask / 255.0

    return mask


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return -np.sum(edge) / 2.
