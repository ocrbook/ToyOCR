import math


def calculate_distance(c1, c2):
    return math.sqrt(math.pow(c1[0]-c2[0], 2) + math.pow(c1[1]-c2[1], 2))


def choose_best_begin_point(points):
    """
    find top-left vertice and resort

    Args:
        points(np.array): points need to process.

    Returns:
        np.array: points processed.
    """

    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]
    x4 = points[3][0]
    y4 = points[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_points = [[xmin, ymin], [
        xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = calculate_distance(combinate[i][0], dst_points[0]) + calculate_distance(combinate[i][1], dst_points[1]) + calculate_distance(
            combinate[i][2], dst_points[2]) + calculate_distance(combinate[i][3], dst_points[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
        # if force_flag != 0:
        #    print("choose one direction!")
    return combinate[force_flag]
