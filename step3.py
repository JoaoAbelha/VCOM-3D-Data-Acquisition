from step2 import camera_position, readIntrinsicParameters
from step1 import camera_calibration
from step5_2 import getShadowPoints
from step6 import calculate3DPoint
import cv2 as cv
import numpy as np
import random as rng

BASE_PLANE = [1, 0, 0, 0]
REAL_PENCIL_HEIGHT_MM = 105

FOREGROUND_THRESHOLD = 50
BACKGROUND_THRESHOLD = 127

cursor_held = False
cursor_position = None


def find_point(img_foreground, imga, nr):
    img_foreground = cv.GaussianBlur(img_foreground, (5, 5), 0)
    cv.imshow('houghrfflines3.jpg', img_foreground)

    contours, _ = cv.findContours(
        img_foreground, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # get the biggest one
    best_area = 0
    for countour in contours:
        current_area = cv.contourArea(countour)

        if current_area > best_area:
            best_area = current_area
            rect = cv.minAreaRect(countour)
            box = cv.boxPoints(rect)
            box = np.int0(box)

    print(box)

    cv.drawContours(imga, [box], 0, (0, 0, 255), 1)
    box = sorted(box, key=lambda k: k[1])
    base_corner = (box[nr], box[nr + 1])
    cv.circle(imga, (int((base_corner[0][0] + base_corner[1][0]) / 2), int((base_corner[0][1] + base_corner[1][1]) / 2)),
              radius=5, color=(0, 255, 0), thickness=1)

    cv.imshow('nr' + str(nr) + '.jpg', imga)


def getTwoPointsOfInterest(img):
    imga = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, im = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV)
    _, img_foreground = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)
    img_foreground = cv.GaussianBlur(img_foreground, (5, 5), 0)

    shadow = cv.subtract(im, img_foreground)
    kernel = np.ones((1, 5), np.uint8)
    img_erosion = cv.erode(shadow, kernel, iterations=2)

    find_point(img_erosion, imga, 0)
    find_point(img_foreground, imga, 2)
    cv.waitKey()

# calculates the intersection of a plan and a line such that:
# line => has p and q points
# plane: ax + bx + cx + d = 0


def intersection(p, q, a, b, c, d):
    px = p[0]
    py = p[1]
    pz = p[2]
    qx = q[0]
    qy = q[1]
    qz = q[2]

    denom = a*(qx-px) + b*(qy-py) + c*(qz-pz)

    if denom == 0:
        return None

    t = - (a*px + b*py + c*pz + d) / denom

    return [px+t*(qx-px), py+t*(qy-py), pz+t*(qz-pz)]

# calculates the minimum distance between two lines
# line1 => vector (v1) and a point (r1)
# idem for line2


def distance_from_two_lines(v1, v2, r1, r2):

    # Find the unit vector perpendicular to both lines
    n = np.cross(v1, v2)
    if (n == [0, 0, 0]).all():
        # se acharem melhor posso retornar na mma a distancia
        # so que neste caso n faz mto sentido serem paralelos de qq forma
        print("lines are parallel")
        exit(1)

    n = n / np.linalg.norm(n)
    # Calculate distance
    d = np.dot(n, np.subtract(r1, r2))

    return d

# mouse callback function


def handler(event, x, y, flags, param):
    global cursor_held, cursor_position
    if event == cv.EVENT_LBUTTONUP and cursor_held:
        cursor_held = False
    if event == cv.EVENT_LBUTTONDOWN and not cursor_held:
        cursor_held = True
        cursor_position = (x, y, 0)
        print("Set position")


def getUVcoords(img):
    global cursor_position
    orig_height, orig_width, _ = img.shape
    new_size = 1200 / orig_width
    imgb = cv.resize(img, (int(new_size * orig_width),
                           int(new_size * orig_height)))
    cursor_position = None
    while cursor_position is None:
        cv.namedWindow('interactive')
        cv.setMouseCallback('interactive', handler)
        cv.imshow('interactive', imgb)
        cv.waitKey(1)
    cv.destroyAllWindows()
    print("Cursor position {}".format(str(cursor_position)))
    uv = np.array([[cursor_position[0] / new_size,
                    cursor_position[1] / new_size, 1]], dtype=np.float32).T
    return uv


def calculate_plane(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return [a, b, c, d]

# returns the A, B, C, and D components of a plane that's perpendicular to x = 0 and contains the points passed as arguments


def get_perpendicular_plane(point1, point2):
    p1, p2 = np.array(point1), np.array(point2)
    p3 = np.copy(p2)
    print(" {}".format(str(p3)))
    p3[0] += 1

    return calculate_plane(p1, p2, p3)


def project_image_point_to_plane(point, plane, mtx, rotM, camera_pos):
    position_aux = np.linalg.inv(mtx).dot(point)  # - tvec
    image_plane_position = np.linalg.inv(np.matrix(rotM)).dot(position_aux)
    flat_image_position = np.squeeze(np.asarray(image_plane_position))
    print("Image plane position {}".format(str(flat_image_position)))
    world_position = intersection(
        flat_image_position, camera_pos, plane[0], plane[1], plane[2], plane[3])
    print("World position {}".format(str(world_position)))
    return world_position


def plane_adjustments_alt(image, mask):
    edges = cv.Canny(image, 60, 75)
    cv.imshow("edges", edges)
    edges_filtered = cv.bitwise_and(edges, edges, mask=mask)
    cv.imshow("edges filtered", edges_filtered)
    cv.waitKey(0)

    # finding contours, can use connectedcomponents aswell
    contours, hierarchy = cv.findContours(
        edges_filtered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros(
        (edges_filtered.shape[0], edges_filtered.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color,
                        2, cv.LINE_8, hierarchy, 0)

    cv.imshow("image", drawing)
    cv.waitKey(0)

    print(len(contours))
    # Choose the first pair of points
    contour_planes = np.zeros((len(contours), 4), dtype=np.float32)
    for i in range(len(contours)):
        contour = contours[i].copy()
        if len(contour) < 2:
            continue

        print(contour)
        index_a = rng.randint(0, len(contour) - 1)
        print(index_a)
        print(contour[index_a])
        print(contour[index_a][0])
        im_point_a = contour[index_a][0]
        point_a = np.array([[im_point_a[0] / image.shape[0],
                             im_point_a[1] / image.shape[1], 1]], dtype=np.float32).T
        contour = np.delete(contour, index_a, 0)
        print(contour)

        index_b = rng.randint(0, len(contour) - 1)
        print(index_b)
        print(contour[index_b])
        print(contour[index_b][0])
        im_point_b = contour[index_b][0]
        point_b = np.array([[im_point_b[0] / image.shape[0],
                             im_point_b[1] / image.shape[1], 1]], dtype=np.float32).T

        world_point_a = project_image_point_to_plane(point_a, [1, 0, 0, 0])
        world_point_b = project_image_point_to_plane(point_b, [1, 0, 0, 0])

        print("A Image point {} : World point {}".format(
            str(point_a), str(world_point_a)))
        print("B Image point {} : World point {}".format(
            str(point_b), str(world_point_b)))

        contour_planes[i] = get_perpendicular_plane(
            world_point_a, world_point_b)

    for i in range(len(contour_planes)):
        plane_a = contour_planes[i]
        vector_a = (plane_a[0], plane_a[1], plane_a[2])
        print(vector_a)
        print(np.linalg.norm(vector_a))
        unit_a = vector_a / np.linalg.norm(vector_a)

        for j in range(i + 1, len(contour_planes)):
            plane_b = contour_planes[j]
            vector_b = (plane_b[0], plane_b[1], plane_b[2])
            print(vector_b)
            print(np.linalg.norm(vector_b))
            unit_b = vector_b / np.linalg.norm(vector_b)

            angle = np.arccos(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))

            #print("Angle between {} and {} is {}".format(str(unit_a), str(unit_b), str(angle)))

            if np.abs(angle - np.pi / 2) < np.pi / 32:
                return [1, 0, 0, 0], plane_a, plane_b


def plane_adjustments(image, mtx, rotM, camera_pos):
    print("Point to the intersection of the three planes")
    uv = getUVcoords(image)
    vertex_position = project_image_point_to_plane(
        uv, [1, 0, 0, 0], mtx, rotM, camera_pos)

    print("Point to the intersection between the x = 0 plane and a second plane")
    uv = getUVcoords(image)
    intersection1_position = project_image_point_to_plane(
        uv, [1, 0, 0, 0], mtx, rotM, camera_pos)

    print("Point to the intersection between the x = 0 plane and a third plane")
    uv = getUVcoords(image)
    intersection2_position = project_image_point_to_plane(
        uv, [1, 0, 0, 0], mtx, rotM, camera_pos)

    plane1 = [1, 0, 0, 0]  # x = 0
    plane2 = get_perpendicular_plane(vertex_position, intersection1_position)
    plane3 = get_perpendicular_plane(vertex_position, intersection2_position)

    return plane1, plane2, plane3


def decompose_image(background, foreground):
    bg_grey = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    _, bg_mask_aux = cv.threshold(
        bg_grey, BACKGROUND_THRESHOLD, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    if foreground is None:
        return bg_mask_aux, None

    difference = cv.subtract(foreground, background)

    diff_grey = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    diff_eq = cv.equalizeHist(diff_grey)
    _, diff_mask_aux = cv.threshold(
        diff_eq, FOREGROUND_THRESHOLD, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    bg_mask = cv.bitwise_and(bg_mask_aux, bg_mask_aux,
                             mask=cv.bitwise_not(diff_mask_aux))
    fg_mask = cv.bitwise_and(diff_mask_aux, bg_mask_aux)

    return bg_mask, fg_mask


def project_image_point_to_space(point, mtx, rotM, camera_pos, planes):
    point1 = project_image_point_to_plane(
        point, planes[0], mtx, rotM, camera_pos)
    print(point1)
    print(camera_pos)
    distance1 = np.linalg.norm(point1 - camera_pos)

    point2 = project_image_point_to_plane(
        point, planes[1], mtx, rotM, camera_pos)
    distance2 = np.linalg.norm(point2 - camera_pos)

    point3 = project_image_point_to_plane(
        point, planes[2], mtx, rotM, camera_pos)
    distance3 = np.linalg.norm(point3 - camera_pos)

    if distance1 >= distance2 and distance1 >= distance3:
        return np.asarray(point1)
    elif distance2 >= distance3:
        return np.asarray(point2)
    else:
        return np.asarray(point3)

# this function only has dummy values for now


def light_calibration(image, mtx, rotM, camera_pos, planes, mask=None):
    flat_camera_position = np.squeeze(np.asarray(camera_pos))

    # Obtain scan line
    grey_scan = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    _, thresh_scan = cv.threshold(grey_scan, 235, 255, cv.THRESH_BINARY)
    rgb_thresh_scan = cv.merge((thresh_scan, thresh_scan, thresh_scan))
    cv.imshow("rgv", rgb_thresh_scan)
    cv.waitKey()
    scan_line = getShadowPoints(rgb_thresh_scan)

    if len(scan_line) < 3:
        return []

    # Choose three random points that are a part of the background
    index_a = rng.randint(0, len(scan_line) - 1)
    point_a = np.asarray(scan_line[index_a])
    point_a = np.array([[point_a[0], point_a[1], 1]], dtype=np.float32).T
    print(point_a)
    p1 = project_image_point_to_space(
        point_a, mtx, rotM, flat_camera_position, planes)

    scan_line = np.delete(scan_line, index_a, 0)

    index_b = rng.randint(0, len(scan_line) - 1)
    point_b = np.asarray(scan_line[index_b])
    point_b = np.array([[point_b[0], point_b[1], 1]], dtype=np.float32).T
    print(point_b)
    p2 = project_image_point_to_space(
        point_b, mtx, rotM, flat_camera_position, planes)

    scan_line = np.delete(scan_line, index_b, 0)

    index_c = rng.randint(0, len(scan_line) - 1)
    point_c = np.asarray(scan_line[index_c])
    point_c = np.array([[point_c[0], point_c[1], 1]], dtype=np.float32).T
    print(point_c)
    p3 = project_image_point_to_space(
        point_c, mtx, rotM, flat_camera_position, planes)
    # Calculate the shadow plane

    return calculate_plane(p1, p2, p3)


def calibrate_planes(mtx, rotM, camera_pos, image, control_image=None):
    flat_camera_position = np.squeeze(np.asarray(camera_pos))

    bg_mask, fg_mask = decompose_image(image, control_image)

    plane1, plane2, plane3 = plane_adjustments(
        image, mtx, rotM, flat_camera_position)

    return bg_mask, fg_mask, (plane1, plane2, plane3)


def shadowPlane(step3Config, image, projection_matrix, mtx, dist, steps):
    shadowPoints = []
    shadowPoints = getShadowPoints(image, steps)

    point1 = max(shadowPoints, key=lambda x: x[0])
    point2 = min(shadowPoints, key=lambda x: x[0])
    point3 = min(shadowPoints, key=lambda x: x[1])

    if steps:
        #img = np.zeros(image)
        cv.circle(image, point1,
                  radius=5, color=(0, 255, 0), thickness=1)
        cv.circle(image, point2,
                  radius=5, color=(255, 0, 0), thickness=1)
        cv.circle(image, point3,
                  radius=5, color=(0, 0, 255), thickness=1)
        cv.imshow('Points to calulate shadow plane', image)
        cv.waitKey(5000)
        cv.destroyAllWindows()

    point13D = calculate3DPoint(
        point1[0], point1[1], projection_matrix, BASE_PLANE)
    point23D = calculate3DPoint(
        point2[0], point2[1], projection_matrix, BASE_PLANE)
    point33D = calculate3DPoint(point3[0], point3[1], projection_matrix, [
                                1, 0, 0, step3Config['Object Height']])

    print(point13D, point23D, point33D)

    return calculate_plane(np.asarray(point13D), np.asarray(point23D), np.asarray(point33D))
