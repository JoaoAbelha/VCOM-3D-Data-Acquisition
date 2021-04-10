from step2 import camera_position, readIntrinsicParameters
from step1 import camera_calibration
import cv2 as cv
import numpy as np

REAL_PENCIL_HEIGHT_MM = 105

cursor_held = False
cursor_position = None

def find_point(img_foreground, imga, nr):
    img_foreground  = cv.GaussianBlur(img_foreground, (5, 5), 0)
    cv.imshow('houghrfflines3.jpg', img_foreground)

    contours, _ = cv.findContours(img_foreground, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #get the biggest one
    best_area = 0
    for countour in contours:
        current_area = cv.contourArea(countour)
        
        if current_area > best_area:
            best_area = current_area
            rect = cv.minAreaRect(countour)
            box = cv.boxPoints(rect)
            box = np.int0(box)

    print(box)

    cv.drawContours(imga,[box],0,(0,0,255),1)
    box = sorted(box, key = lambda k: k[1])
    base_corner = (box[nr], box[nr + 1])
    cv.circle(imga, ( int((base_corner[0][0] + base_corner[1][0]) / 2), int((base_corner[0][1] + base_corner[1][1]) / 2)) , \
        radius=5, color=(0, 255, 0), thickness=1)

    cv.imshow('nr' + str(nr) + '.jpg', imga)


def getTwoPointsOfInterest(img):
    imga = img.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    _, im = cv.threshold(gray, 130, 255,cv.THRESH_BINARY_INV)
    _, img_foreground = cv.threshold(gray, 70, 255,cv.THRESH_BINARY_INV)
    img_foreground  = cv.GaussianBlur(img_foreground, (5, 5), 0)

    shadow = cv.subtract(im, img_foreground)
    kernel = np.ones((1,5), np.uint8)  
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
    
    t = - ( a*px + b*py + c*pz + d ) / denom

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
    d = np.dot(n, np.subtract(r1 , r2))
    
    return d

# mouse callback function
def handler(event,x,y,flags,param):
    global cursor_held,cursor_position
    if event == cv.EVENT_LBUTTONUP and cursor_held:
        cursor_held = False
    if event == cv.EVENT_LBUTTONDOWN and not cursor_held:
        cursor_held = True
        cursor_position = (x,y,0)
        print("Set position")

def getUVcoords(img):
    global cursor_position
    orig_height, orig_width, _ = img.shape
    new_size = 1200 / orig_width
    imgb = cv.resize(img,(int(new_size * orig_width), int(new_size * orig_height)))
    cursor_position = None
    while cursor_position is None:
        cv.namedWindow('interactive')
        cv.setMouseCallback('interactive', handler)
        cv.imshow('interactive', imgb)
        cv.waitKey(1)
    cv.destroyAllWindows()
    print("Cursor position {}".format(str(cursor_position)))
    u = cursor_position[0] - new_size * orig_width / 2
    v = cursor_position[1] - new_size * orig_height / 2
    uv = np.array([[cursor_position[0] / new_size,cursor_position[1] / new_size,1]], dtype=np.float32).T
    return uv

#returns the A, B, C, and D components of a plane that's perpendicular to x = 0 and contains the points passed as arguments
def get_perpendicular_plane(point1, point2):
    p1, p2 = np.array(point1), np.array(point2)
    p3 = np.copy(p2)
    print(" {}".format(str(p3)))
    p3[0] += 1

    vector12 = p2 - p1
    vector12 = vector12 / np.linalg.norm(vector12)
    vector23 = p3 - p2
    vector23 = vector23 / np.linalg.norm(vector23)

    normal = np.cross(vector12, vector23)

    A,B,C = [normal[i] for i in range(0,len(normal))]
    D = A*p1[0] + B*p1[1] + C*p1[2]

    return [A,B,C,D]


def plane_adjustments(image):
    print("Point to the intersection of the three planes")
    uv = getUVcoords(image)
    position_aux = np.linalg.inv(mtx).dot(uv) - tvec
    image_plane_position = np.linalg.inv(np.matrix(rotM)).dot(position_aux)
    flat_image_position = np.squeeze(np.asarray(image_plane_position))
    print("Image plane position {}".format(str(flat_image_position)))
    vertex_position = intersection(flat_image_position,flat_camera_position,1,0,0,0)
    print("World position {}".format(str(vertex_position)))

    print("Point to the intersection between the x = 0 plane and a second plane")
    uv = getUVcoords(image)
    position_aux = np.linalg.inv(mtx).dot(uv) - tvec
    image_plane_position = np.linalg.inv(np.matrix(rotM)).dot(position_aux)
    flat_image_position = np.squeeze(np.asarray(image_plane_position))
    print("Image plane position {}".format(str(flat_image_position)))
    intersection1_position = intersection(flat_image_position,flat_camera_position,1,0,0,0)
    print("World position {}".format(str(intersection1_position)))

    print("Point to the intersection between the x = 0 plane and a third plane")
    uv = getUVcoords(image)
    position_aux = np.linalg.inv(mtx).dot(uv) - tvec
    image_plane_position = np.linalg.inv(np.matrix(rotM)).dot(position_aux)
    flat_image_position = np.squeeze(np.asarray(image_plane_position))
    print("Image plane position {}".format(str(flat_image_position)))
    intersection2_position = intersection(flat_image_position,flat_camera_position,1,0,0,0)
    print("World position {}".format(str(intersection2_position)))

    plane1 = [1,0,0,0]  # x = 0
    plane2 = get_perpendicular_plane(vertex_position, intersection1_position)
    plane3 = get_perpendicular_plane(vertex_position, intersection2_position)

    return plane1,plane2,plane3

# this function only has dummy values for now
def light_calibration(frame):
    image = cv.imread('./imgs/alternate/i ({}).png'.format(frame))
    cv.imshow("img", image)

    # Difference between the control image and the frame being analised

    # Thinning of the scan line

    # Choose three random points that are a part of the background

    # Calculate the shadow plane


(mtx, dist) = readIntrinsicParameters()
img = cv.imread('./imgs/alternate3/checkerboard.png')
camera_position, rvec, tvec, rotM, r_camera_position = camera_position(img)
flat_camera_position = np.squeeze(np.asarray(r_camera_position))

planes = cv.imread('./imgs/alternate3/planes.png')
plane1, plane2, plane3 = plane_adjustments(planes)

control_image = cv.imread('./imgs/alternate3/i (1).png')
light_calibration(13)
#for i in range(2, 2):  
#    light_calibration(i)
