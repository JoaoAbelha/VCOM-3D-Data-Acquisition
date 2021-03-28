from step2 import camera_position 
import cv2 as cv
import numpy as np

REAL_PENCIL_HEIGHT_MM = 105

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

    return {
        'x': (px+t*(qx-px)), 'y': (py+t*(qy-py)), 'z': (pz+t*(qz-pz))
    }

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

# this function only has dummy values for now
def light_calibration(position):
    # 0. Get two images with the pencil
    pencil = cv.imread('./imgs/pencil.jpg')
    pencil = cv.resize(pencil, (300, 300))
    # 1. Having the camara position find a point in the image plane that we want
        #getTwoPointsOfInterest(pencil)

    # 2. Intersect that point with the z = 0 plane (get K , Ts and T (K + h)), h is known (http://nghiaho.com/?page_id=363)
        # not sure se aqui podemos usar tb o triangulatePoints
    print(intersection((1,0,1), (0,1,0), 0, 0, 1, 0))
    # 3. Calculate the line equation for Ts and T for an image (this step is not needed we can place the points in 5 => vector + point)
    # 4. Repeat step3 for the other image
    # 5. uses square difference to get the intersection point
    d = distance_from_two_lines([1,1,2] , [1,1,3], [0,2,-1], [1,0,-1])
    print(abs(d))

    # 6. if we want better results we can use more than one image

#img = cv.imread('./calibration/GOPR0032.jpg')
#position = camera_position(img)
light_calibration(1)