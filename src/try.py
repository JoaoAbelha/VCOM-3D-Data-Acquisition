import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def sobelFilter():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path

    # applying a gaussian blur: it makes disappear noise mainly due to texture and maybe light
    # the bigger the less noise but if is too big we may lose some edges
    blur = cv2.GaussianBlur(img,(31,31),0)

    gray_temp = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray_temp,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    
    # first derivatives: results in matrix with positive and negative value
    # if the kernel size is bigger then more pixels are taken into account in the convulotion (edged can get more blurry)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=31)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=31)

    # combining both derivatives: 
    grad = np.sqrt(grad_x**2 + grad_y**2) # non-negative values
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8) 

    #cv2.imshow('grad X',grad_x)
    #cv2.imshow('grad Y',grad_y)
    # Show some stuff
    imS = ResizeWithAspectRatio(grad_norm, width=400)
    cv2.imshow("result sobel", imS)
    cv2.waitKey()


def nothing(x):
    pass

def cannyFilter():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path
    cv2.namedWindow('canny')
    # add lower and upper threshold slidebars to "canny"
    cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
    cv2.createTrackbar('upper', 'canny', 0, 255, nothing)

    while(1):

        # get current positions of four trackbars
        lower = cv2.getTrackbarPos('lower', 'canny')
        upper = cv2.getTrackbarPos('upper', 'canny')

        edges = cv2.Canny(img, lower, upper)
        imS = ResizeWithAspectRatio(edges, width=400)

        # display images
        #cv2.imshow('original', img)
        cv2.imshow('canny', imS)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:   # hit escape to quit
            break

    cv2.waitKey()

def edgeDetection():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path
    blur = cv2.GaussianBlur(img, (31,31),0) # remove gaussian noise
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # without blur

    #cv2.imshow('gaussian blur', blur)
    # the laplacian operator uses the sobel operator internally: The function calculates the Laplacian of the source image by adding up the second x and y derivatives calculated using the Sobel operator:

    dst = cv2.Laplacian(gray, ddepth = cv2.CV_16S, ksize=3)
    #print(dst)
    abs_dst = cv2.convertScaleAbs(dst) # Scales, calculates absolute values, and converts the result to 8-bit.
    dst2 = cv2.Laplacian(gray2, ddepth = cv2.CV_16S, ksize=3)
    abs_dst2 = cv2.convertScaleAbs(dst2) # Scales, calculates absolute values, and converts the result to 8-bit. but why?


    minLoG = cv2.morphologyEx(dst, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(dst, cv2.MORPH_DILATE, np.ones((3,3)))

    zeroCross = np.logical_or(np.logical_and(minLoG < 0, dst > 0), np.logical_and(maxLoG > 0, dst < 0)) 
    plt.imshow(zeroCross, cmap='gray')
    plt.title('Zero-Crossings')
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey()

def houghTransform():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 70,apertureSize = 5)

    lines = cv2.HoughLines(edges, 1,np.pi/180, 100)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    imS = ResizeWithAspectRatio(img, width=400)

    cv2.imshow('houghlines.jpg',imS)
    cv2.waitKey()

def segmentationThreshold():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The first argument is the source image, which should be a grayscale image
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


def threshold_adaptative():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path
    img = cv2.medianBlur(img,5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


def osu ():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()

def kMeans():
    image = cv2.imread('./../imgs/alternate/Untitled_000037.png') # Change this, according to your image's path

    # reshape image into 2d array and make those numbers floats
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # stop criteria:
        # iterations exceeded (say 100) or if the clusters move less than some epsilon value (let's pick 0.2 here),
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cluster_number = 3

    _, labels, (centers) = cv2.kmeans(pixel_values, cluster_number, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()] # # convert all pixels to the color of the centroids
    # reshape back to the original image dimension
    segmented_image2 = segmented_image.reshape((image.shape))
    imS = ResizeWithAspectRatio(segmented_image2, width=400)
    cv2.imshow('res2', imS)
    cv2.waitKey(0)


def watershed():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,255,0]
    imS = ResizeWithAspectRatio(img, width=400)
    cv2.imshow('res2', imS)

    cv2.waitKey(0)

#ver melhor os parametros
def meanShift():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png')
    dst = cv2.pyrMeanShiftFiltering(img, 10, 255, maxLevel=2)
    imS = ResizeWithAspectRatio(img, width=400)
    cv2.imshow('res2', imS)
    cv2.waitKey()

def grabcut():
    img = cv2.imread('./../imgs/alternate/Untitled_000037.png')
    mask = np.zeros(img.shape[:2], np.uint8) # mask for the output with the same dimension as the intial image
    rect = (10, 10, 250, 250)

    # allocate memory for the two arrays the algorithms uses for the segmentation of the fg and bg
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)

    niter = 5

    mask, bg, fg = cv2.grabCut(img, mask, rect, bg, fg, niter, cv2.GC_INIT_WITH_RECT)

    # all background we are sure or quite sure of is set to 0; the same thing for foregoround which is set to 1
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0 ,1) 
    # scale from 0 to 1 => 0 to 255
    outputMask = (outputMask * 255).astype("uint8")

    image_final = cv2.bitwise_and(img, img, mask = outputMask)
    imS = ResizeWithAspectRatio(image_final, width=400)
    cv2.imshow('res2', imS)
    cv2.waitKey(0)



#sobelFilter()
cannyFilter()
#edgeDetection()
#houghTransform()
#segmentationThreshold()
#threshold_adaptative()
#osu()
#kMeans()
#watershed()
#meanShift()
#grabcut()