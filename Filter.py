import numpy as np
import math
import os
import cv2 as cv
from torch import row_stack
from scipy import signal
from scipy import ndimage

def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = signal.convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

def zeropadding(filter, size):
    row,col = filter.shape
    output = np.zeros(size)
    mid_row = math.floor(size[0]/2)
    mid_col = math.floor(size[1]/2)
    for i in range(row):
        for j in range(col):
            output[mid_row+i][mid_col+j] = filter[i][j]
    return output

def filtering(filterType, img):
    img_row, img_col= img.shape
    mid_row = math.floor(img_row/2)
    mid_col = math.floor(img_col/2)

    ## LPF blurs the image
    if filterType == 1:  
        rows = 5
        col = 5
        filter = np.array([ [1,4,7,4,1], 
                            [4,16,26,16,4], 
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1,4,7,4,1]]) /273
        print("LPF Filter")
        print(filter)
        filter_out = np.zeros((img_col, img_row))
        output = np.zeros((img_row, img_col))
        output = cv.filter2D(src=img, ddepth=-1, kernel=filter)
        
        return output

    ## median filter
    elif filterType == 2: 
        rows = 3
        col = 3
        filter = np.zeros((rows, col))
        for i in range(rows):
            for j in range(col):
                filter[i][j] = 1/(rows*col)
        print("Median Filter")
        print(filter)
        output = np.zeros((img_row, img_col))
        output = ndimage.convolve(img, filter)
        return output

    ##Laplacian filter
    elif filterType == 3:  
        
        rows = 3
        col = 3
        filter = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        print("Laplacian Filter")
        print(filter)
        
        output = np.zeros((img_row, img_col))
        # output = signal.convolve2d(img, filter)
        # output = multi_convolver(img, filter, 1)
        image_sharp = cv.filter2D(src=img, ddepth=-1, kernel=filter)
        return image_sharp

    ##Roberts cross-Gradiant
    elif filterType == 4:
        #  -1  0
        #   0  1
        rows = 2
        col = 2
        filter = np.array([[-1,0], [0,1]])
        print("Roberts Filter")
        print(filter)
        output = np.zeros((img_row, img_col))
        image_sharp = cv.filter2D(src=img, ddepth=-1, kernel=filter)
        # output = signal.convolve2d(img, filter)
        return image_sharp

    ##Sobel Gradiant
    elif filterType == 5:
        # -1 -2 -1
        #  0  0  0
        #  1  2  1
        rows = 3
        col = 3
        filter = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
        print("Sol Filter")
        print(filter)

        output = np.zeros((img_row, img_col))
        image_sharp = cv.filter2D(src=img, ddepth=-1, kernel=filter)
        # output = signal.convolve2d(img, filter)
        return image_sharp

def filter_gray(img_gray, filterchoice):
    ##Filter with Gaussian
    if filterchoice == 'gau':
        output = filtering(1, img_gray)  
        LPF_img = output.astype(np.uint8)
        # cv.imshow('LPF', LPF_img)
        # cv.waitKey(0)
        return LPF_img

    ## Filter with median
    elif filterchoice == 'med':
        output = filtering(2, img_gray)
        median_img = output.astype(np.uint8)
        # cv.imshow('Med', median_img)
        # cv.waitKey(0)
        return median_img

    ## Filter with Laplacian
    elif filterchoice == 'lap':
        output = filtering(3, img_gray)
        Lap_img = output.astype(np.uint8)
        # cv.imshow('Laplacian', Lap_img)
        # cv.waitKey(0)
        return Lap_img

    ## Filter with Roberts
    elif filterchoice == 'rob':
        output = filtering(4, img_gray)
        Rob_img = output.astype(np.uint8)
        # cv.imshow('Roberts', Rob_img)
        # cv.waitKey(0)
        return Rob_img

    ## Filter with Sobel
    elif filterchoice == 'sol':
        output = filtering(5, img_gray)
        Sob_img = output.astype(np.uint8)
        # cv.imshow('Sobel', Sob_img)
        # cv.waitKey(0)
        return Sob_img


def filter_ori(img, filterType):
    if filterType == 'gau':
        kernel = np.array([
                         [1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]
                         ]) / 9
        resulting_image = cv.filter2D(img, -1, kernel)
        return resulting_image

    elif filterType == 'lap':
        kernel = np.array([
                         [ 0, -1,  0],
                         [-1, 5, -1],
                         [ 0, -1,  0]
                         ])
        resulting_image = cv.filter2D(img, -1, kernel)
        return resulting_image
  





