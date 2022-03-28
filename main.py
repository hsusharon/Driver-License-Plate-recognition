import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from Filter import *
from opencv_self import *
from opencv import *
from Noiseadder import *

print("1-Generate noise image  2-Plate recognition")
val = input('Operation:')
print(val)
ori_filename = "test/image4.jpg"
gau_filename = "C:/Users/USER/Documents/Play Ground/Image processing Project/OpenCV/gaussian/gaus_img5.jpg"
pro_gau_filename = "C:/Users/USER/Documents/Play Ground/Image processing Project/OpenCV/gaussian/pro_gaus_img5.jpg"
pep_filename = "C:/Users/USER/Documents/Play Ground/Image processing Project/OpenCV/pepper/pepper_img5.jpg"
pro_pep_filename = "C:/Users/USER/Documents/Play Ground/Image processing Project/OpenCV/pepper/pro_pepper_img5.jpg"


if val == '1':
    noiseadder(ori_filename, gau_filename, pro_gau_filename, pep_filename, pro_pep_filename)

if val == '2':
    val_x = input("Which image? 1-Orig 2-Gaussain 3-Pepper")
    val_y = input("Which image? 1-Noisy image 2-Noise reduction image" )
    ## Process the original image
    if val_x == '1': 
        bifilter_para = (10, 17, 17)
        canny = (30,200)
        poly = 10
        contour_val = 2
        opencv_ori(ori_filename, bifilter_para, canny, poly, contour_val)
    
    ## Process the White noise image
    elif val_x == '2' and val_y == '1':
        bifilter_para = (15, 50, 50)
        canny = (30,50)
        poly = 10
        contour_val = 1
        filter = 0
        opencv_self(gau_filename, bifilter_para, canny, poly, contour_val, filter)
    
    ## Process the Noise reduction image
    elif val_x == '2' and val_y == '2':
        bifilter_para = (10, 17, 17)
        canny = (50,100)
        poly = 10
        contour_val = 2
        filter = 1
        opencv_self(pro_gau_filename, bifilter_para, canny, poly, contour_val, filter)
    
    ## Process Peppered noisy image   
    elif val_x == '3' and val_y == '1':
        bifilter_para = (10, 17, 17)
        canny = (30,200)
        poly = 10
        contour_val = 2
        filter = 0
        opencv_self(pep_filename, bifilter_para, canny, poly, contour_val, filter)
    
    ## Process Noice reduction image
    elif val_x == '3' and val_y == '2':
        bifilter_para = (11, 50, 50)
        canny = (150,200)
        poly = 15
        contour_val = 0
        filter = 0
        opencv_self(pro_pep_filename, bifilter_para, canny, poly, contour_val, filter)
    





