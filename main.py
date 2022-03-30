import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from Filter import *
from opencv_self import *
from opencv import *
from Noiseadder import * 

ori_filename = "test/image3.jpg"
gau_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/gaussian/gaus_img3.jpg"
pro_gau_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/gaussian/pro_gaus_img3.jpg"
pep_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pepper/pepper_img4.jpg"
pro_pep_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pepper/pro_pepper_img4.jpg"
LPF_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pro_image/LPF.jpg"
HPF_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pro_image/HPF.jpg"
ROB_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pro_image/ROB.jpg"
SOL_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pro_image/SOL.jpg"
CAN_filename = "C:/Users/USER/Documents/School/Semester2/ENEE631 IP/Driver-License-Plate-recognition/pro_image/CAN.jpg"

while(1):
    print("1-Generate noise image  2-Plate recognition")
    val = input('Operation : ')

    if val == '1':
        val_x = input("Which Process 1-noise adder 2-LPF 3-HPF 4-Edge detection : ")
        ## Add Gaussian noise and peppered noise
        if val_x == '1':
            noiseadder(ori_filename, gau_filename, pro_gau_filename, pep_filename, pro_pep_filename)

        ## Filter with LPF  
        elif val_x == '2':  
            img = cv.imread(ori_filename)
            plt.imshow(img, cmap='gray')
            plt.show()
            filterchoice = 'gau'
            LPF_img = filter_ori(img, filterchoice)
            cv.imwrite(LPF_filename, LPF_img)
            plt.imshow(LPF_img, cmap='gray')
            plt.show()
        
        elif val_x == '3':
            img = cv.imread(LPF_filename)
            filterchoice = 'lap'
            HPF_img = filter_ori(img, filterchoice)
            cv.imwrite(HPF_filename, HPF_img)

        elif val_x == '4':
            img = cv.imread(ori_filename, 0)
            col, row = img.shape

            filterchoice = 'rob'
            ROB_img = filter_gray(img, filterchoice)
            for i in range(col):
                for j in range(row):
                    if ROB_img[i][j] >= 10:
                        ROB_img[i][j] = 255
                    else:
                        ROB_img[i][j] = 0
            cv.imwrite(ROB_filename, ROB_img)

            filterchoice = 'sol'
            SOL_img = filter_gray(img, filterchoice)
            for i in range(col):
                for j in range(row):
                    if ROB_img[i][j] >= 130:
                        ROB_img[i][j] = 255
                    else:
                        ROB_img[i][j] = 0
            cv.imwrite(SOL_filename, SOL_img)
            
            filterchoice = 'can'
            CAN_img = filter_gray(img, filterchoice)
            cv.imwrite(CAN_filename, CAN_img)


    if val == '2':
        val_x = input("Which image? 1-Orig 2-Gaussain 3-Pepper: ")
        if val_x != '1':
            val_y = input("Which image? 1-Noisy image 2-Noise reduction image : " )
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
    







