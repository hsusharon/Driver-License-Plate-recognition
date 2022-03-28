import cv2 as cv
import numpy as np
from Noise import *
from Filter import *
from matplotlib import pyplot as plt


def noiseadder(ori_file, gau_file, pro_gau_file, pep_file, pro_pep_file):
    img = cv.imread(ori_file,0)
    row,col = img.shape
    plt.imshow(img, cmap = 'gray')
    plt.show()

    # # Add pepper noise to the image
    addPepperNoise(pep_file, img)

    # # Add Gaussian noise to the image
    addGauNoise(gau_file, img)
    

    # # White noise reduction
    img = cv.imread(gau_file,0)
    filterchoice = 'gau'
    pro_img = filter_gray(img, filterchoice)
    pro_img = pro_img.astype(np.uint8)
    cv.imwrite(pro_gau_file, pro_img)
    

    # # Median filter for peppered noise
    img = cv.imread(pep_file,0)
    filterchoice = 'med'
    pro_img = filter_gray(img, filterchoice)
    pro_img = pro_img.astype(np.uint8)
    cv.imwrite(pro_pep_file, pro_img)
    



