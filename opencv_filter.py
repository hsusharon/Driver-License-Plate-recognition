import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import numpy as np
import easyocr
from Filter import *

def opencv_edge(filename, poly, contour_val, filter):
    image = cv.imread(filename)
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # img = cv.imread('Gaussian/gaus_car1.png')
    plt.imshow(img, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    row, col = img.shape

    filter1 = 10 
    filter2 = 17
    filter3 = 17
    if filter == 1:
    ## Sharpen image
        # filterType = 'lap'
        # temp = filter_gray(img, filterType)
        #bfilter = cv.bilateralFilter(img, filter1, filter2, filter3)
        # temp = filter_gray(temp, filterType)
        temp = img
        for i in range(row):
            for j in range(col):
                if temp[i][j] >= 20:
                    temp[i][j] = 255
                else:
                    temp[i][j] = 0
        filterType = 'gau'
        temp = filter_gray(temp, filterType)
        temp = cv.Canny(temp, 30, 200)
        plt.imshow(temp, cmap='gray')
        plt.show()

    edge = temp
    # canny1 = 30
    # canny2 = 100
    ## filtering and edge detection
    # bfilter = cv.bilateralFilter(img, filter1, filter2, filter3) #11, 17, 17
    # edge = cv.Canny(temp, canny1, canny2)
    # plt.imshow(edge, cmap='gray')
    # plt.show()

    


    ## find countours and Apply Masking
    keypoints = cv.findContours(edge.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]

    ## loop over the tree oth find Contours to fing coutours that is rectangular
    location = None
    for contour in contours:
        if contour_val == 1:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.09 * peri, True)
        else:
            approx = cv.approxPolyDP(contour, poly, True)
        if len(approx) == 4:
            location = approx
            break

    if hasattr("location", "all"):
        print('License Foud')
    else:
        print('Cannot find license')

    print(location)

    mask = np.zeros(img.shape, np.uint8)
    print(img.shape)
    new_image = cv.drawContours(mask, [location], 0, 255, -1)
    # plt.imshow(new_image, cmap = 'gray')
    # plt.show()
    new_image = cv.bitwise_and(img, img, mask = mask)
    plt.imshow(new_image, cmap = 'gray')
    plt.show()

    ## Crop the image 
    (x,y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_img = img[x1:x2+1,y1:y2+1]
    plt.imshow(cropped_img, cmap = 'gray')
    plt.show()

    reader = easyocr.Reader(['en'])
    plate = reader.readtext(cropped_img)
    print("Detected plate number:", plate)


    ## Write the result on the image
    text = plate[0][-2]
    font = cv.FONT_HERSHEY_SIMPLEX
    res = cv.putText(image, text=text, org=(approx[0][0][0], approx[1][0][1]+60), 
                        fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
    res = cv.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)
    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    plt.show()


