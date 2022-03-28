import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import numpy as np
import easyocr


def opencv_ori(filename, bifilter, canny_val, poly, contour_val):
    image = cv.imread(filename)
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # plt.imshow(img, cmap = 'gray')

    filter1, filter2, filter3 = bifilter
    canny1, canny2 = canny_val

    ## filtering and edge detection
    bfilter = cv.bilateralFilter(img, filter1, filter2, filter3)
    edge = cv.Canny(bfilter, canny1, canny2)
    plt.imshow(edge, cmap='gray')
    plt.show()

    ## find countours and Apply Masking
    keypoints = cv.findContours(edge.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]

    ## loop over the tree oth find Contours to fing coutours that is rectangular
    location = None
    for contour in contours:
        if contour_val == 1:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.018 * peri, True)
        else:
            approx = cv.approxPolyDP(contour, poly, True)
        if len(approx) == 4:
            location = approx
            break

    if location.all() == None:
        print('Cannot find license')
    print(location)

    mask = np.zeros(img.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0, 255, -1)
    new_image = cv.bitwise_and(img, img, mask = mask)
    # plt.imshow(new_image, cmap = 'gray')
    # plt.show()

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
    #print(plate[0][-2])
    # print("Confidence:")
    # print(plate[0][-1])



    ## Write the result on the image
    text = plate[0][-2]
    font = cv.FONT_HERSHEY_SIMPLEX
    res = cv.putText(image, text=text, org=(approx[0][0][0], approx[1][0][1]+60), 
                        fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
    res = cv.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)
    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    plt.show()


