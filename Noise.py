import numpy as np
import os
import cv2 as cv
import random 

def noisy(noise_typ,image):
  ## add Gaussian noise var = 10
    if noise_typ == "gauss":
      row,col= image.shape
      mean = 0
      var = 10
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
  ## add poisson noise
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy

  ## add speckle noise
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

## add pepper noise
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def addGauNoise(filename, img):
  row,col = img.shape
  noiseType = "gauss"
  img_noise = noisy(noiseType, img)
  for i in range(row):
      for j in range(col):
            img_noise[i][j] = round(img_noise[i][j])             
  img_noise = img_noise.astype(np.uint8)
  cv.imwrite(filename, img_noise)
  

def addPepperNoise(filename, img):
  img_noise = sp_noise(img, 0.01)
  img_noise = img_noise.astype(np.uint8)
  cv.imwrite(filename, img_noise)
  



