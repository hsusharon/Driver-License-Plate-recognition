Implementing different filters and the influence of each with car license detection

This project implemented different filters and also see how different filters affects car license detection.
We could not guarantee the quality of the image we receive and it also depends on various circumstances, so 
it is very important to know how image changes via different filters so that we could use the right one 
when we need it. Add also we could see how filters affects the result of car license detection which is a 
very interesting topic.


Package required to run the code:
PyTorch, NumPy, math, scipy, cv2, easyocr, imutils, matplotlib

Caution:
Please change the directory of the file you want to process 
(directory address are all in main.py line 9-18)


This system has two main functions:

1. Process the image 
	- adding white noise (mean=0 var=10)
	- adding salt and pepper noise (prob = 0.01)
	- Run through LPF (Blur the image)
	- Run through HPF (Sharpen the image)
	- Edge detection with Sobel, Roberts, and Canny filter

2. Car License Plate detection 
	- Original image license plate detection 
	- Gaussain white noise image license plate detection
	- Noise reduction image license plate detection
	- Peppered and Salt noise image license plate detection
	- Median filter image license plate detection

To run the code:
The code will running under python version 3.9.6
Please clone the whole package to prevent data missing
Change the directory in main.py(line 9-18) to your own foldername
run main.py

ML code for car plate license detection is attached in ANPR folder. 
Transfer learning with VGG19 model accuracy 56.76% (with 75 epochs).

The final report is also attached in the report folder.

The presentation link: https://youtu.be/jwSui3u7CAU