# Lane Detection Algorithm
# Marshall Trout
# With credit to kemfic - https://www.hackster.io/kemfic/simple-lane-detection-c3db2f#code
# April 8, 2020

# import libraries
import numpy as np 
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


 # Function to display all 6 test images
def display_images(images, cmap = None):
	plt.figure(figsize = (10,10))
	for i, image in enumerate(images):
		plt.subplot(3,2,i+1)
		plt.imshow(image, cmap)
		plt.autoscale(tight=True)
	plt.show()

# Function to color filter a image given upper and lower bounds bounds
def color_filter(image, lowerBounds, upperBounds):
	mask = cv2.inRange(image, lowerBounds, upperBounds)
	filteredImage = cv2.bitwise_and(image, image, mask=mask)
	return filteredImage

# Function to only look at a designated redgion of interest
def roi(image):
	x = int(image.shape[1])
	y = int(image.shape[0])
	shape = np.array([[int(0),int(y)], [int(x),int(y)], [int(0.55*x),int(0.6*y)], [int(0.45*x),int(0.6*y)]])

	mask = np.zeros_like(image)
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,)*channel_count
	else:
		ignore_mask_color = 255

	# create roi polygon
	cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)

	# apply roi to image
	masked_image = cv2.bitwise_and(image, mask)

	return masked_image

# Find the lines that fit the lane the best and generate an image to indicate this
def line_finder(image, rho = 1, theta = np.pi/180, thresh = 10, minLength = 20, maxGap = 100):
	global rightSlope, leftSlope, rightIntercept, leftIntercept
	# Find lines using Hough Lines
	lines = cv2.HoughLinesP(image, rho, theta, thresh, np.array([]), minLineLength=minLength, maxLineGap=maxGap)

	# Have an image to draw the lines on in RGB
	linesImage = np.zeros((image.shape[0], image.shape[1],3), dtype = np.uint8)

	# look only for lines with approximately the right slope and intercept we expect from a lane
	
	for line in lines:
		for x1,y1,x2,y2 in line:
			slope = (y2-y1)/(x2-x1)
			if slope > 0.3:
				if x1 > 500:
					yint = y2 - slope*x2
					rightSlope.append(slope)
					rightIntercept.append(yint)
				else:
					None
			elif slope < -0.3:
				if x1 < 600:
					yint = y2 - slope*x2
					leftSlope.append(slope)
					leftIntercept.append(yint)
				else:
					None

	# Use the average slopes and intercepts of all the collected lines
	avgLeftSlope = np.mean(leftSlope)
	avgLeftInt = np.mean(leftIntercept)
	avgRightSlope = np.mean(rightSlope)
	avgRightInt = np.mean(rightIntercept)

	try:
		# calculate x1 and x2 for our two lines			
		leftX1 = int((0.65*image.shape[0] - avgLeftInt)/avgLeftSlope)
		leftX2 = int((image.shape[0] - avgLeftInt)/avgLeftSlope)
		rightX1 = int((0.65*image.shape[0] - avgRightInt)/avgRightSlope)
		rightX2 = int((image.shape[0] - avgRightInt)/avgRightSlope)

		# form the shape of our lane
		pts = np.array([[leftX2, image.shape[0]], [leftX1, image.shape[0]*0.65], [rightX1, image.shape[0]*0.65], [rightX2, image.shape[0]]], np.int32)
		pts = pts.reshape((-1,1,2))

		# fill shape in our linesImage
		cv2.fillPoly(linesImage, [pts], (0,255, 0))

		# Draw Lines
		cv2.line(linesImage, (rightX2, int(image.shape[0])), (rightX1, int(0.65*image.shape[0])), [0, 0, 255], 10)
		cv2.line(linesImage, (leftX2, int(image.shape[0])), (leftX1, int(0.65*image.shape[0])), [0, 0, 255], 10)

		return linesImage

	except ValueError:
		pass


# Load test images
imageDir = 'testImg/'
imageFiles = os.listdir(imageDir)
imageList = []
for i in range(0, len(imageFiles)):
	imageList.append(mpimg.imread(imageDir+imageFiles[i]))

# display test images
#display_images(imageList)

# convert images to HSV color space
hsvImages = list(map(cv2.cvtColor, imageList, [cv2.COLOR_RGB2HLS]*len(imageList)))

# upper and lower color bounds
whiteLower = np.array([0,190,0])
whiteUpper = np.array([255,255,255])
yellowLower = np.array([10,0,90])
yellowUpper = np.array([50,255,255])

# filter to get yellow and white images
whiteImages = list(map(color_filter, hsvImages, [whiteLower]*len(hsvImages), [whiteUpper]*len(hsvImages)))
yellowImages = list(map(color_filter, hsvImages, [yellowLower]*len(hsvImages), [yellowUpper]*len(hsvImages)))

# combine resulting images
filteredImages = list(map(cv2.bitwise_or,whiteImages,yellowImages))
#display_images(filteredImages)

# apply roi to images
roiImages = list(map(roi,filteredImages))
#display_images(roiImages)

# convert images to grayscale
rbgImages = list(map(cv2.cvtColor, roiImages, [cv2.COLOR_HLS2RGB]*len(imageList)))
grayImages = list(map(cv2.cvtColor, roiImages, [cv2.COLOR_RGB2GRAY]*len(imageList)))
#display_images(grayImages, cmap = 'gray')

# use canny edge detection
edgeImages = list(map(cv2.Canny, grayImages, [50]*len(imageList), [120]*len(imageList)))
#display_images(edgesImages, cmap = 'gray')

# detect lines and choose the best ones
rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]

lineImages = list(map(line_finder, edgeImages))
#display_images(lineImages)

#Combine the original image with the detected lane
finale = list(map(cv2.addWeighted, imageList, [1]*len(imageList), lineImages, [0.75]*len(imageList), [0]*len(imageList)))
display_images(finale)
	






