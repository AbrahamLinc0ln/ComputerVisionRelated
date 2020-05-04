# Messing with AR techniques
# Marshall Trout
# May, 2020

# Python 2/3 compatibility
from __future__ import print_function

# import libraries
import numpy as np
import cv2

# Given an GRAY edges image, recolors the edges to a rainbow based on the angle judged from the center of the image
# offset is used to adjust the color corresponding to each angle
def edge2rainbowCircle(img, offset=0):

	# convert to HLS color space
	bgrImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	rainbowImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2HLS)
	
	rows, cols = img.shape
	for i in range(rows):

		for j in range(cols):

			if img[i,j] != 0:

				if j >= cols/2:
					angle = np.arctan((rows/2-i)/(cols/2-j))+np.pi/2+offset
					angle=angle%(2*np.pi)
					rainbowImg[i,j,:] = np.array((180*(angle)/(2*np.pi), 127, 255), np.uint8)

				else:
					angle = np.arctan((rows/2-i)/(cols/2-j))+3*np.pi/2+offset
					angle=angle%(2*np.pi)
					rainbowImg[i,j,:] = np.array((180*(angle)/(2*np.pi), 127, 255), np.uint8)
	
	rainbowEdge = cv2.cvtColor(rainbowImg, cv2.COLOR_HLS2BGR)
	return rainbowEdge

# generate sequence of rainbow edges and save them for concatenation
def spinRainbowImages(img, imgCount):

	angleAdj = 2*np.pi/imgCount

	for i in range(imgCount):
		rainbowEdge = edge2rainbowCircle(img, offset = (i)*angleAdj)
		imgName = 'rainbowEdge_' + str(i) + '.png'
		cv2.imwrite(imgName, rainbowEdge)
		print(str(i)+'...')




def main():
    print('loading images...')
    img = cv2.pyrDown(cv2.imread(cv2.samples.findFile('aloeR.png')))
    cv2.imshow('OG', img)

    # convert to grayscale
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # get the edges
    edgeImage = cv2.Canny(grayImage, 50, 120)
    cv2.imshow('Edge', edgeImage)

    # convert edges to rainbow edges
    cEdge = edge2rainbowCircle(edgeImage)
    cv2.imshow('Color Edge', cEdge)

    # recombine with original image
    finalImg = cv2.addWeighted(cEdge, 1, img, 0.9, 0)
    cv2.imshow('Rainbow Edge', finalImg)

    # make sequence of images
    print('Generating images...')
    spinRainbowImages(edgeImage, 5)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()