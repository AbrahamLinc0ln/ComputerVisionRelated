# Messing with AR techniques
# Marshall Trout
# May, 2020

# Python 2/3 compatibility
from __future__ import print_function

# import libraries
import numpy as np
import cv2

# Given an GRAY edges image, recolors the edges to a rainbow based on the angle judged from the center of the image
def edge2rainbowCircle(img):

	# convert to HLS color space
	bgrImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	rainbowImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2HLS)
	
	rows, cols = img.shape
	for i in range(rows):

		for j in range(cols):

			if img[i-1,j-1] != 0:

				if j >= cols/2:
					angle = np.arctan((rows/2-i)/(cols/2-j))+np.pi/2
					rainbowImg[i-1,j-1,:] = np.array((180*(angle)/(2*np.pi), 127, 255), np.uint8)

				else:
					angle = np.arctan((rows/2-i)/(cols/2-j))+3*np.pi/2
					rainbowImg[i-1,j-1,:] = np.array((180*(angle)/(2*np.pi), 127, 255), np.uint8)
	

	return rainbowImg



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
    cEdge = edge2rainbow(edgeImage)
    colorEdge = cv2.cvtColor(cEdge, cv2.COLOR_HLS2BGR)
    cv2.imshow('Color Edge', colorEdge)

    # reccombine with original image
    finalImg = cv2.addWeighted(colorEdge, 1, img, 0.9, 0)
    cv2.imshow('Rainbow Edge', finalImg)

if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()