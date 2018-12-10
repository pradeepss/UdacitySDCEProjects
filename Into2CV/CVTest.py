# #importing some useful packages
#
# # importing opencv CV2 module
# import cv2
#
# # bat.jpg is the batman image.
# img = cv2.imread('curved-lane.jpg')
#
# # make sure that you have saved it in the same folder
# # Averaging
# # You can change the kernel size as you want
# avging = cv2.blur(img,(10,10))
#
# cv2.imshow('Averaging',avging)
# cv2.waitKey(0)
#
# # Gaussian Blurring
# # Again, you can change the kernel size
# gausBlur = cv2.GaussianBlur(img, (5,5),0)
# cv2.imshow('Gaussian Blurring', gausBlur)
# cv2.waitKey(0)
#
# # Median blurring
# medBlur = cv2.medianBlur(img,5)
# cv2.imshow('Media Blurring', medBlur)
# cv2.waitKey(0)
#
# # Bilateral Filtering
# bilFilter = cv2.bilateralFilter(img,9,75,75)
# cv2.imshow('Bilateral Filtering', bilFilter)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # import the required library
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
#
# # read the image
# img = cv2.imread('curved-lane.jpg')
#
# # convert image to gray scale image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # detect corners with the goodFeaturesToTrack function.
# corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
# corners = np.int0(corners)
#
# # we iterate through each corner,
# # making a circle at each point that we think is a corner.
# for i in corners:
#     x, y = i.ravel()
#     cv2.circle(img, (x, y), 3, 255, -1)
#
# plt.imshow(img), plt.show()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('curved-lane.jpg')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

# Define color selection criteria
###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 200
green_threshold = 200
blue_threshold = 200
######

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a boolean or with the "|" character to identify
# pixels below the thresholds
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)

# Uncomment the following code if you are running the code locally and wish to save the image
# mpimg.imsave("test-after.png", color_select)
