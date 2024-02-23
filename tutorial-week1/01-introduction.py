#!/usr/bin/env python
# coding: utf-8

# # Lab 1: Introduction to OpenCV

# The goal of this first lab is to present a small introduction to image processing using OpenCV. In each section, you can find:
# * a small example - analyse the code and try it
# * some exercises

# In[ ]:


# Requirements for this tutorial
get_ipython().system(' pip install opencv-python')
get_ipython().system(' pip install numpy')


# In[2]:


# If you prefer, you can convert this notebook to a Python script by uncommenting the following command
pip install nbconvert
jupyter nbconvert --to script 1-introduction.ipynb


# In[1]:


import cv2
import numpy as np
import os

dataDir = './data'


# ### 1. Images – read, write and display; ROIs

# In[2]:


# Opening an image
img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))

# Showing the image
cv2.imshow("ml.jpg", img)

# Waiting for user to press a key to close the image
cv2.waitKey(0)

# Close the window after user pressed a key
cv2.destroyWindow("ml.jpg")


# In[3]:


# Check image size
h, w, c = img.shape
print(f'height: {h}')
print(f'width: {w}')
print(f'channels: {c}')


# In[4]:


# Saving image in bmp format
cv2.imwrite('ml_new.bmp', img)


# Exercise 1.1 - Read any other color image from a file, show the mouse cursor over the image, and the coordinates and RGB components of the pixel under the cursor. When the user clicks on the mouse, let him modify the RGB components of the selected pixel.

# In[7]:


img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))

def paint_coordinates(event, x, y, flag, params):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f'Coordinates: ({x}, {y})')
        print(f'BGR: ({img[y][x]})')
    if event == cv2.EVENT_LBUTTONDOWN:
        img[y][x] = [255, 0, 0]

cv2.namedWindow('image_window')
cv2.setMouseCallback('image_window', paint_coordinates)

while True:
    cv2.imshow('image_window', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyWindow('image_window')


# Exercise 1.2 - Allow the user to select a region of interest (ROI) in the image, by clicking on two points that identify two opposite corners of the selected ROI, and save the ROI into another file.

# In[2]:


img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))
roi = cv2.selectROI(img)
cv2.destroyAllWindows()

x = int(roi[0])
y = int(roi[1])
width = int(roi[2])
height = int(roi[3])

img_roi = img[y:y+height, x:x+width]
cv2.imwrite('roi_image.jpg', img_roi)
# coordinates_first = None
# 
# def select_coordinates(event, x, y, flag, params):
#     global coordinates_first
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if coordinates_first is None:
#             coordinates_first = [x, y]
#         else:
#             coordinates_second = [x, y]
#             print(coordinates_first)
#             print(coordinates_second)
#             new_img = img[coordinates_first[1]:coordinates_second[1]]
#             print(len(new_img[0]))
#             for idx in range(len(new_img[0])):
#                 new_img[idx] = new_img[idx][coordinates_first[0]:coordinates_second[0]]
#             cv2.imwrite('teste.bmp', new_img)
#         print(f'Coordinates: ({x}, {y})')
#         print(f'BGR: ({img[y][x]})')
#         
# 
# cv2.namedWindow('image_window')
# cv2.setMouseCallback('image_window', select_coordinates)
# 
# while True:
#     cv2.imshow('image_window', img)
#     if cv2.waitKey(1) == ord('q'):
#         break
# 
# cv2.destroyWindow('image_window')


# ### 2. Images – representation, grayscale and color, color spaces

# In[3]:


# Create a white image
m = np.ones((100,200,1), np.uint8)

# Change the intensity to 100
m = m * 100

# Display the image
cv2.imshow('Grayscale image', m)
cv2.waitKey(0)
cv2.destroyWindow('Grayscale image')


# In[4]:


# Draw a line with thickness of 5 px
cv2.line(m, (0,0), (200,100), 255, 5)
cv2.line(m, (200, 0), (0, 100), 255, 5)
cv2.imshow('Grayscale image with diagonals', m)
cv2.waitKey(0)
cv2.destroyWindow('Grayscale image with diagonals')


# Exercise 2.1 - Create a color image with 100(lines)x200(columns) pixels with yellow color; draw the two diagonals of the image, one in red color, the other in blue color. Display the image.

# In[5]:


# Create a white image
image = np.zeros((100, 200, 3), dtype=np.uint8)
image[:,:,1:] = 255

cv2.line(image, (0,0), (200, 100), [0, 0, 255], 5)
cv2.line(image, (200,0), (0, 100), [255, 0, 0], 5)

cv2.imshow('Yellow', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Exercise 2.2 - Read any color image, in RGB format, display it in one window, convert it to grayscale, display the grayscale image in another window and save the grayscale image to a different file

# In[6]:


img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))
img_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('normal', img)
cv2.imshow('grayscale', img_gr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Exercise 2.3 - Split the 3 RGB channels and show each channel in a separate window. Add a constant value to one of the channels, merge the channels into a new color image and show the resulting image.

# In[9]:


img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))

b = img[:][:][0]
print(b)
g = [0, img[:][:][1], 0]

cv2.imshow('blue', b)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Exercise 2.4 - Convert the image to HSV, split the 3 HSV channels and show each channel in a separate window. Add a constant value to saturation channel, merge the channels into a new color image and show the resulting image.

# In[ ]:


# TODO


# ### 3. Video – acquisition and simple processing

# In[10]:


# Define a VideoCapture Object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_nr = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('webcam', frame)

    # Wait for user to press s to save frame
    if cv2.waitKey(1) == ord('s'):
        frame_name = 'frame' + str(frame_nr) + '.png'
        cv2.imwrite(frame_name, frame)
        cv2.imshow("Saved frame: " + frame_name, frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Saved frame: " + frame_name)

    # Wait for user to press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


# Exercise 3.1 - Using the previous example as the baseline, implement a script that acquires the video from the webcam, converts it to grayscale, and shows the frames in binary format (i.e. the intensity of each pixel is 0 or 255); use a threshold value of 128.

# In[11]:


# Define a VideoCapture Object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_nr = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscale_frame, 128, 255, cv2.THRESH_BINARY)
    # Display the resulting frame
    cv2.imshow('webcam', threshold)

    # Wait for user to press s to save frame
    if cv2.waitKey(1) == ord('s'):
        frame_name = 'frame' + str(frame_nr) + '.png'
        cv2.imwrite(frame_name, frame)
        cv2.imshow("Saved frame: " + frame_name, frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Saved frame: " + frame_name)

    # Wait for user to press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


# Exercise 3.2 - Implement a simple detection/tracking algorithm for colored objects, using the following steps:
# 1) take each frame of the video;
# 2) convert from BGR to HSV color-space;
# 3) threshold the HSV image for a range of color values (creating a binary image);
# 4) extract the objects of the selected range (with a bitwise AND operation, using as operands the original and the binary image).

# In[ ]:


# Define a VideoCapture Object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_nr = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask to threshold the blue pixels
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    # Apply the mask to the original HSV frame to keep only the blue pixels
    blue_pixels = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    
    # Convert the result back to BGR for display
    blue_pixels_bgr = cv2.cvtColor(blue_pixels, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('Blue Pixels', blue_pixels_bgr)
    cv2.waitKey(0)

    # Wait for user to press s to save frame
    if cv2.waitKey(1) == ord('s'):
        frame_name = 'frame' + str(frame_nr) + '.png'
        cv2.imwrite(frame_name, frame)
        cv2.imshow("Saved frame: " + frame_name, frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Saved frame: " + frame_name)

    # Wait for user to press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:




