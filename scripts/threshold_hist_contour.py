'''
Author: Mudit Singal
Project: Stop Sign corner detection for camera pose estimation
University: University of Maryland, College Park
'''

# Imporing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime
import cv2

# Reading the path
curr_pwd = os.getcwd()
img_path = curr_pwd + "/.." + '/src_imgs/'
CASE = 3
margin_percent = 1 / 100

'''
stop  -> 397, 115, 458, 177

stop2 -> 595, 120, 655, 184

stop3 -> 494, 178, 535, 222

stop4 -> 696, 166, 726, 210
'''

img1 = cv2.imread(img_path + "stop.jpg")
img2 = cv2.imread(img_path + "stop2.jpg")
img3 = cv2.imread(img_path + "stop3.jpg")
img4 = cv2.imread(img_path + "stop4.jpg")

# Considering the 4 cases of stop signs, later this will be integrated with YOLO to eliminate separate cases
if CASE == 0:
    box_tl_x = 397
    box_tl_y = 115
    box_br_x = 458
    box_br_y = 177
    img = img1

elif CASE == 1:
    box_tl_x = 596
    box_tl_y = 97
    box_br_x = 655
    box_br_y = 162
    img = img2

elif CASE == 2:
    box_tl_x = 494
    box_tl_y = 178
    box_br_x = 535
    box_br_y = 222
    img = img3

elif CASE == 3:
    box_tl_x = 696
    box_tl_y = 166
    box_br_x = 726
    box_br_y = 210
    img = img4


# Adjusted box dimensions
box_w = abs(box_br_x - box_tl_x)
box_h = abs(box_br_y - box_tl_y)
margin_x = int( box_w * margin_percent )
margin_y = int( box_h * margin_percent )


erode_kernel = np.ones((5,5), np.uint8)
dil_kernel = np.ones((5,5), np.uint8)


curr_ts = datetime.datetime.now()
print("Current timestamp: " + str(curr_ts))

stop_isolated = np.copy(img)

stop_isolated[:box_tl_y - margin_y] = 0
stop_isolated[box_br_y + margin_y:] = 0
stop_isolated[:, :box_tl_x - margin_x] = 0
stop_isolated[:, box_br_x + margin_x:] = 0


hist_pad = 8

sub_img = img[box_tl_y - margin_y : box_br_y + margin_y, box_tl_x - margin_x : box_br_x + margin_x]

gray_sub_img = cv2.cvtColor(stop_isolated, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([sub_img[:, :, 2]],[0],None,[256],[0,256])
hist = cv2.calcHist([gray_sub_img],[0],None,[256],[0,256])
sub_hist = hist[8:-8]
max_idx = np.argmax(sub_hist) + 8
hist_roi = hist[max_idx-hist_pad:max_idx+hist_pad+1]
print(hist[max_idx])
# hist = sorted(hist)
print(sub_img.shape[0]*sub_img.shape[1])
print(np.sum(hist))

# plt.plot(hist_roi)
# plt.show()


# mask = (gray_sub_img < hist[max_idx-hist_pad]) | (gray_sub_img > hist[max_idx])
mask = (gray_sub_img < 2) | (gray_sub_img > hist[max_idx])
gray_sub_img[mask] = 0
gray_sub_img = (gray_sub_img > 0).astype(np.uint8)*255

contours, hierarchy = cv2.findContours(gray_sub_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    if cv2.contourArea(contour) < 150:
        continue

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) >= 8:  # Assuming an octagon has 8 vertices
        print(approx)
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)


cv2.imshow("Stop capture", img)
cv2.imwrite("hist_stop4.jpg", img)
# cv2.imshow("HSV Stop capture", img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()