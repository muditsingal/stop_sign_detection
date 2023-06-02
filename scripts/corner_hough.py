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
CASE = 0
margin_percent = 4 / 100

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


# Thresholding 
hue_min1 = 0
hue_max1 = 9
hue_min2 = 166
hue_max2 = 179
sat_min = 20
sat_max = 255
val_min = 10
val_max = 255

erode_kernel = np.ones((5,5), np.uint8)
dil_kernel = np.ones((5,5), np.uint8)

low_thres1 = np.array([hue_min1, sat_min, val_min])
high_thres1 = np.array([hue_max1, sat_max, val_max])

low_thres2 = np.array([hue_min2, sat_min, val_min])
high_thres2 = np.array([hue_max2, sat_max, val_max])


curr_ts = datetime.datetime.now()
print("Current timestamp: " + str(curr_ts))


# img = cv2.imread("/home/mudit/escooter_dev/images/estop1.png")
# img = cv2.imread("/home/mudit/escooter_dev/images/estop2.png")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_hsv[:, :, 1] = cv2.equalizeHist(img_hsv[:, :, 1])
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
print(img_eq.shape)
cv2.imwrite("stop_eq.jpg", img_eq)

mask1 = cv2.inRange(img_hsv, low_thres1, high_thres1)
mask2 = cv2.inRange(img_hsv, low_thres2, high_thres2)
mask = mask1 + mask2

masked_img = cv2.bitwise_and(img_eq, img_eq, mask=mask)

masked_img[:box_tl_y - margin_y] = 0
masked_img[box_br_y + margin_y:] = 0
masked_img[:, :box_tl_x - margin_x] = 0
masked_img[:, box_br_x + margin_x:] = 0


img[:box_tl_y - margin_y] = 0
img[box_br_y + margin_y:] = 0
img[:, :box_tl_x - margin_x] = 0
img[:, box_br_x + margin_x:] = 0


edge_img = cv2.Canny(img_eq, 100, 200)
edge_img[:box_tl_y - margin_y] = 0
edge_img[box_br_y + margin_y:] = 0
edge_img[:, :box_tl_x - margin_x] = 0
edge_img[:, box_br_x + margin_x:] = 0


lines = cv2.HoughLines(edge_img, 1, np.pi / 180, 23, None, 0, 0)

if lines is not None:
    print("Lines exist")
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img_eq, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


cv2.imshow("Stop capture", img)
# cv2.imshow("HSV Stop capture", img_hsv)
cv2.imshow("Equalized img", img_eq)
cv2.imshow("Masked img", masked_img)
# cv2.imshow("Canny image", edge_img)

cv2.waitKey(0)
cv2.destroyAllWindows()