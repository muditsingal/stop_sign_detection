import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime
import cv2
import pytesseract as pts

print(os.getcwd() + '/data.vd')

'''
stop  -> [396.75000, 138.87500, 457.75000, 201.37500]

stop2 -> [595.00000, 120.00000, 655.00000, 184.25000]

stop3 -> [493.75000, 202.25000, 535.00000, 245.50000]


stop4 -> [695.50000, 189.50000, 726.50000, 233.75000]

'''



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

img = cv2.imread("/home/mudit/escooter_dev/images/stop4.jpg")
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
# masked_img[:111] = 0
# masked_img[183:] = 0
# masked_img[:, :396] = 0
# masked_img[:, 459:] = 0


# masked_img[:90] = 0
# masked_img[162:] = 0
# masked_img[:, :591] = 0
# masked_img[:, 659:] = 0
# 595.00000, 120.00000, 655.00000, 184.25000

masked_img[:155] = 0
masked_img[213:] = 0
masked_img[:, :693] = 0
masked_img[:, 729:] = 0
# 695.50000, 189.50000, 726.50000, 233.75000



edge_img = cv2.Canny(img_eq, 100, 200)
# 396.75000, 138.87500, 457.75000, 201.37500
# edge_img[:111] = 0
# edge_img[183:] = 0
# edge_img[:, :396] = 0
# edge_img[:, 459:] = 0

edge_img[:116] = 0
edge_img[187:] = 0
edge_img[:, :591] = 0
edge_img[:, 659:] = 0

stop_img = cv2.cvtColor(img_eq[111:183, 396:459], cv2.COLOR_BGR2GRAY)



############################################################ Corner detection ##################################################################

# corners = cv2.goodFeaturesToTrack(stop_img, 40, 0.05, 10)
# corners = np.int0(corners)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img_eq, (x+396, y+111), 3, 255, -1)

################################################################################################################################################


# lines = cv2.HoughLines(edge_img, 1, np.pi / 180, 19, None, 0, 0)

# if lines is not None:
#     print("Lines exist")
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(img_eq, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

# img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)




############################################################ Corner detection ##################################################################
masked_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
bin_img = (masked_gray > 0).astype(np.uint8)*255



blur_img = cv2.GaussianBlur(masked_gray, (3, 3), 0)
edge_img = cv2.Canny(blur_img, 60, 180)
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 30:
        continue

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) >= 8:  # Assuming an octagon has 8 vertices
        print(approx)
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

# cv2.imshow("Octagon Detection", masked_img)
# # cv2.imshow("nbin Detection", bin_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# contours, hierarchy = cv2.findContours(masked_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnt = sorted(contours, key=cv2.contourArea)
# cnt0 = cnt[-1]
# cv2.drawContours(img, [cnt0], 0, (0,255,0), 1)
# cv2.drawContours(img, cnt, -1, (0,255,0), 2)

################################################################################################################################################


cv2.imshow("Stop capture", img)
# cv2.imshow("HSV Stop capture", img_hsv)
cv2.imshow("Equalized img", img_eq)
cv2.imshow("Masked img", masked_img)
cv2.imshow("Canny image", edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()