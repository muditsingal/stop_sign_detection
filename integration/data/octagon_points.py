import numpy as np
import math
import matplotlib.pyplot as plt

oct_size = 609.6

r = 609.6/(2*math.cos(np.deg2rad(22.5)))

side = 2*r*math.cos(np.deg2rad(67.5))

mid_side = side/2

up_pts_y = mid_side + oct_size/2
down_pts_y = mid_side - oct_size/2

up_pt_x1 = oct_size/2 - side/2
up_pt_x2 = oct_size/2 + side/2

down_pt_x1 = oct_size/2 - side/2
down_pt_x2 = oct_size/2 + side/2


# np.vstack(pts, np.array([0, 0]))
pts = np.array([0, side])
pts = np.vstack((pts, np.array([up_pt_x1, up_pts_y])))
pts = np.vstack((pts, np.array([up_pt_x2, up_pts_y])))
pts = np.vstack((pts, np.array([oct_size, side])))
pts = np.vstack((pts, np.array([0, 0])))
pts = np.vstack((pts, np.array([down_pt_x1, down_pts_y])))
pts = np.vstack((pts, np.array([down_pt_x2, down_pts_y])))
pts = np.vstack((pts, np.array([oct_size, 0])))
zeros = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])

# pts = pts + np.array([0, 178.5477061886])

pts = pts + np.array([-178.5477061886 - mid_side, 178.5477061886])

pts_3d = np.concatenate([pts, zeros], axis=1)

print(pts_3d)

# plt.plot(pts[:, 0], pts[:, 1])
# plt.xlim([-700, 700])
# plt.ylim([-700, 700])
# plt.show()

# print(431.05229381 - 252.50458762)

# print("Points are:")
# print([0, 0])
# print([0, side])
# print([oct_size, 0])
# print([oct_size, side])
# print([up_pt_x1, up_pts_y])
# print([up_pt_x2, up_pts_y])
# print([down_pt_x1, down_pts_y])
# print([down_pt_x2, down_pts_y])
