'''
Author: Mudit Singal
Project: Stop Sign corner detection for camera pose estimation
University: University of Maryland, College Park
'''

'''
Zed camera params
height: 720
width: 1280
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [521.8381958007812, 0.0, 684.0656127929688, 0.0, 521.8381958007812, 350.3512268066406, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [521.8381958007812, 0.0, 684.0656127929688, 0.0, 0.0, 521.8381958007812, 350.3512268066406, 0.0, 0.0, 0.0, 1.0, 0.0]

'''


# Imporing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime
import cv2
import octagon_points as pt_src
from scipy.spatial.transform import Rotation
import base_cam_transformation as trans

# Reading the path
curr_pwd = os.getcwd()
img_path = curr_pwd + "/.." + '/src_imgs/'
CASE = 3
margin_percent = 1 / 100
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
K = np.array([[521.8381958007812, 0.0, 684.0656127929688], 
              [0.0, 521.8381958007812, 350.3512268066406], 
              [0.0, 0.0, 1.0]])
# K = np.array([  [1382.58398,    0,          945.743164], 
#                 [0,             1383.57251, 527.04834], 
#                 [0,             0,          1]])

K_inv = np.linalg.inv(K)


# Construct the A matrix as per the given formula in slides 
def make_A_matrix(image_pts, world_pts):
    A = np.zeros(shape=(2*len(image_pts),9), dtype=np.float32)
    for i in range(0, 2*len(image_pts) - 1, 2):
        x1 = world_pts[i//2][0]
        y1 = world_pts[i//2][1]

        x1_dash = image_pts[i//2][0]
        y1_dash = image_pts[i//2][1]
        A[i] = np.array([x1, y1, 1, 0, 0, 0, -x1_dash*x1, -x1_dash*y1, -x1_dash])
        A[i+1] = np.array([0, 0, 0, x1, y1, 1, -y1_dash*x1, -y1_dash*y1, -y1_dash])

    # print(A.shape)
    return A

# Function to calculate the H matrix as by finding the least eigen vector of A_transpose * A, scaling it by the h22 term 
# and then rearranging into a 3x3 matrix
def calc_H_matrix(A):
    sq_A = np.matmul(A.T, A)
    eig_vals, eig_vecs = np.linalg.eig(sq_A)
    smallest_eig_vec = eig_vecs[:, np.argmin(eig_vals)]
    smallest_eig_vec = smallest_eig_vec / smallest_eig_vec[-1]
    H = smallest_eig_vec.reshape((3,3))

    return H

def compute_pose(H):
    global K_inv    
    r_t_mat = np.matmul(K_inv, H)

    # Finding 2 lambdas from A1 and A2 vectors and taking average to get a better lambda
    lambda1 = np.linalg.norm(r_t_mat[:,0])
    lambda2 = np.linalg.norm(r_t_mat[:,1])
    lambda_ = (lambda1 + lambda2)/2

    # Scaling the matrix by lambda
    r_t_mat_norm = r_t_mat / lambda_

    # Extracting the pose rotation and translation vectors from the above matrix
    r1_vec = r_t_mat_norm[:,0]
    r2_vec = r_t_mat_norm[:,1]
    t_vec = r_t_mat_norm[:,2]
    r3_vec = np.cross(r1_vec, r2_vec)
    R = np.vstack((r1_vec, r2_vec, r3_vec))
    # R = np.vstack(R, r3_vec)
    # R = R.T
    print("R matrix is: \n", R)
    print("T vector is: ", t_vec)
    # print(t_vec)

    return np.column_stack([r1_vec, r2_vec, r3_vec, t_vec])


# Function to compute the transformation matrix for the translation and rotation vector (axis-angle representation) returned by solvePnP cv2 function
def compute_T_from_vecs(retval, r_vec, t_vec):
    if retval == False:
        return None

    R, _ = cv2.Rodrigues(r_vec)
    T = np.column_stack((R, t_vec))
    T = np.vstack((T, [0,0,0,1]))

    return T


def rvec_to_euler_angles(r_vec):
    R, _ = cv2.Rodrigues(r_vec)
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=True)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return roll, pitch, yaw

def rotation_matrix_to_euler_angles(R):
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=True)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return roll, pitch, yaw


def draw_obj_frame(r_vec, t_vec, img, image_points, axis_len=700, axis_thickness=3):
    global K, dist_coeffs
    axes = np.array([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len]], dtype=np.float32)
    img_axes, jacobian = cv2.projectPoints(axes, r_vec, t_vec, K, dist_coeffs)
    img_axes = img_axes.reshape(-1, 2).astype(np.int32)
    axis_origin = image_points[5]
    cv2.line(img, axis_origin, img_axes[0], (255,0,0), axis_thickness)
    cv2.line(img, axis_origin, img_axes[1], (0,255,0), axis_thickness)
    cv2.line(img, axis_origin, img_axes[2], (0,0,255), axis_thickness)

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
print(hist_roi)
print(hist[max_idx])
# hist = sorted(hist)
print(sub_img.shape[0]*sub_img.shape[1])
print(np.sum(hist))

# plt.plot(hist_roi)
# plt.show()


# mask = (gray_sub_img < hist[max_idx-hist_pad]) | (gray_sub_img > hist[max_idx])
mask = (gray_sub_img < 2) | (gray_sub_img > hist[max_idx]+hist_pad)
gray_sub_img[mask] = 0
gray_sub_img = (gray_sub_img > 0).astype(np.uint8)*255

contours, hierarchy = cv2.findContours(gray_sub_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final_img_pts = []

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    if cv2.contourArea(contour) < 150:
        continue

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) >= 8:  # Assuming an octagon has 8 vertices
        print(approx)
        final_img_pts = approx.copy().reshape(approx.shape[0],2)
        # cv2.drawContours(img, [final_img_pts], -1, (0, 255, 0), 2)
        final_img_pts = final_img_pts[final_img_pts[:, 1].argsort()]
        top4_pts = final_img_pts[:4]
        top4_pts = top4_pts[top4_pts[:, 0].argsort()]
        bottom4_pts = final_img_pts[-4:]
        bottom4_pts = bottom4_pts[bottom4_pts[:, 0].argsort()]
        final_img_pts = np.concatenate((top4_pts, bottom4_pts))
        print(final_img_pts.shape)
        print(pt_src.pts.shape)
        print(final_img_pts)

        H, _ = cv2.findHomography(final_img_pts, pt_src.pts)
        compute_pose(H)
        print("H matrix using openCV function: \n", H)
        A = make_A_matrix(final_img_pts, pt_src.pts)
        H2 = calc_H_matrix(A)
        print("H matrix using custom function: \n", H2)
        compute_pose(H2)
        # Uses the Levenberg-Marquardt optimization method
        retval, r_vec, t_vec = cv2.solvePnP(pt_src.pts_3d, final_img_pts.astype(np.float64), K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # Infinitesimal Plane-Based Pose Estimation, the results don't vary significantly when compared to the iterative PnP process 
        # retval, r_vec, t_vec = cv2.solvePnP(pt_src.pts_3d, final_img_pts.astype(np.float64), K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
        # cv2.solvePnPRansac() uses ransac with iterative process, however the results returned are nearly identical when compared to the iterative process

        T = compute_T_from_vecs(retval, r_vec, t_vec)
        # T = T @ trans.T_final
        print("Final t vec is: ", t_vec.reshape((3)))
        print("Transformation matrix is: \n" ,T)
        roll, pitch, yaw = rotation_matrix_to_euler_angles(T[:3, :3])
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        rotation_matrix = r.as_matrix()
        print("The roll, pitch, yaw are: ", roll, pitch, yaw )
        print(rotation_matrix)


        # Draw coordinate frame on image
        draw_obj_frame(r_vec, t_vec, img, image_points=final_img_pts, axis_len=700, axis_thickness=3)



        # print("Rotation matrix", )
        # rotation_matrix_to_euler_angles(R)




        # final_img_pts.roll(-1)/
        # final_img_pts.sorted(key=lambda pt: (pt[0], pt[1]))
        # sorted_indices = np.sort()
        # sorted_points = final_img_pts[sorted_indices]
        # print(final_img_pts)
        


cv2.imshow("Stop capture", img)
cv2.imwrite("hist_stop4.jpg", img)
# cv2.imshow("HSV Stop capture", img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
