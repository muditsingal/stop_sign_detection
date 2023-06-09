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

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import os
import datetime

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from scipy.spatial.transform import Rotation

import data.octagon_points as pt_src

margin_percent = 1 / 100
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
K = np.array([[521.8381958007812, 0.0, 684.0656127929688], 
              [0.0, 521.8381958007812, 350.3512268066406], 
              [0.0, 0.0, 1.0]])
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


def mask_img_hist(img, box_tl_x, box_tl_y, box_br_x, box_br_y, margin_x, margin_y):
    # Masking regions outside the stop sign box
    stop_isolated = np.copy(img)
    stop_isolated[:box_tl_y - margin_y] = 0
    stop_isolated[box_br_y + margin_y:] = 0
    stop_isolated[:, :box_tl_x - margin_x] = 0
    stop_isolated[:, box_br_x + margin_x:] = 0

    hist_pad = 8

    sub_img = img[box_tl_y - margin_y : box_br_y + margin_y, box_tl_x - margin_x : box_br_x + margin_x]

    gray_sub_img = cv2.cvtColor(stop_isolated, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_sub_img],[0],None,[256],[0,256])
    sub_hist = hist[8:-8]
    max_idx = np.argmax(sub_hist) + 8
    hist_roi = hist[max_idx-hist_pad:max_idx+hist_pad+1]


    # mask = (gray_sub_img < hist[max_idx-hist_pad]) | (gray_sub_img > hist[max_idx])
    mask = (gray_sub_img < 2) | (gray_sub_img > hist[max_idx])
    gray_sub_img[mask] = 0
    gray_sub_img = (gray_sub_img > 0).astype(np.uint8)*255
    return gray_sub_img

def find_sign_corners(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_img_pts = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if cv2.contourArea(contour) < 150:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) >= 8:  # Assuming an octagon has 8 vertices
            print(approx)
            final_img_pts = approx.copy().reshape(approx.shape[0],2)
            cv2.drawContours(img, [final_img_pts], -1, (0, 255, 0), 2)
            # final_img_pts.sort()
            final_img_pts = final_img_pts[final_img_pts[:, 1].argsort()]
            top4_pts = final_img_pts[:4]
            top4_pts = top4_pts[top4_pts[:, 0].argsort()]
            bottom4_pts = final_img_pts[-4:]
            bottom4_pts = bottom4_pts[bottom4_pts[:, 0].argsort()]
            final_img_pts = np.concatenate((top4_pts, bottom4_pts))
            return final_img_pts


def find_cam_pose_homography(img_pts, world_pts_2d):
    H, _ = cv2.findHomography(img_pts, world_pts_2d)
    compute_pose(H)
    print("H matrix using openCV function: \n", H)
    A = make_A_matrix(img_pts, world_pts_2d)
    H2 = calc_H_matrix(A)
    print("H matrix using custom function: \n", H2)
    compute_pose(H2)


def find_cam_pose_PnP(img_pts, world_pts_3d):
    global K, dist_coeffs
    retval, r_vec, t_vec = cv2.solvePnP(world_pts_3d, img_pts.astype(np.float64), K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    est_pose = make_pose_xyzrpy(t_vec, r_vec)
    # print("Final t vec is: ", t_vec.reshape((3)))
    return est_pose

# Function to compute the transformation matrix for the translation and rotation vector (axis-angle representation) returned by solvePnP cv2 function
def compute_T_from_vecs(retval, r_vec, t_vec):
    if retval == False:
        return None

    R, _ = cv2.Rodrigues(r_vec)
    T = np.column_stack((R, t_vec))
    T = np.vstack((T, [0,0,0,1]))

    return T

# Function to get euler angles of rotation from rotation vector (axis-angle representation)
def rvec_to_euler_angles(r_vec):
    R, _ = cv2.Rodrigues(r_vec)
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=False)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return roll, pitch, yaw

# Function to get euler angles from rotation matrix
def rotation_matrix_to_euler_angles(R):
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=False)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return roll, pitch, yaw

# Function to draw the estimated co-ordinate frame of object in image frame
def draw_obj_frame(r_vec, t_vec, img, image_points, axis_len=700, axis_thickness=3):
    global K, dist_coeffs
    axes = np.array([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len]], dtype=np.float32)
    img_axes, jacobian = cv2.projectPoints(axes, r_vec, t_vec, K, dist_coeffs)
    img_axes = img_axes.reshape(-1, 2).astype(np.int32)
    axis_origin = (image_points[5]+image_points[6])//2
    cv2.line(img, axis_origin, img_axes[0], (255,0,0), axis_thickness)
    cv2.line(img, axis_origin, img_axes[1], (0,255,0), axis_thickness)
    cv2.line(img, axis_origin, img_axes[2], (0,0,255), axis_thickness)


# Function to make a pose in ROS format viz. x, y, z, roll, pitch, yaw
def make_pose_xyzrpy(pos_vec, r_vec):
    roll, pitch, yaw = rvec_to_euler_angles(r_vec)
    euler_vec = np.array([roll, pitch, yaw], dtype=np.float32)
    curr_pose = np.append(pos_vec, euler_vec)
    return curr_pose, r_vec, pos_vec

# Function to calculate the error in position of 2 poses using L2 norm
def compute_pose_position_err(expected_pose, curr_pose):
    err = np.linalg.norm(expected_pose[:3]-curr_pose[:3])/10  # To make error in cm
    return err

# Function to calculate the rotational difference of 2 poses using rpy in radians and then taking L2 norm
def compute_pose_rotation_err(expected_pose, curr_pose):
    err = np.linalg.norm(expected_pose[3:]-curr_pose[3:])*180/np.pi
    return err

# Function to calculate the overall error in position using sum of position and rotation errors
def compute_pose_overall_err(expected_pose, curr_pose):
    err = compute_pose_position_err(expected_pose, curr_pose) + compute_pose_rotation_err(expected_pose, curr_pose)
    return err

# Function to plot the histogram of pose and rotation errors
def draw_pose_err_hist(expected_pose, r_vec, t_vec):
    curr_pose = make_pose_xyzrpy(t_vec, r_vec)
    pos_err = compute_pose_position_err(expected_pose, curr_pose)
    rot_err = compute_pose_rotation_err(expected_pose, curr_pose)
    tot_err = compute_pose_overall_err(expected_pose, curr_pose)
    names = ['position_err', 'rotation_err', 'total_err']
    err_axis = [pos_err, rot_err, tot_err]
    figure = plt.figure(figsize=(9,6))
    plt.bar(names, err_axis)
    plt.ylabel("Error in cm and degrees")
    plt.show()



# The most important function that uses YOLOv7 to detect objects in frame, detect corners of stop signs and then estimate the 3D pose of them
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for box in det:
                    if box[5] == 11.0:
                        box_np = box.detach().to('cpu').numpy().copy()
                        box_tl_x = int(box_np[0])
                        box_tl_y = int(box_np[1])
                        box_br_x = int(box_np[2])
                        box_br_y = int(box_np[3])
                        box_w = abs(box_br_x - box_tl_x)
                        box_h = abs(box_br_y - box_tl_y)
                        margin_x = int( box_w * margin_percent )
                        margin_y = int( box_h * margin_percent )
                        print(box_np)

                        masked_img = mask_img_hist(im0, box_tl_x, box_tl_y, box_br_x, box_br_y, margin_x, margin_y)
                        corner_img_pts = find_sign_corners(masked_img)
                        estimated_pose, rvec, tvec = find_cam_pose_PnP(corner_img_pts, pt_src.pts_3d)
                        draw_obj_frame(rvec, tvec, im0, image_points=corner_img_pts, axis_len=700, axis_thickness=3)
                        print("Estimated pose is: ", estimated_pose)

            if save_img:
                cv2.imwrite(save_path, im0)


            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect(save_img=True)
