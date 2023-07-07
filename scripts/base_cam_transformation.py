from scipy.spatial.transform import Rotation
import numpy as np

def compute_T_from_vecs(roll, pitch, yaw, x, y, z):
    R_scipy = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    R = R_scipy.as_matrix()
    T = np.column_stack((R, [x, y, z]))
    T = np.vstack((T, [0,0,0,1]))

    return T


def rotation_matrix_to_euler_angles(R):
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=True)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return roll, pitch, yaw

#                                       roll        pitch       yaw     x       y           z
T_base_cam_mount = compute_T_from_vecs(-2.8799, -0.0078747, -1.5414, 0.55275, 0.023161, 0.25643)
T_cam_mount_cam  = compute_T_from_vecs( 1.5708, 0.0, 0.0, -0.042893, -0.012411, -0.018901)
T_cam_center     = compute_T_from_vecs(1.61079632679, -1.57079632679, 0.0, 0.06, 0.0, 0.0)

T_final = np.matmul(T_base_cam_mount, T_cam_mount_cam, T_cam_center)
roll_f, pitch_f, yaw_f = rotation_matrix_to_euler_angles(T_final[:3,:3])
# print(T_final)
# print(roll_f, pitch_f, yaw_f)