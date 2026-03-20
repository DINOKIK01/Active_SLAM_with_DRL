#!/usr/bin/env python3

import numpy as np
import cv2
import math
import os

from numpy import savetxt
import math
import time
import datetime
import logging
from pathlib import Path

def frequency_control(params_update_freq):
    #print("Sleeping for: "+str(1/params_update_freq) +' s')
    time.sleep(1/params_update_freq)

def compute_frequency(t0):
    t1 = time.perf_counter()
    step_time = t1-t0
    t0 = t1
    action_hz = 1./(step_time)
    return action_hz, t1

def process_odom(goal_pose, odom):

    goal_distance = math.sqrt(
        (goal_pose[0]-odom[0])**2
        + (goal_pose[1]-odom[1])**2)

    path_theta = math.atan2(
        goal_pose[1]-odom[1],
        goal_pose[0]-odom[0])

    goal_angle = path_theta - odom[2]

    if goal_angle > math.pi:
        goal_angle -= 2 * math.pi

    elif goal_angle < -math.pi:
        goal_angle += 2 * math.pi

    goal_info = [goal_distance, goal_angle]
    robot_pose = [odom[0], odom[1], odom[2]]

    return goal_info, robot_pose

def create_logdir(policy, sensor, logdir):
    """
    """
    logdir_ = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}_{sensor}_{policy}/"
    Path(os.path.join(logdir, logdir_)).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logdir, logdir_, 'screen_logger.log'), 
        level=logging.INFO)
    return str(Path(os.path.join(logdir, logdir_)))

def euler_from_quaternion(quat):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        return v, norm
        #norm=np.finfo(v.dtype).eps
    return v/norm, norm

def normalize_angle(theta):
    # theta should be in the range [-pi, pi]
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta < -math.pi:
        theta += 2 * math.pi
    return theta

def quat_to_euler(qz, qw):
    """
    """
    t1 = 2*(qw*qz)
    t2 = 1 - 2*(qz*qz)

    Wz = math.atan2(t1,t2)

    return Wz

def euler_to_quat(Wz):
    """
    """
    qz = np.sin(Wz/2)
    qw = np.cos(Wz/2)

    return qz, qw

def tf_compose(robot_pose, goal_pose):
    """
    This method composes two consecutive reference frames.

    For example, it can return the pose of the goal, converted from 
    fixed reference frame to robot reference frame, with robot_pose and 
    goal_pose given in fixed reference frame.
    """
    xr, yr, zqr, wqr = tuple(robot_pose)
    zr = quat_to_euler(zqr, wqr)

    A = np.zeros([4,4], dtype=float)
    A[0,0] = math.cos(zr)
    A[1,0] = math.sin(zr)
    A[0,1] = -A[1,0]
    A[1,1] = A[0,0]
    A[0,3] = xr
    A[1,3] = yr
    A[2,3] = 0
    A[3,3] = 1
    A[2,2] = 1
    A = np.linalg.inv(A)
    
    return np.matmul(A, goal_pose).tolist()
    
def tf_decompose2(robot_pose, goal_pose):
    """
    This method decomposes two consecutive reference frames.

    For example, it can return the pose of the goal, converted from 
    robot reference frame to fixed reference frame, with robot_pose given in 
    fixed reference frame and goal_pose given in robot reference frame.
    """ 
    xr, yr, zqr, wqr = tuple(robot_pose)
    zr = quat_to_euler(zqr, wqr)

    A = np.zeros([4,4], dtype=float)
    A[0,0] = math.cos(zr)
    A[1,0] = math.sin(zr)
    A[0,1] = -A[1,0]
    A[1,1] = A[0,0]
    A[0,3] = xr
    A[1,3] = yr
    A[2,3] = 0
    A[3,3] = 1
    A[2,2] = 1
    
    return np.matmul(A, goal_pose).tolist()

def tf_decompose(robot_pose, goal_pose):
    """
    This method decomposes two consecutive reference frames.

    For example, it can return the pose of the goal, converted from 
    robot reference frame to fixed reference frame, with robot_pose given in 
    fixed reference frame and goal_pose given in robot reference frame.
    """ 
    xr, yr, zr = tuple(robot_pose)

    A = np.zeros([4,4], dtype=float)
    A[0,0] = math.cos(zr)
    A[1,0] = math.sin(zr)
    A[0,1] = -A[1,0]
    A[1,1] = A[0,0]
    A[0,3] = xr
    A[1,3] = yr
    A[2,3] = 0
    A[3,3] = 1
    A[2,2] = 1
    
    return np.matmul(A, goal_pose).tolist()

def display(img, window_name):
    """
    Display the image img.
    """
    try:
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
    except :
        print("Error in opening frame. Image shape: {}".format(img.shape))

def log_check(node):
    """
    Select the ROS2 log level.
    """
    try:
        log_level = int(os.environ['LOG_LEVEL'])
    except:
        log_level = 20
        node.get_logger().info("LOG_LEVEL not defined, setting default: INFO")

    node.get_logger().set_level(log_level)

def get_initial_info_map(layout):
    """
    Return initial info map.
    """
    grid_height = layout[0]
    grid_width = layout[1]
    ret = np.zeros((grid_height, grid_width, 2), dtype=np.int32)
                                            #channel 0 = visited
                                            #channel 1 = visit_count
    return ret

def compute_normed_distances(lidar_measurements):
    # Mean Distances
    # FRONT: wrap-around (end + beginning)
    front = np.concatenate([       # 13 values
        lidar_measurements[30:36],   # indices 31,32,33,34,35,36
        lidar_measurements[0:7]     # indices 0,1,2,3,4,5,6
    ])
    front_mean_dist = np.mean(front)
    # LEFT: 90°
    left = lidar_measurements[7:16]   # 9 values
    left_mean_dist = np.mean(left)
    # BACK: 180°
    back = lidar_measurements[16:21]  # 5 values
    back_mean_dist = np.mean(back)
    # RIGHT: 270°
    right = lidar_measurements[21:30] # 9 values
    right_mean_dist = np.mean(right)

    # Min Distance
    min_dist = np.min(lidar_measurements)
    
    # normalized
    max_dist = 5.5
    front_norm = front_mean_dist / max_dist
    left_norm = left_mean_dist / max_dist
    back_norm = back_mean_dist / max_dist
    right_norm = right_mean_dist / max_dist
    min_norm = min_dist / max_dist
    return front_norm, left_norm, back_norm, right_norm, min_norm

def update_info_map(info_map, robot_pose, cell_size=3, offset=9):
    """
    info_map: shape (H, W, 2)
    robot_pose: [x, y, yaw]
    """

    x, y, _ = robot_pose

    # Koordinaten -> Grid Index
    col = int(np.floor((x + offset) / cell_size))
    row = int(np.floor((offset - y) / cell_size))

    # Bounds check
    H, W, _ = info_map.shape
    if row < 0 or row >= H or col < 0 or col >= W:
        print("!!!!!!!!!!!! Außerhalb der Karte")
        return info_map  # außerhalb der Karte

    # Update:
    info_map[row, col, 0] = 1      # visited
    info_map[row, col, 1] += 1     # counter

    return info_map

def print_info_map(info_map):
    visited = info_map[:, :, 0]
    visit_count = info_map[:, :, 1]

    H, W = visited.shape

    print("\n=== VISITED MAP ===")
    for r in range(H):
        row_str = ""
        for c in range(W):
            row_str += f"{visited[r, c]:3d} "
        print(row_str)

    print("\n=== VISIT COUNT MAP ===")
    for r in range(H):
        row_str = ""
        for c in range(W):
            row_str += f"{visit_count[r, c]:3d} "
        print(row_str)

