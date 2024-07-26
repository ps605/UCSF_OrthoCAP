import tensorflow as tf
import tensorflow_hub as hub
import cv2
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import pandas as pd
import os
import datetime

## INFORMATION
# This script will run the METRABS (https://github.com/isarandi/metrabs) 3D HPE algorithm of videos.
# The METRABS model must be installed locally and best in the same directory as this script.

def resizeWithPad(image: np.array, 
    new_shape: Tuple[int, int], 
    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    # Find scalling value for largest diminsion to fit new size
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    # Find ammount of padding needed to make image square (likely only left/right of smallest dimension required)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    # Resize image with padding 
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image
def plot_results(image, pred, joint_names, joint_edges):

    pose_ax = plt.axes( projection='3d')
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(2000, 5000)
    
    # Extracts XYZ data from pose predition
    poses3d = pred['poses3d'].numpy()
    # Resctructures XYZ to XZ-Y
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    pose_ax.scatter(poses3d[:, 0], poses3d[:, 1], poses3d[:, 2], c = 'red', s = 15, marker = 'o')
 
## SETUP
# Load METRABS model. It does not seem to like relative paths so keep the path absolute when loading the METRABS model. Takes about 3 minutes
model = hub.load('/Users/orthocap_01/Documents/Research/UCSF/Development/Motion_Tracking/metrabs_eff2l_y4_384px_800k_28ds')

# Initialise paths and variables of where data will be read from and outputed to
data_path = '../Study_ACL/In/' 
out_data_path = '../Study_ACL/Out/' 
# Video format (.MOV, .AVI etc.). ! NOTE .MOV prefferable.
video_format = '.MOV'
# This can be changed (https://github.com/isarandi/metrabs/blob/master/docs/API.md#skeleton-conventions). ! NOTE check plotting in animation_script.py as keypoiints will change
keypoint_model = 'kinectv2_25' #'smpl+head_30'

# Check if ./Figures/ path exists if not make folder
if not os.path.exists(out_data_path):
    os.makedirs(out_data_path, exist_ok = True)

# List video files
video_files = os.listdir(data_path)
## Loop through videos in data_path
for video_file in video_files:
    if video_file.endswith(video_format):
        # Open .txt file to log information from processing
        log_file = open(out_data_path + video_file[:-4] + '_log.txt', 'w')
        log_file.write(str(datetime.datetime.now()) + ': METRABS 3D skeleton model used: ' + keypoint_model + '\n')

        # Get joint (ie keypoint) names and connection frame
        joint_names = model.per_skeleton_joint_names[keypoint_model].numpy().astype(str)
        joint_edges = model.per_skeleton_joint_edges[keypoint_model].numpy()
        # Save out joint edges to use in animation later
        np.savetxt(out_data_path + video_file[:-4]  + '_edges.txt', joint_edges)

        # Expand joint names to _x _y and _z 
        joint_names_xyz = []
        for i_joint in range(joint_names.size):
            joint_names_xyz.append(joint_names[i_joint] + '_x')
            joint_names_xyz.append(joint_names[i_joint] + '_y')
            joint_names_xyz.append(joint_names[i_joint] + '_z')

        ## Image handling and pose detection
        # Read in video
        cap = cv2.VideoCapture((data_path +  video_file)) 

        # Initialise variables 
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log_file.write(str(datetime.datetime.now()) + ': Total video frames: ' + str(n_frames) + '\n')
        n_keypoints = np.size(joint_names)        
        i_frame = 0

        keypoint_data_x = np.zeros((n_keypoints, n_frames))
        keypoint_data_y = np.zeros((n_keypoints, n_frames))
        keypoint_data_z = np.zeros((n_keypoints, n_frames))
        keypoint_data_c = np.zeros((n_keypoints, n_frames))
        keypoint_data_t = np.zeros((n_frames,1))

        # Loop through video frames - !NOTE this could possile be changed to for loop to increase performance?
        while cap.isOpened():
        
            # Read frame
            ret, frame = cap.read()
            
            # Exit if no frame returned (workaround for capture open afer final frame)
            if frame is None:
                print(str(datetime.datetime.now()) + ': CAUTION: No read: ' + out_data_path + video_file + ' frame: ' + str(i_frame) + '. Moved to next frame.')
                log_file.write(str(datetime.datetime.now()) + ': CAUTION: No read: ' +out_data_path +  video_file + ' frame: ' + str(i_frame) + '. Moved to next frame.\n')
                i_frame = i_frame + 1
                break

            # # Mirror image
            # frame = cv2.flip(frame,1)

            # Resize image
            image = resizeWithPad(frame,[256,256], [0,0,0]) 
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = model.detect_poses(img, skeleton=keypoint_model)
            pred['poses3d'].shape

            poses3d = pred['poses3d'].numpy()

            # Check if no detection then skip loop
            if poses3d.size == 0:
                print(str(datetime.datetime.now()) + 'CAUTION: No detection: ' + video_file + ' frame: ' + str(i_frame) + '. Moving to next frame.')
                log_file.write(str(datetime.datetime.now()) + 'CAUTION: No detection: ' + video_file + ' frame: ' + str(i_frame) + '. Moving to next frame.\n')
                i_frame = i_frame + 1
                continue
            
            # Create arrays to save out keypoints 
            # Pass frame number
            keypoint_data_t[i_frame,0] = i_frame
            # Pass x values
            keypoint_data_x[:,i_frame] = poses3d[0,:,0]
            # Pass y values
            keypoint_data_y[:,i_frame] = poses3d[0,:,1]
            # Pass confidence values
            keypoint_data_z[:,i_frame] = poses3d[0,:,2]

            #plot_results(img, pred, joint_names, joint_edges)
            
            # print('Read: ' + video_file + ' frame: ' + str(i_frame) + ' - OK')
            # log_file.write('Read: ' + video_file + ' frame: ' + str(i_frame) + ' - OK')

            i_frame = i_frame + 1  

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
            
        cap.release()
        #out_vid.release()
        cv2.destroyAllWindows()


        # Transorm data to XYZ from XZ-Y
        pose_x = np.transpose(keypoint_data_x)
        pose_y = np.transpose(keypoint_data_z)
        pose_z = np.transpose(-keypoint_data_y)

        # Prepare and save out data and log file
        n_poses, n_markers = pose_x.shape
        pose_xyz = np.zeros((n_poses, n_markers*3))
        for i_marker in range(n_markers):
            pose_xyz[:,i_marker*3] = pose_x[:,i_marker]
            pose_xyz[:,i_marker*3 + 1] = pose_y[:,i_marker]
            pose_xyz[:,i_marker*3 + 2] = pose_z[:,i_marker]

        out_xyz = pd.DataFrame(np.squeeze(pose_xyz))
        out_xyz.to_csv((out_data_path + video_file[:-4] + '_3DTracked.csv'), header = joint_names_xyz)

        out_x = pd.DataFrame(pose_x)
        out_x.to_csv((out_data_path + video_file[:-4]  + '_3DTracked_x.csv'), header = joint_names)

        out_y = pd.DataFrame(pose_y)
        out_y.to_csv((out_data_path + video_file[:-4]  + '_3DTracked_y.csv'), header = joint_names)

        out_z = pd.DataFrame(pose_z)
        out_z.to_csv((out_data_path + video_file[:-4]  + '_3DTracked_z.csv'), header = joint_names)

        print(str(datetime.datetime.now()) + '3D pose estimation complete for: ' + out_data_path + video_file[:-4]  + '_3DTracked.csv')
        log_file.write(str(datetime.datetime.now()) + '3D pose estimation complete for: ' + out_data_path + video_file[:-4]  + '_3DTracked.csv\n')
        log_file.close()