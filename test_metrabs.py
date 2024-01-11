import tensorflow as tf
import tensorflow_hub as hub
import cv2
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

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

    #fig = plt.figure()
    #image_ax = fig.add_subplot(1, 2, 1)
    #image_ax.imshow(image.numpy())
    #for x, y, w, h, c in pred['boxes'].numpy():
    #    image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = plt.axes( projection='3d')
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(2000, 5000)
    
    # Extracts XYZ data from pose predition
    poses3d = pred['poses3d'].numpy()
    # Resctructures XYZ to XZ-Y
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    # === Updated plots ====
    
    # === Original Plotting code ===
   # for pose3d, pose2d in zip(poses3d, pred['poses2d'].numpy()):
   #     for i_start, i_end in joint_edges:
   #         #image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
   #         pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
   #     #image_ax.scatter(*pose2d.T, s=2)
   #     pose_ax.scatter(*pose3d.T, s=2)

video_name = 'right_trim'

model = hub.load('/Users/orthocap_01/Documents/Research/UCSF/Development/Motion_Tracking/metrabs_eff2l_y4_384px_800k_28ds')  # Takes about 3 minutes
#! wget -q https://istvansarandi.com/eccv22_demo/test.jpg
#img = tf.image.decode_image(tf.io.read_file('/Users/orthocap_01/Desktop/download.jpeg'))


joint_names = model.per_skeleton_joint_names['smpl+head_30'].numpy().astype(str)
joint_edges = model.per_skeleton_joint_edges['smpl+head_30'].numpy()

# Image handling and pose detection
cap = cv2.VideoCapture(('./video_data/' +  video_name + '.mp4')) 

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Initialise variables 
i_frame = 0
keypoint_data_x = np.zeros((30, n_frames))
keypoint_data_y = np.zeros((30, n_frames))
keypoint_data_z = np.zeros((30, n_frames))
keypoint_data_c = np.zeros((30, n_frames))
keypoint_data_t = np.zeros((n_frames,1))

while cap.isOpened():
   
    # Read frame
    ret, frame = cap.read()
    
    # Exit if no frame returned (workaround for capture open afer final frame)
    if frame is None:
        break

    # Mirror image
    frame = cv2.flip(frame,1)

    # Resize image
    image = resizeWithPad(frame,[256,256], [0,0,0]) 
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred = model.detect_poses(img, skeleton='smpl+head_30')
    pred['poses3d'].shape

    poses3d = pred['poses3d'].numpy()
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
    
        
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
cap.release()
#out_vid.release()
cv2.destroyAllWindows()

print('done')