import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from typing import Tuple
from scipy import signal
import os


# Parameters and flags
data_path = '../Study_ACL/In/' 
confidence_threshold = 0.3
video_files = os.listdir(data_path)

# Filtering         
f_order = 2
f_cutoff = 3
f_sampling = 30
f_nyquist = f_cutoff/(f_sampling/2)
b, a = signal.butter(2, f_nyquist, 'lowpass', analog = False)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Create list with keypoints
keypoint_names = list()
keypoint_names.append('nose')
keypoint_names.append('eye_l')
keypoint_names.append('eye_r')
keypoint_names.append('ear_l')
keypoint_names.append('ear_r')
keypoint_names.append('shoulder_l')
keypoint_names.append('shoulder_r')
keypoint_names.append('elbow_l')
keypoint_names.append('elbow_r')
keypoint_names.append('wrist_l')
keypoint_names.append('wrist_r')
keypoint_names.append('hip_l')
keypoint_names.append('hip_r')
keypoint_names.append('knee_l')
keypoint_names.append('knee_r')
keypoint_names.append('ankle_l')
keypoint_names.append('ankle_r')

# ML Pose detection model from TensorFlow (download or reference url)
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
movenet = model.signatures['serving_default']

# Functions
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

def draw_keypoints_cor(frame, kp, confidence_threshold):
    y, x, c = frame.shape
    
    lamda = 100/56.25
    beta = -lamda*0.21875
    
    kp_y_scale_offset = 0 #kp[:,0]*0.25 - 0.125
    kp_x_scale_offset = kp[:,1]*lamda + beta

    kp_corrected = np.zeros([17,3])
    kp_corrected[:,0] = kp[:,0] + kp_y_scale_offset
    kp_corrected[:,1] = kp_x_scale_offset
    kp_corrected[:,2] = kp[:,2]
    
    shaped = np.squeeze(np.multiply(kp_corrected, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections_cor(frame, kp, edges, confidence_threshold):
    y, x, c = frame.shape
    
    lamda = 100/56.25
    beta = -lamda*0.21875

    kp_y_scale_offset = 0 #kp[:,0]*0.25 - 0.125
    kp_x_scale_offset = kp[:,1]*lamda + beta

    kp_corrected = np.zeros([17,3])
    kp_corrected[:,0] = kp[:,0] + kp_y_scale_offset
    kp_corrected[:,1] = kp_x_scale_offset
    kp_corrected[:,2] = kp[:,2]
    
    shaped = np.squeeze(np.multiply(kp_corrected, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

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

# Loop through videos 
for video_file in video_files:
    if video_file.endswith('.MOV'):
        
        video_name = video_file[:-4]

        # Image handling and pose detection
        cap = cv2.VideoCapture((data_path +  video_name + '.mov')) 
        out_vid = cv2.VideoWriter((data_path + video_name + '_tracked.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (720, 1280))
        out_vid_256 = cv2.VideoWriter((data_path + video_name + '_256_tracked.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (256, 256))

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialise variables 
        i_frame = 0
        keypoint_data_x = np.zeros((17, n_frames))
        keypoint_data_y = np.zeros((17, n_frames))
        keypoint_data_c = np.zeros((17, n_frames))
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
            input_img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
            input_img = tf.cast(input_img, dtype=tf.int32)
        
            # Detection section
            results = movenet(input_img)
            keypoints_with_scores = results['output_0']
            kp = np.squeeze(keypoints_with_scores)

            
            # Create arrays to save out keypoints 
            # Pass frame number
            keypoint_data_t[i_frame,0] = i_frame
            # Pass x values
            keypoint_data_x[:,i_frame] = np.transpose(kp[:,0])
            # Pass y values
            keypoint_data_y[:,i_frame] = np.transpose(kp[:,1])
            # Pass confidence values
            keypoint_data_c[:,i_frame] = np.transpose(kp[:,2])
            
            out_vid_256.write(image)
            cv2.imshow("Padded image", image)

            i_frame = i_frame + 1 
            
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break

        cap.release()

        kp =[]
        kp.append(keypoint_data_x)
        kp.append(keypoint_data_y)
        kp.append(keypoint_data_c)
        kp = np.array(kp)
        kp_filt = signal.filtfilt(b, a, kp, axis=2)

        cap = cv2.VideoCapture((data_path +  video_name + '.mov')) 
        i_frame = 0
        while cap.isOpened():
        
            # Read frame
            ret, frame = cap.read()
            
            # Exit if no frame returned (workaround for capture open afer final frame)
            if frame is None:
                break

            # Mirror image
            frame = cv2.flip(frame,1)
            # Render keypoints 
            draw_connections_cor(frame, np.transpose(kp_filt[:,:,i_frame]), EDGES, confidence_threshold)
            draw_keypoints_cor(frame, np.transpose(kp_filt[:,:,i_frame]), confidence_threshold)

            out_vid.write(frame)
            cv2.imshow('Movenet Single Pose', frame)

            i_frame = i_frame + 1 
            
                

        cap.release()
        out_vid.release()
        out_vid_256.release()
        cv2.destroyAllWindows()

        # # Plot the x vs y vs frame data
        # fig1 = plt

        # for i_kp in [5, 6, 11, 12, 13, 14, 15, 16]:
        #     fig1.scatter(x=keypoint_data_y[i_kp,:]*256, y=keypoint_data_x[i_kp,:]*-256, c=keypoint_data_t, cmap='jet', s=10, alpha=0.25)

        # fig1.title('Joints')
        # fig1.xlabel('Pixel (#)')
        # fig1.ylabel('Pixel (#)')
        # fig1.xlim(0,256)
        # fig1.ylim(-256, 0)
        # fig1.colorbar(label='Frame Number', orientation='vertical')

        # fig1.savefig('./Out/Figures/' + video_name + '_joint_xyvst.png')

        # # Save out data as .csv files
        # out_headers = pd.DataFrame(np.transpose(keypoint_names))
        # out_headers.to_csv(('./Out/Data/' + video_name + '_tracked_headers.csv'))

        # out_x = pd.DataFrame(np.transpose(keypoint_data_x))
        # out_x.to_csv(('./Out/Data/' + video_name + '_tracked_x.csv'))

        # out_y = pd.DataFrame(np.transpose(keypoint_data_y))
        # out_y.to_csv(('./Out/Data/' + video_name + '_tracked_y.csv'))

        # out_c = pd.DataFrame(np.transpose(keypoint_data_c))
        # out_c.to_csv(('./Out/Data/' + video_name + '_tracked_c.csv'))

        # out_t = pd.DataFrame(np.transpose(keypoint_data_t))
        # out_t.to_csv(('./Out/Data/' + video_name + '_tracked_t.csv'))