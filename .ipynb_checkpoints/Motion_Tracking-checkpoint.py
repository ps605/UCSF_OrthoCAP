import tf as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
import cv2
from typing import Tuple

# Parameters and flags
confidence_threshold = 0.3

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

def draw_keypoints_cor(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    
    kp = np.squeeze(keypoints_with_scores)
    kp_y_scale_offset = kp[:,0]*0.25 - 0.125

    kp_corrected = np.zeros([17,3])
    kp_corrected[:,0] = kp[:,0] + kp_y_scale_offset
    kp_corrected[:,1] = kp[:,1]
    kp_corrected[:,2] = kp[:,2]
    
    shaped = np.squeeze(np.multiply(kp_corrected, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections_cor(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    
    kp = np.squeeze(keypoints_with_scores)
    kp_y_scale_offset = kp[:,0]*0.25 - 0.125

    kp_corrected = np.zeros([17,3])
    kp_corrected[:,0] = kp[:,0] + kp_y_scale_offset
    kp_corrected[:,1] = kp[:,1]
    kp_corrected[:,2] = kp[:,2]
    
    shaped = np.squeeze(np.multiply(kp_corrected, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


def resize_with_pad(image: np.array, 
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
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# Image handling and pose detection
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Mirror image
    frame = cv2.flip(frame,1)

    # Resize image
    image = resize_with_pad(frame,[256,256], [0,0,0]) 
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
  
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0']
    #print(keypoints_with_scores)
    
    # Render keypoints 
    #draw_connections_cor(frame, keypoints_with_scores, EDGES, confidence_threshold)
    #draw_keypoints_cor(frame, keypoints_with_scores, confidence_threshold)

    #cv2.imshow('Movenet Single Pose', frame)
    
      
    draw_connections(image, keypoints_with_scores, EDGES, confidence_threshold)
    draw_keypoints(image, keypoints_with_scores, confidence_threshold)
    
    cv2.imshow("Padded image", image)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()