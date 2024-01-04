import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Parameters and flags
confidence_threshold = 0.5

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
    img = frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
  
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0']
    #print(keypoints_with_scores)
    
    # Render keypoints 
    draw_connections(frame, keypoints_with_scores, EDGES, confidence_threshold)
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)

    cv2.imshow('Movenet Single Pose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

