from matplotlib import pyplot as plt
import numpy as np
import cv2
from typing import Tuple

# Parameters and flags
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

## Image handling and pose detection
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    # Mirror image
    frame = cv2.flip(frame,1)
    
    # Resize image
   # img = frame.copy()
  #  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  

    cv2.imshow('Raw', frame)
  #  cv2.imshow('RGB2BGR', img)
    image = resize_with_pad(frame,[256,256], [0,0,0])
    cv2.imshow("Padded image", image)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

