import numpy as np
import cv2
from PIL import Image


def preprocess_image(image_path: str):
    image = Image.open(image_path)
    np_image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    np_image = cv2.Canny(np_image, low_threshold, high_threshold)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)

    return image, canny_image
