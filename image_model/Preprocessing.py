#UTILS

from skimage import exposure
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image

def enhance_contrast(image_array):
    image_array = image_array.astype(np.float32) / 255.0
    image_array_eq = exposure.equalize_hist(image_array)
    return (image_array_eq * 255).astype(np.uint8)

def adaptive_hist_eq(image_array):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(image_array)
    return cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)  # Convert back to RGB

def preprocess_image(image):
    image = enhance_contrast(image).astype(np.uint8)
    image = adaptive_hist_eq(image).astype(np.float32)
    image = tf.image.resize(image,(224,224))
    image = np.array(image)
    return image
