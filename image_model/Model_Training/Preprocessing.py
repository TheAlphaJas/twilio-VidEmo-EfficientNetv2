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


def process_and_save_images(source_dir, dest_dir):
    """
    Process images from the source directory and save them in the destination directory.
    Args:
        source_dir (str): The path to the source directory containing class folders.
        dest_dir (str): The path to the destination directory where processed images will be saved.
    """
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through each class folder in the source directory
    for class_name in os.listdir(source_dir):
        if (class_name[0] == "."):
            continue
        class_source_dir = os.path.join(source_dir, class_name)
        class_dest_dir = os.path.join(dest_dir, class_name)
        
        # Ensure class folder exists in the destination directory
        os.makedirs(class_dest_dir, exist_ok=True)
        
        # Iterate through each image in the class folder
        for img_name in os.listdir(class_source_dir):
            img_path = os.path.join(class_source_dir, img_name)
            img = plt.imread(img_path)
            processed_img = preprocess_image(img).astype(np.uint8)
#             print(processed_img.type)
            processed_img = Image.fromarray(processed_img)
                # Save the processed image in the destination folder
            processed_img.save(os.path.join(class_dest_dir, img_name))
            # Load and preprocess the image
#             with img_path as img:
#                 img = plt.imread(img)
#                 processed_img = preprocess_image(img)
                
#                 # Save the processed image in the destination folder
#                 processed_img.save(os.path.join(class_dest_dir, img_name))

# Define source and destination directories
source_directory = '/kaggle/input/sentiment-data/sentiment_dataset'
destination_directory = '/kaggle/working/NewData2'

# Process and save images
process_and_save_images(source_directory, destination_directory)

print("Processing complete. Processed images saved to:", destination_directory)
