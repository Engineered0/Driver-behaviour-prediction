import cv2
import os
import numpy as np


image_dir = 'data/dataset/images'


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values to [0,1]
    return image


images = []
for img_file in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_file)
    images.append(preprocess_image(img_path))

images = np.array(images)
