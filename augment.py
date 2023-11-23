import os
import random
from PIL import Image
import cv2
import numpy as np

def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def horizontal_and_vertical_flip(image):
    return image.transpose(Image.TRANSPOSE)

def add_salt_and_pepper_noise(image, probability=0.05):
    image_array = np.array(image)
    h, w = image_array.shape[:2]
    num_salt_and_pepper_pixels = int(probability * h * w)
    coords = [np.random.randint(0, i, num_salt_and_pepper_pixels) for i in image_array.shape[:2]]
    if len(image_array.shape) == 3:  # Color image (RGB)
        for x, y in zip(coords[0], coords[1]):
            if random.random() < 0.5:
                image_array[x, y] = [0, 0, 0]  # Set to black (pepper noise)
            else:
                image_array[x, y] = [255, 255, 255]  # Set to white (salt noise)
    else:  # Grayscale image
        for x, y in zip(coords[0], coords[1]):
            if random.random() < 0.5:
                image_array[x, y] = 0  # Set to black (pepper noise)
            else:
                image_array[x, y] = 255  # Set to white (salt noise)
    return Image.fromarray(image_array)

def augment_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_path):
            try:
                image = Image.open(input_path)
                
                image = image.convert("RGB")

                # Perform horizontal flip
                image_horizontal = horizontal_flip(image)
                image_horizontal.save(os.path.join(output_folder, f"{filename[:-4]}_horizontal.jpg"))

                # Perform vertical flip
                image_vertical = vertical_flip(image)
                image_vertical.save(os.path.join(output_folder, f"{filename[:-4]}_vertical.jpg"))

                # Perform horizontal and vertical flip
                image_both = horizontal_and_vertical_flip(image)
                image_both.save(os.path.join(output_folder, f"{filename[:-4]}_both.jpg"))

                # Perform salt and pepper noise
                image_salt_pepper = add_salt_and_pepper_noise(image)
                image_salt_pepper.save(os.path.join(output_folder, f"{filename[:-4]}_salt_pepper.jpg"))

            except Exception as e:
                print(f"Error processing image {filename}: {e}")

if __name__ == "__main__":
    input_folder = ""  # Change this to the path of the folder containing the input images
    output_folder = ""  # Change this to the path where augmented images will be saved

    augment_images(input_folder, output_folder)
