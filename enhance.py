import os
import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image

def wiener_filter_denoise(image, kernel, noise_var):
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    image_fft = np.fft.fft2(image)
    psd = np.abs(kernel_fft)**2
    snr = 1.0 / noise_var
    psd = np.where(psd < 1.0 / snr, snr, psd)
    result_fft = np.conj(kernel_fft) / psd * image_fft
    result = np.fft.ifft2(result_fft).real
    return np.uint8(np.clip(result, 0, 255))

def z_score_normalization(image):
    # Convert image to numpy array
    image_array = np.array(image)

    # Compute mean and standard deviation of the image
    mean = np.mean(image_array)
    std_dev = np.std(image_array)

    # Z-score normalization
    normalized_image_array = (image_array - mean) / std_dev

    # Clip values to [0, 1] range (in case standard deviation is close to 0)
    normalized_image_array = np.clip(normalized_image_array, 0, 1)

    # Convert back to image and return
    normalized_image = Image.fromarray((normalized_image_array * 255).astype(np.uint8))

    return normalized_image

def denoise_images(input_folder, output_folder, kernel_size=1, noise_var=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            denoised_image = wiener_filter_denoise(image, kernel, noise_var)

            img_normalized = z_score_normalization(denoised_image)
            img_normalized = np.array(img_normalized)

            img_enhanced = cv2.equalizeHist(img_normalized)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img_enhanced)

if __name__ == "__main__":
    input_folder = "" #Path to flder containing the images to enhance before training the model.
    output_folder = "" #Path to flder you want to save the enhance images to.
    denoise_images(input_folder, output_folder)
