import os
import numpy as np
from skimage import io, color, exposure, restoration, data, img_as_float
from sklearn.svm import SVC
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import joblib
import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image

kernel_size = 1
noise_var = 2

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

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Denoise using weiner filter
    img_denoised = wiener_filter_denoise(img, kernel, noise_var)
    
    # Apply Z-Score normalization
    img_normalized = z_score_normalization(img_denoised)
    img_normalized = np.array(img_normalized)
    
    # Apply histogram equalization for brightness enhancement
    img_enhanced = cv2.equalizeHist(img_normalized)
    
    # Resize the image to (224, 224) for VGG16
    img_resized = cv2.resize(img_enhanced, (224, 224))
    
    return img_resized


# Load the trained model
model_path = 'D:\Liquid Retina\Fruits_Model.joblib'  # Specify the path to the trained model
classifier = joblib.load(model_path)

# Function to extract features
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_features(image):
    img_array = img_to_array(image)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array, verbose=0)
    return features.flatten()  # Flatten the features

# Specify the folder path containing images
folder_path = 'D:\Liquid Retina\Testing'  # Replace with actual path

# Make predictions and save results
results = {}

for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Convert preprocessed image to 3-channel for VGG
    preprocessed_image = color.gray2rgb(preprocessed_image)
    
    # Extract features
    features = extract_features(preprocessed_image)
    features = features.reshape(1, -1)  # Reshape the features array
    
    # Make prediction
    predicted_label = classifier.predict(features)
    results[image_name] = predicted_label[0]

# Save results to a text file
output_path = 'predictions.txt'
with open(output_path, 'w') as f:
    for image, prediction in results.items():
        f.write(f"{image}: {prediction}\n")

print(f"Predictions saved to {output_path}")
