import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Step 1: Data Preprocessing

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array, verbose=0)
    return features.flatten()  # Flatten the features

train_path = ''
X = []
y = []
for label in os.listdir(train_path):
    label_dir = os.path.join(train_path, label)
    for image in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image)
        features = extract_features(image_path)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train the model (SVM classifier with a linear kernel)
classifier = SVC(kernel='linear', C=1, decision_function_shape='ovr')  # 'ovr' for multi-class
classifier.fit(X, y)

# Save the trained model
import joblib
model_path = ''
joblib.dump(classifier, model_path)

print(f"Model saved to {model_path}")
