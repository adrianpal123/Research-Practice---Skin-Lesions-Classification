import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from sklearn.utils import shuffle
from PIL import Image
from keras.layers import BatchNormalization
from PreprocessingForNeuralNetworksUtil import DataPreprocessingUtil,metadata_path,image_dir

# Initialize DataPreprocessingUtil
data_util = DataPreprocessingUtil(metadata_path, image_dir)

tile_df = pd.read_csv(metadata_path)

# Load data and split into training and testing sets
X_train, X_test, y_train, y_test = data_util.binary_train_test_split(tile_df, (100, 100))

# Model configuration
input_shape = (100, 100, 3)
num_classes_binary = 1  # Binary classification

# Define the binary classification model
model_binary = Sequential()
model_binary.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_binary.add(BatchNormalization())
model_binary.add(MaxPooling2D(pool_size=(2, 2)))
model_binary.add(Dropout(0.25))
model_binary.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_binary.add(Flatten())
model_binary.add(Dense(32, activation='relu'))
model_binary.add(Dropout(0.3))
model_binary.add(Dense(num_classes_binary, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the binary model
initial_learning_rate_binary = 0.0001
opt_binary = Adam(learning_rate=initial_learning_rate_binary)
model_binary.compile(loss='binary_crossentropy',
                     optimizer=opt_binary,
                     metrics=['accuracy'])

# Print the binary model summary
model_binary.summary()

# Fit the binary model
history_binary = model_binary.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test), verbose=1)

# After the model training

# Predict on the test set
y_pred = model_binary.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = np.round(y_pred)

# Calculate the number of benign and non-benign/malignant samples
num_benign = np.sum(y_test == 1)
num_non_benign = np.sum(y_test == 0)

# Print the results
print("Number of benign samples:", num_benign)
print("Number of non-benign/malignant samples:", num_non_benign)
