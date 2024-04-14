import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from PreprocessingForNeuralNetworksUtil import DataPreprocessingUtil, metadata_path, image_dir
from ModelPerformanceUtil import ModelPerformanceUtil

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Initialize DataPreprocessingUtil
data_util = DataPreprocessingUtil(metadata_path, image_dir)
data_util.check_device()
print("Number of images:", data_util.count_images())


# Load data and split into training and testing sets
X_train, X_test, y_train, y_test = data_util.train_test_split()
print("Number of training data:", len(X_train))
print("Number of testing data:", len(X_test))
print("Number of training labels:", len(y_train))
print("Number of testing labels:", len(y_test))

# Model configuration
input_shape = (*data_util.target_size, 3)
num_classes = data_util.num_classes()

# Define the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Record the start time
start_time = time.time()

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

history = model.fit(train_dataset, epochs=25, validation_data=test_dataset)

# Record the end time
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Make predictions on the test set
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Initialize ModelPerformanceUtil with the model name
performance_util = ModelPerformanceUtil("CustomSimpleNeuralNetwork - 1 Epochs")

# Generate and save plots along with performance metrics
performance_util.generate_and_save_plots(history, y_test, y_pred)

# Append the time taken to the txt file
performance_util.record_training_time(start_time, end_time)