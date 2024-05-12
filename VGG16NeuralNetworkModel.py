'''
Author: Adrian Pal
Summary: This script builds, trains, and evaluates a custom classification model using the Vgg16 architecture for Skin Lesion classification.
It utilizes transfer learning by loading the pre-trained Vgg16 model and fine-tunes it for the specific task.
The script includes data preprocessing, model compilation, training, evaluation, and performance analysis.
'''

import time
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from ModelPerformanceUtil import ModelPerformanceUtil
from PreprocessingForNeuralNetworksUtil import DataPreprocessingUtil, metadata_path, image_dir

# Initialize DataPreprocessingUtil
data_util = DataPreprocessingUtil(metadata_path, image_dir)
data_util.check_device()

# Load data and split into training and testing sets
X_train, X_test, y_train, y_test = data_util.train_test_split()

# Model configuration
input_shape = (*data_util.target_size, 3)
num_classes = data_util.num_classes()

# Load pre-trained VGG16 model
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze convolutional layers
for layer in vgg16_base.layers:
    layer.trainable = False

# Define the model
model = Sequential([
    vgg16_base,
    Flatten(),
    # Dense(512, activation='relu'),
    # Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Record the start time
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Record the end time
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Make predictions on the test set
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Initialize ModelPerformanceUtil with the model name
performance_util = ModelPerformanceUtil("VGG16- 15 Epochs")

# Generate and save plots along with performance metrics
performance_util.generate_and_save_plots(history, y_test, y_pred)

# Append the time taken to the txt file
performance_util.record_training_time(start_time, end_time)