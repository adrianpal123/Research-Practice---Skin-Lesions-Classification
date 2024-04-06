from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from PreprocessingForNeuralNetworksUtil import DataPreprocessingUtil,metadata_path,image_dir
import numpy as np
from ModelPerformanceUtil import ModelPerformanceUtil

# Initialize DataPreprocessingUtil
data_util = DataPreprocessingUtil(metadata_path, image_dir)

# Load data and split into training and testing sets
X_train, X_test, y_train, y_test = data_util.train_test_split()

# Model configuration
input_shape = (*data_util.target_size, 3)
num_classes = data_util.num_classes()

# Load pre-trained InceptionV3 model
inceptionV3_base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# Define the model
model = Sequential([
    inceptionV3_base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Make predictions on the test set
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Initialize ModelPerformanceUtil with the model name
performance_util = ModelPerformanceUtil("InceptionV3")

# Generate and save plots along with performance metrics
performance_util.generate_and_save_plots(history, y_test, y_pred)

# Make predictions on the test set
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Initialize ModelPerformanceUtil with the model name
performance_util = ModelPerformanceUtil("InceptionV3 -10Epochs")

# Generate and save plots along with performance metrics
performance_util.generate_and_save_plots(history, y_test, y_pred)