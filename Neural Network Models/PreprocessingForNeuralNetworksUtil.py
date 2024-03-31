import os
from random import shuffle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from PIL import Image


metadata_path = r'C:\Users\Adrian\Desktop\Skin_Lesions_Disertation_Practical_Part\Skin_Lesions\HAM10000_metadata.csv'
image_dir = r'C:\Users\Adrian\Desktop\Skin_Lesions_Disertation_Practical_Part\Skin_Lesions\HAM10000_images'

class DataPreprocessingUtil:
    def __init__(self, metadata_path, image_dir, target_size=(100, 100)):
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.target_size = target_size

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        return img_array

    def load_data(self):
        tile_df = pd.read_csv(self.metadata_path)

        # Map diagnosis to numerical labels
        diagnosis_to_label = {
            'nv': 0,  # Melanocytic nevi
            'mel': 1,  # Melanoma
            'bkl': 2,  # Benign keratosis-like lesions
            'bcc': 3,  # Basal cell carcinoma
            'akiec': 4,  # Actinic keratoses
            'vasc': 5,  # Vascular lesions
            'df': 6  # Dermatofibroma
        }

        # Add numerical labels to DataFrame
        tile_df['cell_type_idx'] = tile_df['dx'].map(diagnosis_to_label)

        X = []
        y = []
        for index, row in tile_df.iterrows():
            image_path = os.path.join(self.image_dir, row['image_id'] + '.jpg')
            if os.path.exists(image_path):
                X.append(self.preprocess_image(image_path))
                y.append(row['cell_type_idx'])

        return np.array(X), np.array(y)

    def train_test_split(self, test_size=0.2, random_state=42):
        X, y = self.load_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def num_classes(self):
        return 7  # Since there are 7 different classes in the dataset

    def binary_train_test_split(tile_df, input_dims, test_size=0.25, random_state=42):
        # Map 'Benign' to 1 and the rest to 0
        tile_df['benign_type'] = tile_df['benign_status'].apply(lambda x: 1 if x == 'Benign' else 0)

        # Load images and preprocess
        tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize(input_dims)))
        x_data = np.asarray(tile_df['image'].tolist())
        y_data = np.asarray(tile_df['benign_type'].tolist())

        # Normalize pixel values
        x_mean = np.mean(x_data)
        x_std = np.std(x_data)
        x_data = (x_data - x_mean) / x_std

        # Shuffle the data
        x_data, y_data = shuffle(x_data, y_data, random_state=random_state)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, stratify=y_data,
                                                            random_state=random_state)

        return x_train, x_test, y_train, y_test



