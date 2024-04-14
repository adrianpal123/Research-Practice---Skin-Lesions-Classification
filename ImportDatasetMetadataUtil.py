import os
import shutil
from zipfile import ZipFile

import numpy as np

# Set the base directory and image directories
def get_base_skin_dir():
    base_skin_dir = r'C:\Users\Adrian\Desktop\Research-Practice---Skin-Lesions-Classification\Neural Network Models'
    base_skin_metadata = os.path.join(base_skin_dir, 'Skin_Lesions/HAM10000_metadata.csv')
    image_zip_files = [
        os.path.join(base_skin_dir, 'HAM10000_images_part_1.zip'),
        os.path.join(base_skin_dir, 'HAM10000_images_part_2.zip')
    ]

    destination_dir = os.path.join(base_skin_dir, 'Skin_Lesions')

    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Copy metadata CSV to the destination folder
    shutil.copy(base_skin_metadata, destination_dir)

    # Create a subfolder to store the extracted images if it doesn't exist
    image_folder = os.path.join(destination_dir, 'HAM10000_images')
    if not os.path.exists(image_folder):
        # Create image folder
        os.makedirs(image_folder)

        # Unzip image files and move them to the destination folder
        for zip_file in image_zip_files:
            with ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(image_folder)

    # Update base_skin_dir to point to the folder containing all images
    base_skin_dir = image_folder

    return base_skin_dir

np.random.seed(123)
