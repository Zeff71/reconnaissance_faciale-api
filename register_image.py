# register_image.py

import os
import shutil
import uuid

# Chemin absolu sécurisé (adapté peu importe où tu exécutes FastAPI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, 'dataset_lfw', 'new_images')

def register_image(image_path, person_name):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"❌ Image introuvable à {image_path}")

    dest_dir = os.path.join(RAW_DATASET, person_name)
    os.makedirs(dest_dir, exist_ok=True)

    dest_file = os.path.join(dest_dir, f"{uuid.uuid4()}.jpg")
    shutil.copy(image_path, dest_file)

    print(f"✅ Image enregistrée de '{person_name}' dans : {dest_file}")
    return dest_file
