# train_incremental.py
import os
import json
import numpy as np
import tensorflow as tf
import cv2
from medium_facenet_tutorial.align_dlib import AlignDlib

MODEL_PATH = 'facenet_model/20170511-185253/20170511-185253.pb'
SHAPE_PREDICTOR_PATH = 'medium_facenet_tutorial/shape_predictor_68_face_landmarks.dat'
RAW_DATASET_DIR = 'dataset_lfw/new_images'
OUTPUT_PATH = 'output/embeddings_labels.npz'
CLASS_NAMES_PATH = 'output/class_names.json'
IMAGE_SIZE = 224

align_dlib = AlignDlib(SHAPE_PREDICTOR_PATH)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj

def _buffer_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise ValueError(f"Image vide ou illisible: {filename}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _align_image(image, crop_dim=224):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    if bb is None:
        return None
    x, y, w, h = bb.left(), bb.top(), bb.width(), bb.height()
    margin_ratio = 0.3
    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)
    x1 = max(x - x_margin, 0)
    y1 = max(y - int(y_margin * 1.5), 0)
    x2 = min(x + w + x_margin, image.shape[1])
    y2 = min(y + h + y_margin, image.shape[0])
    cropped_face = image[y1:y2, x1:x2]
    aligned = cv2.resize(cropped_face, (crop_dim, crop_dim))
    return aligned

def load_model(model_path):
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return json.load(f)
    return []

def save_class_names(class_names):
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)

def load_embeddings():
    if os.path.exists(OUTPUT_PATH):
        data = np.load(OUTPUT_PATH)
        return list(data['embeddings']), list(data['labels'])
    return [], []

def save_embeddings(embeddings, labels):
    np.savez(OUTPUT_PATH, embeddings=np.array(embeddings), labels=np.array(labels))

def extract_embedding(sess, image_path, images_placeholder, embeddings_tensor, phase_train_placeholder):
    img = _buffer_image(image_path)
    aligned = _align_image(img, IMAGE_SIZE)
    if aligned is None:
        raise ValueError("Aucun visage détecté.")
    prewhitened = prewhiten(aligned)
    emb = sess.run(embeddings_tensor, feed_dict={
        images_placeholder: [prewhitened],
        phase_train_placeholder: False
    })
    return emb[0]

def train_incremental():
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        load_model(MODEL_PATH)
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

        embeddings_list, labels_list = load_embeddings()
        class_names = load_class_names()

        for person_name in os.listdir(RAW_DATASET_DIR):
            person_dir = os.path.join(RAW_DATASET_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            if person_name not in class_names:
                class_names.append(person_name)

            label_id = class_names.index(person_name)

            for img_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, img_name)
                try:
                    emb = extract_embedding(sess, image_path, images_placeholder, embeddings_tensor, phase_train_placeholder)
                    embeddings_list.append(emb)
                    labels_list.append(label_id)
                    print(f"✅ Embedding ajouté pour {person_name} - {img_name}")
                except Exception as e:
                    print(f"❌ Échec pour {image_path} : {e}")

        save_embeddings(embeddings_list, labels_list)
        save_class_names(class_names)
        print("🎉 Mise à jour terminée.")

# Pour exécuter manuellement si besoin
if __name__ == "__main__":
    train_incremental()

#Latvia 