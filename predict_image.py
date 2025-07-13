import os
import numpy as np
import tensorflow as tf
import cv2
import json
from sklearn.metrics.pairwise import cosine_similarity
from medium_facenet_tutorial.align_dlib import AlignDlib

# Chemins
MODEL_PATH = 'facenet_model/20170511-185253/20170511-185253.pb'
EMBEDDINGS_PATH = 'output/embeddings_labels.npz'
CLASS_NAMES_PATH = 'output/class_names.json'
SHAPE_PREDICTOR_PATH = 'medium_facenet_tutorial/shape_predictor_68_face_landmarks.dat'

IMAGE_SIZE = 224
align = AlignDlib(SHAPE_PREDICTOR_PATH)

# -----------------------------
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj

def load_and_align_image(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgb)
    if bb is None:
        raise ValueError("❌ Aucun visage détecté")

    x, y, w, h = bb.left(), bb.top(), bb.width(), bb.height()
    margin_ratio = 0.3
    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)

    x1 = max(x - x_margin, 0)
    y1 = max(y - int(y_margin * 1.5), 0)
    x2 = min(x + w + x_margin, rgb.shape[1])
    y2 = min(y + h + y_margin, rgb.shape[0])

    cropped = rgb[y1:y2, x1:x2]
    aligned = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
    # Sauvegarde pour debug (optionnel)
    #cv2.imwrite('debug_aligned.jpg', cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    return prewhiten(aligned)

# -----------------------------
def load_model(model_path):
    model_exp = os.path.expanduser(model_path)
    with tf.io.gfile.GFile(model_exp, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

# -----------------------------
def predict_cosine(image_path, threshold=0.7):
    # Charger base d'embeddings
    data = np.load(EMBEDDINGS_PATH)
    known_embeddings = data['embeddings']
    known_labels = data['labels']

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)

    # Embedding de l’image cible
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            load_model(MODEL_PATH)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            aligned = load_and_align_image(image_path)
            aligned = np.expand_dims(aligned, axis=0)

            feed_dict = {
                images_placeholder: aligned,
                phase_train_placeholder: False
            }
            embedding = sess.run(embeddings_tensor, feed_dict=feed_dict)

    # Comparaison avec embeddings connus
    similarities = cosine_similarity(embedding, known_embeddings)[0]
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    best_label = known_labels[best_index]
    best_name = class_names[best_label]

    if best_similarity >= threshold:
        return {
            "status": "recognized",
            "name": best_name,
            "confidence": round(float(best_similarity), 4)
        }
    else:
        return {
            "status": "unrecognized",
            "confidence": round(float(best_similarity), 4)
        }

# -----------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python predict_image.py chemin/vers/image.jpg"
        }))
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_cosine(image_path)
    print(json.dumps(result, indent=2))
