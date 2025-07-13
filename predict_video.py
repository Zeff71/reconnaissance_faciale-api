import cv2
import numpy as np
import tensorflow as tf
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from medium_facenet_tutorial.align_dlib import AlignDlib

# Chemins
MODEL_PATH = 'facenet_model/20170511-185253/20170511-185253.pb'
EMBEDDINGS_PATH = 'output/embeddings_labels.npz'
CLASS_NAMES_PATH = 'output/class_names.json'
SHAPE_PREDICTOR_PATH = 'medium_facenet_tutorial/shape_predictor_68_face_landmarks.dat'

# Initialisation
align = AlignDlib(SHAPE_PREDICTOR_PATH)
IMAGE_SIZE = 224  # même dimension que lors du preprocessing

# -------------------
# Preprocessing utils
# -------------------
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj


def load_model(model_path):
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def align_and_crop(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgb)
    if bb is None:
        return None

    x, y, w, h = bb.left(), bb.top(), bb.width(), bb.height()
    margin = 0.3
    x1 = max(x - int(w * margin), 0)
    y1 = max(y - int(h * 1.5 * margin), 0)
    x2 = min(x + w + int(w * margin), rgb.shape[1])
    y2 = min(y + h + int(h * margin), rgb.shape[0])

    cropped = rgb[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
    return prewhiten(resized)

# -------------------
# Prédiction
# -------------------
def predict_from_video(video_path, threshold=0.6, max_frames=15):
    tf.compat.v1.reset_default_graph()

    # Charger les embeddings et les noms
    data = np.load(EMBEDDINGS_PATH)
    known_embeddings = data['embeddings']
    known_labels = data['labels']

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)

    results = []

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            load_model(MODEL_PATH)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                aligned = align_and_crop(frame)
                if aligned is None:
                    continue

                aligned = np.expand_dims(aligned, axis=0)
                emb_array = sess.run(embeddings_tensor, feed_dict={
                    images_placeholder: aligned,
                    phase_train_placeholder: False
                })

                # Comparaison cosine
                similarities = cosine_similarity(emb_array, known_embeddings)[0]
                best_index = np.argmax(similarities)
                best_score = similarities[best_index]

                if best_score >= threshold:
                    name = class_names[int(known_labels[best_index])]
                else:
                    name = "unknown"

                results.append({
                    "frame": frame_count,
                    "name": name,
                    "confidence": round(float(best_score), 4)
                })

            cap.release()

    return results

if __name__ == "__main__":
    video = "videos/abdoulaye_video.mp4"
    results = predict_from_video(video)

    for r in results:
        print(f"🖼️ Frame {r['frame']:03d} → {r['name']} ({r['confidence']})")
