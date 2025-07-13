import cv2
import os
from medium_facenet_tutorial.align_dlib import AlignDlib

# === IMPORTER LA MÊME FONCTION QUE TU AS CORRIGÉE MANUELLEMENT ===

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    if bb is None:
        return None

    x, y, w, h = bb.left(), bb.top(), bb.width(), bb.height()

    margin_ratio = 0.3
    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)

    x1 = max(x - x_margin, 0)
    y1 = max(y - int(y_margin * 1.5), 0)  # Plus de marge en haut pour inclure le front
    x2 = min(x + w + x_margin, image.shape[1])
    y2 = min(y + h + y_margin, image.shape[0])

    cropped_face = image[y1:y2, x1:x2]
    aligned = cv2.resize(cropped_face, (crop_dim, crop_dim))

    return aligned

# === TEST D'UNE IMAGE ===

image_path = "C:/Users/Zeff/reconnaissance_faciale/app/dataset_lfw/lfw-deepfunneled/Abdullah_Gul/Abdullah_Gul_0013.jpg"  # <-- Modifie ici avec un vrai chemin
image = cv2.imread(image_path)

if image is None:
    print("❌ Image introuvable.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
aligned = _align_image(image, crop_dim=224)

if aligned is None:
    print("❌ Aucune face détectée.")
else:
    cv2.imwrite("aligned_test1.jpg", cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    print("✅ Image alignée sauvegardée dans aligned_test.jpg")
