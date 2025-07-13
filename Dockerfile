# Utiliser une image Python légère
FROM python:3.11-slim

# Installer les outils nécessaires
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail principal
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans l’image
COPY . .

# Télécharger le modèle shape_predictor_68_face_landmarks.dat
RUN mkdir -p medium_facenet_tutorial && \
    curl -L -o medium_facenet_tutorial/shape_predictor_68_face_landmarks.dat \
    https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat

# Télécharger et extraire le dataset LFW
WORKDIR /app/dataset_raw
RUN wget https://archive.org/download/lfw-dataset/lfw-dataset.zip && \
    unzip lfw-dataset.zip -d /app/dataset_lfw && \
    rm lfw-dataset.zip

# Revenir au dossier principal et télécharger le modèle FaceNet
WORKDIR /app
RUN mkdir -p /app/facenet_model && \
    python download_and_extract_model.py --model-dir /app/facenet_model

# Exposer le port de l’API
EXPOSE 8000

# Démarrer l’application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]