from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import base64
import numpy as np
import cv2
import tempfile
import shutil
import os
import uuid
import subprocess

from predict_image import predict_cosine
from predict_video import predict_from_video

from register_image import register_image
from train_incremental import train_incremental 

app = FastAPI()

UPLOAD_DIR = "/app/uploads"
NEW_IMAGES_DIR = "/app/dataset_lfw/new_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(NEW_IMAGES_DIR, exist_ok=True)

# 📥 1. Upload image vers dataset_lfw/new_images/<nom>/
@app.post("/upload")
async def upload_face(file: UploadFile = File(...), name: str = Form(...)):
    try:
        # 1. Sauvegarde temporaire de l'image uploadée
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            with open(tmp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 2. Enregistrement définitif dans dataset_lfw/new_images/<nom>/
        final_path = register_image(tmp_path, name)

        # 3. Nettoyage du fichier temporaire
        os.remove(tmp_path)

        return {
            "status": "success",
            "message": f"Image enregistrée pour {name}",
            "saved_path": final_path
        }

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# 🧠 2. Lance entraînement incrémental avec train_incremental.py
@app.post("/train")
async def train_model():
    try:
        train_incremental()
        return {
            "status": "success",
            "message": "Entraînement incrémental terminé."
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 🧠 Image → Prédiction
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_cosine(file_path)
        return result
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 🎥 Vidéo → Prédiction
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4()}.mp4"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = predict_from_video(file_path)
        return {"status": "success", "results": results}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 📸 WebSocket → reconnaissance en live (image par image)
@app.websocket("/ws/stream")
async def stream_faces(websocket: WebSocket):
    await websocket.accept()
    print("📡 WebSocket connecté.")

    try:
        while True:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)

            try:
                result = predict_cosine(temp_path, threshold=0.7)
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })

            os.remove(temp_path)

    except WebSocketDisconnect:
        print("❌ WebSocket client déconnecté.")
