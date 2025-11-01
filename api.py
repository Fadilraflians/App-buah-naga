"""
API RESTful untuk Klasifikasi Kematangan Buah Naga
Menggunakan FastAPI untuk menyediakan endpoint API
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ==============================================================================
# KONFIGURASI PATH
# ==============================================================================

# Path ke model dan metrik - otomatis detect dari lokasi file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_RESULTS_DIR = os.path.join(BASE_DIR, 'model_results')

# Fallback ke path absolut jika tidak ada di relatif
if not os.path.exists(MODEL_RESULTS_DIR):
    MODEL_RESULTS_DIR = r"E:\TUGAS\Skripsi\model_results"

VGG16_MODEL_PATH = os.path.join(MODEL_RESULTS_DIR, 'best_vgg16_model.h5')
MOBILENETV2_MODEL_PATH = os.path.join(MODEL_RESULTS_DIR, 'best_mobilenetv2_model.h5')

# Parameter gambar
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['Defect Dragon Fruit', 'Immature Dragon Fruit', 'Mature Dragon Fruit']

# ==============================================================================
# INISIALISASI FASTAPI
# ==============================================================================

app = FastAPI(
    title="Dragon Fruit Classification API",
    description="API untuk klasifikasi tingkat kematangan buah naga menggunakan CNN VGG16 dan MobileNetV2",
    version="1.0.0"
)

# CORS middleware untuk mengizinkan akses dari web browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# MEMUAT MODEL (Dilakukan sekali saat startup)
# ==============================================================================

model_vgg16 = None
model_mobilenetv2 = None

@app.on_event("startup")
async def load_models():
    """Memuat model saat aplikasi startup"""
    global model_vgg16, model_mobilenetv2
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Model results directory: {MODEL_RESULTS_DIR}")
    print(f"VGG16 path: {VGG16_MODEL_PATH}")
    print(f"MobileNetV2 path: {MOBILENETV2_MODEL_PATH}")
    
    # List files di model_results jika folder ada
    if os.path.exists(MODEL_RESULTS_DIR):
        print(f"Files in {MODEL_RESULTS_DIR}:")
        try:
            for f in os.listdir(MODEL_RESULTS_DIR):
                print(f"  - {f}")
        except Exception as e:
            print(f"Error listing files: {e}")
    
    print("\nMemuat model VGG16...")
    if os.path.exists(VGG16_MODEL_PATH):
        try:
            model_vgg16 = tf.keras.models.load_model(VGG16_MODEL_PATH)
            print(f"✅ Model VGG16 berhasil dimuat dari '{VGG16_MODEL_PATH}'")
        except Exception as e:
            print(f"❌ Gagal memuat model VGG16: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ File model VGG16 tidak ditemukan: {VGG16_MODEL_PATH}")
        print("⚠️ API akan tetap berjalan, tapi endpoint VGG16 tidak akan tersedia")
    
    print("\nMemuat model MobileNetV2...")
    if os.path.exists(MOBILENETV2_MODEL_PATH):
        try:
            model_mobilenetv2 = tf.keras.models.load_model(MOBILENETV2_MODEL_PATH)
            print(f"✅ Model MobileNetV2 berhasil dimuat dari '{MOBILENETV2_MODEL_PATH}'")
        except Exception as e:
            print(f"❌ Gagal memuat model MobileNetV2: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ File model MobileNetV2 tidak ditemukan: {MOBILENETV2_MODEL_PATH}")
        print("⚠️ API akan tetap berjalan, tapi endpoint MobileNetV2 tidak akan tersedia")
    
    print("\n✅ Startup selesai!")

# ==============================================================================
# FUNGSI PREPROCESSING DAN PREDIKSI
# ==============================================================================

def preprocess_image(img):
    """
    Melakukan pre-processing pada gambar agar sesuai dengan input model CNN.
    """
    try:
        # Konversi RGBA/LA/P ke RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisasi
        return img_array
    except Exception as e:
        return None

def predict_image(model, img_array):
    """
    Melakukan prediksi menggunakan model.
    Mengembalikan (nama_kelas, confidence, scores, stats)
    """
    try:
        predictions = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(predictions[0])
        scores_numpy = scores.numpy()
        
        max_confidence = np.max(scores_numpy) * 100
        predicted_class_index = np.argmax(scores_numpy)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        # Hitung statistik
        sorted_scores = np.sort(scores_numpy)[::-1]
        confidence_diff = (sorted_scores[0] - sorted_scores[1]) * 100 if len(sorted_scores) > 1 else 100
        entropy = -np.sum(scores_numpy * np.log(scores_numpy + 1e-10))
        max_entropy = np.log(len(CLASS_NAMES))
        
        # Deteksi "Tidak Valid"
        is_invalid = False
        if max_confidence < 70:
            is_invalid = True
        elif max_confidence >= 75 and max_confidence < 85:
            if confidence_diff < 30 or (entropy > max_entropy * 0.50 and confidence_diff < 40):
                is_invalid = True
        elif max_confidence >= 85 and max_confidence < 98:
            if confidence_diff < 50 or (entropy > max_entropy * 0.45 and confidence_diff < 60):
                is_invalid = True
        
        if is_invalid:
            predicted_class_name = "Tidak Valid - Bukan Buah Naga"
        
        stats = {
            "confidence_diff": float(confidence_diff),
            "entropy": float(entropy),
            "max_entropy": float(max_entropy),
            "is_valid": not is_invalid
        }
        
        # Konversi scores ke list
        scores_dict = {CLASS_NAMES[i]: float(scores_numpy[i]) * 100 for i in range(len(CLASS_NAMES))}
        
        return predicted_class_name, float(max_confidence), scores_dict, stats
    except Exception as e:
        return None, 0.0, {}, {}

# ==============================================================================
# MODEL RESPONSE
# ==============================================================================

class PredictionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float
    scores: dict
    statistics: dict

class CombinedPredictionResponse(BaseModel):
    vgg16: Optional[PredictionResponse]
    mobilenetv2: Optional[PredictionResponse]
    message: str

# ==============================================================================
# ENDPOINT API
# ==============================================================================

@app.get("/")
async def root():
    """Endpoint root - informasi API"""
    return {
        "message": "Dragon Fruit Classification API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "vgg16": model_vgg16 is not None,
            "mobilenetv2": model_mobilenetv2 is not None
        },
        "endpoints": {
            "predict_vgg16": "/api/predict/vgg16",
            "predict_mobilenetv2": "/api/predict/mobilenetv2",
            "predict_both": "/api/predict/both",
            "health": "/api/health",
            "docs": "/docs"
        },
        "usage": "Visit /docs for interactive API documentation"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vgg16_loaded": model_vgg16 is not None,
        "mobilenetv2_loaded": model_mobilenetv2 is not None
    }

@app.post("/api/predict/vgg16", response_model=PredictionResponse)
async def predict_vgg16(file: UploadFile = File(...)):
    """
    Endpoint untuk prediksi menggunakan model VGG16
    
    - **file**: File gambar (JPG, JPEG, PNG)
    - Returns: Hasil prediksi dengan confidence score
    """
    if model_vgg16 is None:
        raise HTTPException(status_code=503, detail="Model VGG16 tidak dimuat")
    
    # Validasi file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG, JPEG, PNG)")
    
    try:
        # Baca file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocess
        img_array = preprocess_image(img)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Gagal memproses gambar")
        
        # Predict
        prediction, confidence, scores, stats = predict_image(model_vgg16, img_array)
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Gagal melakukan prediksi")
        
        return PredictionResponse(
            model="VGG16",
            prediction=prediction,
            confidence=confidence,
            scores=scores,
            statistics=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/predict/mobilenetv2", response_model=PredictionResponse)
async def predict_mobilenetv2(file: UploadFile = File(...)):
    """
    Endpoint untuk prediksi menggunakan model MobileNetV2
    
    - **file**: File gambar (JPG, JPEG, PNG)
    - Returns: Hasil prediksi dengan confidence score
    """
    if model_mobilenetv2 is None:
        raise HTTPException(status_code=503, detail="Model MobileNetV2 tidak dimuat")
    
    # Validasi file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG, JPEG, PNG)")
    
    try:
        # Baca file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocess
        img_array = preprocess_image(img)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Gagal memproses gambar")
        
        # Predict
        prediction, confidence, scores, stats = predict_image(model_mobilenetv2, img_array)
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Gagal melakukan prediksi")
        
        return PredictionResponse(
            model="MobileNetV2",
            prediction=prediction,
            confidence=confidence,
            scores=scores,
            statistics=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/predict/both", response_model=CombinedPredictionResponse)
async def predict_both(file: UploadFile = File(...)):
    """
    Endpoint untuk prediksi menggunakan kedua model (VGG16 dan MobileNetV2)
    
    - **file**: File gambar (JPG, JPEG, PNG)
    - Returns: Hasil prediksi dari kedua model
    """
    # Validasi file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG, JPEG, PNG)")
    
    try:
        # Baca file sekali
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocess
        img_array = preprocess_image(img)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Gagal memproses gambar")
        
        vgg16_result = None
        mobilenetv2_result = None
        
        # Predict dengan VGG16
        if model_vgg16 is not None:
            try:
                prediction, confidence, scores, stats = predict_image(model_vgg16, img_array)
                if prediction is not None:
                    vgg16_result = PredictionResponse(
                        model="VGG16",
                        prediction=prediction,
                        confidence=confidence,
                        scores=scores,
                        statistics=stats
                    )
            except Exception as e:
                print(f"Error VGG16: {e}")
        
        # Predict dengan MobileNetV2
        if model_mobilenetv2 is not None:
            try:
                prediction, confidence, scores, stats = predict_image(model_mobilenetv2, img_array)
                if prediction is not None:
                    mobilenetv2_result = PredictionResponse(
                        model="MobileNetV2",
                        prediction=prediction,
                        confidence=confidence,
                        scores=scores,
                        statistics=stats
                    )
            except Exception as e:
                print(f"Error MobileNetV2: {e}")
        
        if vgg16_result is None and mobilenetv2_result is None:
            raise HTTPException(status_code=500, detail="Gagal melakukan prediksi dengan kedua model")
        
        return CombinedPredictionResponse(
            vgg16=vgg16_result,
            mobilenetv2=mobilenetv2_result,
            message="Prediksi berhasil"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ==============================================================================
# RUN SERVER (untuk development)
# ==============================================================================

if __name__ == "__main__":
    # Untuk development, jalankan dengan: python api.py
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload saat ada perubahan
    )

