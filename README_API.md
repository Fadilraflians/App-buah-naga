# API RESTful untuk Klasifikasi Buah Naga

## 📋 Deskripsi

API RESTful menggunakan FastAPI untuk klasifikasi tingkat kematangan buah naga. API ini memungkinkan akses programmatic ke model CNN VGG16 dan MobileNetV2.

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Jalankan API Server

```bash
python api.py
```

Atau dengan uvicorn langsung:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API akan berjalan di: `http://localhost:8000`

### 3. Akses Dokumentasi API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoint API

### 1. Health Check
```
GET /api/health
```
Cek status API dan model yang dimuat.

### 2. Prediksi VGG16
```
POST /api/predict/vgg16
Content-Type: multipart/form-data
Body: file (gambar)
```

### 3. Prediksi MobileNetV2
```
POST /api/predict/mobilenetv2
Content-Type: multipart/form-data
Body: file (gambar)
```

### 4. Prediksi Kedua Model
```
POST /api/predict/both
Content-Type: multipart/form-data
Body: file (gambar)
```

## 💻 Contoh Penggunaan

### Python (requests)

```python
import requests

url = "http://localhost:8000/api/predict/both"
files = {'file': open('gambar_buah_naga.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/predict/both" \
  -F "file=@gambar_buah_naga.jpg"
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/predict/both', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🌐 Hosting API

### Opsi 1: Railway.app (Gratis)
1. Push code ke GitHub
2. Login ke Railway.app
3. Connect GitHub repository
4. Deploy otomatis

### Opsi 2: Render.com (Gratis)
1. Buat file `render.yaml`:
```yaml
services:
  - type: web
    name: dragon-fruit-api
    env: python
    buildCommand: pip install -r requirements_api.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
```

### Opsi 3: PythonAnywhere
1. Upload file ke PythonAnywhere
2. Install dependencies via Bash console
3. Setup Web app dengan manual config

### Opsi 4: Heroku
1. Buat file `Procfile`:
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```
2. Deploy via Heroku CLI

## 📝 Response Format

### Single Model Response
```json
{
  "model": "VGG16",
  "prediction": "Mature Dragon Fruit",
  "confidence": 95.3,
  "scores": {
    "Defect Dragon Fruit": 2.1,
    "Immature Dragon Fruit": 1.5,
    "Mature Dragon Fruit": 96.4
  },
  "statistics": {
    "confidence_diff": 94.3,
    "entropy": 0.15,
    "max_entropy": 1.099,
    "is_valid": true
  }
}
```

### Both Models Response
```json
{
  "vgg16": { ... },
  "mobilenetv2": { ... },
  "message": "Prediksi berhasil"
}
```

## 🔒 Security Notes

Untuk production:
- Ganti `allow_origins=["*"]` dengan domain spesifik
- Tambahkan authentication (API keys)
- Gunakan HTTPS
- Rate limiting

