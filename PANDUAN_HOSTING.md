# 🚀 Panduan Hosting Web Streamlit dan API

## 📋 Ringkasan

Anda memiliki **2 aplikasi terpisah**:
1. **Web Streamlit** (`app_naga.py`) - Interface untuk pengguna
2. **API FastAPI** (`api.py`) - API RESTful untuk akses programmatic

## 🌐 Opsi Hosting

### **Opsi 1: Streamlit Cloud (Gratis) - Untuk Web Streamlit**

**Keuntungan:**
- ✅ Gratis
- ✅ Mudah setup
- ✅ Auto-deploy dari GitHub
- ✅ URL publik langsung

**Cara Setup:**
1. Push code ke GitHub repository
2. Login ke [Streamlit Cloud](https://streamlit.io/cloud)
3. Klik "New app"
4. Pilih repository dan file `app_naga.py`
5. Deploy!

**URL:** `https://namarepo-streamlit-app.streamlit.app`

---

### **Opsi 2: Railway.app (Gratis) - Untuk API**

**Keuntungan:**
- ✅ Gratis (dengan limit)
- ✅ Mudah setup
- ✅ Auto-deploy dari GitHub
- ✅ Support FastAPI

**Cara Setup:**
1. Push code ke GitHub
2. Login ke [Railway.app](https://railway.app)
3. Klik "New Project" → "Deploy from GitHub repo"
4. Pilih repository
5. Railway akan auto-detect Python
6. Set start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
7. Deploy!

**Environment Variables** (jika perlu):
- `MODEL_RESULTS_DIR` (opsional, jika struktur folder berbeda)

**URL:** `https://namaproject.up.railway.app`

---

### **Opsi 3: Render.com (Gratis) - Untuk API**

**Keuntungan:**
- ✅ Gratis (dengan limit)
- ✅ Mudah setup
- ✅ Support FastAPI

**Cara Setup:**
1. Push code ke GitHub (termasuk `render.yaml`)
2. Login ke [Render.com](https://render.com)
3. Klik "New Web Service"
4. Connect GitHub repository
5. Render akan auto-detect `render.yaml`
6. Deploy!

**File yang diperlukan:**
- `render.yaml` ✅ (sudah dibuat)
- `requirements_api.txt` ✅ (sudah dibuat)

**URL:** `https://dragon-fruit-api.onrender.com`

---

### **Opsi 4: PythonAnywhere (Gratis/Paid) - Untuk Keduanya**

**Keuntungan:**
- ✅ Free tier tersedia
- ✅ Support Streamlit dan FastAPI
- ✅ Full control

**Cara Setup Streamlit:**
1. Login ke PythonAnywhere
2. Upload file ke server
3. Install dependencies via Bash console
4. Setup Web app dengan manual config
5. Run Streamlit dengan port forwarding

**Cara Setup API:**
1. Upload `api.py` dan model files
2. Install dependencies
3. Run: `uvicorn api:app --host 0.0.0.0 --port 8000`

---

## 🔧 Konfigurasi untuk Hosting

### **1. Update Path di `api.py` (SUDAH FIX)**

Path sudah otomatis detect dari lokasi file, jadi akan bekerja di hosting.

### **2. Pastikan File Model Ada**

Saat hosting, pastikan file model (`best_vgg16_model.h5`, `best_mobilenetv2_model.h5`) ikut ter-upload.

**Cara:**
- Upload ke GitHub (tapi file besar mungkin perlu Git LFS)
- Atau upload langsung ke hosting platform
- Atau gunakan cloud storage (S3, Google Cloud Storage)

### **3. Update CORS di `api.py` untuk Production**

**Saat ini (development):**
```python
allow_origins=["*"]  # Izinkan semua origin
```

**Untuk production (ganti dengan):**
```python
allow_origins=[
    "https://namastreamlit.streamlit.app",
    "https://yourdomain.com"
]  # Domain spesifik saja
```

---

## 📦 Struktur File untuk Hosting

```
your-repo/
├── app_naga.py              # Web Streamlit
├── api.py                   # FastAPI
├── requirements_api.txt      # Dependencies untuk API
├── requirements.txt         # Dependencies untuk Streamlit (opsional)
├── Procfile                 # Untuk Heroku/Railway
├── render.yaml              # Untuk Render.com
├── model_results/           # Folder model (perlu diupload)
│   ├── best_vgg16_model.h5
│   ├── best_mobilenetv2_model.h5
│   ├── model_metrics.json
│   └── *.png, *.txt         # Grafik dan report
└── README.md
```

---

## 🔗 Integrasi Web Streamlit dengan API

Jika ingin Streamlit menggunakan API (opsional):

```python
# Di app_naga.py, tambahkan opsi menggunakan API
API_URL = "https://your-api-url.com"

def predict_via_api(image_bytes):
    """Prediksi via API"""
    import requests
    files = {'file': image_bytes}
    response = requests.post(f"{API_URL}/api/predict/both", files=files)
    return response.json()
```

---

## 🧪 Testing API Lokal

1. Install dependencies:
```bash
pip install -r requirements_api.txt
```

2. Jalankan API:
```bash
python api.py
```

3. Test dengan script:
```bash
python test_api.py
```

4. Atau test manual di browser:
   - http://localhost:8000/docs (Swagger UI)
   - Upload gambar dan test langsung!

---

## 📝 Checklist Sebelum Deploy

- [ ] Push semua file ke GitHub
- [ ] Pastikan `model_results/` folder ada dan berisi model
- [ ] Update CORS di `api.py` untuk production
- [ ] Test API lokal dulu
- [ ] Test Streamlit lokal dulu
- [ ] Siapkan environment variables jika perlu

---

## 🆘 Troubleshooting

### API tidak bisa load model
- Pastikan path `model_results/` benar
- Check apakah file model ada di hosting
- Cek log error di hosting platform

### CORS Error
- Update `allow_origins` di `api.py`
- Pastikan URL web Streamlit sudah ditambahkan

### Model file terlalu besar untuk GitHub
- Gunakan Git LFS: `git lfs track "*.h5"`
- Atau upload model langsung ke hosting platform
- Atau gunakan cloud storage

---

## 🎯 Rekomendasi Final

**Untuk Development/Testing:**
- Streamlit: Jalankan lokal (`streamlit run app_naga.py`)
- API: Jalankan lokal (`python api.py`)

**Untuk Production:**
- Streamlit: Streamlit Cloud (gratis, mudah)
- API: Railway.app atau Render.com (gratis, mudah)

**Alternatif:**
- Gunakan satu platform yang support kedua (misal: PythonAnywhere dengan custom setup)

