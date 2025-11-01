# 🚀 Quick Start Guide

## 📦 File yang Telah Dibuat

1. ✅ **`api.py`** - FastAPI untuk RESTful API
2. ✅ **`requirements_api.txt`** - Dependencies untuk API
3. ✅ **`requirements.txt`** - Dependencies untuk Streamlit
4. ✅ **`test_api.py`** - Script testing API
5. ✅ **`render.yaml`** - Config untuk Render.com
6. ✅ **`Procfile`** - Config untuk Heroku/Railway
7. ✅ **`contoh_penggunaan_api.html`** - Contoh frontend menggunakan API

## 🏃 Cara Menjalankan (Lokal)

### 1. Install Dependencies API
```bash
pip install -r requirements_api.txt
```

### 2. Jalankan API Server
```bash
python api.py
```

API akan berjalan di: **http://localhost:8000**

### 3. Test API
- Buka browser: http://localhost:8000/docs
- Atau jalankan: `python test_api.py`

### 4. Jalankan Streamlit (Terpisah)
```bash
streamlit run app_naga.py
```

Streamlit akan berjalan di: **http://localhost:8501**

## 🌐 Hosting (Production)

### Untuk Streamlit (Web Interface):
1. Push ke GitHub
2. Deploy di **Streamlit Cloud**: https://streamlit.io/cloud
3. URL: `https://repo-name-streamlit-app.streamlit.app`

### Untuk API:
1. Push ke GitHub (termasuk `render.yaml` dan `Procfile`)
2. Deploy di:
   - **Railway.app** (Recommended) - https://railway.app
   - **Render.com** - https://render.com
   - **PythonAnywhere** - https://www.pythonanywhere.com
3. URL: `https://your-api-url.railway.app`

## 📝 Catatan Penting

- ✅ API dan Streamlit adalah **2 aplikasi terpisah**
- ✅ Keduanya bisa di-host di platform berbeda
- ✅ API bisa digunakan oleh siapa saja yang punya URL
- ✅ Streamlit untuk user interface, API untuk programmatic access

## 🔗 Integrasi

Setelah API di-hosting, Anda bisa:
1. Menggunakan API dari web frontend (HTML/JavaScript)
2. Menggunakan API dari aplikasi mobile
3. Menggunakan API dari web Streamlit (opsional)
4. Menggunakan API dari aplikasi Python lainnya

---

**Selesai! API dan Web sudah siap untuk di-hosting! 🎉**

