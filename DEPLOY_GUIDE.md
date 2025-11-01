# 🚀 Panduan Deploy Streamlit Cloud + Render.com API

## 📋 Checklist Sebelum Deploy

- [x] Model files ada di `model_results/`
- [x] File `app_naga.py` sudah siap
- [x] File `api.py` sudah siap
- [x] Dependencies terdaftar (`requirements.txt` dan `requirements_api.txt`)

---

## 🌐 PART 1: Setup GitHub Repository

### 1.1 Inisialisasi Git (Jika belum)

```bash
cd E:\TUGAS\Skripsi
git init
```

### 1.2 Setup Git LFS untuk Model Files

File model (`.h5`) biasanya besar, jadi perlu Git LFS:

```bash
# Install Git LFS (jika belum)
# Download dari: https://git-lfs.github.com/

# Setup Git LFS di repository
git lfs install
git lfs track "*.h5"
git lfs track "*.png"  # Untuk plot files juga (opsional)
```

### 1.3 Buat .gitattributes

File ini sudah otomatis dibuat saat `git lfs track`, tapi pastikan ada:

```bash
# Cek apakah .gitattributes ada
dir .gitattributes
```

### 1.4 Commit Semua Files

```bash
# Add semua files
git add .

# Commit
git commit -m "Initial commit: Streamlit app + FastAPI + models"
```

### 1.5 Push ke GitHub

```bash
# Buat repository baru di GitHub dulu
# Lalu:
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

**Catatan:** Push pertama mungkin lama karena file model besar.

---

## 🎨 PART 2: Deploy Streamlit Cloud

### 2.1 Login ke Streamlit Cloud

1. Buka https://streamlit.io/cloud
2. Login dengan GitHub account
3. Klik "New app"

### 2.2 Konfigurasi Deploy

- **Repository:** Pilih repository yang baru dibuat
- **Branch:** `main`
- **Main file path:** `app_naga.py`
- **App URL:** (otomatis generate, atau custom)

### 2.3 Advanced Settings (Jika perlu)

**Python version:**
- `3.10` atau `3.11`

**Dependencies:**
- Streamlit Cloud akan otomatis detect `requirements.txt`
- Tapi bisa juga set manual di dashboard

### 2.4 Deploy

Klik "Deploy" dan tunggu build selesai.

**URL Streamlit:** `https://repo-name-streamlit-app.streamlit.app`

---

## 🔌 PART 3: Deploy API ke Render.com

### 3.1 Login ke Render.com

1. Buka https://render.com
2. Login dengan GitHub account
3. Klik "New" → "Web Service"

### 3.2 Connect Repository

- **Repository:** Pilih repository yang sama
- **Name:** `dragon-fruit-api`
- **Region:** Pilih terdekat (Singapore recommended)

### 3.3 Build & Start Commands

Render akan auto-detect dari `render.yaml`, tapi pastikan:

**Build Command:**
```
pip install -r requirements_api.txt
```

**Start Command:**
```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

### 3.4 Advanced Settings

**Environment Variables:**
- Tidak perlu tambah manual (kecuali ada secret)

**Plan:**
- Free plan (sudah set di `render.yaml`)

### 3.5 Deploy

Klik "Create Web Service" dan tunggu build selesai.

**URL API:** `https://dragon-fruit-api.onrender.com`

**Catatan:** Render.com free tier akan sleep setelah 15 menit tidak aktif. Request pertama setelah sleep butuh waktu ~30 detik untuk wake up.

---

## ✅ PART 4: Verifikasi Deploy

### 4.1 Test Streamlit

1. Buka URL Streamlit Cloud
2. Upload gambar buah naga
3. Pastikan prediksi muncul

### 4.2 Test API

**Health Check:**
```
https://dragon-fruit-api.onrender.com/api/health
```

**Documentation:**
```
https://dragon-fruit-api.onrender.com/docs
```

**Root:**
```
https://dragon-fruit-api.onrender.com/
```

### 4.3 Test API dari Browser

Buka Swagger UI:
```
https://dragon-fruit-api.onrender.com/docs
```

1. Klik endpoint `/api/predict/both`
2. Klik "Try it out"
3. Upload gambar
4. Klik "Execute"
5. Lihat hasil

---

## 🔧 Troubleshooting

### Streamlit: Model tidak ditemukan

**Problem:** "File not found: model_results/best_vgg16_model.h5"

**Solution:**
1. Pastikan folder `model_results/` ada di GitHub
2. Pastikan file `.h5` ter-commit (cek dengan Git LFS)
3. Redeploy di Streamlit Cloud

**Cek di GitHub:**
- Buka repository di browser
- Pastikan folder `model_results/` visible
- Klik folder, pastikan file `.h5` ada

### API: "Not Found" Error

**Problem:** URL menampilkan "Not Found"

**Solution:**
1. Cek logs di Render.com dashboard
2. Pastikan build **success** (hijau)
3. Pastikan start command benar
4. Test endpoint `/api/health` dulu

### API: Model tidak dimuat

**Problem:** Health check return `vgg16_loaded: false`

**Solution:**
1. Cek logs di Render.com
2. Pastikan path model benar (harus relative ke `api.py`)
3. Pastikan file model ter-upload ke GitHub
4. Pastikan Git LFS sudah setup dengan benar

### Git LFS: File tidak ter-upload

**Problem:** File model terlalu besar atau tidak ter-track

**Solution:**
```bash
# Cek status Git LFS
git lfs ls-files

# Jika tidak ada, tambah manual
git lfs track "model_results/*.h5"
git add .gitattributes
git add model_results/*.h5
git commit -m "Add model files with LFS"
git push
```

---

## 📝 File Structure untuk GitHub

```
your-repo/
├── .gitattributes          ← Git LFS config (auto-generated)
├── .gitignore             ← Ignore temp files
├── app_naga.py            ← Streamlit app
├── api.py                 ← FastAPI
├── requirements.txt       ← Dependencies Streamlit
├── requirements_api.txt   ← Dependencies API
├── render.yaml            ← Render.com config
├── Procfile               ← Heroku/Railway config
├── README.md              ← Documentation
└── model_results/         ← Model files (MUST BE IN GITHUB)
    ├── best_vgg16_model.h5
    ├── best_mobilenetv2_model.h5
    ├── model_metrics.json
    ├── *.png              ← Plot files
    └── *.txt              ← Report files
```

---

## 🎯 Quick Commands Summary

```bash
# Setup Git LFS
git lfs install
git lfs track "*.h5"

# Add dan commit
git add .
git commit -m "Deploy ready: Streamlit + API + Models"

# Push ke GitHub
git push origin main

# Setelah push, deploy di:
# 1. Streamlit Cloud: https://streamlit.io/cloud
# 2. Render.com: https://render.com
```

---

## 🔗 URLs Setelah Deploy

**Streamlit:**
- URL: `https://repo-name-streamlit-app.streamlit.app`
- Untuk: User interface web

**API:**
- URL: `https://dragon-fruit-api.onrender.com`
- Docs: `https://dragon-fruit-api.onrender.com/docs`
- Untuk: API access, integration dengan app lain

---

## ✨ Tips

1. **Git LFS Quota:** GitHub free tier punya 1GB storage + 1GB bandwidth/month untuk Git LFS. Cukup untuk 2 file model.

2. **Render.com Free Tier:** 
   - Sleep setelah 15 menit idle
   - Wake up butuh ~30 detik
   - Cocok untuk development/testing

3. **Streamlit Cloud:**
   - Free tier unlimited
   - Auto-deploy dari GitHub
   - Perfect untuk web interface

4. **Alternative untuk Production:**
   - Jika butuh API yang selalu aktif: Railway.app (free tier lebih baik)
   - Jika butuh custom domain: Render.com paid atau Railway.app

---

**Selamat! Deploy Anda siap! 🎉**

