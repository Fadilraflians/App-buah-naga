# 🚀 LANGKAH-LANGKAH DEPLOY

## ⚡ Quick Deploy (Auto Setup)

Jalankan script PowerShell ini untuk setup otomatis:

```powershell
.\setup_deploy.ps1
```

Script akan:
- ✅ Check Git & Git LFS
- ✅ Initialize Git repository
- ✅ Setup Git LFS untuk file .h5
- ✅ Check model files
- ✅ Show next steps

---

## 📝 Manual Setup (Step-by-Step)

### **STEP 1: Install Git LFS**

1. Download Git LFS: https://git-lfs.github.com/
2. Install dan restart terminal/PowerShell
3. Verify: `git lfs version`

**Atau dengan winget:**
```powershell
winget install GitHub.GitLFS
```

---

### **STEP 2: Setup Git Repository**

```powershell
# Masuk ke folder project
cd E:\TUGAS\Skripsi

# Initialize Git (jika belum)
git init

# Setup Git LFS
git lfs install

# Track file besar dengan Git LFS
git lfs track "*.h5"
git lfs track "model_results/*.h5"
git lfs track "*.png"

# Add semua files
git add .

# Commit
git commit -m "Initial commit: Streamlit app + FastAPI + Models"
```

---

### **STEP 3: Push ke GitHub**

#### 3.1 Buat Repository di GitHub

1. Login ke https://github.com
2. Klik "New repository"
3. Nama: `dragon-fruit-classification` (atau sesuai keinginan)
4. **JANGAN** centang "Initialize with README"
5. Klik "Create repository"

#### 3.2 Push Code

```powershell
# Add remote (ganti USERNAME dan REPO_NAME)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# Push ke GitHub
git branch -M main
git push -u origin main
```

**⚠️ Catatan:** Push pertama mungkin lama (5-10 menit) karena file model besar.

---

### **STEP 4: Deploy Streamlit Cloud**

#### 4.1 Login ke Streamlit Cloud

1. Buka https://streamlit.io/cloud
2. Login dengan GitHub account
3. Authorize Streamlit Cloud untuk akses GitHub

#### 4.2 Deploy App

1. Klik **"New app"**
2. Pilih repository: `USERNAME/REPO_NAME`
3. Pilih branch: `main`
4. **Main file path:** `app_naga.py`
5. **App URL:** (biarkan auto-generate, atau custom)
6. Klik **"Deploy"**

#### 4.3 Tunggu Deploy

- Build biasanya butuh 2-5 menit
- Jika berhasil, akan muncul URL: `https://repo-name-streamlit-app.streamlit.app`
- Klik URL untuk test

#### 4.4 Verify Model Files

Jika error "model tidak ditemukan":
1. Cek di GitHub: Apakah folder `model_results/` ada?
2. Cek apakah file `.h5` visible di GitHub
3. Jika file terlalu besar dan tidak muncul, mungkin Git LFS belum benar
4. Redeploy di Streamlit Cloud

---

### **STEP 5: Deploy API ke Render.com**

#### 5.1 Login ke Render.com

1. Buka https://render.com
2. Login dengan GitHub account
3. Authorize Render untuk akses GitHub

#### 5.2 Create Web Service

1. Klik **"New"** → **"Web Service"**
2. Pilih repository: `USERNAME/REPO_NAME`
3. **Name:** `dragon-fruit-api`
4. **Region:** Pilih terdekat (Singapore recommended)

#### 5.3 Configure Service

Render akan auto-detect dari `render.yaml`, tapi pastikan:

**Build Command:**
```
pip install -r requirements_api.txt
```

**Start Command:**
```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Environment:**
- Python (auto-detect)
- Plan: Free

#### 5.4 Deploy

1. Klik **"Create Web Service"**
2. Tunggu build selesai (biasanya 5-10 menit)
3. Jika berhasil, akan muncul URL: `https://dragon-fruit-api.onrender.com`

#### 5.5 Verify API

Test endpoint:
- Root: https://dragon-fruit-api.onrender.com/
- Health: https://dragon-fruit-api.onrender.com/api/health
- Docs: https://dragon-fruit-api.onrender.com/docs

---

## ✅ VERIFIKASI DEPLOYMENT

### Test Streamlit

1. Buka URL Streamlit Cloud
2. Upload gambar buah naga
3. Pastikan prediksi muncul
4. Pastikan grafik muncul

### Test API

1. Buka https://dragon-fruit-api.onrender.com/docs
2. Klik endpoint `/api/predict/both`
3. Klik "Try it out"
4. Upload gambar
5. Klik "Execute"
6. Pastikan response muncul

---

## 🔧 TROUBLESHOOTING

### ❌ Git LFS: File tidak ter-upload

**Problem:** File `.h5` tidak muncul di GitHub

**Solution:**
```powershell
# Cek apakah file di-track
git lfs ls-files

# Jika kosong, add manual
git lfs track "*.h5"
git add .gitattributes
git add model_results/*.h5
git commit -m "Add model files with LFS"
git push
```

---

### ❌ Streamlit: Model tidak ditemukan

**Problem:** Error "File not found: model_results/best_vgg16_model.h5"

**Checklist:**
1. ✅ Folder `model_results/` ada di GitHub?
2. ✅ File `.h5` visible di GitHub?
3. ✅ Git LFS sudah setup?
4. ✅ Redeploy di Streamlit Cloud?

**Fix:**
1. Cek di GitHub: https://github.com/USERNAME/REPO/tree/main/model_results
2. Pastikan file visible
3. Jika tidak, push ulang dengan Git LFS
4. Redeploy di Streamlit Cloud

---

### ❌ Render: "Not Found"

**Problem:** URL menampilkan "Not Found"

**Checklist:**
1. ✅ Build status **success** (hijau)?
2. ✅ Logs tidak ada error fatal?
3. ✅ Start command benar?

**Fix:**
1. Buka Render dashboard → Logs
2. Cek error message
3. Pastikan model files ada
4. Pastikan path model benar (relative ke `api.py`)
5. Redeploy

---

### ❌ Render: Model tidak dimuat

**Problem:** Health check return `vgg16_loaded: false`

**Checklist:**
1. ✅ File model ada di GitHub?
2. ✅ Path model benar?
3. ✅ Cek logs untuk error spesifik

**Fix:**
1. Buka Render dashboard → Logs
2. Cari error "File not found"
3. Pastikan struktur folder sama dengan lokal
4. Path harus: `model_results/best_vgg16_model.h5` (relative)

---

## 📋 CHECKLIST FINAL

Sebelum deploy, pastikan:

- [x] Git LFS sudah install
- [x] Git repository sudah initialized
- [x] `.gitattributes` sudah ada (auto dari `git lfs track`)
- [x] Model files ada di `model_results/`
- [x] `requirements.txt` dan `requirements_api.txt` ada
- [x] `app_naga.py` dan `api.py` sudah final
- [x] `render.yaml` ada untuk Render.com
- [x] Code sudah di-push ke GitHub
- [x] File model visible di GitHub (via Git LFS)

---

## 🎯 URLs Setelah Deploy

**Streamlit Web:**
```
https://repo-name-streamlit-app.streamlit.app
```

**API:**
```
https://dragon-fruit-api.onrender.com
```

**API Documentation:**
```
https://dragon-fruit-api.onrender.com/docs
```

---

## 💡 Tips

1. **Git LFS Quota:** GitHub free tier = 1GB storage + 1GB bandwidth/month. Cukup untuk 2 model files.

2. **Render.com Free Tier:**
   - Sleep setelah 15 menit idle
   - Request pertama setelah sleep butuh ~30 detik
   - Perfect untuk development/testing

3. **Streamlit Cloud:**
   - Free tier unlimited
   - Auto-deploy dari GitHub (setiap push)
   - Perfect untuk web interface

4. **Alternative untuk Production:**
   - Railway.app (free tier lebih baik untuk API)
   - PythonAnywhere (jika butuh custom setup)

---

**Selamat! Semua sudah siap untuk deploy! 🎉**

