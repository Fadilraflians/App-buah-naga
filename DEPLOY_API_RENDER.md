# 🚀 Panduan Deploy API ke Render.com

## ✅ Prasyarat

- ✅ Repository GitHub sudah ada: `Fadilraflians/App-buah-naga`
- ✅ File `api.py` sudah ada
- ✅ File `render.yaml` sudah ada
- ✅ File `requirements_api.txt` sudah ada

---

## 📝 Langkah-Langkah Deploy

### **1. Login ke Render.com**

1. Buka: https://render.com
2. Klik **"Get Started for Free"** atau **"Sign In"**
3. Pilih **"Sign in with GitHub"**
4. Authorize Render untuk akses GitHub repository

---

### **2. Create New Web Service**

1. Setelah login, di dashboard klik **"New +"**
2. Pilih **"Web Service"**

---

### **3. Connect Repository**

1. **Connect account:** Pilih GitHub account Anda
2. **Repository:** Pilih `Fadilraflians/App-buah-naga`
3. Klik **"Connect"**

---

### **4. Configure Service**

Render akan auto-detect dari `render.yaml`, tapi pastikan setting berikut:

**Basic Settings:**
- **Name:** `dragon-fruit-api` (atau nama lain)
- **Region:** Pilih terdekat (Singapore recommended untuk Indonesia)
- **Branch:** `main`
- **Root Directory:** (biarkan kosong, atau `./` jika perlu)

**Build & Deploy:**
- **Environment:** `Python 3`
- **Build Command:** 
  ```
  pip install -r requirements_api.txt
  ```
- **Start Command:**
  ```
  uvicorn api:app --host 0.0.0.0 --port $PORT
  ```

**Advanced (opsional):**
- **Auto-Deploy:** `Yes` (auto-deploy setiap push ke GitHub)
- **Plan:** `Free` (untuk testing)

---

### **5. Deploy!**

1. Klik **"Create Web Service"** di bagian bawah
2. Tunggu build selesai (biasanya 5-10 menit)
3. Build process:
   - Clone repository
   - Install dependencies dari `requirements_api.txt`
   - Start API server

---

### **6. Verify API**

Setelah deploy selesai, Render akan memberikan URL:
- Contoh: `https://dragon-fruit-api.onrender.com`

**Test Endpoints:**
1. **Root:** `https://dragon-fruit-api.onrender.com/`
2. **Health Check:** `https://dragon-fruit-api.onrender.com/api/health`
3. **Documentation:** `https://dragon-fruit-api.onrender.com/docs`

---

## 🔧 Troubleshooting

### **Build Failed**

**Cek:**
1. Tab **"Logs"** di Render dashboard
2. Pastikan `requirements_api.txt` ada di root repository
3. Pastikan semua dependencies terinstall

**Fix:**
- Pastikan file `requirements_api.txt` ada
- Pastikan `api.py` ada di root (bukan di subfolder)

---

### **API Error: Model Not Found**

**Problem:** Path model tidak ditemukan di Render

**Solution:**
- Path sudah auto-detect dari `api.py` (menggunakan `os.path.dirname(__file__)`)
- Pastikan folder `model_results/` ada di GitHub
- Pastikan file `.h5` ter-upload dengan Git LFS

---

### **Service Keeps Crashing**

**Cek Logs:**
1. Buka service di Render dashboard
2. Tab **"Logs"**
3. Lihat error message

**Common Issues:**
- Memory limit (free tier = 512MB)
- Model files terlalu besar
- Dependencies conflict

---

### **"Not Found" Error**

**Problem:** URL menampilkan "Not Found"

**Solution:**
1. Pastikan service status **"Live"** (bukan "Failed")
2. Pastikan start command benar: `uvicorn api:app --host 0.0.0.0 --port $PORT`
3. Test endpoint `/api/health` dulu

---

## ⚙️ Update API Code

Jika perlu update `api.py`:

1. Edit `api.py` di lokal
2. Commit dan push ke GitHub:
   ```bash
   git add api.py
   git commit -m "Update API"
   git push
   ```
3. Render akan **auto-redeploy** (jika auto-deploy enabled)

---

## 🔒 Production Settings (Opsional)

Untuk production, update di Render dashboard:

1. **Environment Variables:**
   - `PYTHON_VERSION=3.11`
   - `MODEL_RESULTS_DIR=/opt/render/project/src/model_results` (jika perlu)

2. **Health Check Path:**
   - `/api/health`

3. **Custom Domain (paid):**
   - Add custom domain jika perlu

---

## 📊 Monitoring

Di Render dashboard, Anda bisa lihat:
- **Metrics:** CPU, Memory usage
- **Logs:** Real-time logs
- **Events:** Deploy history
- **Health:** Service status

---

## 💡 Tips

1. **Free Tier Limitations:**
   - Service akan **sleep** setelah 15 menit idle
   - Request pertama setelah sleep butuh ~30 detik untuk wake up
   - Perfect untuk development/testing

2. **Keep Alive (Opsional):**
   - Setup cron job atau uptime monitor untuk ping service setiap 10 menit
   - Atau upgrade ke paid plan

3. **Large Files:**
   - Model files besar (~385 MB) mungkin butuh waktu untuk download di first deploy
   - Pastikan Git LFS sudah setup dengan benar

---

**Siap untuk deploy! Ikuti langkah-langkah di atas! 🚀**

