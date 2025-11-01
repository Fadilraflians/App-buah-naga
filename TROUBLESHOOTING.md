# 🔧 Troubleshooting "Not Found" Error di Render.com

## ❌ Masalah: "Not Found" saat mengakses URL

Jika Anda melihat **"Not Found"** di browser, kemungkinan penyebabnya:

### 1. ✅ **Cek Logs di Render.com Dashboard**

1. Login ke Render.com
2. Pilih service `dragon-fruit-api`
3. Klik tab **"Logs"**
4. Lihat error message

**Kemungkinan error:**
- Model file tidak ditemukan
- Dependencies tidak terinstall
- Path model salah

---

### 2. ✅ **Pastikan Model Files Diupload**

Model files (`.h5`) **HARUS** ada di repository GitHub:

```
your-repo/
├── api.py
├── requirements_api.txt
├── render.yaml
└── model_results/
    ├── best_vgg16_model.h5       ← HARUS ADA
    └── best_mobilenetv2_model.h5  ← HARUS ADA
```

**Cara upload:**
1. Pastikan folder `model_results/` ada di GitHub
2. File model harus ikut ter-commit
3. Jika file terlalu besar (>100MB), gunakan Git LFS:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   git add model_results/*.h5
   git commit -m "Add model files"
   git push
   ```

---

### 3. ✅ **Cek Build Status**

Di Render.com dashboard:
- Build harus **success** (hijau)
- Jika **failed** (merah), lihat error di logs

**Kemungkinan masalah build:**
- Dependencies tidak terinstall
- Python version tidak sesuai
- Memory tidak cukup

---

### 4. ✅ **Test Endpoint yang Benar**

**JANGAN akses langsung root domain**, coba endpoint ini:

✅ **Yang benar:**
- `https://dragon-fruit-api.onrender.com/` (root - harusnya OK)
- `https://dragon-fruit-api.onrender.com/api/health`
- `https://dragon-fruit-api.onrender.com/docs`

❌ **Jangan akses:**
- `https://dragon-fruit-api.onrender.com/api` (tanpa trailing slash bisa error)

---

### 5. ✅ **Perbaiki Konfigurasi Render**

Pastikan `render.yaml` sudah benar:

```yaml
services:
  - type: web
    name: dragon-fruit-api
    env: python
    buildCommand: pip install -r requirements_api.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Atau set manual di dashboard Render:**
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements_api.txt`

---

### 6. ✅ **Alternatif: Deploy Manual tanpa render.yaml**

Jika `render.yaml` tidak bekerja:

1. Di Render dashboard, pilih service
2. Settings → Manual Deploy
3. Set:
   - **Build Command**: `pip install -r requirements_api.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Redeploy

---

### 7. ✅ **Cek Model Path**

API akan otomatis detect path model. Tapi jika masih error:

**Test di local dulu:**
```bash
python api.py
```

Cek console output untuk:
- `Base directory: ...`
- `Model results directory: ...`
- `Files in model_results: ...`

Jika path salah di local, akan salah juga di hosting.

---

### 8. ✅ **Fix: Handle Error Gracefully**

API sudah diupdate untuk tetap berjalan meski model tidak ditemukan. Tapi tetap:

1. **Pastikan model files ada di GitHub**
2. **Pastikan folder structure benar**
3. **Cek logs untuk error spesifik**

---

### 9. ✅ **Test dengan Health Check**

Setelah deploy, test endpoint health:

```bash
curl https://dragon-fruit-api.onrender.com/api/health
```

Atau buka di browser:
```
https://dragon-fruit-api.onrender.com/api/health
```

Response seharusnya:
```json
{
  "status": "healthy",
  "vgg16_loaded": true,
  "mobilenetv2_loaded": true
}
```

---

### 10. ✅ **Solusi Cepat: Railway.app Alternative**

Jika Render.com masih bermasalah, coba **Railway.app**:

1. Login ke https://railway.app
2. New Project → Deploy from GitHub
3. Pilih repo
4. Railway auto-detect, tapi set manual:
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add environment variable (opsional):
   - `PORT` = (otomatis)
6. Deploy!

Railway biasanya lebih mudah dan reliable.

---

## 🎯 Checklist Debugging

- [ ] Model files ada di GitHub repository
- [ ] Build di Render.com **success** (hijau)
- [ ] Logs tidak ada error fatal
- [ ] Start command benar: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- [ ] Dependencies terinstall (cek di logs)
- [ ] Mengakses endpoint yang benar (`/` atau `/api/health`)
- [ ] Service status **Live** (bukan "Failed")

---

## 🆘 Masih Error?

**Kirim info ini:**
1. Screenshot logs dari Render.com
2. Response dari `/api/health` endpoint
3. Struktur folder di GitHub (apakah `model_results/` ada?)
4. Ukuran file model (apakah terlalu besar?)

