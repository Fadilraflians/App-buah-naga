# 🐉 Dragon Fruit Classification System

Sistem klasifikasi tingkat kematangan buah naga menggunakan CNN VGG16 dan MobileNetV2.

## 📋 Deskripsi

Sistem ini terdiri dari:
- **Web Interface (Streamlit)**: Aplikasi web untuk klasifikasi gambar buah naga
- **RESTful API (FastAPI)**: API untuk akses programmatic

## 🚀 Quick Start

### Local Development

**Streamlit Web App:**
```bash
pip install -r requirements.txt
streamlit run app_naga.py
```

**API Server:**
```bash
pip install -r requirements_api.txt
python api.py
```

API akan berjalan di: http://localhost:8000
Documentation: http://localhost:8000/docs

## 🌐 Deployment

### Streamlit Cloud
1. Push code ke GitHub
2. Deploy di https://streamlit.io/cloud
3. Pilih file `app_naga.py`

### Render.com (API)
1. Push code ke GitHub (termasuk `render.yaml`)
2. Deploy di https://render.com
3. Service akan auto-detect dari `render.yaml`

Lihat `DEPLOY_GUIDE.md` untuk panduan lengkap.

## 📁 Struktur Project

```
├── app_naga.py              # Streamlit web application
├── api.py                   # FastAPI RESTful API
├── requirements.txt         # Dependencies untuk Streamlit
├── requirements_api.txt     # Dependencies untuk API
├── model_results/          # Model files (.h5)
│   ├── best_vgg16_model.h5
│   ├── best_mobilenetv2_model.h5
│   └── model_metrics.json
├── render.yaml             # Render.com configuration
└── DEPLOY_GUIDE.md         # Panduan deployment
```

## 📚 Dokumentasi

- `DEPLOY_GUIDE.md` - Panduan deploy lengkap
- `README_API.md` - Dokumentasi API
- `PANDUAN_HOSTING.md` - Opsi hosting
- `TROUBLESHOOTING.md` - Troubleshooting guide

## 🔧 Requirements

- Python 3.10+
- TensorFlow 2.13+
- Streamlit 1.28+
- FastAPI 0.104+
- Model files (VGG16 & MobileNetV2)

## 📝 License

Project untuk Tugas Akhir - Fadil Rafliansyah

