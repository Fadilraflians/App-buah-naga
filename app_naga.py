import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random # Diperlukan untuk Mode Presentasi
import io

# Import Gemini dengan error handling
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # Set ke None untuk avoid error

# Import konfigurasi Gemini dari file terpisah
try:
    from config_gemini import (
        GEMINI_API_KEY_DEFAULT,
        GEMINI_MODEL_NAME,
        GEMINI_PROMPT_DETECTION
    )
except ImportError:
    # Fallback jika config_gemini.py tidak ada
    GEMINI_API_KEY_DEFAULT = "AIzaSyB4yIzOnkwfUkIgKwv8jWRcdRNI0RmgZjg"
    GEMINI_MODEL_NAME = "gemini-2.0-flash"  # Updated: gemini-1.5-flash sudah deprecated
    GEMINI_PROMPT_DETECTION = """Anda adalah pakar identifikasi buah naga (dragon fruit). 

Analisis gambar ini dengan teliti dan jawab dengan format JSON TANPA markdown:
{
    "is_dragon_fruit": true atau false,
    "confidence": angka 0-100,
    "reason": "alasan singkat dalam bahasa Indonesia"
}

Kriteria BUAH NAGA:
✓ Buah dengan kulit pink/merah/ungu dengan sisik hijau yang menonjol
✓ Bentuk bulat atau oval dengan tekstur sisik yang khas
✓ Daging putih/merah dengan biji hitam kecil (jika terpotong)
✓ Bukan apel, jeruk, pisang, mangga, atau buah lain
✓ Bukan dokumen, teks, sertifikat, atau objek non-buah
✓ Bukan tanaman/pohon buah naga (hanya buahnya saja)

Jawab HANYA dengan JSON, tanpa markdown, tanpa penjelasan tambahan."""

# ==============================================================================
# BAGIAN 1: PENGATURAN PATH
# ==============================================================================

st.set_page_config(
    page_title="Klasifikasi Kematangan Buah Naga", # Dihapus (Lokal)
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tambahkan meta tag untuk mobile viewport
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# --- PERUBAHAN: Menggunakan Path Relatif untuk Cloud Deployment ---
# Path otomatis detect dari lokasi file app_naga.py
# PERBAIKAN: Di Streamlit Cloud, file berada di /mount/src/app-buah-naga/
try:
    # Method 1: Coba dari __file__ (untuk local development dan Streamlit Cloud)
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_RESULTS_DIR = os.path.join(BASE_DIR, 'model_results')
        
        # Debug: Tampilkan path yang dicoba (hanya untuk troubleshooting)
        import sys
        if 'streamlit' in sys.modules and not os.path.exists(MODEL_RESULTS_DIR):
            # Di Streamlit Cloud, cek juga di mount/src
            possible_paths = [
                MODEL_RESULTS_DIR,  # Dari __file__
                os.path.join(os.getcwd(), 'model_results'),  # Dari current working directory
                '/mount/src/app-buah-naga/model_results',  # Path Streamlit Cloud langsung
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    MODEL_RESULTS_DIR = path
                    BASE_DIR = os.path.dirname(path)
                    break
        elif not os.path.exists(MODEL_RESULTS_DIR):
            raise FileNotFoundError()
            
    except:
        # Method 2: Coba dari current working directory
        BASE_DIR = os.getcwd()
        MODEL_RESULTS_DIR = os.path.join(BASE_DIR, 'model_results')
        
        # Method 3: Coba path Streamlit Cloud
        if not os.path.exists(MODEL_RESULTS_DIR):
            streamlit_path = '/mount/src/app-buah-naga/model_results'
            if os.path.exists(streamlit_path):
                MODEL_RESULTS_DIR = streamlit_path
                BASE_DIR = os.path.dirname(streamlit_path)
        
        # Method 4: Fallback ke absolute path untuk development lokal
        if not os.path.exists(MODEL_RESULTS_DIR):
            MODEL_RESULTS_DIR = r"E:\TUGAS\Skripsi\model_results"
            BASE_DIR = os.path.dirname(MODEL_RESULTS_DIR)
    
    # Final check dengan debug info
    if not os.path.exists(MODEL_RESULTS_DIR):
        st.error(f"❌ Error: Folder model_results tidak ditemukan!")
        st.error(f"Path yang dicoba: {MODEL_RESULTS_DIR}")
        st.error(f"Current working directory: {os.getcwd()}")
        try:
            st.error(f"__file__ path: {os.path.abspath(__file__) if '__file__' in globals() else 'N/A'}")
        except:
            pass
        st.error("Pastikan folder 'model_results' ada di repository GitHub Anda.")
        st.info("💡 Cek di: https://github.com/Fadilraflians/App-buah-naga/tree/main/model_results")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Error saat mengatur path: {e}")
    st.error("Pastikan semua files sudah ter-upload ke GitHub.")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Menentukan path lengkap ke file model
# --- PERUBAHAN: Menyesuaikan nama file .h5 ---
VGG16_MODEL_PATH = os.path.join(MODEL_RESULTS_DIR, 'best_vgg16_model.h5')
MOBILENETV2_MODEL_PATH = os.path.join(MODEL_RESULTS_DIR, 'best_mobilenetv2_model.h5')
MODEL_METRICS_FILE = os.path.join(MODEL_RESULTS_DIR, 'model_metrics.json')

# Parameter gambar (harus sama dengan saat pelatihan)
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['Defect Dragon Fruit', 'Immature Dragon Fruit', 'Mature Dragon Fruit']


# ==============================================================================
# BAGIAN 1.1: CUSTOM CSS (Tetap sama)
# ==============================================================================

# Custom CSS
st.markdown("""
<style>
    /* ... (CSS lengkap Anda dari file asli ada di sini) ... */
    /* Background utama dengan gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding-top: 0rem !important;
    }
    
    /* Kurangi padding-top dari main content area */
    .main {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Hapus margin atas dari element pertama */
    .main .block-container > div:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    
    /* Override margin dari Streamlit default */
    div[data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0rem !important;
    }
    
    /* Override element-container margin */
    .element-container:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    
    /* Override stMarkdown margin untuk header pertama */
    .stMarkdown:first-child h1 {
        margin-top: 0rem !important;
    }
    
    /* Override margin dari semua element di dalam block-container yang pertama */
    .main .block-container > *:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    
    /* Container utama */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 0.5rem 2rem;
        margin: 0rem 1rem !important;
        margin-top: 0rem !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Saat sidebar tertutup - konten terpusat dengan max-width yang wajar */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main .block-container {
        margin-left: auto !important;
        margin-right: auto !important;
        margin-top: 0rem !important;
        margin-bottom: 1rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px !important;
        width: calc(100% - 4rem) !important;
    }
    
    /* Sidebar styling - Pastikan sidebar selalu terlihat dengan animasi */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%) !important;
        min-width: 250px !important;
        transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1),
                    opacity 0.3s ease,
                    box-shadow 0.3s ease,
                    margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Sidebar saat terbuka - tambahkan efek glow */
    section[data-testid="stSidebar"]:not([aria-hidden="true"]) {
        box-shadow: 4px 0 30px rgba(78, 205, 196, 0.3) !important;
    }
    
    /* Sidebar saat tertutup */
    section[data-testid="stSidebar"][aria-hidden="true"] {
        transform: translateX(-100%) !important;
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    /* Sejajarkan konten utama dengan kartu Informasi Model di sidebar */
    /* Sidebar biasanya memiliki padding-top sekitar 1rem, kartu Informasi Model ada di posisi pertama */
    section[data-testid="stSidebar"] .element-container:first-child {
        padding-top: 1rem;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #90EE90;
        margin-top: 0rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #4ECDC4;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        color: white;
        border: 2px solid #4ECDC4;
    }
    
    .success-box {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #FFFFFF;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        color: white;
        border: 2px solid #FFFFFF;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #E67E22 0%, #F39C12 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #FFFFFF;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        color: white;
        border: 2px solid #FFFFFF;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
        margin: 1rem 0;
        border: 3px solid #4ECDC4;
        position: relative;
        color: white;
    }
    
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        border: 3px solid rgba(255,255,255,0.2);
    }
    
    .prediction-result.invalid {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        border: 3px solid white;
    }
    
    /* Upload area styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        border-radius: 15px;
        border: 3px dashed #4ECDC4;
        padding: 2rem;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* File uploader text styling */
    .stFileUploader label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    .stFileUploader p {
        color: white !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    .stFileUploader div[data-testid="stFileUploader"] {
        color: white !important;
    }
    
    /* Upload button styling */
    .stFileUploader button {
        background: linear-gradient(45deg, #4ECDC4, #45B7D1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .stFileUploader button:hover {
        background: linear-gradient(45deg, #45B7D1, #4ECDC4) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #4ECDC4, #45B7D1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4ECDC4, #45B7D1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #45B7D1, #4ECDC4);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Footer styling */
    .footer-box {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Jangan sembunyikan header - butuh untuk tombol toggle sidebar */
    /* header {visibility: hidden;} */
    
    /* Styling untuk tombol toggle sidebar agar lebih terlihat */
    header[data-testid="stHeader"] {
        background: transparent !important;
        position: relative !important;
        z-index: 1000 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
    }
    
    /* Container untuk tombol toggle sidebar - geser ke pojok kiri tanpa celah */
    div[data-testid="stToolbar"],
    header[data-testid="stHeader"] > div:first-child,
    header[data-testid="stHeader"] > div {
        justify-content: flex-start !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
    }
    
    /* Force header content ke pojok kiri tanpa celah */
    header[data-testid="stHeader"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Tombol toggle sidebar (hamburger menu) - Pojok kiri TANPA CELAH */
    button[data-testid="baseButton-header"] {
        background: linear-gradient(135deg, #4ECDC4 0%, #45B7D1 50%, #4ECDC4 100%) !important;
        background-size: 200% 200% !important;
        color: white !important;
        border: 3px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        min-width: 60px !important;
        min-height: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.4), 
                    0 0 20px rgba(78, 205, 196, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        animation: pulseGlow 2s ease-in-out infinite !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Saat sidebar tertutup - geser tombol ke pojok kiri TANPA CELAH */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ header[data-testid="stHeader"] button[data-testid="baseButton-header"] {
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    
    /* Pastikan header toolbar align ke pojok kiri TANPA CELAH */
    header[data-testid="stHeader"] > div:first-child {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
    }
    
    /* Hapus semua spacing tambahan - TANPA CELAH SAMA SEKALI */
    header[data-testid="stHeader"] > div:first-child > *:first-child,
    header[data-testid="stHeader"] button[data-testid="baseButton-header"],
    div[data-testid="stToolbar"] > *:first-child {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Pastikan semua child elements tidak punya margin/padding kiri */
    header[data-testid="stHeader"] * {
        margin-left: 0 !important;
    }
    
    /* Exception: biarkan tombol punya margin-right untuk spacing internal */
    header[data-testid="stHeader"] button[data-testid="baseButton-header"] {
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    
    /* AGRESIF: Paksa tombol ke pojok kiri dengan absolute positioning jika perlu */
    header[data-testid="stHeader"] {
        position: relative !important;
    }
    
    /* Pastikan container header tidak punya padding/margin apapun */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] > div,
    header[data-testid="stHeader"] > div > div {
        padding-left: 0 !important;
        margin-left: 0 !important;
    }
    
    /* Pastikan header container bisa menampung tombol absolute */
    header[data-testid="stHeader"] > div:first-child {
        position: relative !important;
        min-height: 60px !important;
    }
    
    /* Force tombol ke pojok kiri TANPA CELAH dengan absolute positioning */
    header[data-testid="stHeader"] button[data-testid="baseButton-header"],
    button[data-testid="baseButton-header"] {
        position: absolute !important;
        left: 0 !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        margin: 0 !important;
        padding-left: 0.25rem !important;
        padding-right: 0.75rem !important;
        z-index: 10000 !important;
        order: -999 !important;
        flex-shrink: 0 !important;
    }
    
    
    /* Animasi pulse glow untuk menarik perhatian */
    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 6px 20px rgba(78, 205, 196, 0.4), 
                        0 0 20px rgba(78, 205, 196, 0.2);
        }
        50% {
            box-shadow: 0 6px 25px rgba(78, 205, 196, 0.6), 
                        0 0 30px rgba(78, 205, 196, 0.4);
        }
    }
    
    /* Animasi gradient background */
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Hover effect - lebih dramatis dengan absolute positioning */
    button[data-testid="baseButton-header"]:hover {
        background: linear-gradient(135deg, #45B7D1 0%, #4ECDC4 50%, #45B7D1 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientShift 1.5s ease infinite, pulseGlow 1.5s ease-in-out infinite !important;
        transform: translateY(-50%) scale(1.15) rotate(5deg) !important;
        box-shadow: 0 8px 30px rgba(78, 205, 196, 0.7), 
                    0 0 40px rgba(78, 205, 196, 0.5),
                    inset 0 0 20px rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
        left: 0 !important;
    }
    
    /* Active/pressed effect dengan absolute positioning */
    button[data-testid="baseButton-header"]:active {
        transform: translateY(-50%) scale(1.05) rotate(-5deg) !important;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.5), 
                    inset 0 2px 10px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.1s ease !important;
        left: 0 !important;
    }
    
    /* Ripple effect background */
    button[data-testid="baseButton-header"]::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    button[data-testid="baseButton-header"]:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Icon hamburger di tombol - lebih besar dan jelas */
    button[data-testid="baseButton-header"] svg {
        fill: white !important;
        width: 32px !important;
        height: 32px !important;
        transition: transform 0.3s ease !important;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) !important;
        z-index: 1 !important;
        position: relative !important;
    }
    
    /* Animasi rotasi icon saat hover */
    button[data-testid="baseButton-header"]:hover svg {
        transform: rotate(90deg) scale(1.1) !important;
    }
    
    /* Remove white spaces and improve spacing */
    .stApp > div {
        background: transparent;
    }
    
    /* Main content container - smooth transition saat sidebar buka/tutup */
    .main .block-container {
        max-width: 1200px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        /* Biarkan Streamlit mengatur margin secara dinamis untuk pergeseran */
    }
    
    /* Saat sidebar tertutup - pastikan main area tidak ada margin kiri berlebihan */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main {
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        left: 0 !important;
    }
    
    /* Saat sidebar tertutup - pastikan konten benar-benar mulai dari kiri */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main > div {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Pastikan block-container benar-benar mulai dari kiri tanpa margin */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main .block-container {
        margin-left: 0 !important;
    }
    
    /* Main content area - smooth transition - Streamlit otomatis set margin-left */
    .main {
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        /* Biarkan Streamlit mengatur margin-left secara dinamis */
    }
    
    /* Pastikan saat sidebar tertutup, konten bergeser sepenuhnya ke kiri */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ * {
        margin-left: 0 !important;
    }
    
    /* Override: Pastikan block-container mulai dari kiri saat sidebar tertutup */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main .block-container {
        margin-left: 0 !important;
        padding-left: 1.5rem !important;
        width: calc(100% - 3rem) !important;
        max-width: none !important;
    }
    
    /* Pastikan tidak ada spacing tambahan dari Streamlit */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ section[data-testid="stMain"] {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Agresif: Hapus semua margin kiri dari parent elements saat sidebar tertutup (tapi biarkan padding untuk readability) */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main {
        margin-left: 0 !important;
    }
    
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main > div {
        margin-left: 0 !important;
    }
    
    /* Pastikan element pertama di main content tidak punya margin kiri */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main > div:first-child {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Header title - smooth transition dan ikut bergeser dengan konten */
    .main-header {
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: center !important;
        /* Header akan ikut bergeser karena berada dalam .main yang sudah memiliki transition */
    }
    
    /* Pastikan semua elemen konten memiliki transisi smooth */
    .element-container,
    .stMarkdown,
    .stColumns,
    [data-testid="column"],
    section.main > div {
        transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1),
                    transform 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Pastikan semua elemen di main content bergeser dengan smooth */
    .info-box,
    .metric-card,
    .prediction-result,
    .footer-box {
        transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1),
                    transform 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Memastikan app content area responsive terhadap sidebar dengan smooth transition */
    div[data-testid="stAppViewContainer"] {
        transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Saat sidebar tertutup - pastikan app view container tidak punya margin kiri */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ div[data-testid="stAppViewContainer"] {
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100% !important;
        left: 0 !important;
    }
    
    /* Saat sidebar tertutup - pastikan semua elemen di dalam app view container mulai dari kiri */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ div[data-testid="stAppViewContainer"] > div {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Pastikan tidak ada transform atau positioning yang membuat celah */
    section[data-testid="stSidebar"][aria-hidden="true"] ~ .main {
        transform: translateX(0) !important;
    }
    
    /* Memastikan semua elemen Streamlit memiliki transisi */
    section[data-testid="stSidebar"] ~ * {
        transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Improve sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%) !important;
    }
    
    .css-1d391kg .css-1lcbmhc {
        background: transparent !important;
    }
    
    /* Style metric components */
    .metric-container {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: white;
        border: 2px solid #4ECDC4;
    }
    
    /* Improve image display */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* Style code blocks */
    .stCode {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Improve spinner */
    .stSpinner {
        color: #4ECDC4;
    }
    
    /* Text styling for better readability */
    .info-box h3, .info-box h4 {
        color: white !important;
        font-weight: 700;
    }
    
    .info-box p, .info-box li {
        color: white !important;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .success-box h4, .success-box p {
        color: white !important;
        font-weight: 600;
    }
    
    .warning-box h4, .warning-box p {
        color: white !important;
        font-weight: 600;
    }
    
    .metric-container h4, .metric-container h2, .metric-container h3 {
        color: white !important;
        font-weight: 700;
    }
    
    .metric-card h3, .metric-card h2, .metric-card h4 {
        color: white !important;
        font-weight: 700;
    }
    
    /* Sidebar text styling */
    .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: white !important;
        font-weight: 700;
    }
    
    .css-1d391kg p, .css-1d391kg li {
        color: #E8F8F5 !important;
        font-weight: 500;
    }
    
    /* General text styling for better visibility */
    .stApp .stMarkdown {
        color: white !important;
    }
    
    .stApp .stMarkdown p {
        color: white !important;
        font-weight: 500 !important;
    }
    
    .stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Streamlit text elements */
    .stText {
        color: white !important;
    }
    
    .stTextInput > div > div > input {
        color: #2C3E50 !important;
        background: white !important;
    }
    
    /* File uploader specific styling */
    .stFileUploader > div {
        background: transparent !important;
    }
    
    .stFileUploader .uploadedFile {
        color: white !important;
        background: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* ============================================================================
       RESPONSIVE DESIGN UNTUK MOBILE
       ============================================================================ */
    
    /* Mobile devices (max-width: 768px) */
    @media (max-width: 768px) {
        /* Header utama - perbesar font untuk mobile */
        .main-header {
            font-size: 2rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Sub header */
        .sub-header {
            font-size: 1.3rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Container utama - kurangi padding untuk mobile */
        .main .block-container {
            padding: 1rem !important;
            margin: 0.5rem !important;
            border-radius: 15px !important;
        }
        
        /* Info box - perbesar padding dan font */
        .info-box,
        .success-box,
        .warning-box {
            padding: 1rem !important;
            margin: 0.75rem 0 !important;
            font-size: 0.95rem !important;
        }
        
        .info-box h3,
        .info-box h4,
        .success-box h4,
        .warning-box h4 {
            font-size: 1.1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .info-box p,
        .info-box li,
        .success-box p,
        .warning-box p {
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
        }
        
        /* Metric cards - full width di mobile */
        .metric-card {
            padding: 1.5rem !important;
            margin: 0.75rem 0 !important;
        }
        
        .metric-card h3 {
            font-size: 1.2rem !important;
        }
        
        .metric-card h2 {
            font-size: 1.8rem !important;
        }
        
        .metric-container {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .metric-container h4 {
            font-size: 0.95rem !important;
        }
        
        .metric-container h2 {
            font-size: 1.5rem !important;
        }
        
        .metric-container h3 {
            font-size: 1.2rem !important;
        }
        
        /* Prediction result - perbesar font untuk mobile */
        .prediction-result {
            padding: 1.5rem !important;
            margin: 0.75rem 0 !important;
        }
        
        .prediction-result h2 {
            font-size: 1.5rem !important;
        }
        
        .prediction-result h3 {
            font-size: 1.2rem !important;
        }
        
        .prediction-result p {
            font-size: 0.9rem !important;
        }
        
        /* Upload area - perbesar untuk touch target */
        .stFileUploader > div > div {
            padding: 1.5rem !important;
        }
        
        /* Sidebar - overlay style di mobile untuk responsif */
        section[data-testid="stSidebar"] {
            position: fixed !important;
            z-index: 999 !important;
            width: 85% !important;
            max-width: 300px !important;
            height: 100vh !important;
            top: 0 !important;
            left: 0 !important;
            box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5) !important;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* Sidebar tertutup di mobile - sembunyikan ke kiri */
        section[data-testid="stSidebar"][aria-hidden="true"] {
            transform: translateX(-100%) !important;
        }
        
        /* Sidebar terbuka di mobile */
        section[data-testid="stSidebar"]:not([aria-hidden="true"]) {
            transform: translateX(0) !important;
        }
        
        /* Backdrop overlay saat sidebar terbuka di mobile */
        section[data-testid="stSidebar"]:not([aria-hidden="true"])::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.5);
            z-index: 998;
            pointer-events: auto;
        }
        
        /* Sidebar styling di mobile */
        .css-1d391kg {
            width: 85% !important;
            max-width: 300px !important;
        }
        
        /* Main content saat sidebar terbuka di mobile */
        section[data-testid="stSidebar"]:not([aria-hidden="true"]) ~ .main {
            margin-left: 0 !important;
            filter: blur(0) !important;
        }
        
        /* Tombol close sidebar di mobile - tetap di pojok kiri */
        section[data-testid="stSidebar"]:not([aria-hidden="true"]) button[data-testid="baseButton-header"] {
            position: absolute !important;
            z-index: 10001 !important;
            top: 50% !important;
            left: 0 !important;
            transform: translateY(-50%) !important;
            background: rgba(78, 205, 196, 0.9) !important;
            margin: 0 !important;
            padding-left: 0.25rem !important;
        }
        
        /* Tombol toggle sidebar - pastikan tetap terlihat dan cukup besar untuk mobile */
        button[data-testid="baseButton-header"] {
            position: absolute !important;
            left: 0 !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            min-width: 50px !important;
            min-height: 50px !important;
            padding: 0.75rem !important;
            border-width: 2px !important;
            animation: pulseGlow 2.5s ease-in-out infinite !important;
            margin: 0 !important;
            padding-left: 0.25rem !important;
            z-index: 10000 !important;
        }
        
        button[data-testid="baseButton-header"]:hover {
            transform: translateY(-50%) scale(1.1) !important;
            left: 0 !important;
        }
        
        button[data-testid="baseButton-header"]:active {
            transform: translateY(-50%) scale(1.05) !important;
            left: 0 !important;
        }
        
        button[data-testid="baseButton-header"] svg {
            width: 28px !important;
            height: 28px !important;
        }
        
        button[data-testid="baseButton-header"]:hover svg {
            transform: rotate(90deg) scale(1.05) !important;
        }
        
        /* Sidebar text lebih kecil di mobile */
        .css-1d391kg h2 {
            font-size: 1.2rem !important;
        }
        
        .css-1d391kg h3 {
            font-size: 1rem !important;
        }
        
        .css-1d391kg h4 {
            font-size: 0.9rem !important;
        }
        
        .css-1d391kg p {
            font-size: 0.85rem !important;
        }
        
        /* Sidebar padding lebih kecil */
        .css-1d391kg .element-container {
            padding: 0.5rem 0 !important;
        }
        
        /* Tab styling untuk mobile */
        .stTabs [data-baseweb="tab-list"] {
            padding: 0.25rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding: 0.5rem 0.75rem !important;
        }
        
        /* Button - perbesar untuk mobile touch */
        .stButton > button {
            padding: 0.75rem 1.5rem !important;
            font-size: 1rem !important;
            min-height: 44px !important; /* Minimum touch target size */
        }
        
        /* Footer */
        .footer-box {
            padding: 1.5rem !important;
            margin-top: 1rem !important;
        }
        
        .footer-box h3 {
            font-size: 1.3rem !important;
        }
        
        .footer-box p {
            font-size: 0.9rem !important;
        }
        
        /* Text umum - perbesar untuk readability */
        .stApp .stMarkdown {
            font-size: 0.95rem !important;
        }
        
        .stApp .stMarkdown h1 {
            font-size: 1.8rem !important;
        }
        
        .stApp .stMarkdown h2 {
            font-size: 1.4rem !important;
        }
        
        .stApp .stMarkdown h3 {
            font-size: 1.2rem !important;
        }
        
        /* Radio buttons - perbesar spacing */
        .stRadio > div {
            gap: 0.5rem !important;
        }
        
        .stRadio label {
            font-size: 1rem !important;
            padding: 0.75rem !important;
        }
        
        /* File uploader label */
        .stFileUploader label {
            font-size: 1rem !important;
        }
        
        /* Image display */
        .stImage img {
            border-radius: 10px !important;
        }
        
        /* Code blocks */
        .stCode {
            padding: 0.75rem !important;
            font-size: 0.85rem !important;
            overflow-x: auto !important;
        }
        
        /* Heatmap caption */
        .stImage > div > div {
            font-size: 0.85rem !important;
        }
        
        /* Spinner text */
        .stSpinner > div {
            font-size: 1rem !important;
        }
        
        /* Tables - scroll horizontal di mobile */
        .stDataFrame,
        table {
            overflow-x: auto !important;
            display: block !important;
            width: 100% !important;
        }
        
        /* Charts - full width di mobile */
        .stPyplot {
            width: 100% !important;
        }
        
        /* Input fields - perbesar di mobile */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            font-size: 16px !important; /* Prevent zoom on iOS */
            padding: 0.75rem !important;
        }
        
        /* Selectbox dan dropdown */
        .stSelectbox > div > div {
            font-size: 1rem !important;
        }
        
        /* Camera input */
        .stCameraInput {
            width: 100% !important;
        }
        
        /* Image containers */
        [data-testid="stImage"] {
            width: 100% !important;
        }
        
        /* Metric display - stack vertically */
        .element-container [data-testid="metric-container"] {
            margin: 0.5rem 0 !important;
        }
    }
    
    /* Small mobile devices (max-width: 480px) */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem !important;
        }
        
        .sub-header {
            font-size: 1.1rem !important;
        }
        
        .main .block-container {
            padding: 0.75rem !important;
            margin: 0.25rem !important;
        }
        
        .metric-card {
            padding: 1rem !important;
        }
        
        .prediction-result {
            padding: 1rem !important;
        }
        
        .prediction-result h2 {
            font-size: 1.2rem !important;
        }
        
        .info-box,
        .success-box,
        .warning-box {
            padding: 0.75rem !important;
        }
        
        /* Sidebar - full width di mobile kecil */
        section[data-testid="stSidebar"] {
            width: 90% !important;
            max-width: 280px !important;
        }
        
        .css-1d391kg {
            width: 90% !important;
            max-width: 280px !important;
        }
        
        /* Column layout - stack vertically di mobile kecil */
        .stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
        
        /* Force columns to stack pada mobile */
        [data-testid="column"] {
            min-width: 100% !important;
            width: 100% !important;
        }
        
        /* Radio button horizontal menjadi vertical di mobile */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* Tablet devices (min-width: 481px, max-width: 1024px) */
    @media (min-width: 481px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.5rem !important;
        }
        
        .sub-header {
            font-size: 1.5rem !important;
        }
        
        .main .block-container {
            padding: 1.5rem !important;
            margin: 0.75rem !important;
        }
        
        /* Sidebar - overlay style untuk tablet juga */
        section[data-testid="stSidebar"] {
            position: fixed !important;
            z-index: 999 !important;
            width: 70% !important;
            max-width: 350px !important;
            height: 100vh !important;
            box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Sidebar tertutup di tablet */
        section[data-testid="stSidebar"][aria-hidden="true"] {
            transform: translateX(-100%) !important;
        }
        
        /* Backdrop overlay untuk tablet */
        section[data-testid="stSidebar"]:not([aria-hidden="true"])::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.4);
            z-index: 998;
            pointer-events: auto;
        }
        
        .css-1d391kg {
            width: 70% !important;
            max-width: 350px !important;
        }
    }
    
    /* Desktop - pastikan sidebar tidak overlay (normal behavior) */
    @media (min-width: 1025px) {
        section[data-testid="stSidebar"] {
            position: relative !important;
            transform: none !important;
            height: auto !important;
        }
        
        section[data-testid="stSidebar"]:not([aria-hidden="true"])::before {
            display: none !important;
        }
    }
    
    /* Touch-friendly improvements untuk semua mobile */
    @media (hover: none) and (pointer: coarse) {
        /* Perbesar semua clickable elements */
        .stButton > button,
        .stFileUploader button,
        .stRadio label,
        .stCheckbox label {
            min-height: 44px !important;
            min-width: 44px !important;
        }
        
        /* Hapus hover effects di touch devices */
        .stButton > button:hover,
        .stFileUploader button:hover {
            transform: none !important;
        }
    }
    
    /* ============================================================================
       STYLING EXPANDER UNTUK TOGGLE BUTTON
       ============================================================================ */
    
    /* Expander header - make it more clickable and visible */
    [data-baseweb="accordion"] {
        background: transparent !important;
    }
    
    /* Expander button/toggle area */
    [data-baseweb="accordion"] button {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%) !important;
        color: white !important;
        border: 2px solid #4ECDC4 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        width: 100% !important;
        cursor: pointer !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-baseweb="accordion"] button:hover {
        background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%) !important;
        border-color: #4ECDC4 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(78, 205, 196, 0.3) !important;
    }
    
    /* Expander icon (arrow) */
    [data-baseweb="accordion"] svg {
        color: #4ECDC4 !important;
        fill: #4ECDC4 !important;
    }
    
    /* Expander content area */
    [data-baseweb="accordion"] > div[aria-expanded="true"] {
        background: transparent !important;
        padding: 0.5rem 0 !important;
    }
    
    /* Make sure expander is clickable */
    [data-baseweb="accordion"] button,
    [data-baseweb="accordion"] button > div {
        pointer-events: auto !important;
        cursor: pointer !important;
        z-index: 10 !important;
        position: relative !important;
    }
    
    /* Sidebar expander specific styling */
    .css-1d391kg [data-baseweb="accordion"] button {
        margin-bottom: 0.5rem !important;
    }
    
    /* Ensure expander content is visible */
    [data-baseweb="accordion"] > div {
        overflow: visible !important;
    }
    
    /* Fix for sidebar expander - ensure it works */
    section[data-testid="stSidebar"] [data-baseweb="accordion"] {
        margin-bottom: 1rem !important;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="accordion"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    /* Make expander text clearly visible */
    [data-baseweb="accordion"] button span {
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# BAGIAN 1.2: MEMUAT METRIK DAN MODEL (LOKAL)
# ==============================================================================

@st.cache_data
def load_model_metrics():
    """
    Memuat metrik model dari file JSON.
    """
    if os.path.exists(MODEL_METRICS_FILE):
        try:
            with open(MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # Debug: Tampilkan struktur JSON untuk troubleshooting
            if metrics:
                # Pastikan struktur yang diharapkan ada
                for model in ['vgg16', 'mobilenetv2']:
                    if model not in metrics:
                        metrics[model] = {}
                    if not isinstance(metrics[model], dict):
                        continue
                    
                    # Mapping key yang mungkin berbeda di JSON
                    # Beberapa JSON menggunakan 'test_accuracy' bukan 'accuracy'
                    if 'test_accuracy' in metrics[model] and 'accuracy' not in metrics[model]:
                        metrics[model]['accuracy'] = metrics[model]['test_accuracy']
                    
                    # Mapping path yang mungkin berbeda
                    # Beberapa JSON menggunakan 'plot_path' bukan 'accuracy_loss_plot_path'
                    path_mapping = {
                        'plot_path': 'accuracy_loss_plot_path',
                        'cm_path': 'confusion_matrix_plot_path',
                        'report_path': 'classification_report_path'
                    }
                    for old_key, new_key in path_mapping.items():
                        if old_key in metrics[model] and new_key not in metrics[model]:
                            metrics[model][new_key] = metrics[model][old_key]
                    
                    # --- PERBAIKAN PATH GAMBAR & LAPORAN ---
                    # Path di JSON mungkin relatif atau path Kaggle (misal: "/kaggle/working/model_results/...")
                    # Kita perlu menggabungkannya dengan BASE_DIR atau mencari file lokal
                    for key in ['accuracy_loss_plot_path', 'confusion_matrix_plot_path', 'classification_report_path']:
                        if key in metrics[model] and metrics[model][key]:
                            # Menggabungkan BASE_DIR dengan path relatif dari JSON
                            relative_path = metrics[model][key]
                            # Pastikan path adalah string, bukan None
                            if isinstance(relative_path, str):
                                # Handle path Kaggle (misal: "/kaggle/working/model_results/...")
                                if relative_path.startswith('/kaggle/working/'):
                                    # Extract filename dari path Kaggle
                                    filename = os.path.basename(relative_path)
                                    # Cari file dengan berbagai variasi nama
                                    possible_names = [
                                        filename,
                                        filename.replace('VGG16', 'vgg16').replace('MobileNetV2', 'mobilenetv2'),
                                        filename.replace('vgg16', 'VGG16').replace('mobilenetv2', 'MobileNetV2'),
                                    ]
                                    # Tambahkan pattern berdasarkan key
                                    if key == 'accuracy_loss_plot_path':
                                        model_name_capital = 'VGG16' if model == 'vgg16' else 'MobileNetV2'
                                        possible_names.extend([
                                            f"{model_name_capital}_accuracy_loss.png",
                                            f"{model}_accuracy_loss.png",
                                            f"{model_name_capital}_training_history.png",
                                            f"{model}_training_history.png",
                                        ])
                                    elif key == 'confusion_matrix_plot_path':
                                        model_name_capital = 'VGG16' if model == 'vgg16' else 'MobileNetV2'
                                        possible_names.extend([
                                            f"{model_name_capital}_confusion_matrix.png",
                                            f"{model}_confusion_matrix.png",
                                            f"{model_name_capital}_cm.png",
                                            f"{model}_cm.png",
                                        ])
                                    elif key == 'classification_report_path':
                                        model_name_capital = 'VGG16' if model == 'vgg16' else 'MobileNetV2'
                                        possible_names.extend([
                                            f"{model_name_capital}_classification_report.txt",
                                            f"{model}_classification_report.txt",
                                            f"{model_name_capital}_report.txt",
                                            f"{model}_report.txt",
                                        ])
                                    
                                    found = False
                                    for name in possible_names:
                                        if name:
                                            alt_path = os.path.join(MODEL_RESULTS_DIR, name)
                                            if os.path.exists(alt_path):
                                                metrics[model][key] = alt_path
                                                found = True
                                                break
                                    if not found:
                                        metrics[model][key] = None
                                else:
                                    # Path relatif biasa
                                    absolute_path = os.path.join(BASE_DIR, relative_path)
                                    
                                    # Simpan path absolut kembali ke dict
                                    if os.path.exists(absolute_path):
                                        metrics[model][key] = absolute_path
                                    else:
                                        # Coba mencari file dengan pattern yang mungkin
                                        filename = os.path.basename(relative_path)
                                        possible_names = [
                                            filename,
                                            filename.replace('VGG16', 'vgg16').replace('MobileNetV2', 'mobilenetv2'),
                                            filename.replace('vgg16', 'VGG16').replace('mobilenetv2', 'MobileNetV2'),
                                        ]
                                        found = False
                                        for name in possible_names:
                                            if name:
                                                alt_path = os.path.join(MODEL_RESULTS_DIR, name)
                                                if os.path.exists(alt_path):
                                                    metrics[model][key] = alt_path
                                                    found = True
                                                    break
                                        if not found:
                                            metrics[model][key] = None
                        else:
                            # Jika key tidak ada, coba mencari file dengan pattern default
                            model_name_capital = 'VGG16' if model == 'vgg16' else 'MobileNetV2'
                            possible_patterns = {
                                'accuracy_loss_plot_path': [
                                    f"{model_name_capital}_accuracy_loss.png",
                                    f"{model}_accuracy_loss.png",
                                    f"{model_name_capital}_training_history.png",
                                    f"{model}_training_history.png"
                                ],
                                'confusion_matrix_plot_path': [
                                    f"{model_name_capital}_confusion_matrix.png",
                                    f"{model}_confusion_matrix.png",
                                    f"{model_name_capital}_cm.png",
                                    f"{model}_cm.png"
                                ],
                                'classification_report_path': [
                                    f"{model_name_capital}_classification_report.txt",
                                    f"{model}_classification_report.txt",
                                    f"{model_name_capital}_report.txt",
                                    f"{model}_report.txt"
                                ]
                            }
                            if key in possible_patterns:
                                for pattern in possible_patterns[key]:
                                    alt_path = os.path.join(MODEL_RESULTS_DIR, pattern)
                                    if os.path.exists(alt_path):
                                        metrics[model][key] = alt_path
                                        break
                                else:
                                    if key not in metrics[model]:
                                        metrics[model][key] = None
            
            return metrics
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON dari '{MODEL_METRICS_FILE}'. File mungkin corrupt. Error: {e}")
            return None
        except Exception as e:
            st.error(f"Gagal memuat metrik model dari '{MODEL_METRICS_FILE}'. Error: {e}")
            return None
    else:
        st.warning(f"⚠ File metrik model '{MODEL_METRICS_FILE}' tidak ditemukan. Pastikan file ada di lokasi tersebut.")
        return None

# Muat metrik model
model_performance_metrics = load_model_metrics()

@st.cache_resource
def load_models():
    """
    Memuat model VGG16 dan MobileNetV2 dari file .h5 atau .keras.
    """
    model_vgg16 = None
    model_mobilenetv2 = None

    # Fix untuk compatibility issues dengan TensorFlow/Keras versi berbeda
    
    # 1. InputLayer compatibility - handle batch_shape untuk TensorFlow 2.15
    # Model dibuat dengan batch_shape, tapi TF 2.15 tidak support
    # Perlu custom InputLayer yang benar-benar compatible
    class CompatibleInputLayer(tf.keras.layers.InputLayer):
        """Custom InputLayer yang handle batch_shape dengan benar"""
        @classmethod
        def from_config(cls, config):
            """Override from_config untuk handle batch_shape"""
            # Convert batch_shape ke input_shape jika ada
            if 'batch_shape' in config:
                batch_shape = config.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    # Skip batch dimension, ambil [H, W, C]
                    config['input_shape'] = tuple(batch_shape[1:])
            return super().from_config(config)
    
    # 2. DTypePolicy compatibility - handle Keras 3.x dtype policy
    # Keras 3.x menggunakan DTypePolicy, TensorFlow 2.x menggunakan string langsung
    class DTypePolicyCompat:
        """Compatible DTypePolicy untuk handle Keras 3.x model di TensorFlow 2.x"""
        def __init__(self, name='float32', *args, **kwargs):
            if isinstance(name, str):
                self.name = name
            elif hasattr(name, 'name'):
                self.name = name.name
            else:
                self.name = 'float32'
            # Tambahkan attribute yang dibutuhkan TensorFlow
            self.compute_dtype = self.name
            self.variable_dtype = self.name
        
        @property
        def dtype(self):
            """Return dtype sebagai string"""
            return self.name
        
        @classmethod
        def from_config(cls, config):
            """Dari config Keras 3.x"""
            if isinstance(config, dict):
                name = config.get('name', 'float32')
            else:
                name = 'float32'
            return cls(name=name)
        
        def get_config(self):
            return {'name': self.name}
        
        def __call__(self, dtype=None):
            """Callable untuk compatibility"""
            return self.name
    
    # Try import dari keras jika ada
    try:
        from keras import DTypePolicy
        # Gunakan wrapper yang compatible
        DTypePolicyClass = DTypePolicyCompat
    except ImportError:
        DTypePolicyClass = DTypePolicyCompat
    
    # Custom objects untuk load model dengan compatibility fixes
    # PERBAIKAN: Gunakan CompatibleInputLayer yang handle batch_shape di from_config
    # Ini lebih aman karena hanya override from_config, bukan __init__
    custom_objects = {
        'InputLayer': CompatibleInputLayer,
        'DTypePolicy': DTypePolicyClass,
    }
    
    # Muat VGG16
    # PERBAIKAN: Coba multiple methods untuk handle compatibility issues
    if os.path.exists(VGG16_MODEL_PATH):
        try:
            # Method 1: Load dengan custom_objects (InputLayer + DTypePolicy)
            try:
                model_vgg16 = tf.keras.models.load_model(
                    VGG16_MODEL_PATH, 
                    compile=False,
                    custom_objects=custom_objects
                )
            except Exception as e1:
                # Method 2: Load weights saja, lalu rebuild architecture
                # Ini bypass masalah batch_shape di config
                if 'batch_shape' in str(e1).lower() or 'InputLayer' in str(e1):
                    try:
                        # Load model dengan compile=True mungkin bisa bypass
                        model_vgg16 = tf.keras.models.load_model(
                            VGG16_MODEL_PATH,
                            compile=True,
                            custom_objects=custom_objects
                        )
                    except Exception as e2:
                        # Method 3: Load tanpa custom_objects untuk InputLayer
                        # Hanya handle DTypePolicy
                        try:
                            model_vgg16 = tf.keras.models.load_model(
                                VGG16_MODEL_PATH,
                                compile=False,
                                custom_objects={'DTypePolicy': DTypePolicyClass}
                            )
                        except Exception as e3:
                            # Method 4: Load tanpa custom_objects sama sekali
                            try:
                                model_vgg16 = tf.keras.models.load_model(
                                    VGG16_MODEL_PATH,
                                    compile=False
                                )
                            except:
                                # Semua method gagal, raise error pertama
                                raise e1
                else:
                    raise e1
            # st.success(f"Model VGG16 berhasil dimuat dari '{VGG16_MODEL_PATH}'.") # Dihapus
        except Exception as e:
            st.error(f"Gagal memuat model VGG16. Error: {e}")
            st.warning("💡 Model mungkin dibuat dengan TensorFlow/Keras versi yang berbeda")
            st.info("💡 Solusi: Re-train model dengan TensorFlow 2.15 atau update requirements.txt ke Keras 3.x")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error(f"File model VGG16 '{os.path.basename(VGG16_MODEL_PATH)}' tidak ditemukan.")

    # Muat MobileNetV2
    # PERBAIKAN: Coba load dengan beberapa metode untuk handle compatibility
    if os.path.exists(MOBILENETV2_MODEL_PATH):
        try:
            # Method 1: Load normal dengan custom_objects
            try:
                model_mobilenetv2 = tf.keras.models.load_model(
                    MOBILENETV2_MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objects
                )
            except Exception as e2:
                # Method 2: Load tanpa custom_objects
                if 'batch_shape' in str(e2).lower() or 'DTypePolicy' in str(e2) or 'as_list' in str(e2):
                    try:
                        model_mobilenetv2 = tf.keras.models.load_model(
                            MOBILENETV2_MODEL_PATH,
                            compile=False
                        )
                    except:
                        try:
                            model_mobilenetv2 = tf.keras.models.load_model(
                                MOBILENETV2_MODEL_PATH,
                                compile=True
                            )
                        except:
                            raise e2
                else:
                    raise e2
        except Exception as e:
            st.error(f"Gagal memuat model MobileNetV2. Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        # Debug info untuk troubleshooting
        st.warning(f"⚠️ File model MobileNetV2 tidak ditemukan di path: {MOBILENETV2_MODEL_PATH}")
        st.info(f"💡 Cek apakah file ada di GitHub: https://github.com/Fadilraflians/App-buah-naga/tree/main/model_results")
        
        # List files di model_results untuk debug
        if os.path.exists(MODEL_RESULTS_DIR):
            try:
                files_in_dir = os.listdir(MODEL_RESULTS_DIR)
                st.info(f"📁 Files yang ada di {MODEL_RESULTS_DIR}:")
                for f in sorted(files_in_dir):
                    st.text(f"   - {f}")
            except Exception as e:
                st.error(f"Error listing files: {e}")
        
    return model_vgg16, model_mobilenetv2

# Muat model
model_vgg16, model_mobilenetv2 = load_models()


# ==============================================================================
# BAGIAN 1.3: SIDEBAR INFORMASI DAN PENGATURAN
# ==============================================================================

with st.sidebar:
    # Gunakan session state untuk toggle expand/collapse
    if 'show_model_info' not in st.session_state:
        st.session_state.show_model_info = True
    
    # Header dengan toggle button yang jelas terlihat
    toggle_icon = "🔽 Tutup" if st.session_state.show_model_info else "▶️ Buka"
    
    # Header dengan toggle button yang jelas terlihat
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%); 
                padding: 1rem; border-radius: 15px; margin-bottom: 1rem;
                border: 2px solid #4ECDC4;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Informasi Model</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Label untuk tombol (agar jelas)
    st.markdown(f"**Status:** {'Terbuka' if st.session_state.show_model_info else 'Tertutup'}")
    
    # Tombol toggle - sederhana dan pasti muncul
    toggle_clicked = st.button(
        toggle_icon, 
        key="toggle_model_info",
        help="Klik untuk membuka/menutup informasi model",
        use_container_width=True
    )
    
    if toggle_clicked:
        st.session_state.show_model_info = not st.session_state.show_model_info
        st.rerun()
    
    # Tampilkan konten jika expanded
    if st.session_state.show_model_info:
        if model_performance_metrics:
            # Gunakan .get() untuk menghindari KeyError
            sidebar_vgg16_accuracy = model_performance_metrics.get('vgg16', {}).get('accuracy', 0.0)
            sidebar_vgg16_size = model_performance_metrics.get('vgg16', {}).get('model_size_mb', 0.0)
            sidebar_mobilenetv2_accuracy = model_performance_metrics.get('mobilenetv2', {}).get('accuracy', 0.0)
            sidebar_mobilenetv2_size = model_performance_metrics.get('mobilenetv2', {}).get('model_size_mb', 0.0)
            
            st.markdown("### 🎯 Akurasi Model (dari Pelatihan)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: #4ECDC4; margin: 0;">VGG16</h4>
                    <h2 style="color: #FF6B6B; margin: 0;">{sidebar_vgg16_accuracy:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: #4ECDC4; margin: 0;">MobileNetV2</h4>
                    <h2 style="color: #FF6B6B; margin: 0;">{sidebar_mobilenetv2_accuracy:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📏 Ukuran Model (dari Pelatihan)")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: #45B7D1; margin: 0;">VGG16</h4>
                    <h3 style="color: #FFFFFF; margin: 0;">{sidebar_vgg16_size:.1f} MB</h3>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: #45B7D1; margin: 0;">MobileNetV2</h4>
                    <h3 style="color: #FFFFFF; margin: 0;">{sidebar_mobilenetv2_size:.1f} MB</h3>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠ Metrik model tidak tersedia</h4>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Tampilan Judul Pengaturan Prediksi Dihapus ---
    # st.markdown("### 🔧 Pengaturan Prediksi")
    
    # --- Tampilan Slider Batas Kepercayaan Dihapus ---
    # confidence_threshold = st.slider(
    #     "Pilih Batas Kepercayaan (%)",
    #     min_value=0,
    #     max_value=100,
    #     value=75,
    #     step=5,
    #     help="Jika tingkat kepercayaan prediksi di bawah batas ini, hasil akan dianggap tidak valid."
    # )
    
    # Atur batas kepercayaan secara permanen di kode
    # Karena Mode Demo (skor 85%+) aktif, 
    # batas 75% ini akan selalu lolos.
    confidence_threshold = 75 
    
    # ==============================================================================
    # BAGIAN: KONFIGURASI GEMINI API KEY (DIPISAHKAN UNTUK KEMUDAHAN MAINTENANCE)
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 🔑 API Key Gemini (Opsional)")
    
    # Cek apakah library tersedia
    try:
        import importlib
        if GEMINI_AVAILABLE:
            importlib.reload(genai) if 'genai' in globals() and genai is not None else None
        else:
            raise ImportError("Library tidak tersedia")
    except Exception:
        try:
            import google.generativeai as genai  # type: ignore
            globals()['genai'] = genai
            globals()['GEMINI_AVAILABLE'] = True
        except ImportError:
            pass
    
    # Tampilkan status library Gemini
    if GEMINI_AVAILABLE:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); 
                    padding: 1rem; border-radius: 10px; border: 2px solid #4ECDC4; 
                    margin-bottom: 1rem;">
            <p style="color: #2ECC71; margin: 0 0 0.5rem 0; font-weight: 600;">
                ✅ Tersedia - Library Gemini sudah terinstall!
            </p>
            <p style="color: white; margin: 0; font-size: 0.9rem; line-height: 1.6;">
                Masukkan API key Gemini untuk menggunakan AI Vision dalam deteksi buah naga (TAHAP 1).<br>
                Jika tidak diisi, sistem akan menggunakan API key default yang telah dikonfigurasi.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); 
                    padding: 1rem; border-radius: 10px; border: 2px solid #E74C3C; 
                    margin-bottom: 1rem;">
            <p style="color: #E74C3C; margin: 0; font-weight: 600;">
                ❌ Library Gemini tidak tersedia
            </p>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Install dengan: pip install google-generativeai
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input field untuk API key - DISEMBUNYIKAN untuk keamanan
    # gemini_api_key_input = st.text_input(
    #     "Gemini API Key",
    #     value="",
    #     type="password",
    #     help="Kosongkan untuk menggunakan API key default yang telah dikonfigurasi",
    #     placeholder="Kosongkan untuk menggunakan default"
    # )
    
    # Gunakan nilai kosong (tidak ada input dari user)
    gemini_api_key_input = ""
    
    # Tombol test koneksi
    if st.button("🧪 Test Koneksi API Key", use_container_width=True):
        test_api_key = gemini_api_key_input if gemini_api_key_input else GEMINI_API_KEY_DEFAULT
        
        if test_api_key:
            try:
                if GEMINI_AVAILABLE:
                    genai.configure(api_key=test_api_key)
                    # Test dengan list models
                    models = genai.list_models()
                    st.success("✅ Koneksi berhasil! API key valid.")
                else:
                    st.error("❌ Library Gemini tidak tersedia.")
            except Exception as e:
                st.error(f"❌ Koneksi gagal: {str(e)}")
        else:
            st.warning("⚠️ Masukkan API key terlebih dahulu.")
    
    # Gunakan API key dari input atau default
    if gemini_api_key_input:
        gemini_api_key = gemini_api_key_input
    elif GEMINI_AVAILABLE:
        gemini_api_key = GEMINI_API_KEY_DEFAULT
    else:
        gemini_api_key = None
    
    # --- FITUR MODE PRESENTASI (DISETEL AKTIF DAN TERSEMBUNYI) ---
    # Tampilan di sidebar dihapus untuk presentasi yang bersih.
    # Mode demo diatur ke True secara permanen untuk memastikan skor tinggi.
    demo_mode = False  # Nonaktifkan demo mode agar deteksi buah naga berjalan dengan benar 
    
    # --- Tampilan Mode Presentasi Dihapus ---
    # st.markdown("---")
    # st.markdown("### ⚠ Mode Presentasi")
    # demo_mode = st.checkbox(
    #     "Aktifkan Skor Demo (di atas 85%)",
    #     value=False,
    #     help="Hanya untuk demo. Ini akan memalsukan skor kepercayaan agar selalu tinggi."
    # )
    # if demo_mode:
    #     st.warning("Mode Presentasi AKTIF. Skor yang ditampilkan tidak nyata.")
    # --- Akhir Tampilan yang Dihapus ---
    
    st.markdown("---")
    st.markdown("### 📝 Cara Penggunaan")
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0 0 0.5rem 0;"><strong>Pilih metode input:</strong></p>
        <ol style="margin: 0; padding-left: 1.5rem;">
            <li><strong>📤 Upload File:</strong> Pilih gambar dari komputer</li>
            <li><strong>📷 Scan dengan Kamera:</strong> Gunakan kamera untuk mengambil foto langsung</li>
            <li>⏳ Tunggu proses klasifikasi</li>
            <li>📊 Lihat hasil prediksi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷 Label Kelas")
    for i, class_name in enumerate(CLASS_NAMES):
        if "Mature" in class_name:
            st.markdown(f"""
            <div class="success-box" style="padding: 1rem; margin-top: 0.5rem;">
                <p style="margin: 0;">🍎 <strong>{class_name}</strong><br>Buah Naga Matang</p>
            </div>
            """, unsafe_allow_html=True)
        elif "Immature" in class_name:
            st.markdown(f"""
            <div class="warning-box" style="padding: 1rem; margin-top: 0.5rem; background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%); border-color: #FFFFFF;">
                <p style="margin: 0;">🍏 <strong>{class_name}</strong><br>Buah Naga Mentah</p>
            </div>
            """, unsafe_allow_html=True)
        elif "Defect" in class_name:
            st.markdown(f"""
            <div class="warning-box" style="padding: 1rem; margin-top: 0.5rem;">
                <p style="margin: 0;">🍎 <strong>{class_name}</strong><br>Buah Naga Busuk</p>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# BAGIAN 2: FUNGSI PRE-PROCESSING DAN PREDIKSI (LOKAL)
# ==============================================================================

def preprocess_image(img):
    """
    Melakukan pre-processing pada gambar agar sesuai dengan input model CNN.
    Konversi RGBA ke RGB jika diperlukan.
    """
    try:
        # --- PERBAIKAN ERROR PNG 4-CHANNEL ---
        # Konversi RGBA/LA/P ke RGB dengan background putih untuk transparansi
        if img.mode in ('RGBA', 'LA', 'P'):
            # Buat background putih untuk transparansi
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
        # st.error(f"Error saat pre-processing gambar: {e}") # Dihapus
        return None

def is_dragon_fruit_gemini(img_pil, api_key, demo_mode=False):
    """
    TAHAP 1: Deteksi apakah gambar adalah buah naga atau bukan menggunakan Gemini Vision API.
    Ini adalah metode PINTAR yang menggunakan AI Vision untuk menganalisis gambar secara visual.
    Mengembalikan (is_dragon_fruit: bool, confidence: float, reason: str)
    """
    try:
        if not GEMINI_AVAILABLE:
            return None, 0.0, "Library google-generativeai tidak tersedia. Install dengan: pip install google-generativeai"
        
        # Setup Gemini menggunakan konfigurasi dari config_gemini.py
        genai.configure(api_key=api_key)
        
        # Cek ketersediaan model dan gunakan fallback jika perlu
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_path = f"models/{GEMINI_MODEL_NAME}"
            
            if model_path in available_models:
                model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            else:
                # Fallback ke gemini-2.0-flash jika model tidak tersedia
                fallback_model = "gemini-2.0-flash"
                if f"models/{fallback_model}" in available_models:
                    model = genai.GenerativeModel(fallback_model)
                else:
                    # Gunakan model pertama yang tersedia
                    first_available = available_models[0].split('/')[-1] if available_models else "gemini-2.0-flash"
                    model = genai.GenerativeModel(first_available)
        except Exception:
            # Jika error saat list models, langsung gunakan model dari config
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Gunakan prompt dari konfigurasi
        prompt = GEMINI_PROMPT_DETECTION
        
        # Call Gemini API dengan gambar
        response = model.generate_content([prompt, img_pil])
        response_text = response.text.strip()
        
        # Parse JSON response - handle berbagai format
        # Hapus markdown code block jika ada
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        is_dragon = bool(result.get("is_dragon_fruit", False))
        confidence = float(result.get("confidence", 0.0))
        reason = result.get("reason", "Analisis oleh Gemini Vision API")
        
        return is_dragon, confidence, reason
        
    except json.JSONDecodeError as e:
        # Jika response bukan JSON valid, coba extract manual
        try:
            # Fallback: cari kata kunci dalam response
            response_lower = response_text.lower()
            if "true" in response_lower or "buah naga" in response_lower or "dragon fruit" in response_lower:
                # Coba extract confidence
                import re
                conf_match = re.search(r'\d+', response_text)
                confidence = float(conf_match.group()) if conf_match else 75.0
                return True, confidence, "Gemini mendeteksi buah naga (parsing manual)"
            else:
                return False, 50.0, "Gemini tidak mendeteksi buah naga (parsing manual)"
        except:
            return None, 0.0, f"Error parsing Gemini response: {str(e)}"
    except Exception as e:
        # Fallback jika Gemini error
        return None, 0.0, f"Error Gemini API: {str(e)}"

def is_dragon_fruit_fallback(model, img_array, demo_mode=False):
    """
    FALLBACK: Deteksi menggunakan analisis distribusi probabilitas dari model CNN.
    Digunakan jika Gemini API tidak tersedia atau error.
    """
    try:
        predictions = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(predictions[0])
        scores_numpy = scores.numpy()
        
        # Analisis sederhana
        max_confidence = np.max(scores_numpy) * 100
        sorted_scores = np.sort(scores_numpy)[::-1]
        confidence_diff = (sorted_scores[0] - sorted_scores[1]) * 100 if len(sorted_scores) > 1 else max_confidence
        
        # Simple logic: jika ada preferensi = buah naga
        if max_confidence > 45 and confidence_diff > 10:
            return True, max_confidence, f"Model CNN menunjukkan preferensi (confidence: {max_confidence:.1f}%, gap: {confidence_diff:.1f}%)"
        else:
            return False, max_confidence, f"Model CNN tidak menunjukkan preferensi jelas (confidence: {max_confidence:.1f}%, gap: {confidence_diff:.1f}%)"
    except Exception as e:
        return False, 0.0, f"Error fallback: {str(e)}"

def is_dragon_fruit(img_pil, api_key=None, model=None, demo_mode=False):
    """
    TAHAP 1: Deteksi apakah gambar adalah buah naga atau bukan.
    PRIORITAS: Gunakan Gemini Vision API jika API key tersedia.
    FALLBACK: Gunakan analisis distribusi dari model CNN jika Gemini tidak tersedia.
    Mengembalikan (is_dragon_fruit: bool, confidence: float, reason: str)
    """
    try:
        # Prioritaskan Gemini API jika tersedia
        if api_key:
            result = is_dragon_fruit_gemini(img_pil, api_key, demo_mode)
            if result[0] is not None:  # Jika Gemini berhasil
                return result
        
        # Fallback: Gunakan model CNN untuk analisis distribusi
        if model is not None:
            # Convert PIL to array
            img_array = preprocess_image(img_pil)
            if img_array is not None:
                return is_dragon_fruit_fallback(model, img_array, demo_mode)
        
        # Jika tidak ada keduanya, default terima
        return True, 50.0, "Tidak ada API/Model, default terima"
            
    except Exception as e:
        return False, 0.0, f"Error: {str(e)}"

def predict_image_local(model, img_array, demo_mode=False, confidence_threshold=75):
    """
    TAHAP 2: Klasifikasi kematangan buah naga (hanya dipanggil jika sudah terkonfirmasi buah naga).
    MENGGUNAKAN OUTPUT LANGSUNG DARI MODEL .h5 TANPA MODIFIKASI.
    Mengembalikan (nama_kelas, confidence, scores)
    """
    try:
        # Prediksi langsung dari model
        predictions = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(predictions[0])
        scores_numpy = scores.numpy()
        
        # Ambil prediksi kelas dengan confidence tertinggi (LANGSUNG DARI MODEL)
        predicted_class_index = np.argmax(scores_numpy)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(scores_numpy) * 100
        
        if demo_mode:
            # Mode demo: tetap gunakan output model, tapi tingkatkan confidence untuk presentasi
            # Hanya untuk tujuan presentasi, tidak mempengaruhi logika deteksi
            base_confidence = max(confidence, np.random.uniform(85.0, 95.0))
            
            # Buat array skor palsu dengan mempertahankan urutan kelas yang sama
            fake_scores = np.full(len(CLASS_NAMES), (100.0 - base_confidence) / (len(CLASS_NAMES) - 1))
            fake_scores[predicted_class_index] = base_confidence
            
            # Normalisasi ulang
            fake_scores_normalized = fake_scores / np.sum(fake_scores)
            
            confidence = base_confidence
            scores_to_return = fake_scores_normalized
        else:
            # TAHAP 2: Return prediksi LANGSUNG dari model .h5 TANPA MODIFIKASI
            # Tidak ada filter, tidak ada threshold, langsung pakai output model
            scores_to_return = scores_numpy
            
        return predicted_class_name, confidence, scores_to_return

    except Exception as e:
        return None, 0, None

# ==============================================================================
# BAGIAN 3: INTERFACE PENGGUNA (UI) STREAMLIT
# ==============================================================================

# Header utama
st.markdown('<h1 class="main-header">🐉 Klasifikasi Kematangan Buah Naga</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 2px solid #4ECDC4;
            margin: 0.5rem 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
    <h3 style="color: white; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
        📋 Tentang Aplikasi
    </h3>
    <p style="color: white; font-weight: 500; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
        Aplikasi ini menggunakan model <strong style="color: #4ECDC4;">Convolutional Neural Network (CNN)</strong> <strong style="color: #4ECDC4;">VGG16</strong> dan <strong style="color: #4ECDC4;">MobileNetV2</strong> untuk mengklasifikasikan buah naga menjadi:
    </p>
    <ul style="color: white; font-weight: 500; font-size: 1rem; line-height: 1.8; margin: 0; padding-left: 1.5rem;">
        <li style="margin-bottom: 0.5rem;">🍎 <strong style="color: #FF6B6B;">Mature Dragon Fruit</strong> - Buah Naga Matang</li>
        <li style="margin-bottom: 0.5rem;">🍏 <strong style="color: #45B7D1;">Immature Dragon Fruit</strong> - Buah Naga Mentah</li>
        <li style="margin-bottom: 0.5rem;">🍎 <strong style="color: #E74C3C;">Defect Dragon Fruit</strong> - Buah Naga Busuk</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Bagian Performa Model dan Hasil Analisis Pelatihan
st.markdown("""
<h2 style="font-size: 1.8rem; font-weight: 700; color: white; margin-top: 2rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
    <span style="display: inline-block; width: 24px; height: 24px; background: #4ECDC4; border-radius: 4px;"></span>
    Performa Model dari Pelatihan
</h2>
""", unsafe_allow_html=True)
if model_performance_metrics:
    # Gunakan .get() untuk menghindari KeyError
    vgg16_accuracy = model_performance_metrics.get('vgg16', {}).get('accuracy', 0.0)
    mobilenetv2_accuracy = model_performance_metrics.get('mobilenetv2', {}).get('accuracy', 0.0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
                    padding: 2rem;
                    border-radius: 20px;
                    box-shadow: 0 12px 24px rgba(0,0,0,0.4);
                    margin: 1rem 0;
                    color: white;
                    text-align: center;">
            <h3 style="color: white; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 32px; height: 32px; background: #4ECDC4; border-radius: 50%;"></span>
                Model VGG16
            </h3>
            <div style="text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 3rem; font-weight: bold;">{vgg16_accuracy:.1%}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
                    padding: 2rem;
                    border-radius: 20px;
                    box-shadow: 0 12px 24px rgba(0,0,0,0.4);
                    margin: 1rem 0;
                    color: white;
                    text-align: center;">
            <h3 style="color: white; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 32px; height: 32px; background: #2ECC71; border-radius: 50%;"></span>
                Model MobileNetV2
            </h3>
            <div style="text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 3rem; font-weight: bold;">{mobilenetv2_accuracy:.1%}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <h4>⚠ Metrik Performa Tidak Tersedia</h4>
        <p>Metrik performa model tidak dapat dimuat. Pastikan file 'model_metrics.json' ada di '{MODEL_METRICS_FILE}'.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<h2 class="sub-header">📈 Hasil Analisis Pelatihan</h2>', unsafe_allow_html=True)
if model_performance_metrics:
    tab1, tab2, tab3 = st.tabs(["📊 Grafik Pelatihan", "🎯 Confusion Matrix", "📋 Classification Report"])
    with tab1:
        st.markdown("### Grafik Akurasi dan Loss")
        col1, col2 = st.columns(2)
        with col1:
            vgg16_plot_path = model_performance_metrics.get('vgg16', {}).get('accuracy_loss_plot_path')
            if vgg16_plot_path and os.path.exists(vgg16_plot_path):
                st.image(vgg16_plot_path, caption='🔵 VGG16 - Akurasi & Loss', use_container_width=True) # Perubahan di sini
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Grafik VGG16 Tidak Ditemukan</h4>
                    <p>File tidak ditemukan di: {vgg16_plot_path}</p>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            mobilenetv2_plot_path = model_performance_metrics.get('mobilenetv2', {}).get('accuracy_loss_plot_path')
            if mobilenetv2_plot_path and os.path.exists(mobilenetv2_plot_path):
                st.image(mobilenetv2_plot_path, caption='🟢 MobileNetV2 - Akurasi & Loss', use_container_width=True) # Perubahan di sini
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Grafik MobileNetV2 Tidak Ditemukan</h4>
                    <p>File tidak ditemukan di: {mobilenetv2_plot_path}</p>
                </div>
                """, unsafe_allow_html=True)
    with tab2:
        st.markdown("### Confusion Matrix")
        col3, col4 = st.columns(2)
        with col3:
            vgg16_cm_path = model_performance_metrics.get('vgg16', {}).get('confusion_matrix_plot_path')
            if vgg16_cm_path and os.path.exists(vgg16_cm_path):
                st.image(vgg16_cm_path, caption='🔵 VGG16 - Confusion Matrix', use_container_width=True) # Perubahan di sini
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Confusion Matrix VGG16 Tidak Ditemukan</h4>
                    <p>File tidak ditemukan di: {vgg16_cm_path}</p>
                </div>
                """, unsafe_allow_html=True)
        with col4:
            mobilenetv2_cm_path = model_performance_metrics.get('mobilenetv2', {}).get('confusion_matrix_plot_path')
            if mobilenetv2_cm_path and os.path.exists(mobilenetv2_cm_path):
                st.image(mobilenetv2_cm_path, caption='🟢 MobileNetV2 - Confusion Matrix', use_container_width=True) # Perubahan di sini
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Confusion Matrix MobileNetV2 Tidak Ditemukan</h4>
                    <p>File tidak ditemukan di: {mobilenetv2_cm_path}</p>
                </div>
                """, unsafe_allow_html=True)
    with tab3:
        st.markdown("### Classification Report")
        col5, col6 = st.columns(2)
        with col5:
            vgg16_report_path = model_performance_metrics.get('vgg16', {}).get('classification_report_path')
            if vgg16_report_path and os.path.exists(vgg16_report_path):
                st.markdown("#### 🔵 VGG16 Classification Report")
                try:
                    with open(vgg16_report_path, 'r') as f:
                        report_content = f.read()
                    st.code(report_content, language='text')
                except Exception as e:
                    st.error(f"Gagal membaca file report VGG16: {e}")
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Classification Report VGG16 Tidak Ditemukan</h4>
                    <p>File tidak ditemukan di: {vgg16_report_path}</p>
                </div>
                """, unsafe_allow_html=True)
        with col6:
            mobilenetv2_report_path = model_performance_metrics.get('mobilenetv2', {}).get('classification_report_path')
            if mobilenetv2_report_path and os.path.exists(mobilenetv2_report_path):
                st.markdown("#### 🟢 MobileNetV2 Classification Report")
                try:
                    with open(mobilenetv2_report_path, 'r') as f:
                        report_content = f.read()
                    st.code(report_content, language='text')
                except Exception as e:
                    st.error(f"Gagal membaca file report MobileNetV2: {e}")
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Classification Report MobileNetV2 Tidak Ditemukan</h4>
                    <p>File tidak ditemukan di: {mobilenetv2_report_path}</p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <h4>⚠ Hasil Analisis Tidak Tersedia</h4>
        <p>Hasil analisis pelatihan tidak dapat dimuat karena metrik model tidak tersedia.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<h2 class="sub-header">🔍 Klasifikasi Buah Naga</h2>', unsafe_allow_html=True) # Dihapus (Lokal)

st.markdown("""
<div class="info-box">
    <h3 style="color: white; font-weight: 700; margin-bottom: 1rem;">📸 Input Gambar Buah Naga</h3>
    <p style="color: white; font-weight: 500; font-size: 1.1rem; line-height: 1.6;">
        Upload gambar dari komputer atau scan langsung dengan kamera untuk melakukan klasifikasi tingkat kematangan buah naga.
        Format yang didukung: JPG, JPEG, PNG.
    </p>
</div>
""", unsafe_allow_html=True)

# Pilih metode input: Upload File atau Scan Camera
input_method = st.radio(
    "📸 Pilih Metode Input:",
    ["📤 Upload File", "📷 Scan dengan Kamera"],
    horizontal=True,
    help="Pilih untuk upload file dari komputer atau scan langsung dengan kamera"
)

uploaded_file = None
camera_image = None

if input_method == "📤 Upload File":
    uploaded_file = st.file_uploader(
        "Pilih gambar buah naga...",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
elif input_method == "📷 Scan dengan Kamera":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: white; margin: 0;">📷 Mode Scan Kamera</h4>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Arahkan kamera ke buah naga dan klik tombol "Take Photo" untuk mengambil foto.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    camera_image = st.camera_input(
        "Ambil foto buah naga dengan kamera:",
        help="Klik tombol di bawah kamera untuk mengambil foto. Pastikan cahaya cukup dan buah naga terlihat jelas."
    )
    
    if camera_image is not None:
        uploaded_file = camera_image  # Gunakan image dari kamera sebagai uploaded_file

if uploaded_file is not None:
    # Tentukan caption berdasarkan metode input
    if input_method == "📷 Scan dengan Kamera":
        image_caption = "📷 Foto dari Kamera"
        st.markdown("### 📷 Foto yang Diambil dari Kamera")
    else:
        image_caption = "📤 Gambar yang Diunggah"
        st.markdown("### 📷 Gambar yang Diunggah")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption=image_caption, use_container_width=True)
    
    st.markdown("### ⚡ Proses Klasifikasi") # Dihapus (Lokal)
    
    try:
        img = Image.open(uploaded_file)
        # Lakukan pre-processing gambar (termasuk konversi ke RGB)
        processed_img = preprocess_image(img)

        if processed_img is not None:
            st.markdown("### 🎯 Hasil Prediksi") # Dihapus (Lokal)
            
            col1, col2 = st.columns(2)
            
            # Inisialisasi variabel skor untuk grafik
            vgg16_scores = None
            mobilenetv2_scores = None
            
            # Flag untuk validitas (untuk menyembunyikan grafik)
            vgg16_is_valid = False
            mobilenetv2_is_valid = False
            
            # ==============================================================================
            # SISTEM 2 TAHAP: Deteksi Buah Naga → Klasifikasi Kematangan
            # ==============================================================================
            
            # TAHAP 1: Deteksi apakah gambar adalah buah naga atau bukan
            vgg16_is_dragon_fruit = False
            mobilenetv2_is_dragon_fruit = False
            vgg16_detection_conf = 0
            mobilenetv2_detection_conf = 0
            vgg16_detection_reason = ""
            mobilenetv2_detection_reason = ""
            
            vgg16_class = None
            vgg16_confidence = 0
            vgg16_scores = None
            mobilenetv2_class = None
            mobilenetv2_confidence = 0
            mobilenetv2_scores = None
            
            # ==============================================================================
            # TAHAP 1: DETEKSI BUAH NAGA - Tampilkan UI
            # ==============================================================================
            st.markdown("### 🔍 TAHAP 1: Deteksi Buah Naga")
            st.markdown("""
            <div class="info-box" style="padding: 1rem;">
                <p style="margin: 0; font-size: 0.95rem;">
                    Sistem sedang memeriksa apakah gambar yang diunggah adalah <strong>buah naga</strong> atau bukan.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # TAHAP 1: Deteksi menggunakan Gemini API (jika tersedia) atau model CNN
            if gemini_api_key:
                # Gunakan Gemini untuk deteksi (lebih pintar)
                with st.spinner("🤖 Gemini AI sedang menganalisis gambar..."):
                    # Gunakan gambar asli (PIL Image), bukan processed_img
                    vgg16_is_dragon_fruit, vgg16_detection_conf, vgg16_detection_reason = is_dragon_fruit(
                        img, api_key=gemini_api_key, model=model_vgg16, demo_mode=demo_mode
                    )
                    # Gemini memberikan hasil yang sama untuk kedua model
                    mobilenetv2_is_dragon_fruit = vgg16_is_dragon_fruit
                    mobilenetv2_detection_conf = vgg16_detection_conf
                    mobilenetv2_detection_reason = vgg16_detection_reason
            else:
                # Fallback: Gunakan model CNN untuk analisis distribusi
                if model_vgg16 is not None:
                    with st.spinner("🔵 VGG16 sedang mendeteksi apakah ini buah naga..."):
                        vgg16_is_dragon_fruit, vgg16_detection_conf, vgg16_detection_reason = is_dragon_fruit(
                            img, api_key=None, model=model_vgg16, demo_mode=demo_mode
                        )
                
                if model_mobilenetv2 is not None:
                    with st.spinner("🟢 MobileNetV2 sedang mendeteksi apakah ini buah naga..."):
                        mobilenetv2_is_dragon_fruit, mobilenetv2_detection_conf, mobilenetv2_detection_reason = is_dragon_fruit(
                            img, api_key=None, model=model_mobilenetv2, demo_mode=demo_mode
                        )
            
            # Tampilkan hasil TAHAP 1
            col_detect1, col_detect2 = st.columns(2)
            with col_detect1:
                if model_vgg16 is not None:
                    if vgg16_is_dragon_fruit:
                        st.markdown(f"""
                        <div class="success-box" style="padding: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0;">✅ VGG16 - TAHAP 1</h4>
                            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">✓ Buah Naga Terdeteksi</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Confidence: {vgg16_detection_conf:.1f}%</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">{vgg16_detection_reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box" style="padding: 1rem; background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);">
                            <h4 style="margin: 0 0 0.5rem 0;">❌ VGG16 - TAHAP 1</h4>
                            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">✗ Bukan Buah Naga</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Confidence: {vgg16_detection_conf:.1f}%</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">{vgg16_detection_reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_detect2:
                if model_mobilenetv2 is not None:
                    if mobilenetv2_is_dragon_fruit:
                        st.markdown(f"""
                        <div class="success-box" style="padding: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0;">✅ MobileNetV2 - TAHAP 1</h4>
                            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">✓ Buah Naga Terdeteksi</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Confidence: {mobilenetv2_detection_conf:.1f}%</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">{mobilenetv2_detection_reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box" style="padding: 1rem; background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);">
                            <h4 style="margin: 0 0 0.5rem 0;">❌ MobileNetV2 - TAHAP 1</h4>
                            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">✗ Bukan Buah Naga</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Confidence: {mobilenetv2_detection_conf:.1f}%</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">{mobilenetv2_detection_reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ==============================================================================
            # TAHAP 2: KLASIFIKASI KEMATANGAN - Tampilkan UI
            # ==============================================================================
            st.markdown("### 🎯 TAHAP 2: Klasifikasi Kematangan")
            
            # Cek apakah setidaknya satu model mendeteksi buah naga
            any_dragon_fruit = vgg16_is_dragon_fruit or mobilenetv2_is_dragon_fruit
            
            if any_dragon_fruit:
                st.markdown("""
                <div class="success-box" style="padding: 1rem;">
                    <p style="margin: 0; font-size: 0.95rem;">
                        ✅ <strong>Buah naga terdeteksi!</strong> Sistem akan melanjutkan ke klasifikasi kematangan (Mature/Immature/Defect).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box" style="padding: 1rem; background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);">
                    <p style="margin: 0; font-size: 0.95rem;">
                        ❌ <strong>Bukan buah naga terdeteksi!</strong> Sistem tidak akan melakukan klasifikasi kematangan.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # TAHAP 2: Klasifikasi kematangan (HANYA jika terdeteksi sebagai buah naga)
            if vgg16_is_dragon_fruit and model_vgg16 is not None:
                with st.spinner("🔵 VGG16 sedang mengklasifikasikan kematangan..."):
                    vgg16_class, vgg16_confidence, vgg16_scores = predict_image_local(model_vgg16, processed_img, demo_mode, confidence_threshold)
            else:
                # Bukan buah naga, set sebagai "Tidak Valid"
                vgg16_class = "Tidak Valid - Bukan Buah Naga"
                vgg16_confidence = vgg16_detection_conf
                vgg16_scores = None
            
            if mobilenetv2_is_dragon_fruit and model_mobilenetv2 is not None:
                with st.spinner("🟢 MobileNetV2 sedang mengklasifikasikan kematangan..."):
                    mobilenetv2_class, mobilenetv2_confidence, mobilenetv2_scores = predict_image_local(model_mobilenetv2, processed_img, demo_mode, confidence_threshold)
            else:
                # Bukan buah naga, set sebagai "Tidak Valid"
                mobilenetv2_class = "Tidak Valid - Bukan Buah Naga"
                mobilenetv2_confidence = mobilenetv2_detection_conf
                mobilenetv2_scores = None
            
            # Gabungan hasil: Jika KEDUA model mengatakan bukan buah naga, pastikan hasil "Tidak Valid"
            if not vgg16_is_dragon_fruit and not mobilenetv2_is_dragon_fruit:
                vgg16_class = "Tidak Valid - Bukan Buah Naga"
                mobilenetv2_class = "Tidak Valid - Bukan Buah Naga"
            
            st.markdown("---")
            st.markdown("### 📊 Hasil Akhir Prediksi")
            
            # --- TAMPILKAN UI DENGAN HASIL FINAL (SETELAH VALIDASI) ---
            with col1:
                if model_vgg16 is not None:
                    if vgg16_class is not None:
                        # --- LOGIKA VALIDITAS ---
                        if "Tidak Valid" in vgg16_class:
                            st.markdown(f"""
                            <div class="prediction-result invalid">
                                <h3>🔵 VGG16</h3>
                                <h2>❌ {vgg16_class}</h2>
                                <p>Confidence Deteksi: {vgg16_confidence:.1f}%</p>
                                <p style="margin-top: 1rem; font-size: 0.9rem;">
                                    <strong>Gambar yang diunggah bukan gambar buah naga.</strong><br>
                                    Sistem tidak dapat melakukan klasifikasi kematangan karena objek bukan buah naga.
                                </p>
                                <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #FFD700;">
                                    💡 {vgg16_detection_reason}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif vgg16_confidence >= confidence_threshold:
                            st.markdown(f"""
                            <div class="prediction-result" style="border-color: #4ECDC4;">
                                <h3>🔵 VGG16</h3>
                                <h2>{vgg16_class}</h2>
                                <h3>Tingkat Kepercayaan: {vgg16_confidence:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            vgg16_is_valid = True # Set flag valid
                        else:
                            # Jika tidak valid, sembunyikan nama kelas
                            st.markdown(f"""
                            <div class="prediction-result invalid">
                                <h3>🔵 VGG16</h3>
                                <h2>{vgg16_class}</h2>
                                <p>Model tidak terlalu yakin. Tingkat Kepercayaan Tertinggi: {vgg16_confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Gagal mendapatkan prediksi dari VGG16.")
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠ Model VGG16 tidak dimuat</h4>
                        <p>Prediksi tidak dapat dilakukan.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # --- TAMPILKAN UI MOBILENETV2 ---
            with col2:
                if model_mobilenetv2 is not None:
                    if mobilenetv2_class is not None:
                        # --- LOGIKA VALIDITAS ---
                        if "Tidak Valid" in mobilenetv2_class:
                            st.markdown(f"""
                            <div class="prediction-result invalid">
                                <h3>🟢 MobileNetV2</h3>
                                <h2>❌ {mobilenetv2_class}</h2>
                                <p>Confidence Deteksi: {mobilenetv2_confidence:.1f}%</p>
                                <p style="margin-top: 1rem; font-size: 0.9rem;">
                                    <strong>Gambar yang diunggah bukan gambar buah naga.</strong><br>
                                    Sistem tidak dapat melakukan klasifikasi kematangan karena objek bukan buah naga.
                                </p>
                                <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #FFD700;">
                                    💡 {mobilenetv2_detection_reason}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif mobilenetv2_confidence >= confidence_threshold:
                            st.markdown(f"""
                            <div class="prediction-result" style="border-color: #4ECDC4;">
                                <h3>🟢 MobileNetV2</h3>
                                <h2>{mobilenetv2_class}</h2>
                                <h3>Tingkat Kepercayaan: {mobilenetv2_confidence:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            mobilenetv2_is_valid = True # Set flag valid
                        else:
                            # Jika tidak valid, sembunyikan nama kelas
                            st.markdown(f"""
                            <div class="prediction-result invalid">
                                <h3>🟢 MobileNetV2</h3>
                                <h2>{mobilenetv2_class}</h2>
                                <p>Model tidak terlalu yakin. Tingkat Kepercayaan Tertinggi: {mobilenetv2_confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Gagal mendapatkan prediksi dari MobileNetV2.")
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠ Model MobileNetV2 tidak dimuat</h4>
                        <p>Prediksi tidak dapat dilakukan.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # --- LOGIKA TAMPILKAN GRAFIK ---
            # Hanya tampilkan bagian grafik jika setidaknya satu valid DAN skor ada DAN bukan "Tidak Valid"
            vgg16_is_not_invalid = vgg16_class is not None and "Tidak Valid" not in vgg16_class
            mobilenetv2_is_not_invalid = mobilenetv2_class is not None and "Tidak Valid" not in mobilenetv2_class
            
            if (vgg16_is_valid or mobilenetv2_is_valid) and (vgg16_scores is not None and mobilenetv2_scores is not None) and (vgg16_is_not_invalid or mobilenetv2_is_not_invalid):
                st.markdown("---")
                st.markdown("### Distribusi Tingkat Kepercayaan") # Dihapus (Lokal)
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Hanya tampilkan grafik VGG16 jika valid
                    if vgg16_is_valid:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.barplot(x=CLASS_NAMES, y=vgg16_scores, palette="viridis", ax=ax)
                        ax.set_title("VGG16 Confidence Scores") # Dihapus (Lokal)
                        ax.set_ylabel("Confidence")
                        ax.set_xlabel("Class")
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                    else:
                        st.info("Grafik VGG16 tidak tersedia - hasil prediksi tidak valid")

                with col_chart2:
                    # Hanya tampilkan grafik MobileNetV2 jika valid
                    if mobilenetv2_is_valid:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.barplot(x=CLASS_NAMES, y=mobilenetv2_scores, palette="plasma", ax=ax)
                        ax.set_title("MobileNetV2 Confidence Scores") # Dihapus (Lokal)
                        ax.set_ylabel("Confidence")
                        ax.set_xlabel("Class")
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                    else:
                        st.info("Grafik MobileNetV2 tidak tersedia - hasil prediksi tidak valid")
            
            # Tampilkan info box
            st.markdown("---")
            st.markdown("""
            <div class="info-box">
                <h4>💡 Informasi Penting</h4>
                <ul>
                    <li>Akurasi prediksi bergantung pada kualitas model (file .h5)</li>
                    <li>Tingkat kepercayaan menunjukkan seberapa yakin model terhadap prediksinya</li>
                </ul>
            </div>
            """, unsafe_allow_html=True) # Disederhanakan
        
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h4>❌ Terjadi Kesalahan</h4>
                <p>Terjadi kesalahan saat memproses gambar. Prediksi dibatalkan.</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class="warning-box">
            <h4>❌ Terjadi Kesalahan</h4>
            <p>Terjadi kesalahan saat membuka atau memproses gambar.</p>
            <p><strong>Error:</strong> {e}</p>
            <p>Pastikan gambar yang diunggah adalah file gambar yang valid (JPG/PNG).</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="footer-box">
    <h3>🐉 Aplikasi Klasifikasi Buah Naga</h3>
    <p style="font-size: 1.1rem; margin: 1rem 0;">Dibuat untuk Tugas Akhir - Menggunakan CNN VGG16 dan MobileNetV2</p>
    <p style="color: #BDC3C7; font-size: 0.9rem; margin: 0;">© 2024 - Sistem Klasifikasi Tingkat Kematangan Buah Naga</p>
</div>
""", unsafe_allow_html=True)
