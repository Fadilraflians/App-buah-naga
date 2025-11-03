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
    import google.generativeai as genai
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
    }
    
    /* Container utama */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
    header {visibility: hidden;}
    
    /* Remove white spaces and improve spacing */
    .stApp > div {
        background: transparent;
    }
    
    .main .block-container {
        max-width: 1200px;
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
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%); padding: 1rem; border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Informasi Model</h2>
    </div>
    """, unsafe_allow_html=True) # Mengganti judul
    
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
    
    # Cek apakah library tersedia dengan lebih akurat
    try:
        # Cek ulang import untuk memastikan
        import importlib
        if GEMINI_AVAILABLE:
            # Double check dengan import langsung
            importlib.reload(genai) if 'genai' in globals() and genai is not None else None
            gemini_status = "✅ Tersedia"
            gemini_status_color = "green"
        else:
            raise ImportError("Library tidak tersedia")
    except Exception:
        # Coba import ulang
        try:
            import google.generativeai as genai
            gemini_status = "✅ Tersedia"
            gemini_status_color = "green"
            # Update global
            globals()['genai'] = genai
            globals()['GEMINI_AVAILABLE'] = True
        except ImportError:
            gemini_status = "❌ Tidak Tersedia"
            gemini_status_color = "red"
    
    # Tampilkan status library
    if GEMINI_AVAILABLE:
        st.markdown(f"""
        <div class="info-box" style="padding: 1rem; border-left: 4px solid #4ECDC4;">
            <p style="margin: 0; font-size: 0.95rem;">
                <strong style="color: {gemini_status_color};">{gemini_status}</strong> - Library Gemini sudah terinstall!<br>
                Masukkan API key Gemini untuk menggunakan AI Vision dalam deteksi buah naga (TAHAP 1).<br>
                Jika tidak diisi, sistem akan menggunakan API key default yang telah dikonfigurasi.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input API Key (dengan default dari config)
        gemini_api_key_input = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            help="API key dari Google AI Studio. Kosongkan untuk menggunakan default dari konfigurasi.",
            placeholder="Kosongkan untuk menggunakan default"
        )
        
        # Gunakan input user atau default dari config
        gemini_api_key = gemini_api_key_input if gemini_api_key_input.strip() else GEMINI_API_KEY_DEFAULT
        
        # Jangan tampilkan info tentang API key yang digunakan (untuk keamanan)
        # Info dihapus untuk mencegah exposure API key
        
        # Test koneksi API key (opsional)
        with st.expander("🧪 Test Koneksi API Key", expanded=False):
            if st.button("Test Koneksi", help="Cek apakah API key Gemini valid dan terkoneksi"):
                if gemini_api_key:
                    try:
                        genai.configure(api_key=gemini_api_key)
                        
                        # List available models untuk verifikasi
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        
                        # Cek apakah model yang dipilih tersedia
                        model_path = f"models/{GEMINI_MODEL_NAME}"
                        if model_path in available_models:
                            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                            # Test dengan prompt sederhana
                            test_response = model.generate_content("Hello")
                            st.success("✅ **API Key Gemini valid dan terkoneksi!**")
                            st.info(f"📝 Menggunakan model: **{GEMINI_MODEL_NAME}**")
                            st.success(f"✅ Model tersedia dan didukung")
                        else:
                            # Gunakan model yang tersedia (fallback ke gemini-2.0-flash)
                            fallback_model = "gemini-2.0-flash"
                            if f"models/{fallback_model}" in available_models:
                                st.warning(f"⚠️ Model **{GEMINI_MODEL_NAME}** tidak tersedia!")
                                st.info(f"🔄 Menggunakan model fallback: **{fallback_model}**")
                                model = genai.GenerativeModel(fallback_model)
                                test_response = model.generate_content("Hello")
                                st.success("✅ **API Key Gemini valid dan terkoneksi!**")
                                # Update model name di config (optional)
                                st.info(f"💡 Update `GEMINI_MODEL_NAME` di `config_gemini.py` menjadi `{fallback_model}`")
                            else:
                                st.error(f"❌ Model {GEMINI_MODEL_NAME} dan {fallback_model} tidak tersedia!")
                                st.info(f"📋 Model yang tersedia: {', '.join([m.split('/')[-1] for m in available_models[:5]])}...")
                    except Exception as e:
                        st.error(f"❌ **Error koneksi API Key:** {str(e)}")
                        st.info("💡 Pastikan API key valid dari https://aistudio.google.com/app/apikey")
                        st.info("💡 Pastikan billing sudah di-setup di Google AI Studio (walaupun free tier)")
    else:
        st.error("⚠️ **Library 'google-generativeai' belum terinstall!**")
        st.markdown("""
        <div class="warning-box" style="padding: 1rem;">
            <p style="margin: 0; font-size: 0.9rem;">
                <strong>Cara Install:</strong><br>
                1. Buka terminal/command prompt<br>
                2. Jalankan command: <code style="background: #f0f0f0; padding: 0.2rem 0.5rem; border-radius: 3px;">pip install google-generativeai</code><br>
                3. Restart aplikasi Streamlit<br><br>
                <strong>Catatan:</strong> Setelah install, pastikan restart aplikasi Streamlit agar library terdeteksi.
            </p>
        </div>
        """, unsafe_allow_html=True)
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
st.markdown('<h1 class="main-header">🐉 Klasifikasi Kematangan Buah Naga</h1>', unsafe_allow_html=True) # Dihapus (Versi Lokal)
st.markdown("""
<div class="info-box">
    <h3 style="color: white; font-weight: 700; margin-bottom: 1rem;">📋 Tentang Aplikasi</h3>
    <p style="color: white; font-weight: 500; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
        Aplikasi ini menggunakan model <strong style="color: #4ECDC4;">Convolutional Neural Network (CNN)</strong> VGG16 dan MobileNetV2 untuk mengklasifikasikan buah naga menjadi:
    </p>
    <ul style="color: white; font-weight: 500; font-size: 1rem; line-height: 1.8;">
        <li style="margin-bottom: 0.5rem;">🍎 <strong style="color: #FF6B6B;">Mature Dragon Fruit</strong> - Buah Naga Matang</li>
        <li style="margin-bottom: 0.5rem;">🍏 <strong style="color: #45B7D1;">Immature Dragon Fruit</strong> - Buah Naga Mentah</li>
        <li style="margin-bottom: 0.5rem;">🍎 <strong style="color: #E74C3C;">Defect Dragon Fruit</strong> - Buah Naga Busuk</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Bagian Performa Model dan Hasil Analisis Pelatihan
st.markdown('<h2 class="sub-header">📊 Performa Model dari Pelatihan</h2>', unsafe_allow_html=True)
if model_performance_metrics:
    # Gunakan .get() untuk menghindari KeyError
    vgg16_accuracy = model_performance_metrics.get('vgg16', {}).get('accuracy', 0.0)
    vgg16_size = model_performance_metrics.get('vgg16', {}).get('model_size_mb', 0.0)
    mobilenetv2_accuracy = model_performance_metrics.get('mobilenetv2', {}).get('accuracy', 0.0)
    mobilenetv2_size = model_performance_metrics.get('mobilenetv2', {}).get('model_size_mb', 0.0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔵 Model VGG16</h3>
            <div style="text-align: center;">
                <h2 style="color: #4ECDC4; margin: 0;">{vgg16_accuracy:.1%}</h2>
                <p style="margin: 0;">Akurasi Test Set</p>
                <hr style="margin: 1rem 0;">
                <h4 style="color: #FF6B6B; margin: 0;">{vgg16_size:.1f} MB</h4>
                <p style="margin: 0;">Ukuran Model</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🟢 Model MobileNetV2</h3>
            <div style="text-align: center;">
                <h2 style="color: #4ECDC4; margin: 0;">{mobilenetv2_accuracy:.1%}</h2>
                <p style="margin: 0;">Akurasi Test Set</p>
                <hr style="margin: 1rem 0;">
                <h4 style="color: #FF6B6B; margin: 0;">{mobilenetv2_size:.1f} MB</h4>
                <p style="margin: 0;">Ukuran Model</p>
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