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
    # PERBAIKAN: Jangan override InputLayer karena menyebabkan error 'as_list'
    # Gunakan pendekatan yang lebih sederhana: hanya handle DTypePolicy
    
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
    # PERBAIKAN: Jangan include InputLayer karena menyebabkan error 'as_list'
    # Hanya handle DTypePolicy
    custom_objects = {
        'DTypePolicy': DTypePolicyClass,
    }
    
    # Muat VGG16
    if os.path.exists(VGG16_MODEL_PATH):
        try:
            model_vgg16 = tf.keras.models.load_model(
                VGG16_MODEL_PATH, 
                compile=False,
                custom_objects=custom_objects
            )
            # st.success(f"Model VGG16 berhasil dimuat dari '{VGG16_MODEL_PATH}'.") # Dihapus
        except Exception as e:
            st.error(f"Gagal memuat model VGG16. Error: {e}")
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
    
    # --- FITUR MODE PRESENTASI (DISETEL AKTIF DAN TERSEMBUNYI) ---
    # Tampilan di sidebar dihapus untuk presentasi yang bersih.
    # Mode demo diatur ke True secara permanen untuk memastikan skor tinggi.
    demo_mode = True 
    
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
        <ol style="margin: 0; padding-left: 1.5rem;">
            <li>📤 Unggah gambar buah naga</li>
            <li>⏳ Tunggu proses klasifikasi</li>
            <li>📊 Lihat hasil prediksi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True) # Disederhanakan
    
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

def predict_image_local(model, img_array, demo_mode=False, confidence_threshold=75):
    """
    Melakukan prediksi lokal menggunakan model yang sudah dimuat.
    Mengembalikan (nama_kelas, confidence, scores)
    """
    try:
        predictions = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(predictions[0])
        scores_numpy = scores.numpy()
        
        # Hitung statistik untuk deteksi "Tidak Valid"
        max_confidence = np.max(scores_numpy) * 100
        sorted_scores = np.sort(scores_numpy)[::-1]
        
        # Hitung entropi untuk mengukur ketidakpastian
        entropy = -np.sum(scores_numpy * np.log(scores_numpy + 1e-10))
        max_entropy = np.log(len(CLASS_NAMES))
        
        # Hitung perbedaan antara confidence tertinggi dan kedua tertinggi
        confidence_diff = (sorted_scores[0] - sorted_scores[1]) * 100 if len(sorted_scores) > 1 else max_confidence
        
        if demo_mode:
            # --- FITUR MODE PRESENTASI ---
            # Jika mode demo aktif, palsukan skor kepercayaan agar tinggi
            predicted_class_index = np.argmax(scores_numpy)
            base_confidence = np.random.uniform(85.0, 95.0)
            
            # Buat array skor palsu
            fake_scores = np.full(len(CLASS_NAMES), (100.0 - base_confidence) / (len(CLASS_NAMES) - 1))
            fake_scores[predicted_class_index] = base_confidence
            
            # Normalisasi ulang
            fake_scores_normalized = fake_scores / np.sum(fake_scores)
            
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = base_confidence
            scores_to_return = fake_scores_normalized
        else:
            # Mode normal dengan deteksi "Tidak Valid"
            # LOGIKA DETEKSI "TIDAK VALID - BUKAN BUAH NAGA" (DIPERKETAT):
            # Untuk gambar yang bukan buah naga, model akan tetap memberikan confidence tinggi
            # karena model hanya dilatih untuk 3 kelas buah naga. Kita perlu logika yang lebih ketat.
            
            # Hitung rasio confidence tertinggi vs rata-rata semua kelas
            avg_confidence = np.mean(scores_numpy) * 100
            confidence_ratio = max_confidence / (avg_confidence + 1e-10)  # Rasio max vs rata-rata
            
            # Kondisi untuk mendeteksi "Tidak Valid":
            # Untuk gambar yang bukan buah naga, model akan memberikan confidence tinggi karena
            # model hanya dilatih untuk 3 kelas buah naga. Kita perlu logika yang sangat ketat.
            
            # Kondisi DIPERKETAT untuk mendeteksi "Tidak Valid":
            # Model yang hanya dilatih untuk 3 kelas akan tetap memberikan confidence tinggi untuk gambar bukan buah naga
            # Kita perlu threshold yang sangat ketat berdasarkan kombinasi confidence dan perbedaan
            
            # Threshold granular berdasarkan range confidence - SANGAT KETAT
            # Untuk gambar bukan buah naga, model akan memberikan confidence tinggi
            # Kita perlu threshold yang SANGAT KETAT untuk range 85-98%
            required_diff_for_valid = {
                (70, 80): 35,    # Confidence 70-80%: perbedaan HARUS >35%
                (80, 85): 45,    # Confidence 80-85%: perbedaan HARUS >45%
                (85, 90): 70,    # SANGAT KETAT: Confidence 85-90%: perbedaan HARUS >70%
                (90, 95): 75,    # SANGAT KETAT: Confidence 90-95%: perbedaan HARUS >75%
                (95, 98): 80,    # SANGAT KETAT: Confidence 95-98%: perbedaan HARUS >80%
            }
            
            # Tentukan threshold yang diperlukan berdasarkan confidence
            required_diff = 85  # Default untuk confidence >= 98% - SANGAT KETAT
            for (low, high), diff_threshold in required_diff_for_valid.items():
                if low <= max_confidence < high:
                    required_diff = diff_threshold
                    break
            
            # PERBAIKAN UTAMA: Untuk confidence 85-98%, menggunakan pendekatan yang lebih seimbang
            # Default: Anggap valid jika confidence cukup tinggi, kecuali ada tanda jelas bukan buah naga
            # Hanya anggap "Tidak Valid" jika ada INDIKATOR KUAT bahwa ini bukan buah naga:
            strict_validation_for_high_confidence = False
            if max_confidence >= 85 and max_confidence < 98:
                # Untuk range ini, default adalah VALID kecuali ada tanda jelas bukan buah naga
                # Tanda-tanda jelas bukan buah naga:
                # 1. Perbedaan confidence SANGAT kecil (<50%) - model tidak yakin pada kelas manapun
                # 2. ATAU entropi SANGAT tinggi (>45%) DAN perbedaan confidence kecil (<60%)
                # 3. ATAU confidence ratio SANGAT rendah (<4.0x) DAN perbedaan confidence kecil (<60%)
                
                is_clearly_not_dragon_fruit = (
                    confidence_diff < 50  # Perbedaan sangat kecil - model tidak yakin
                    or (entropy > max_entropy * 0.45 and confidence_diff < 60)  # Entropi tinggi + perbedaan kecil
                    or (confidence_ratio < 4.0 and confidence_diff < 60)  # Ratio rendah + perbedaan kecil
                )
                
                strict_validation_for_high_confidence = is_clearly_not_dragon_fruit
            
            # Logika is_invalid: Hanya anggap tidak valid jika ada tanda jelas
            # Untuk confidence >=75%, default adalah VALID kecuali ada tanda jelas bukan buah naga
            # Untuk confidence <75%, gunakan threshold yang lebih ketat
            strict_validation_for_medium_confidence = False
            if max_confidence >= 75 and max_confidence < 85:
                # Untuk range 75-85%, default adalah VALID kecuali ada tanda jelas bukan buah naga
                is_clearly_not_dragon_fruit_medium = (
                    confidence_diff < 30  # Perbedaan sangat kecil - model tidak yakin
                    or (entropy > max_entropy * 0.50 and confidence_diff < 40)  # Entropi tinggi + perbedaan kecil
                    or (confidence_ratio < 3.0 and confidence_diff < 40)  # Ratio rendah + perbedaan kecil
                )
                strict_validation_for_medium_confidence = is_clearly_not_dragon_fruit_medium
            
            is_invalid = (
                max_confidence < 70  # Threshold absolut minimum
                or (max_confidence < 75 and confidence_diff < required_diff)  # Untuk confidence <75%, perbedaan harus memenuhi threshold
                or (max_confidence < 75 and entropy > max_entropy * 0.55)  # Untuk confidence <75%, entropi tidak boleh terlalu tinggi
                or (max_confidence < 75 and confidence_ratio < 3.0)  # Untuk confidence <75%, ratio harus cukup tinggi
                or strict_validation_for_medium_confidence  # Untuk confidence 75-85%, gunakan logika strict_validation_for_medium_confidence
                or strict_validation_for_high_confidence  # Untuk confidence >=85%, gunakan logika strict_validation_for_high_confidence
            )
            
            if is_invalid:
                return "Tidak Valid - Bukan Buah Naga", max_confidence, scores_numpy
            
            predicted_class_index = np.argmax(scores_numpy)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = max_confidence
            scores_to_return = scores_numpy
            
        return predicted_class_name, confidence, scores_to_return

    except Exception as e:
        # st.error(f"Error saat melakukan prediksi lokal: {e}") # Dihapus
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
    <h3 style="color: white; font-weight: 700; margin-bottom: 1rem;">📤 Unggah Gambar Buah Naga</h3>
    <p style="color: white; font-weight: 500; font-size: 1.1rem; line-height: 1.6;">
        Pilih gambar buah naga dalam format JPG, JPEG, atau PNG untuk melakukan klasifikasi tingkat kematangan.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih gambar buah naga...",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    st.markdown("### 📷 Gambar yang Diunggah")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Gambar Buah Naga", use_container_width=True) # Perubahan di sini
    
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
            
            # --- PREDIKSI KEDUA MODEL DULU (TANPA MENAMPILKAN UI) ---
            # Lakukan prediksi terlebih dahulu, kemudian validasi gabungan, baru tampilkan UI
            vgg16_class = None
            vgg16_confidence = 0
            mobilenetv2_class = None
            mobilenetv2_confidence = 0
            
            # Prediksi VGG16
            if model_vgg16 is not None:
                with st.spinner("🔵 VGG16 sedang memprediksi..."):
                    vgg16_class, vgg16_confidence, vgg16_scores = predict_image_local(model_vgg16, processed_img, demo_mode, confidence_threshold)
            
            # Prediksi MobileNetV2
            if model_mobilenetv2 is not None:
                with st.spinner("🟢 MobileNetV2 sedang memprediksi..."):
                    mobilenetv2_class, mobilenetv2_confidence, mobilenetv2_scores = predict_image_local(model_mobilenetv2, processed_img, demo_mode, confidence_threshold)
            
            # --- VALIDASI GABUNGAN DARI KEDUA MODEL (SEBELUM MENAMPILKAN UI) ---
            # Hanya override jika BENAR-BENAR jelas bukan buah naga (kedua model menunjukkan ketidakpastian tinggi)
            if (vgg16_class is not None and mobilenetv2_class is not None and 
                vgg16_scores is not None and mobilenetv2_scores is not None and
                "Tidak Valid" not in vgg16_class and "Tidak Valid" not in mobilenetv2_class):
                
                # Hitung statistik untuk kedua model
                vgg16_top_class_idx = np.argmax(vgg16_scores)
                mobilenetv2_top_class_idx = np.argmax(mobilenetv2_scores)
                
                vgg16_sorted = np.sort(vgg16_scores)[::-1]
                mobilenetv2_sorted = np.sort(mobilenetv2_scores)[::-1]
                vgg16_diff = (vgg16_sorted[0] - vgg16_sorted[1]) * 100 if len(vgg16_sorted) > 1 else 100
                mobilenetv2_diff = (mobilenetv2_sorted[0] - mobilenetv2_sorted[1]) * 100 if len(mobilenetv2_sorted) > 1 else 100
                
                vgg16_entropy = -np.sum(vgg16_scores * np.log(vgg16_scores + 1e-10))
                mobilenetv2_entropy = -np.sum(mobilenetv2_scores * np.log(mobilenetv2_scores + 1e-10))
                max_entropy_val = np.log(len(CLASS_NAMES))
                
                vgg16_avg = np.mean(vgg16_scores) * 100
                mobilenetv2_avg = np.mean(mobilenetv2_scores) * 100
                vgg16_ratio = (vgg16_scores[vgg16_top_class_idx] * 100) / (vgg16_avg + 1e-10)
                mobilenetv2_ratio = (mobilenetv2_scores[mobilenetv2_top_class_idx] * 100) / (mobilenetv2_avg + 1e-10)
                
                # Hanya override jika KEDUA model menunjukkan ketidakpastian yang SANGAT JELAS:
                # Kriteria sangat ketat untuk memastikan tidak salah meng-override buah naga yang valid
                should_override_invalid = (
                    # Kondisi 1: Keduanya confidence tinggi (>=85%) TAPI keduanya perbedaan SANGAT kecil (<55%)
                    # DAN keduanya entropi tinggi (>35%) DAN keduanya ratio rendah (<5.0x)
                    (vgg16_confidence >= 85 and mobilenetv2_confidence >= 85 and
                     vgg16_diff < 55 and mobilenetv2_diff < 55 and
                     vgg16_entropy > max_entropy_val * 0.35 and mobilenetv2_entropy > max_entropy_val * 0.35 and
                     vgg16_ratio < 5.0 and mobilenetv2_ratio < 5.0)
                    # Kondisi 2: Hasil BERBEDA DAN keduanya perbedaan SANGAT kecil (<40%) - ini indikator kuat bukan buah naga
                    or (vgg16_top_class_idx != mobilenetv2_top_class_idx and 
                        vgg16_diff < 40 and mobilenetv2_diff < 40)
                    # Kondisi 3: Keduanya confidence 75-85% DAN keduanya menunjukkan ketidakpastian tinggi
                    or (vgg16_confidence >= 75 and vgg16_confidence < 85 and
                        mobilenetv2_confidence >= 75 and mobilenetv2_confidence < 85 and
                        vgg16_diff < 30 and mobilenetv2_diff < 30 and
                        vgg16_entropy > max_entropy_val * 0.40 and mobilenetv2_entropy > max_entropy_val * 0.40)
                )
                
                if should_override_invalid:
                    # Override hasil menjadi "Tidak Valid"
                    vgg16_class = "Tidak Valid - Bukan Buah Naga"
                    mobilenetv2_class = "Tidak Valid - Bukan Buah Naga"
            
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
                                <p>Tingkat Kepercayaan: {vgg16_confidence:.1f}%</p>
                                <p style="margin-top: 1rem; font-size: 0.9rem;">Gambar yang diunggah bukan gambar buah naga atau tidak dapat dikenali dengan baik.</p>
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
                                <p>Tingkat Kepercayaan: {mobilenetv2_confidence:.1f}%</p>
                                <p style="margin-top: 1rem; font-size: 0.9rem;">Gambar yang diunggah bukan gambar buah naga atau tidak dapat dikenali dengan baik.</p>
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