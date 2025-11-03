# ==============================================================================
# KONFIGURASI GEMINI API KEY
# ==============================================================================
# File ini berisi konfigurasi untuk Gemini API Key
# Pisahkan dari kode utama untuk kemudahan maintenance

import os

# API Key Gemini - Default dari Google AI Studio
# Jika ingin menggunakan API key sendiri, ubah value di bawah ini
GEMINI_API_KEY_DEFAULT = "AIzaSyB4yIzOnkwfUkIgKwv8jWRcdRNI0RmgZjg"

# Atau bisa juga ambil dari environment variable untuk keamanan lebih baik
# Untuk production, lebih baik gunakan environment variable:
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY_DEFAULT)

# Konfigurasi Model Gemini
# Model yang tersedia: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro
# Update: gemini-1.5-flash sudah tidak tersedia, gunakan gemini-2.0-flash atau gemini-2.5-flash
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # Model yang digunakan untuk vision (faster, good for vision)
# Alternatif: "gemini-2.5-flash" untuk lebih cepat, atau "gemini-2.5-pro" untuk lebih akurat

# Prompt untuk deteksi buah naga
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

