# 📷 Fitur Scan Kamera - Panduan Lengkap

## 🎯 Fitur Baru: Scan dengan Kamera

Aplikasi Streamlit sekarang mendukung **dua metode input** untuk klasifikasi buah naga:

1. **📤 Upload File** - Upload gambar dari komputer
2. **📷 Scan dengan Kamera** - Ambil foto langsung dengan kamera/webcam

---

## 🚀 Cara Menggunakan Fitur Scan

### **Langkah 1: Pilih Mode Scan**

1. Buka aplikasi Streamlit Anda
2. Di bagian utama, pilih **"📷 Scan dengan Kamera"** (radio button)

### **Langkah 2: Izinkan Akses Kamera**

1. Browser akan meminta **izin akses kamera**
2. Klik **"Izinkan"** atau **"Allow"**

### **Langkah 3: Arahkan Kamera ke Buah Naga**

1. **Posisikan buah naga** di depan kamera
2. Pastikan:
   - ✅ Cahaya cukup
   - ✅ Buah naga terlihat jelas
   - ✅ Tidak ada objek lain yang mengganggu
   - ✅ Buah naga mengisi sebagian besar frame

### **Langkah 4: Ambil Foto**

1. Klik tombol **"Take Photo"** di bawah preview kamera
2. Foto akan otomatis diambil dan diproses
3. Sistem akan melakukan klasifikasi secara otomatis

### **Langkah 5: Lihat Hasil**

1. Tunggu proses klasifikasi selesai (2-3 detik)
2. Hasil prediksi akan muncul:
   - Prediksi VGG16
   - Prediksi MobileNetV2
   - Distribusi tingkat kepercayaan
   - Grafik visualisasi

---

## 💡 Tips untuk Hasil Terbaik

### **Posisi Kamera:**

- ✅ **Jarak optimal:** 30-50 cm dari buah naga
- ✅ **Sudut:** Lurus ke buah naga (tidak miring)
- ✅ **Stabil:** Gunakan tripod atau letakkan kamera di permukaan datar

### **Pencahayaan:**

- ✅ **Cahaya cukup:** Gunakan ruangan terang atau dekat jendela
- ✅ **Hindari silau:** Jangan ada sumber cahaya langsung di belakang buah
- ✅ **Cahaya alami:** Lebih baik menggunakan cahaya alami (matahari)

### **Komposisi Foto:**

- ✅ **Focus jelas:** Buah naga harus tajam dan jelas
- ✅ **Background:** Background polos lebih baik (putih/hitam)
- ✅ **Ukuran:** Buah naga harus mengisi minimal 50% dari frame
- ✅ **Posisi:** Buah naga di tengah frame

### **Kondisi Buah Naga:**

- ✅ **Satu buah:** Lebih baik scan satu buah naga per foto
- ✅ **Bersih:** Pastikan buah naga bersih (tidak ada kotoran menempel)
- ✅ **Utuh:** Buah naga harus utuh, tidak terpotong

---

## 🔧 Troubleshooting

### **Kamera Tidak Muncul**

**Penyebab:**
- Browser tidak mengizinkan akses kamera
- Kamera sedang digunakan aplikasi lain
- Kamera tidak terdeteksi

**Solusi:**
1. **Check izin browser:**
   - Chrome: Klik icon 🔒 di address bar → Settings → Camera → Allow
   - Firefox: Preferences → Privacy & Security → Permissions → Camera → Allow
   - Edge: Settings → Site permissions → Camera → Allow

2. **Close aplikasi lain yang pakai kamera:**
   - Zoom, Teams, Skype, dll.

3. **Refresh halaman** (F5)

4. **Coba browser lain** jika masih tidak muncul

---

### **Foto Tidak Jelas/Blur**

**Penyebab:**
- Gerakan saat mengambil foto
- Cahaya kurang
- Jarak terlalu dekat/jauh

**Solusi:**
1. **Stabilkan kamera** sebelum ambil foto
2. **Tingkatkan pencahayaan**
3. **Atur jarak** optimal (30-50 cm)
4. **Ambil ulang foto** jika hasil blur

---

### **Prediksi Tidak Akurat**

**Penyebab:**
- Foto kurang jelas
- Posisi/angle tidak optimal
- Background mengganggu
- Cahaya tidak cukup

**Solusi:**
1. **Ambil foto ulang** dengan kondisi lebih baik
2. **Ikuti tips** di atas untuk hasil terbaik
3. **Coba beberapa angle** berbeda
4. **Gunakan mode Upload File** jika scan tidak berhasil

---

### **Browser Tidak Support Kamera**

**Browser yang Support:**
- ✅ Chrome (Desktop & Mobile)
- ✅ Firefox (Desktop & Mobile)
- ✅ Edge (Desktop)
- ✅ Safari (macOS & iOS)
- ✅ Opera (Desktop)

**Browser yang Tidak Support:**
- ❌ IE 11 (Internet Explorer)

**Solusi:**
- Gunakan browser modern (Chrome, Firefox, Edge)

---

## 📱 Mobile vs Desktop

### **Desktop (Laptop/PC):**

- ✅ **Webcam eksternal:** Biasanya lebih baik kualitasnya
- ✅ **Stabil:** Lebih mudah posisikan kamera
- ✅ **Layar besar:** Lebih mudah lihat preview

### **Mobile (Smartphone/Tablet):**

- ✅ **Kamera bagus:** Smartphone biasanya punya kamera bagus
- ✅ **Portable:** Mudah dibawa ke mana-mana
- ⚠️ **Perlu stabil:** Gunakan kedua tangan untuk stabil
- ⚠️ **Cahaya penting:** Perhatikan pencahayaan

---

## 🎯 Perbandingan: Upload vs Scan

| Fitur | 📤 Upload File | 📷 Scan Kamera |
|-------|---------------|----------------|
| **Kecepatan** | Cepat (langsung) | Cepat (real-time) |
| **Kualitas** | Bisa tinggi (jika foto bagus) | Tergantung kamera |
| **Kemudahan** | Sangat mudah | Mudah (perlu izin kamera) |
| **Use Case** | Foto yang sudah ada | Scan langsung di lapangan |
| **Flexibility** | Bisa edit dulu | Langsung ambil foto |

---

## ✅ Checklist Sebelum Scan

- [ ] Browser sudah mengizinkan akses kamera
- [ ] Kamera berfungsi dengan baik
- [ ] Buah naga sudah siap (bersih, utuh)
- [ ] Pencahayaan cukup
- [ ] Kamera dalam posisi stabil
- [ ] Buah naga dalam frame kamera

---

## 🚀 Tips Lanjutan

### **1. Multiple Scan:**

- Ambil beberapa foto dari angle berbeda
- Bandingkan hasil prediksi
- Gunakan hasil dengan confidence tertinggi

### **2. Kombinasi Upload + Scan:**

- Scan untuk preview cepat
- Upload file untuk hasil lebih akurat (jika foto sudah di-edit/dioptimalkan)

### **3. Quality Check:**

- Setelah scan, cek kualitas foto di preview
- Jika kurang jelas, ambil ulang sebelum proses klasifikasi

---

## 📊 Statistik & Performance

**Kecepatan Scan:**
- Capture: < 1 detik
- Processing: 2-3 detik
- Total: ~3-4 detik

**Akurasi:**
- Sama dengan mode Upload File
- Tergantung kualitas foto yang diambil

---

## 🔒 Privacy & Security

- ✅ Foto hanya diproses **di browser Anda**
- ✅ Foto **tidak disimpan** secara permanen
- ✅ Foto **tidak dikirim** ke server eksternal (kecuali untuk prediksi)
- ✅ Izin kamera bisa di-revoke kapan saja

---

**Selamat menggunakan fitur scan kamera! 📷✨**

