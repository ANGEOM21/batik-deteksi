# 📸 Deteksi Motif Batik - Mobile App

Aplikasi mobile berbasis React + Capacitor untuk mendeteksi motif batik tradisional Indonesia menggunakan model YOLOv8 (versi `.tflite`) secara real-time.

![Demo App](./assets/demo-mega-mendung.png)

---

## 🧠 Fitur Utama

- 🔍 Deteksi motif batik dengan AI (YOLOv8 `.tflite`)
- 📷 Ambil gambar langsung dari kamera atau galeri
- ✅ Output deteksi lengkap dengan nama motif
- ⚡ Cepat & bisa berjalan offline di perangkat Android
- 🎨 Motif yang dikenali:
  - Kawung
  - Mega Mendung
  - Parang
  - Truntum
  - Bukan Batik (negative class)

---

## 🛠️ Teknologi yang Digunakan

| Bagian | Teknologi |
|-------|-----------|
| Frontend | React Vite + TypeScript |
| Styling | Tailwind CSS + DaisyUI |
| Android Bridge | Capacitor Plugin (Native Android) |
| Model AI | YOLOv8 (converted ke `.tflite`) |
| Post-processing | Custom JS decoder YOLO output |

---

## 📦 Instalasi

```
# 1. Clone repositori ini
git clone https://github.com/username/batik-detector-app.git
cd batik-detector-app

# 2. Install dependencies
npm install

# 3. Jalankan di browser
npm run dev

# 4. Build untuk Android
npx cap sync
npx cap open android
```