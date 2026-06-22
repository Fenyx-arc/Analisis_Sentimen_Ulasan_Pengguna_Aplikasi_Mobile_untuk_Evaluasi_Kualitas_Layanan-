# Laporan Proyek Machine Learning

**Nama Pembuat:**
1. Rifqi Fakhrezi Pasya
2. TRY DYMAZ PRAYOGA

---

# Analisis Sentimen Ulasan Pengguna Aplikasi Mobile untuk Evaluasi Kualitas Layanan

## Pendahuluan
Proyek ini mengacu pada standar metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) untuk mengklasifikasikan sentimen ulasan pengguna aplikasi *mobile* menjadi kategori **Positif** atau **Negatif**. Hasil dari analisis ini ditujukan untuk mengevaluasi kualitas layanan secara otomatis dan mengidentifikasi area yang memerlukan perbaikan berdasarkan *feedback* pengguna.

---

## 1. Business Understanding

### Latar Belakang Masalah
Volume ulasan mentah yang sangat besar di platform distribusi aplikasi (seperti *Google Play Store* atau *App Store*) membuat evaluasi secara manual menjadi sangat tidak efisien, memakan waktu, dan rentan terhadap bias subjektivitas pembaca. 

### Tujuan Proyek
* Mengukur tingkat kepuasan pengguna aplikasi *mobile* secara otomatis dan *real-time*.
* Mengidentifikasi keluhan utama pengguna dengan cepat untuk mempercepat respons perbaikan (*bug fixing* atau peningkatan layanan).

### Solusi yang Diusulkan
Membangun model klasifikasi teks (*Natural Language Processing*) menggunakan teknik ekstraksi fitur **TF-IDF Vectorizer** dipadukan dengan algoritma Machine Learning **[SEBUTKAN ALGORITMA FINAL ANDA, misal: Support Vector Machine (SVM) / Logistic Regression]**. Model ini bertugas untuk memprediksi probabilitas sebuah ulasan masuk ke dalam kelas sentimen positif atau negatif.

---

## 2. Data Understanding

Tahap ini berfokus pada pengenalan data ulasan yang digunakan, pemeriksaan struktur teks, dan analisis distribusi kelas sentimen.

### Analisis Grafik Distribusi Sentimen
Berdasarkan eksplorasi data (*Exploratory Data Analysis*), ditemukan karakteristik penting pada dataset terkait ketidakseimbangan kelas (*class imbalance*):

[![data_understanding](https://private-user-images.githubusercontent.com/73056994/611205577-5ab8c338-801d-4a9c-92a0-1f80bd7650ce.png)](https://private-user-images.githubusercontent.com/73056994/611205577-5ab8c338-801d-4a9c-92a0-1f80bd7650ce.png)

* **Jumlah Data:** Sentimen **Negatif** mendominasi dengan jumlah sekitar **1.100 ulasan**, sedangkan sentimen **Positif** berada di kisaran **600 ulasan**. Total dataset berkisar di angka 1.700 ulasan.
* **Insight:** Dominasi ulasan negatif ini wajar dan mencerminkan perilaku riil pengguna (*user behavior*). Pengguna cenderung lebih termotivasi untuk meluangkan waktu menulis ulasan ketika mereka mengalami kendala teknis (seperti *bug*, aplikasi lambat, atau transaksi gagal) dibandingkan saat aplikasi berjalan normal. Fenomena ini menguntungkan untuk tujuan evaluasi layanan karena memberikan banyak sampel data keluhan untuk dipelajari oleh model.

---

## 3. Data Preparation

Data mentah tidak dapat langsung diproses oleh algoritma. Oleh karena itu, dilakukan tahapan *Text Preprocessing* dengan rincian sebagai berikut:

1. **Case Folding:** Mengubah seluruh karakter teks menjadi huruf kecil (*lowercase*) untuk menjaga konsistensi.
2. **Cleansing:** Menghapus elemen yang tidak relevan seperti angka, tanda baca, emoji, karakter khusus, dan *URL* menggunakan modul *Regular Expression* (`re`).
3. **Stopwords Removal (Filtering):** Menghapus kata-kata hubung atau kata umum yang sering muncul namun tidak membawa makna sentimen spesifik (seperti "yang", "dan", "di", "ke").
4. **Feature Extraction (TF-IDF):** Mengonversi teks yang sudah bersih menjadi representasi matriks numerik menggunakan **TF-IDF (*Term Frequency-Inverse Document Frequency*)**. Teknik ini memberikan bobot lebih tinggi pada kata-kata yang unik dan krusial dalam menentukan sentimen.
5. **Train-Test Split:** Membagi dataset menjadi **80% data latih (Training)** dan **20% data uji (Testing)** untuk memastikan model dievaluasi pada data yang belum pernah dilihat sebelumnya.

---

## 4. Modeling

Pada tahap ini, matriks TF-IDF yang dihasilkan dari data latih dimasukkan ke dalam algoritma klasifikasi **[SEBUTKAN ALGORITMA ANDA, misal: Support Vector Machine / Logistic Regression]** menggunakan *library* Scikit-Learn. 

*(Catatan: Anda bisa menambahkan 1-2 kalimat di sini yang menjelaskan mengapa algoritma tersebut dipilih, misalnya: "Algoritma ini dipilih karena kemampuannya yang sangat baik dalam menangani data teks berdimensi tinggi secara efisien".)*

---

## 5. Evaluation

Model diuji menggunakan 20% data uji (sekitar 344 sampel ulasan). Kinerja model divisualisasikan secara komprehensif melalui metrik **Confusion Matrix**.

[![confusion_matrix](https://private-user-images.githubusercontent.com/73056994/611212270-b6743166-cb45-4c54-9d9c-54ade428950e.png)](https://private-user-images.githubusercontent.com/73056994/611212270-b6743166-cb45-4c54-9d9c-54ade428950e.png)

### Analisis Confusion Matrix
Dari visualisasi *heatmap* di atas, rincian performa model adalah sebagai berikut:
* **True Negative (TN) = 215:** Model berhasil menebak 215 ulasan Negatif dengan benar.
* **True Positive (TP) = 93:** Model berhasil menebak 93 ulasan Positif dengan benar.
* **False Negative (FN) = 24:** Sebanyak 24 ulasan Positif salah diprediksi sebagai Negatif.
* **False Positive (FP) = 12:** Sebanyak 12 ulasan Negatif salah diprediksi sebagai Positif.

### Evaluasi Metrik Lanjutan
Karena dataset mengalami *imbalance*, akurasi saja tidak cukup. Berikut adalah kalkulasi metrik kinerja dari matriks di atas:
1. **Akurasi (Accuracy) ~89.5%:** Proporsi total prediksi yang benar ((215 + 93) / 344). Kombinasi TF-IDF dan model terbukti efektif.
2. **Precision & Recall Kelas Negatif:** Model sangat unggul dalam mendeteksi kelas negatif dengan *Recall* yang sangat tinggi (sekitar **94.7%** ulasan negatif asli berhasil dideteksi). Hal ini karena model mendapatkan lebih banyak variasi kosakata keluhan selama proses *training*.
3. **Precision & Recall Kelas Positif:** *Precision* berada di kisaran **88.5%** (ketika model memprediksi positif, 88.5% benar), namun *Recall*-nya lebih rendah di angka **79.5%**, yang menjelaskan mengapa ada 24 ulasan positif yang "bocor" terprediksi sebagai negatif.

---

## 6. Conclusion & Deployment

### Kesimpulan
Proyek ini sukses mengembangkan model klasifikasi sentimen otomatis dengan tingkat akurasi mendekati **90%**. Model sangat andal dan sensitif dalam menyaring ulasan negatif (keluhan). Hal ini memenuhi tujuan bisnis awal, di mana tim *Customer Service* atau *Developer* dapat memfokuskan sumber daya mereka untuk menganalisis dan merespons ulasan negatif yang difilter oleh model secara cepat guna memperbaiki layanan.

### Deployment Simulation
Model klasifikasi beserta *TF-IDF Vectorizer* telah disimpan dan disimulasikan dalam bentuk fungsi inferensi. Fungsi ini dapat menerima input teks ulasan baru dari *user* dan secara *real-time* mengembalikan prediksi apakah ulasan tersebut bersentimen "Positif" atau "Negatif".
