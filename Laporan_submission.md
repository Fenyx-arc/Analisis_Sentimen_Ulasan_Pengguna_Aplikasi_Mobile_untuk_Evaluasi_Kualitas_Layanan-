# Laporan Proyek Machine Learning - Rifqi Fakhrezi Pasya & Try Dymaz Prayoga

## Project Overview

Dalam era digital saat ini, ulasan pengguna di platform distribusi aplikasi (seperti *Google Play Store* atau *App Store*) sangat krusial bagi kelangsungan sebuah aplikasi *mobile*. Namun, volume ulasan mentah yang sangat besar membuat evaluasi kualitas layanan secara manual menjadi sangat tidak efisien, memakan waktu, dan rentan terhadap bias subjektivitas. Oleh karena itu, diperlukan sebuah sistem otomatis menggunakan *Machine Learning* yang mampu mengklasifikasikan sentimen ulasan menjadi kategori **Positif** atau **Negatif**. Proyek ini bertujuan untuk membangun model analisis sentimen guna mengevaluasi kualitas layanan dan mengidentifikasi area yang memerlukan perbaikan secara *real-time*.

💡 **Manfaat Proyek:**
✔ Membantu tim pengembang dan *Customer Service* mengidentifikasi keluhan pengguna secara otomatis.
✔ Mempercepat respons terhadap *bug* atau masalah teknis pada aplikasi.
✔ Meningkatkan kualitas layanan berdasarkan *feedback* nyata dari pengguna.

---

## Business Understanding

📝 **Problem Statements**
* Bagaimana cara mengevaluasi ulasan pengguna aplikasi yang sangat banyak secara otomatis dan efisien?
* Bagaimana membangun model *Machine Learning* yang mampu mengklasifikasikan sentimen pengguna menjadi kategori positif atau negatif dengan tingkat akurasi yang tinggi?

🎯 **Goals**
* Mengembangkan model *Machine Learning* berbasis *Natural Language Processing* (NLP) untuk menganalisis sentimen ulasan.
* Membantu memisahkan ulasan negatif (keluhan) agar segera dapat ditindaklanjuti oleh pihak *developer*.

🛠 **Solution Approach**
✔ **Ekstraksi Fitur Teks:** Menggunakan **TF-IDF Vectorizer** untuk mengubah teks ulasan menjadi representasi matriks numerik berdasarkan frekuensi dan bobot kata.
✔ **Algoritma Klasifikasi:** Menerapkan algoritma *Machine Learning* **Logistic Regression** untuk memprediksi probabilitas dan kelas sentimen.

---

## Data Understanding

Dataset yang digunakan berisi ulasan pengguna terhadap sebuah aplikasi *mobile* yang dikumpulkan dari *platform* publik, terdiri dari sekitar 1.700 ulasan.

📂 **Dataset Components:**

| Fitur | Deskripsi | Tipe Data |
|---|---|---|
| **Ulasan** | Teks ulasan atau *review* mentah dari pengguna aplikasi | Teks (*String*) |
| **Sentimen** | Label kelas sentimen ulasan (Positif / Negatif) | Kategorikal |

📌 **Uraian Fitur:**
* **Ulasan:** Fitur independen (*predictor*) yang berisi kalimat yang ditulis pengguna saat menilai aplikasi.
* **Sentimen:** Target klasifikasi (Fitur dependen) yang memuat kelas sentimen dari ulasan tersebut.

🔍 **Kondisi Data:**
Berdasarkan *Exploratory Data Analysis* (EDA), terdapat karakteristik *class imbalance* pada dataset:
* **Sentimen Negatif:** Berjumlah sekitar **1.100 ulasan**.
* **Sentimen Positif:** Berjumlah sekitar **600 ulasan**.
* **Insight:** Dominasi ulasan negatif ini wajar dan mencerminkan perilaku riil pengguna (*user behavior*). Pengguna cenderung lebih termotivasi untuk menulis ulasan saat mereka mengalami kendala teknis (seperti aplikasi lambat atau transaksi gagal) dibandingkan saat aplikasi berjalan normal. Hal ini sangat menguntungkan model karena lebih banyak belajar mendeteksi kata-kata keluhan.

---

## Data Preparation

Tahapan ini berfokus pada *Text Preprocessing* agar data teks mentah dapat dipahami dan diproses secara matematis oleh algoritma.

1. **Case Folding:**
   Mengubah seluruh karakter teks menjadi huruf kecil (*lowercase*). Hal ini penting agar kata "Error" dan "error" dianggap sebagai entitas yang sama.
2. **Cleansing:**
   Membersihkan teks dari elemen pengganggu seperti angka, tanda baca, karakter khusus, *link URL*, dan *emoji* menggunakan modul Regular Expression (`re`). Ini bertujuan untuk mengurangi *noise* pada data.
3. **Filtering (Stopwords Removal):**
   Menghapus kata-kata umum yang sering muncul namun tidak membawa makna sentimen yang spesifik (seperti "yang", "dan", "di", "ke").
4. **Feature Extraction (TF-IDF):**
   Mengonversi teks yang telah dibersihkan menjadi representasi matriks angka menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**. Teknik ini memberikan bobot tinggi pada kata yang unik dan sering muncul dalam sebuah ulasan tetapi jarang muncul di ulasan lain (kata kunci utama).
5. **Train-Test Split:**
   Membagi dataset menjadi **80% data latih (Training)** dan **20% data uji (Testing)**. Pembagian ini bertujuan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya untuk menghindari indikasi *overfitting*.

---

## Modeling

Model dalam proyek ini dibangun menggunakan algoritma **Logistic Regression** menggunakan *library* `scikit-learn`.

**Tahapan Pemodelan:**
1. Matriks TF-IDF yang diekstrak dari data latih (*training data*) dimasukkan ke dalam algoritma klasifikasi Logistic Regression.
2. Model mempelajari pola kata yang saling terhubung (misalnya kata *"lemot"*, *"bug"*, *"kecewa"* berbobot besar pada kelas Negatif, dan *"bagus"*, *"keren"* pada kelas Positif). Algoritma ini dipilih karena sangat efisien dan cepat dalam mengklasifikasikan data teks yang memiliki dimensi fitur yang tinggi (seperti hasil dari TF-IDF).
3. Setelah model berhasil dilatih, dilakukan pengujian prediksi menggunakan 20% data uji (*testing data*).

---

## Evaluation

Model diuji menggunakan 20% data pengujian yang berjumlah sekitar 344 sampel. Metrik evaluasi utama divisualisasikan melalui **Confusion Matrix** di bawah ini:

![Distribusi Sentimen](<img width="526" height="475" alt="confusion_matrix" src="https://github.com/user-attachments/assets/25024315-65b3-4484-b14c-4725318da136" />)


📊 **Analisis Confusion Matrix:**
Berdasarkan visualisasi *heatmap* di atas, rincian performa model adalah:
* **True Negative (TN) = 215:** Model berhasil menebak 215 ulasan Negatif dengan benar.
* **True Positive (TP) = 93:** Model berhasil menebak 93 ulasan Positif dengan benar.
* **False Negative (FN) = 24:** Sebanyak 24 ulasan Positif salah diprediksi sebagai Negatif.
* **False Positive (FP) = 12:** Sebanyak 12 ulasan Negatif salah diprediksi sebagai Positif.

📌 **Kesimpulan Evaluasi:**
1. **Akurasi Sangat Baik (~89.53%):** Tingkat tebakan total yang benar adalah $308$ dari total $344$ sampel. Kombinasi metode TF-IDF dengan model *Logistic Regression* terbukti sangat optimal.
2. **Performa Kelas Mayoritas (Negatif) Lebih Unggul:** Model jauh lebih sensitif dan akurat dalam mendeteksi ulasan Negatif. Tingkat kesalahannya sangat minim (hanya 12 dari 227 data aktual negatif salah diklasifikasikan). Hal ini terjadi karena model mendapatkan eksposur kosakata keluhan yang jauh lebih banyak selama proses *training* akibat fenomena *class imbalance*.
3. **Deployment Siap Pakai:** Mengingat tujuan bisnis adalah mengidentifikasi keluhan, performa model ini sangat mendukung karena jarang melewatkan ulasan negatif. Sistem ini sudah sangat layak untuk diimplementasikan secara otomatis dalam *pipeline* layanan keluhan pengguna.
