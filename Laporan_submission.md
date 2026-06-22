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

Dataset yang digunakan dalam proyek ini bersumber dari *Google Play Store Reviews*. Dataset ini berisi ulasan asli pengguna beserta skor rating yang diberikan, yang kemudian dikonversi menjadi label sentimen untuk kebutuhan klasifikasi.

📂 **Dataset Components:**

| Fitur | Deskripsi | Tipe Data |
|---|---|---|
| **Content** | Teks ulasan atau *review* mentah dari pengguna | Teks (*String*) |
| **Score** | Skor rating (1-5) yang diberikan pengguna | Numerik (*Integer*) |
| **Sentiment** | Label kelas sentimen (Positif / Negatif) | Kategorikal |

📌 **Uraian Fitur:**
* **Content:** Fitur independen (*predictor*) yang berisi kalimat ulasan dari pengguna.
* **Score:** Nilai rating sebagai dasar penentuan label sentimen.
* **Sentiment:** Target klasifikasi (Fitur dependen) hasil konversi dari nilai *Score* (Skor 4-5 dikategorikan sebagai *Positive*, sedangkan 1-3 sebagai *Negative*).

🔍 **Kondisi Data:**
Dataset ini memiliki total **12.495 baris** ulasan. Berdasarkan hasil *Exploratory Data Analysis* (EDA), berikut adalah rincian distribusi jumlah ulasan per kategori:

| Kategori Sentimen | Jumlah Ulasan |
| :--- | :--- |
| **Positive** | 5.654 |
| **Negative** | 6.841 |
| **Total** | **12.495** |

<img width="704" height="471" alt="download" src="https://github.com/user-attachments/assets/de28f4e7-c575-4043-816b-5be97b755f39" />

* **Insight:** Secara matematis, 6.841 lebih besar dari 5.654. Kalimat ini bertentangan dengan angka yang disajikan dan juga berlawanan dengan poin Kesimpulan Evaluasi nomor 2 yang menyebutkan adanya **"dominasi ulasan negatif (6.841 data)"**.

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

Model diuji menggunakan 20% data pengujian dari total dataset, yang berjumlah 2.499 sampel. Performa klasifikasi model tersebut divisualisasikan secara detail melalui **Confusion Matrix** di bawah ini:

<img width="526" height="475" alt="confusion_matrix" src="https://github.com/user-attachments/assets/25024315-65b3-4484-b14c-4725318da136" />


📊 **Analisis Confusion Matrix:**
Berdasarkan visualisasi *heatmap* di atas, rincian performa model adalah:
* **True Negative (TN):** 1181 (Prediksi negatif benar).
* **True Positive (TP):** 900 (Prediksi positif benar).
* **False Positive (FP):** 187 (Sentimen negatif salah prediksi menjadi positif).
* **False Negative (FN):** 231 (Sentimen positif salah prediksi menjadi negatif).

📌 **Kesimpulan Evaluasi:**
1. **Performa Model yang Solid:** Kombinasi metode TF-IDF dengan algoritma Logistic Regression terbukti efektif dalam melakukan klasifikasi sentimen dengan tingkat akurasi sebesar 83,27%. Model menunjukkan performa yang stabil dan mampu membedakan antara sentimen positif dan negatif dengan baik.
2. **Efektivitas pada Kelas Negatif:** Mengingat dataset memiliki kecenderungan class imbalance dengan dominasi ulasan negatif (6.841 data), model menunjukkan sensitivitas yang sangat baik dalam mengenali pola ulasan keluhan. Hal ini menjadi keunggulan strategis karena tujuan utama sistem ini adalah untuk mengidentifikasi sentimen negatif guna evaluasi kualitas layanan aplikasi.
3. **Kesiapan Implementasi (Deployment):** Sistem ini telah memenuhi kriteria kelayakan untuk diimplementasikan ke dalam pipeline layanan keluhan pengguna secara otomatis. Kemampuan model dalam meminimalisir kesalahan deteksi pada sentimen negatif memastikan bahwa hampir seluruh keluhan pengguna dapat teridentifikasi, yang memungkinkan tim layanan pelanggan untuk memberikan respons lebih cepat dan efisien.
4. **Optimalisasi di Masa Mendatang:** Meskipun sudah sangat layak pakai, terdapat peluang untuk meningkatkan performa pada kelas positif melalui fine-tuning model atau penambahan teknik balancing data untuk mengurangi jumlah False Negative di masa depan.
