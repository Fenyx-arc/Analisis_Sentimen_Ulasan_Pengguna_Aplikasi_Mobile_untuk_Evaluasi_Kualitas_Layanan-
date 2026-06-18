# Laporan Proyek Machine Learning
[Rifqi Fakhrezi Pasya]
[TRY DYMAZ PRAYOGA]
## Domain Proyek
Dalam industri pengembangan perangkat lunak dan aplikasi mobile, ulasan pengguna (*user reviews*) di platform seperti Google Play Store merupakan aset penting yang mencerminkan kualitas layanan secara langsung. Ulasan tersebut berisi umpan balik (*feedback*) kritis mengenai kepuasan pengguna, keluhan *bug*, stabilitas sistem, hingga permintaan fitur baru. 

Namun, volume ulasan yang masuk setiap harinya bisa mencapai ribuan, sehingga proses evaluasi secara manual menjadi sangat tidak efisien, membutuhkan waktu lama, dan rentan terhadap subjektivitas manusia. Oleh karena itu, otomatisasi analisis menggunakan teknik *Natural Language Processing* (NLP) dan *Machine Learning* untuk mengklasifikasikan sentimen menjadi solusi yang sangat krusial. Proyek ini berfokus pada pembangunan model analisis sentimen untuk memisahkan ulasan pengguna ke dalam kategori **Positif** dan **Negatif** guna mempercepat proses evaluasi kualitas layanan aplikasi mobile.

---

## Business Understanding
Proyek ini dikembangkan untuk membantu tim pengembang (*developer*) dan manajemen produk dalam mengidentifikasi masalah utama aplikasi secara *real-time*.

### Problem Statements
1. Bagaimana cara memproses data teks ulasan pengguna yang mentah, tidak terstruktur, dan memiliki banyak gangguan (*noise*) agar dapat dipahami oleh model *Machine Learning*?
2. Bagaimana membangun model klasifikasi teks yang mampu membedakan ulasan positif dan negatif dengan tingkat akurasi dan performa yang optimal?

### Goals
1. Melakukan tahapan *Data Preparation* yang komprehensif (pembersihan data teks dan ekstraksi fitur) agar siap digunakan untuk pelatihan model.
2. Mengembangkan model *Machine Learning* berbasis klasifikasi biner untuk mendeteksi sentimen ulasan dengan target akurasi yang tinggi.

### Solution Statement
Untuk mencapai tujuan tersebut, solusi yang diterapkan adalah sebagai berikut:
* **Ekstraksi Fitur Teks:** Menggunakan metode **TF-IDF (Term Frequency-Inverse Document Frequency)**. Metode ini dipilih karena mampu memberikan bobot matematis pada kata-kata yang unik dan penting dalam dokumen, serta efektif dalam menyoroti kata-kata pembawa sentimen (seperti "bagus", "kecewa", "lemot").
* **Algoritma Modeling:** Menggunakan **Support Vector Machine (SVM)** dengan kernel linear. SVM terbukti sangat tangguh untuk klasifikasi teks berdimensi tinggi karena bekerja dengan mencari *hyperplane* pembatas maksimal (*maximum margin*) untuk memisahkan dua kelas target secara optimal.

---

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah **Google Play Store Reviews** yang diunduh secara otomatis dari platform Kaggle menggunakan pustaka `kagglehub`.

### Variabel-variabel pada Dataset:
* `content`: Berisi teks ulasan mentah yang ditulis oleh pengguna aplikasi.
* `score`: Berisi rating angka berskala 1 hingga 5 yang diberikan oleh pengguna.

### Pelabelan Sentimen (Labeling):
Karena dataset asli hanya menyediakan rating angka (`score`), dilakukan konversi otomatis untuk membentuk kelas target biner (`sentiment`):
* **Positive (Positif):** Untuk ulasan dengan skor rating 4 dan 5.
* **Negative (Negatif):** Untuk ulasan dengan skor rating 1, 2, dan 3.

### Exploratory Data Analysis (EDA):
Pada tahapan analisis data awal, visualisasi menggunakan grafik batang (*countplot*) diterapkan untuk melihat keseimbangan distribusi jumlah ulasan positif dan ulasan negatif. Hal ini penting untuk memastikan model tidak bias terhadap salah satu kelas mayoritas saat proses pelatihan.

---

## Data Preparation
Tahap penyiapan data dilakukan secara sistematis untuk mengubah teks mentah menjadi matriks numerik yang siap diproses oleh algoritma:

1. **Text Cleaning (Pembersihan Teks):**
   * *Case Folding:* Mengubah seluruh karakter huruf menjadi huruf kecil agar kata yang sama tidak dianggap berbeda akibat perbedaan kapitalisasi.
   * *Regex Filtering:* Menghapus angka, tanda baca, dan karakter khusus yang tidak memiliki nilai kontekstual dalam analisis sentimen linear.
   * *Trimming:* Menghapus spasi putih tidak berguna di awal dan akhir teks ulasan.
2. **Data Splitting (Pembagian Data):**
   Dataset dibagi menjadi **80% untuk data latih (Training Set)** dan **20% untuk data uji (Testing Set)**. Pembagian ini menggunakan parameter `stratify=y` untuk menjaga agar proporsi label sentimen positif dan negatif tetap seimbang di kedua subset data tersebut.
3. **Feature Extraction (Vektorisasi TF-IDF):**
   Menggunakan `TfidfVectorizer` dengan batas maksimal fitur sebesar 5000 kata teratas (`max_features=5000`) untuk membatasi ukuran memori serta menghindari masalah *curse of dimensionality*.

---

## Modeling
Model klasifikasi dibangun menggunakan algoritma **Support Vector Machine (SVM)** melalui kelas `SVC` dari pustaka *scikit-learn*. 

### Parameter Model:
* `kernel='linear'`: Dipilih karena data teks yang telah ditransformasikan oleh TF-IDF memiliki ruang dimensi yang sangat besar. Kernel linear terbukti secara empiris sangat efisien, cepat, dan memiliki risiko *overfitting* yang rendah pada data teks.
* `C=1.0`: Parameter regularisasi standar untuk menyeimbangkan antara pembentukan margin yang lebar dengan toleransi kesalahan klasifikasi pada data latih.

Proses pelatihan dilakukan dengan mencocokkan matriks fitur data latih (`X_train_tfidf`) terhadap label targetnya (`y_train`).

---

## Evaluation
Performa model diuji secara ketat menggunakan data uji (*testing set*) yang belum pernah dilihat oleh model selama masa pelatihan. Metrik evaluasi yang digunakan meliputi:

1. **Accuracy:** Menghitung persentase total prediksi benar dari keseluruhan data uji.
2. **Precision:** Mengukur ketepatan model dalam memprediksi kelas positif (berapa banyak yang benar-benar positif dari seluruh hasil prediksi positif).
3. **Recall:** Mengukur sensitivitas model dalam menemukan ulasan positif yang ada di lapangan.
4. **F1-Score:** Nilai rata-rata harmonik antara *Precision* dan *Recall* untuk memberikan penilaian performa yang seimbang.

### Confusion Matrix Analysis:
Selain angka metrik di atas, evaluasi divisualisasikan dalam bentuk grafik **Confusion Matrix Heatmap**. Grafik ini menjabarkan performa model secara detail berdasarkan empat kondisi:
* *True Positive (TP):* Ulasan positif yang berhasil diprediksi positif.
* *True Negative (TN):* Ulasan negatif yang berhasil diprediksi negatif.
* *False Positive (FP):* Ulasan negatif yang salah diprediksi sebagai positif.
* *False Negative (FN):* Ulasan positif yang salah diprediksi sebagai negatif.

---

## Kesimpulan & Referensi
Model klasifikasi sentimen berbasis SVM ini memberikan hasil performa yang sangat baik dalam mendeteksi ekspresi kepuasan maupun keluhan pengguna aplikasi. Dengan adanya sistem otomatisasi ini, manajemen kualitas layanan aplikasi mobile dapat dilakukan secara efisien tanpa harus membaca ulasan satu per satu secara manual.

### Referensi:
* Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
* Dataset: KaggleHub - Google Play Store Reviews (`prakharrathi25/google-play-store-reviews`).
