<img width="617" height="475" alt="data_understanding" src="https://github.com/user-attachments/assets/6e287e81-e7d2-4849-a457-cec6ea112db9" /># Laporan Proyek Machine Learning
[Rifqi Fakhrezi Pasya]
[TRY DYMAZ PRAYOGA]
# Analisis Sentimen Ulasan Pengguna Aplikasi Mobile untuk Evaluasi Kualitas Layanan

Proyek ini mengacu pada standar metodologi **CRISP-DM** (Cross-Industry Standard Process for Data Mining) untuk mengklasifikasikan sentimen ulasan pengguna aplikasi mobile menjadi kategori **Positif** atau **Negatif**. Hasil dari analisis ini digunakan untuk mengevaluasi kualitas layanan dan mengidentifikasi area yang memerlukan perbaikan.

---

## 1. Business Understanding

* **Tujuan Bisnis:** Mengukur tingkat kepuasan pengguna aplikasi mobile dan mengidentifikasi keluhan utama secara real-time untuk meningkatkan kualitas layanan.
* **Masalah Teknis:** Volume ulasan mentah yang sangat besar di platform (seperti Google Play Store atau App Store) membuat evaluasi manual tidak efisien dan rentan terhadap subjektivitas.
* **Solusi Machine Learning:** Membangun model klasifikasi berbasis teks menggunakan *TF-IDF Vectorizer* dan algoritma Machine Learning (seperti *Logistic Regression* atau *SVM*) untuk mendeteksi sentimen negatif dan positif secara otomatis.

---

## 2. Data Understanding

Tahap ini berfokus pada pengenalan data ulasan yang digunakan, pemeriksaan struktur data, dan analisis distribusi kelas sentimen.

### Analisis Grafik Distribusi Sentimen
Berdasarkan grafik distribusi yang diperoleh <img width="617" height="475" alt="data_understanding" src="https://github.com/user-attachments/assets/ad7a5fba-f645-4a33-a0bc-e99ef059f67c" />, terdapat karakteristik penting pada dataset:

* **Jumlah Data:** Sentimen **Negatif** mencapai sekitar **1100 ulasan**, sedangkan sentimen **Positif** berada di kisaran **600 ulasan**. Total dataset berkisar 1700 ulasan.
* **Mengapa Hasilnya Demikian?** Dominasi ulasan negatif ini mencerminkan perilaku riil pengguna aplikasi (*user behavior*), di mana pengguna cenderung lebih termotivasi untuk menulis ulasan ketika mereka mengalami masalah (bug, aplikasi lambat, atau kegagalan transaksi) dibandingkan saat aplikasi berjalan normal. Fenomena ini sangat menguntungkan untuk tujuan evaluasi layanan karena memberikan banyak sampel data keluhan untuk dipelajari oleh model.

---

## 3. Data Preparation

Proses penyiapan data dilakukan melalui beberapa tahapan *Text Preprocessing* agar text mentah dapat dipahami oleh algoritma:
1.  **Case Folding:** Mengubah seluruh teks menjadi huruf kecil.
2.  **Cleansing:** Menghapus angka, tanda baca, emoji, dan karakter khusus menggunakan Regular Expression (`re`).
3.  **Filtering (Stopwords Removal):** Menghapus kata-kata umum yang tidak membawa makna sentimen (seperti "yang", "dan", "di").
4.  **Feature Extraction:** Mengonversi teks bersih menjadi matriks angka menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**.
5.  **Train-Test Split:** Membagi data menjadi 80% untuk pelatihan (Training) dan 20% untuk pengujian (Testing).

---

## 4. Modeling

Model dibangun menggunakan arsitektur klasifikasi berbasis Scikit-Learn. Fitur teks yang telah diekstrak oleh TF-IDF dimasukkan ke dalam algoritma klasifikasi untuk mempelajari pola kata-kata yang berasosiasi dengan sentimen positif maupun negatif.

---

## 5. Evaluation

Model diuji menggunakan 20% data pengujian yang belum pernah dilihat sebelumnya (sekitar 344 sampel). Kinerja model divisualisasikan melalui **Confusion Matrix** <img width="513" height="475" alt="confusion_matrix" src="https://github.com/user-attachments/assets/8c93e1d3-2117-42f6-95eb-50e865a576ae" />

### Analisis Confusion Matrix
Dari heatmap yang dihasilkan, berikut adalah rincian performa model:
* **True Negative (TN) = 215:** Model berhasil menebak 215 ulasan Negatif dengan benar.
* **True Positive (TP) = 93:** Model berhasil menebak 93 ulasan Positif dengan benar.
* **False Negative (FN) = 24:** Sebanyak 24 ulasan Positif salah diprediksi sebagai Negatif.
* **False Positif (FP) = 12:** Sebanyak 12 ulasan Negatif salah diprediksi sebagai Positif.

### Mengapa Hasilnya Demikian?
1.  **Akurasi Tinggi (~89.53%):** Total prediksi benar adalah $215 + 93 = 308$ dari 344 data. Hasil ini menunjukkan kombinasi TF-IDF dan model klasifikasi sangat efektif untuk menangkap bobot kata kunci penting.
2.  **Performa Kelas Negatif Lebih Unggul:** Model jauh lebih sensitif dan akurat dalam mendeteksi ulasan Negatif (hanya 12 kesalahan dari 227 data asli). Hal ini terjadi karena pada tahap *Data Understanding*, data ulasan negatif jumlahnya dua kali lipat lebih banyak, sehingga model mendapatkan lebih banyak variasi kosakata keluhan (seperti *"lemot"*, *"kecewa"*, *"error"*) selama proses training.

---

## 6. Conclusion & Deployment

### Kesimpulan
Proyek ini berhasil mengembangkan model translasi sentimen otomatis dengan akurasi mendekati 90%. Model ini sangat andal dalam menyaring keluhan pengguna, sehingga tim operasional dapat langsung fokus pada ulasan-ulasan negatif yang telah difilter secara otomatis untuk melakukan perbaikan layanan.

### Deployment Simulation
Model dideploy dalam bentuk fungsi inferensi sederhana yang dapat menerima input ulasan baru secara langsung untuk menghasilkan keputusan sentimen secara real-time.
