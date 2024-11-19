# Energy Efficiency on Building - Wildan Aziz Hidayat
![1_Waste-Energy](https://github.com/user-attachments/assets/fbdd2882-956b-42e6-ab03-526439e95a7b)

## Domain Proyek

Pada bagian ini, kami akan membahas latar belakang terkait efisiensi energi pada bangunan. Penelitian ini bertujuan untuk menilai kebutuhan beban pemanasan dan pendinginan bangunan sebagai fungsi dari parameter bangunan yang berbeda. Efisiensi energi sangat penting untuk mengurangi konsumsi energi dan biaya operasional, serta mengurangi dampak lingkungan.

- Mengatasi masalah ini penting untuk membantu meningkatkan efisiensi energi pada bangunan, yang pada gilirannya dapat mengurangi emisi karbon.
- Menggunakan beberapa model machine learning untuk memprediksi HL (Beban Pemanasan) dan CL (Beban Pendinginan) pada data komponen yang diberikan. Model yang akan digunakan adalah model regresi. Kemudian memilih model dengan error terkecil, beberapa model yang akan digunakan adalah:
a. K-Nearest Neighbors
b. Random Forest
c. XGBoost Regressor

- Memilih model dengan error terkecil untuk digunakan dalam memprediksi efisiensi bangunan yang muncul
- Contoh referensi terkait: ![Predictive Modelling for Heating and Cooling Load Systems of Residential Building](https://ieeexplore.ieee.org/document/10234992) 

## Business Understanding
Seperti yang telah dijelaskan sebelumnya, terkadang bangunan seperti rumah tinggal dapat mempengaruhi lingkungan. Penggunaan model pembelajaran mesin regresi dapat membantu memprediksi energi bangunan yang lebih efisien untuk mengurangi dampak lingkungan. Bagian ini bertujuan untuk menjelaskan masalah bisnis dan tujuan yang ingin dicapai melalui proyek ini.

### Problem Statements

- Bagaimana cara memprediksi kebutuhan beban pemanasan sebuah gedung?
- Bagaimana cara memprediksi kebutuhan beban pendinginan gedung?
- Bagaimana mendapatkan model pembelajaran mesin dengan tingkat kesalahan di bawah 1%?

### Goals

- Memprediksi beban pemanasan untuk meningkatkan efisiensi energi.
- Memprediksi beban pendinginan untuk mengoptimalkan penggunaan energi.
- Berhasil mendapatkan model dengan tingkat kesalahan kurang dari 1%.

### Solution Statements
- Gunakan EDA untuk memahami sifat data dan mengidentifikasi fitur yang memengaruhi Beban Pemanasan dan Beban Pendinginan
- Menggunakan algoritma pembelajaran mesin seperti KNN, Random Forest dan XGBoody untuk memprediksi beban pemanasan dan pendinginan.
- Melakukan penyetelan hyperparameter untuk meningkatkan kinerja model dan memilih model terbaik berdasarkan metrik evaluasi seperti **Root Mean Square Error (RMSE)**.

## Data Understanding

Dataset yang digunakan dalam proyek ini diperoleh dari ![UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency). Dataset ini berisi parameter bangunan yang terkait dengan efisiensi energi.

### Variabel-variabel yang terdapat pada Dataset
Berikut ini adalah variabel-variabel yang terdapat di dalam dataset beserta deskripsinya:
- **X1**: Relative Compactness - Kepadatan relatif bangunan
- **X2**: Surface Area - Luas permukaan bangunan
- **X3**: Wall Area - Luas dinding bangunan
- **X4**: Roof Area - Luas atap bangunan
- **X5**: Overall Heigt - Ketinggian total bangunan
- **X6**: Orientation - Orientasi bangunan
- **X7**: Glazing Area - Luas Distribusi
- **X8**: Glazing Area Distribution - Distribusi area kaca
- **Y1**: Heating Load - Beban pemanasan. Ini sebagai target dataset
- **Y2**: Cooling Load - Beban pendinginan. Ini sebagai target dataset

### Data Information ###
Dataset ini berisi 720 sampel

| # | Column     | Non-Null Count | Dtype  
|---|------------|----------------|---------
| 0 | Relative_Compactness   | 720 non-null       | float64
| 1 | Surface Area   | 720 non-null       | float64
| 2 | Wall_Area   | 720 non-null       | float64
| 3 | Roof_Area   | 720 non-null       | float64
| 4 | Overall_Height   | 720 non-null       | float64
| 5 | Orientation   | 720 non-null       | int64
| 6 | Glazing_Area   | 720 non-null       | float64
| 7 | Glazing_Area_Distribution   | 720 non-null       | int64
| 8 | Heating Load   | 720 non-null       | float64
| 9 | Cooling Load   | 720 non-null       | float64


Informasi missing value

| # | Column     | Missing Value 
|---|------------|----------------
| 0 | Relative_Compactness   | 0 
| 1 | Surface Area   | 0 
| 2 | Wall_Area   | 0 
| 3 | Roof_Area   | 0 
| 4 | Overall_Height   | 0 
| 5 | Orientation   | 0 
| 6 | Glazing_Area   | 48 
| 7 | Glazing_Area_Distribution   | 48 
| 8 | Heating Load   | 0 
| 8 | Cooling Load   | 0 

Pada beberapa komponen seperti *Area Kaca*, *Distribusi Area Kaca*, memiliki nilai 0.
Dalam beberapa kasus, nilai 0 akan dianggap sebagai *nilai yang hilang*.

### Data Visualization
- Univariate Analysis

![Univar](https://github.com/user-attachments/assets/6c0b8566-a99a-490e-8cfb-92793b426ba5)

Figure 1. Univariate Analysis

Dari hasil analisis univariat pada dataset, terlihat bahwa kedua variabel target, yaitu Heating Load dan Cooling Load, memiliki distribusi data yang bervariasi. Heating Load cenderung tersebar dalam rentang 10 hingga 40, dengan pola distribusi yang relatif merata, menunjukkan adanya variasi yang signifikan antar sampel. 

Sementara itu, Cooling Load memiliki rentang antara 15 hingga 45 dengan distribusi yang lebih terfokus pada nilai-nilai di tengah rentang tersebut. Untuk fitur-fitur prediktor, seperti Relative Compactness, Surface Area, Wall Area, Roof Area, dan Overall Height, terlihat bahwa distribusi data cenderung terkonsentrasi pada nilai-nilai tertentu, yang mengindikasikan sifat data diskret atau kategori. Roof Area, misalnya, memiliki sebagian besar data terkonsentrasi pada beberapa kategori spesifik, sedangkan Wall Area menunjukkan distribusi yang lebih bervariasi. Selain itu, fitur Orientation, Glazing Area, dan Glazing Area Distribution juga didominasi oleh nilai-nilai diskret tertentu. 

Secara keseluruhan, karakteristik dataset ini menunjukkan bahwa ada kemungkinan hubungan non-linear antara beberapa fitur dan target, sehingga analisis lanjutan seperti korelasi atau analisis feature importance diperlukan untuk menentukan pengaruh masing-masing fitur terhadap Heating Load dan Cooling Load.

- Multivariate Analysis
![corr](https://github.com/user-attachments/assets/cc82b1fc-fecb-4de1-b00e-d1cb4ac429b1)
Figure 2. Correlation Matrix

Dari correlation matrix yang ditampilkan, dapat disimpulkan bahwa Heating Load memiliki korelasi positif yang sangat kuat dengan Overall Height (nilai korelasi sebesar 0.91). Ini menunjukkan bahwa semakin tinggi bangunan, semakin besar kemungkinan nilai Heating Load akan meningkat. Selain itu, terdapat korelasi positif sedang dengan Relative Compactness (0.64) dan korelasi negatif yang cukup kuat dengan Roof Area (-0.88) dan Surface Area (-0.68). Hal ini menunjukkan bahwa luas atap dan permukaan yang lebih besar cenderung berhubungan dengan penurunan Heating Load.

Fitur seperti Wall Area menunjukkan korelasi positif yang lebih lemah dengan Heating Load (0.46), sementara Glazing Area dan Glazing Area Distribution memiliki korelasi sangat rendah (di bawah 0.21), yang mengindikasikan pengaruhnya terhadap Heating Load kemungkinan kecil. Variabel Orientation juga menunjukkan korelasi hampir nol, sehingga tidak signifikan dalam memengaruhi Heating Load.

Kesimpulannya, fitur yang paling berpengaruh terhadap Heating Load adalah Overall Height, diikuti oleh Roof Area, Surface Area, dan Relative Compactness. Sementara fitur seperti Glazing Area, Glazing Area Distribution, dan Orientation tampaknya tidak memiliki pengaruh yang signifikan. Hal ini memberikan gambaran awal tentang fitur yang perlu lebih diperhatikan dalam membangun model prediksi untuk Heating Load.

## Data Preparation

Langkah-langkah persiapan data dilakukan untuk memastikan dataset bersih, konsisten, dan cocok untuk melatih model machine learning.
Langkah-langkah ini dilakukan dalam Persiapan Data:

- **Pengecekan Missing Value:** Pada data yang dilampirkan bahwa terdapat beberapa kolom yang mengandung nilai 0 seperti *Glazing Area* dan *Glazing Area Distribution*. Biasanya nilai 0 ini mengandung dianggap sebagai nilai yang hilang.
- **Data Cleaning dan Handling Outliers:** Nilai yang hilang ditangani dengan memperhitungkannya dengan mean atau median untuk mencegah hilangnya data, sementara outlier dikelola menggunakan metode seperti pembatasan atau penghapusan untuk meningkatkan ketahanan model. Dalam proyek ini kami menggunakan Metode IQR dengan menghapus outlier 1,5 IQR di bawah Q1 dan 1,5 IQR di atas Q3.
- **Data Splitting:** Dataset dibagi menjadi set pelatihan dan set pengujian (pembagian 80:20) untuk menilai kemampuan generalisasi model dan menghindari overfitting.
- **Feature Scaling:** Sementara Random Forest dan XGBRegressor adalah model berbasis pohon yang tidak sensitif terhadap penskalaan fitur, KNN membutuhkan normalisasi atau standarisasi untuk memastikan semua fitur berkontribusi secara merata pada perhitungan jarak.

## Modelling
Pada bagian ini, beberapa model pembelajaran mesin digunakan untuk memprediksi beban pemanasan dan pendinginan gedung berdasarkan parameter gedung. Berikut ini adalah model-model yang dipilih, alasan pemilihannya, kelebihan dan kekurangannya, serta langkah-langkah penyetelan hiperparameter.

### Model yang Digunakan

#### 1. Random Forest Regressor
Random Forest adalah algoritma ensemble yang menggunakan beberapa pohon keputusan untuk membuat prediksi. Algoritma ini cocok digunakan untuk data yang bersifat non-linear.

**Kelebihan:**
- Baik dalam menangani dataset dengan banyak fitur dan interaksi non-linear.
- Relatif tahan terhadap overfitting pada dataset yang besar.

**Kekurangan:**
- Bisa lambat dan membutuhkan memori yang besar pada set data yang sangat besar.
- Interpretasi model lebih sulit daripada model sederhana.

**Penyetelan Hyperparameter:**
- `n_estimator`: Menentukan jumlah pohon di hutan. Karena dataset ini berukuran kecil, kami menggunakan n_estimator yang besar (n_estimator: 1000)
- `max_depth`: Membatasi kedalaman setiap pohon untuk mengurangi overfitting.
- `min_samples_split` dan `min_samples_leaf`: Mengontrol jumlah minimum sampel untuk membagi sebuah node dan jumlah minimum sampel pada setiap node daun.
- `max_features`: Ini adalah jumlah maksimum fitur yang dapat dicoba oleh Random Forest pada setiap pohon. Dalam proyek ini kami menggunakan 'sqrt' yang mengambil akar kuadrat dari jumlah total fitur dalam setiap proses. Sebagai contoh, jika jumlah total variabel adalah 100, kita hanya dapat mengambil 10 variabel dalam pohon individu.

#### 2. XGBoost Regressor
XGBoost adalah algoritma boosting yang menggabungkan beberapa model yang lemah menjadi model yang kuat secara berulang. Algoritma ini sangat efisien dan sering digunakan dalam kompetisi data.

**Kelebihan:**
- Memiliki kinerja prediksi yang sangat baik, terutama pada data yang kompleks.
- Mendukung regularisasi yang membantu mengurangi overfitting.

**Kekurangan:**
- Membutuhkan penyetelan hiperparameter yang lebih kompleks.
- Dapat memakan waktu komputasi yang lebih lama dibandingkan dengan model lainnya.

**Penyetelan Hiperparameter:**
- `learning_rate`: Mengontrol jumlah langkah yang diambil setiap pohon dalam membuat prediksi. Nilai yang umum adalah 0,01 hingga 0,2.
- `n_estimator`: Menentukan jumlah pohon dalam model.
- `max_depth`, `min_child_weight`: Mengatur kompleksitas model dan mencegah overfitting.
- `eval_metric`: Parameter eval_metric adalah metrik yang digunakan untuk memantau kinerja selama pelatihan dan untuk penghentian awal. . Misalnya 'rmse' untuk root mean square error. Penting untuk memilih metrik yang sesuai untuk masalah yang dihadapi.
- `reg_lambda`: Parameter lambda adalah istilah regularisasi L2 pada bobot. Nilai yang lebih besar berarti model yang lebih konservatif, hal ini membantu untuk mengurangi overfitting dengan menambahkan istilah penalti pada loss function.
- `reg_alpha`; Parameter alpha adalah istilah regularisasi L1 pada bobot. Nilai yang lebih besar berarti model yang lebih konservatif, hal ini membantu untuk mengurangi overfitting dengan menambahkan istilah penalti ke loss function.
- 'subsampel': Parameter subsampel mengontrol fraksi pengamatan yang digunakan untuk setiap pohon. Nilai subsampel yang lebih kecil menghasilkan model yang lebih kecil dan tidak terlalu kompleks, yang dapat membantu mencegah overfitting.
- `colsample_bytree`: Parameter colsample_bytree mengontrol fraksi fitur yang digunakan untuk setiap pohon. Nilai colsample_bytree yang lebih kecil menghasilkan model yang lebih kecil dan tidak terlalu kompleks, yang dapat membantu mencegah overfitting.
- `n_jobs`: Dalam proyek ini kami menggunakan `n_jobs = -1` yang berarti menggunakan semua core CPU yang tersedia untuk komputasi paralel.
- `max_features`: Jumlah total karakteristik unik dalam dataset.

#### 3. Pengklasifikasi K-Nearest Neighbors (KNN)
KNN adalah algoritma yang menggunakan kedekatan data untuk memprediksi nilai berdasarkan data terdekat. Algoritma ini lebih sederhana dan bekerja dengan baik pada data yang kecil.

**Kelebihan:** 
- Mudah dimengerti dan diimplementasikan.
- Tidak memerlukan banyak asumsi tentang data.

**Kekurangan:**
- Performa dapat menurun pada dataset yang besar atau data berdimensi tinggi.
- Sangat terpengaruh oleh penskalaan fitur, sehingga membutuhkan normalisasi atau standarisasi.

**Penyetelan Hyperparameter:**
- `n_neighbors`: Menentukan jumlah tetangga terdekat untuk memprediksi nilai. Biasanya diuji dengan nilai yang berbeda.

### Pemilihan dan Perbaikan Model

Berdasarkan model yang digunakan, penyetelan hyperparameter dilakukan untuk meningkatkan kinerja setiap model. Hasil terbaik dipilih berdasarkan metrik evaluasi yang telah ditetapkan, yaitu **Root Mean Squared Error (RMSE)**.
Setelah dilakukan tuning, **XGBoost Regressor** terpilih sebagai model terbaik, dengan performa sebagai berikut:
- **RMSE Latih:** 0,876174  
- **RMSE Uji:** 0,988739 

Hasil ini menunjukkan bahwa XGBoost memberikan kinerja generalisasi terbaik di antara model yang diuji. RMSE yang rendah pada dataset pelatihan dan pengujian menunjukkan bahwa model ini menangkap pola dalam data secara efektif tanpa overfitting.

## Evaluasi
**Root Mean Squared Error (RMSE) ** digunakan sebagai metrik evaluasi. Hal ini diimplementasikan dengan menggunakan fungsi `mean_squared_error` dari `sklearn`, diikuti dengan menerapkan akar kuadrat dengan `numpy.sqrt()` untuk menghitung nilai RMSE.

RMSE mengukur kesalahan dengan mengkuadratkan perbedaan antara nilai sebenarnya (`y_true`) dan nilai prediksi (`y_pred`), rata-rata perbedaan kuadrat ini, dan kemudian mengambil akar kuadrat.
Rumus untuk RMSE adalah:

<img width="450" alt="RMSE Metrics" src="https://github.com/user-attachments/assets/5c1ea65a-f121-4be4-aaf9-31c10a0eaf2b">

Di mana:  
-  RMSE**: Kesalahan Rata-rata Kuadrat Akar (Root Mean Squared Error)  
- **y**: Nilai aktual  
- **ŷ**: Nilai prediksi  
- **i**: Indeks data  
- **n**: Jumlah titik data  

Metrik ini membantu dalam melatih model dan mengevaluasi kesalahannya secara efektif.

Tabel di bawah ini menunjukkan loss untuk setiap model:

| Model | Train RMSE | Test RMSE |
|-------|------------|-----------|
| KNN   | 1.915688   | 2.169956  |
| RF    | 1.287957   | 1.465254  |
| XGR   | 0.876174   | 0.988739  |

Plot di bawah ini menunjukkan loss untuk setiap model:

![bar_train_test](https://github.com/user-attachments/assets/a41334ba-83a1-42a6-8358-16f6b81d0bd6)


Tabel di bawah ini menunjukkan hasil prediksi untuk setiap model:

| Heating Load (true) | Cooling Load (true) | knn_Heating Load (pred) | knn_Cooling Load (pred) | rf_Heating Load (pred) | rf_Cooling Load (pred) | xgbr_Heating Load (pred) | xgbr_Cooling Load (pred) |
|----------------------|---------------------|--------------------------|--------------------------|-------------------------|-------------------------|--------------------------|--------------------------|
| 11.07               | 14.42              | 10.9                    | 14.4                    | 12.0                   | 15.0                   | 11.200000               | 14.300000               |
| 15.19               | 19.30              | 13.3                    | 16.1                    | 16.0                   | 19.6                   | 14.900000               | 18.900000               |
| 39.72               | 39.80              | 39.9                    | 39.6                    | 38.3                   | 38.1                   | 39.700001               | 38.700001               |
| 40.68               | 40.36              | 39.5                    | 39.1                    | 38.5                   | 38.2                   | 39.799999               | 39.000000               |
| 19.42               | 22.53              | 17.2                    | 19.0                    | 18.3                   | 21.3                   | 19.100000               | 22.100000               |
| 34.24               | 37.26              | 35.7                    | 35.8                    | 33.5                   | 34.6                   | 35.099998               | 36.500000               |
| 36.86               | 37.28              | 36.3                    | 37.2                    | 37.4                   | 37.6                   | 36.900002               | 37.099998               |
| 28.67               | 29.43              | 28.6                    | 29.7                    | 29.1                   | 30.5                   | 28.600000               | 30.000000               |
| 12.77               | 16.22              | 14.3                    | 17.5                    | 12.7                   | 15.7                   | 13.200000               | 16.000000               |
| 36.13               | 37.58              | 36.9                    | 37.1                    | 37.4                   | 37.5                   | 36.900002               | 37.099998               |



Dari data tersebut, terlihat bahwa model regresi dapat memprediksi Beban Pemanasan dan Beban Pendinginan berdasarkan data komponen yang diberikan. Di antara model-model tersebut, model XGBRegressor menonjol dengan tingkat kesalahan kurang dari 1%, yang menunjukkan prediksi yang sangat sesuai dengan nilai aktual, meskipun terkadang terdapat sedikit perbedaan.

Namun demikian, masih ada ruang untuk perbaikan, khususnya dalam proses pemodelan. Meningkatkan kinerja model XGBRegressor melalui penyetelan hiperparameter dapat mengurangi kesalahan lebih lanjut, sehingga memungkinkan prediksi yang lebih akurat untuk beban pemanasan dan beban pendinginan dengan deviasi yang minimal.

## Kesimpulan

Dalam evaluasi model, hasilnya menunjukkan bahwa model XGBRegressor memberikan kinerja terbaik, mencapai RMSE 0,98 pada set data uji. Hal ini menunjukkan bahwa XGBRegressor dapat memprediksi beban pemanasan dan beban pendinginan dengan kesalahan kurang dari 1% dari nilai aktual.

Model lain, seperti Random Forest, juga menghasilkan kesalahan yang relatif rendah, meskipun model KNN menunjukkan kesalahan yang sedikit lebih tinggi sebagai perbandingan. Meskipun demikian, semua model tersebut efektif dalam memprediksi beban pemanasan dan beban pendinginan dengan akurasi yang masuk akal.

Proyek ini berhasil memenuhi tujuannya untuk memprediksi beban pemanasan dan beban pendinginan menggunakan model pembelajaran mesin. Namun, penyetelan hiperparameter lebih lanjut dari XGBRegressor dapat mengurangi kesalahan lebih banyak lagi, memastikan keandalan yang lebih besar dalam prediksi.

# References
##### S. K. Tiwari, J. Kaur, dan R. Kaur, "Predictive Modelling for Heating and Cooling Load Systems of Residential Building," 2024. [Online]. Tersedia: https://ieeexplore.ieee.org/document/10503016/authors#authors.
