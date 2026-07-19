# Mengimpor library pandas untuk membaca dan memanipulasi data berbentuk tabel (dataframe)
import pandas as pd
# Mengimpor library numpy untuk melakukan perhitungan matematika kompleks dan array
import numpy as np
# Mengimpor pyplot dari matplotlib untuk menggambar grafik dasar
import matplotlib.pyplot as plt
# Mengimpor seaborn untuk membuat grafik yang lebih cantik dan berwarna
import seaborn as sns
# Mengimpor os untuk berinteraksi dengan sistem folder di komputer
import os

def prepare_data(train_file, test_file):
    # Membaca file CSV data latih (train) dan menyimpannya ke dalam variabel train_df
    train_df = pd.read_csv(train_file)
    # Membaca file CSV data uji (test) dan menyimpannya ke dalam variabel test_df
    test_df = pd.read_csv(test_file)
    
    # Menghapus baris yang memiliki nilai kosong (NaN) di data latih agar tidak error
    train_df = train_df.dropna().reset_index(drop=True)
    # Menghapus baris yang memiliki nilai kosong (NaN) di data uji agar tidak error
    test_df = test_df.dropna().reset_index(drop=True)
    
    # Mendefinisikan daftar nama kolom yang akan dijadikan faktor penentu (fitur)
    features = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 
                'surface_pressure', 'cloud_cover', 'wind_speed_10m']
    # Mendefinisikan nama kolom yang akan ditebak/diprediksi (target)
    target = 'rain_next_6h'
    
    # Mengambil hanya kolom fitur dari data latih, lalu mengubahnya menjadi matriks angka (array)
    X_train = train_df[features].values
    # Mengambil hanya kolom target dari data latih, lalu mengubahnya menjadi array angka (0 atau 1)
    y_train = train_df[target].values
    # Mengambil hanya kolom fitur dari data uji, lalu mengubahnya menjadi matriks angka (array)
    X_test = test_df[features].values
    # Mengambil hanya kolom target dari data uji, lalu mengubahnya menjadi array angka (0 atau 1)
    y_test = test_df[target].values
    
    # Menghitung nilai rata-rata dari setiap kolom fitur pada data latih
    mean_X = np.mean(X_train, axis=0)
    # Menghitung standar deviasi (sebaran data) dari setiap kolom fitur pada data latih
    std_X = np.std(X_train, axis=0)
    
    # Mengecilkan skala data latih (Standarisasi) agar semua fitur seimbang dan model tidak error
    X_train_scaled = (X_train - mean_X) / std_X
    # Mengecilkan skala data uji menggunakan rata-rata dan standar deviasi dari data latih
    X_test_scaled = (X_test - mean_X) / std_X
    
    # Mengembalikan semua data yang sudah siap digunakan ke program utama
    return train_df, X_train_scaled, y_train, X_test_scaled, y_test, features


class CustomLogisticRegression:
    # Fungsi inisialisasi saat model pertama kali dibuat
    def __init__(self, learning_rate=0.01, iterasi=1000):
        # Menyimpan nilai kecepatan belajar (seberapa besar bobot diubah tiap langkah)
        self.lr = learning_rate
        # Menyimpan jumlah putaran (iterasi) model akan belajar dari data
        self.iterasi = iterasi
        # Menyiapkan tempat kosong untuk menyimpan bobot (pengaruh setiap fitur cuaca)
        self.bobot = None
        # Menyiapkan tempat kosong untuk menyimpan angka bias (angka penyeimbang dasar)
        self.bias = None

    # Fungsi matematika Sigmoid untuk mengubah angka bebas menjadi probabilitas (0 sampai 1)
    def _sigmoid(self, z):
        # Membatasi angka agar tidak lebih dari 250 atau kurang dari -250 (mencegah error komputer)
        z = np.clip(z, -250, 250)
        # Mengembalikan hasil rumus sigmoid
        return 1 / (1 + np.exp(-z))

    # Fungsi untuk melatih model menggunakan data (proses belajar)
    def fit(self, X, y):
        # Mengambil jumlah baris (n_samples) dan jumlah kolom (n_features) dari data
        n_samples, n_features = X.shape
        
        # Memberikan nilai awal 0 untuk semua bobot, sebanyak jumlah fitur cuaca
        self.bobot = np.zeros(n_features)
        # Memberikan nilai awal 0 untuk bias
        self.bias = 0

        # Memulai perulangan belajar sebanyak batas iterasi yang ditentukan
        for _ in range(self.iterasi):
            # Mengalikan data cuaca dengan bobot, lalu ditambah bias (rumus garis lurus)
            model_linear = np.dot(X, self.bobot) + self.bias
            # Mengubah hasil garis lurus menjadi peluang hujan (0% - 100%)
            y_pred = self._sigmoid(model_linear)

            # Menghitung arah error bobot (turunan) dengan membandingkan tebakan dan jawaban asli
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            # Menghitung arah error bias (turunan)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Memperbarui nilai bobot agar tebakan berikutnya lebih akurat
            self.bobot -= self.lr * dw
            # Memperbarui nilai bias
            self.bias -= self.lr * db

    # Fungsi untuk menebak data baru setelah model selesai belajar
    def predict(self, X):
        # Menghitung nilai matematis menggunakan bobot dan bias yang sudah pintar
        model_linear = np.dot(X, self.bobot) + self.bias
        # Mengubah hasil menjadi probabilitas (peluang)
        y_pred = self._sigmoid(model_linear)
        
        # Jika peluang di atas 0.5 (50%), maka tebak Hujan (1), jika tidak tebak Cerah (0)
        y_pred_kelas = [1 if i > 0.5 else 0 for i in y_pred]
        # Mengembalikan kumpulan hasil tebakan dalam format array
        return np.array(y_pred_kelas)


def train_model(X_train, y_train):
    # Membuat "otak" model baru dengan kecepatan belajar 0.1 dan 1000 kali putaran belajar
    model = CustomLogisticRegression(learning_rate=0.1, iterasi=1000)
    # Memasukkan data cuaca dan jawaban aslinya agar model mulai belajar
    model.fit(X_train, y_train)
    # Mengembalikan model yang sudah pintar
    return model


def evaluate_model(y_test, y_pred):
    # Menghitung akurasi: jumlah tebakan benar dibagi total tebakan
    akurasi = np.sum(y_test == y_pred) / len(y_test)
    
    # Menghitung True Positive: Cuaca asli Hujan (1) dan ditebak Hujan (1)
    TP = np.sum((y_test == 1) & (y_pred == 1))
    # Menghitung True Negative: Cuaca asli Cerah (0) dan ditebak Cerah (0)
    TN = np.sum((y_test == 0) & (y_pred == 0))
    # Menghitung False Positive: Cuaca asli Cerah (0) TAPI ditebak Hujan (1)
    FP = np.sum((y_test == 0) & (y_pred == 1))
    # Menghitung False Negative: Cuaca asli Hujan (1) TAPI ditebak Cerah (0)
    FN = np.sum((y_test == 1) & (y_pred == 0))
    
    # Menghitung tingkat Kepastian saat menebak Cerah (mencegah dibagi angka nol)
    prec_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    # Menghitung tingkat Kepekaan menemukan cuaca Cerah
    rec_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # Menghitung tingkat Kepastian saat menebak Hujan
    prec_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Menghitung tingkat Kepekaan menemukan cuaca Hujan
    rec_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Mengembalikan semua angka hasil evaluasi ke program utama
    return akurasi, TP, TN, FP, FN, prec_0, rec_0, prec_1, rec_1


def save_visualizations(df, features, target, output_dir):
    # =========================================================================
    # GRAFIK 1: DIAGRAM BATANG (BAR CHART) - DISTRIBUSI TARGET
    # =========================================================================
    # Membuat kanvas gambar berukuran 8x6 inci untuk grafik pertama
    plt.figure(figsize=(8, 6))
    
    # PERBAIKAN WARNING 1: Menambahkan hue=target dan legend=False
    # Menggambar diagram batang untuk menghitung jumlah Hujan dan Tidak Hujan
    ax = sns.countplot(data=df, x=target, hue=target, palette=['#FF9999', '#66B2FF'], legend=False)
    
    # Menambahkan judul di atas grafik (Dwibahasa)
    plt.title('Proporsi Kejadian Cuaca di Dataset Historis\n'
              '(Weather Event Proportions in Historical Dataset)', 
              fontsize=12, pad=15, fontweight='bold')
    
    # Menambahkan label teks di sumbu X (bawah) (Dwibahasa)
    plt.xlabel('Kondisi Cuaca / Weather Condition\n(0 = Tidak Hujan / No Rain, 1 = Hujan / Rain)', fontsize=11)
    
    # Menambahkan label teks di sumbu Y (samping) (Dwibahasa)
    plt.ylabel('Jumlah Observasi (Jam)\n(Number of Observations in Hours)', fontsize=11)
    
    # Melakukan perulangan untuk setiap batang di dalam grafik
    for p in ax.patches:
        # Menempelkan angka jumlah data persis di pucuk (atas) setiap batang
        ax.annotate(f'{p.get_height():,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Mengatur jarak otomatis agar teks pinggiran tidak terpotong
    plt.tight_layout()
    # Menyimpan gambar grafik pertama ke dalam folder output dengan format PNG
    plt.savefig(os.path.join(output_dir, 'distribusi_target.png'))
    # Menutup kanvas grafik pertama agar memori komputer tidak penuh
    plt.close()
    
    
    # =========================================================================
    # GRAFIK 2: PETA PANAS (HEATMAP) - MATRIKS KORELASI
    # =========================================================================
    # Membuat kanvas gambar baru berukuran 12x10 inci agar muat teks lebih banyak
    plt.figure(figsize=(12, 10))
    
    # Menghitung rumus korelasi antar fitur cuaca (hubungan sebab-akibat antar angka)
    korelasi = df[features + [target]].corr()
    
    # Menggambar peta panas (heatmap) berwarna berdasarkan angka korelasi tadi
    sns.heatmap(korelasi, annot=True, cmap='RdYlBu', fmt=".2f", linewidths=1, 
                vmin=-1, vmax=1, 
                cbar_kws={'label': 'Tingkat Korelasi / Correlation Level'})
    
    # PERBAIKAN WARNING 2: Menghapus emoji bulat merah & biru agar didukung semua komputer
    # Menambahkan judul panjang untuk grafik peta panas (Dwibahasa)
    plt.title('Peta Panas Korelasi Fitur Cuaca vs Peluang Hujan\n'
              '(Heatmap of Weather Features Correlation vs Rain Probability)\n\n'
              '[+] Merah / Red = Meningkatkan Peluang (Increases Probability)\n'
              '[-] Biru / Blue = Menurunkan Peluang (Decreases Probability)', 
              fontsize=13, pad=15, fontweight='bold')
    
    # Memiringkan teks di sumbu X agar tulisan nama fiturnya tidak bertumpuk
    plt.xticks(rotation=45, ha='right', fontsize=10)
    # Memastikan teks di sumbu Y mudah dibaca
    plt.yticks(fontsize=10)
    
    # Mengatur jarak otomatis agar teks pinggiran tidak terpotong saat disimpan
    plt.tight_layout()
    # Menyimpan gambar grafik kedua ke folder output
    plt.savefig(os.path.join(output_dir, 'matriks_korelasi.png'))
    # Menutup kanvas grafik kedua
    plt.close()