# NAMA              : Hensa Katelu
# NIM               : 25.11.6415
# Kelas             : 25 S1 IF 03
# Mata Kuliah       : Pengantar Sains Data
# Dosen Pengampu    : Ferian Fauzi Abdulloh, S.Kom., M.Kom

# Mengimpor library os untuk mengatur file dan folder
import os
# Mengimpor fungsi-fungsi yang sudah dibuat dari file utils.py
from utils import prepare_data, train_model, evaluate_model, save_visualizations

def main():
    # Menyimpan lokasi file dataset untuk proses training
    train_path = 'Dataset/train.csv'
    # Menyimpan lokasi file dataset untuk proses testing
    test_path = 'Dataset/test.csv'
    # Menentukan nama folder tempat menyimpan hasil akhir
    output_dir = 'output'
    
    # Mengecek apakah folder 'output' belum ada di komputer
    if not os.path.exists(output_dir):
        # Jika belum ada, maka buat foldernya secara otomatis
        os.makedirs(output_dir)
        # Menampilkan pesan ke layar bahwa folder berhasil dibuat
        print(f"[*] Membuat direktori penyimpanan '{output_dir}'...")
        
    # Menampilkan pesan bahwa program mulai membaca data
    print("[*] 1/5 - Membaca dan melakukan pra-pemrosesan data cuaca...")
    # Mencoba menjalankan blok kode di bawah (Try)
    try:
        # Memanggil fungsi prepare_data dan mengambil 6 hasil kembaliannya
        train_df, X_train, y_train, X_test, y_test, features = prepare_data(train_path, test_path)
    # Jika gagal karena file CSV tidak ditemukan, maka tangkap errornya
    except FileNotFoundError:
        # Menampilkan pesan peringatan error kepada pengguna
        print("[!] ERROR FATAL: Dataset tidak ditemukan! Pastikan train.csv & test.csv ada di dalam folder 'Dataset/'")
        # Menghentikan seluruh program secara paksa
        return

    # Menampilkan pesan pembuatan grafik ke layar terminal
    print("[*] 2/5 - Menganalisis pola cuaca dan mengekspor grafik (PNG)...")
    # Memanggil fungsi save_visualizations untuk menyimpan 2 gambar PNG
    save_visualizations(train_df, features, 'rain_next_6h', output_dir)
    
    # Menampilkan pesan bahwa proses belajar AI akan dimulai
    print("[*] 3/5 - Memulai proses belajar (Training) Logistic Regression kustom...")
    # Memanggil fungsi train_model dan menyimpan otak buatan ke variabel 'model'
    model = train_model(X_train, y_train)
    
    # Menampilkan pesan bahwa model akan dites
    print("[*] 4/5 - Menguji ketepatan model pada data Test...")
    # Menyuruh model untuk menebak data ujian (X_test) yang belum pernah ia lihat
    y_pred = model.predict(X_test)
    # Menghitung hasil raport ujian model dengan membandingkan tebakan dan jawaban asli (y_test)
    akurasi, TP, TN, FP, FN, prec_0, rec_0, prec_1, rec_1 = evaluate_model(y_test, y_pred)
    
    # Menampilkan pesan bahwa program akan mencetak teks laporan
    print("[*] 5/5 - Menyusun laporan hasil akhir ke dalam file teks...")
    
    # Membuat path lengkap untuk file teks (output/hasil_analisis.txt)
    laporan_path = os.path.join(output_dir, 'hasil_analisis.txt')
    # Membuka file tersebut dengan mode 'w' (write/tulis). Jika belum ada, akan otomatis dibuat.
    with open(laporan_path, 'w') as f:
        # Menulis garis pembatas atas ke dalam file
        f.write("=" * 60 + "\n")
        # Menulis judul laporan
        f.write("      LAPORAN ANALISIS PREDIKSI CUACA (MACHINE LEARNING)      \n")
        # Menulis garis pembatas bawah judul
        f.write("=" * 60 + "\n\n")
        
        # Menulis teks bagian 1 (Informasi Dataset)
        f.write("1. INFORMASI DATASET:\n")
        # Mencetak jumlah baris data yang digunakan untuk belajar (format dengan koma)
        f.write(f"   - Total Data Latih (Belajar)   : {len(X_train):,} observasi\n")
        # Mencetak jumlah baris data yang digunakan untuk ujian
        f.write(f"   - Total Data Uji (Tes)         : {len(X_test):,} observasi\n")
        # Mencetak nama algoritma
        f.write(f"   - Algoritma yang Digunakan     : Custom Logistic Regression (Numpy Pure)\n\n")
        
        # Menulis teks bagian 2 (Kinerja Utama)
        f.write("2. KINERJA UTAMA:\n")
        # Mencetak nilai akurasi dalam bentuk persentase (dikali 100) dengan 2 angka di belakang koma
        f.write(f"   - TINGKAT AKURASI KESELURUHAN  : {akurasi * 100:.2f}%\n")
        # Mencetak penjelasan bahasa manusia dari nilai akurasi tersebut
        f.write(f"     (Artinya model menjawab benar {akurasi * 100:.2f} kali dari setiap 100 tebakan)\n\n")
        
        # Menulis teks bagian 3 (Rincian Tebakan)
        f.write("3. RINCIAN KETEPATAN (CONFUSION MATRIX):\n")
        # Mencetak jumlah tebakan jitu saat tidak hujan
        f.write(f"   - Tebakan BENAR Tidak Hujan (TN): {TN:,} kali\n")
        # Mencetak jumlah tebakan jitu saat hujan
        f.write(f"   - Tebakan BENAR Turun Hujan (TP): {TP:,} kali\n")
        # Mencetak jumlah peringatan palsu
        f.write(f"   - Salah Alarm (Prediksi Hujan, padahal Cerah) (FP) : {FP:,} kali\n")
        # Mencetak jumlah luput/kecolongan
        f.write(f"   - Kecolongan (Prediksi Cerah, padahal Hujan) (FN)  : {FN:,} kali\n\n")
        
        # Menulis teks bagian 4 (Analisis mendalam)
        f.write("4. ANALISIS PER KELAS:\n")
        # Subjudul kondisi cerah
        f.write("   A. Kondisi Cuaca Cerah (0):\n")
        # Mencetak angka precision kondisi cerah
        f.write(f"      - Kepastian (Precision) : {prec_0 * 100:.2f}% (Jika model bilang cerah, seberapa sering ia benar?)\n")
        # Mencetak angka recall kondisi cerah
        f.write(f"      - Kepekaan (Recall)     : {rec_0 * 100:.2f}% (Dari semua waktu cerah, berapa persen yang terdeteksi?)\n\n")
        
        # Subjudul kondisi hujan
        f.write("   B. Kondisi Cuaca Hujan (1):\n")
        # Mencetak angka precision kondisi hujan
        f.write(f"      - Kepastian (Precision) : {prec_1 * 100:.2f}%\n")
        # Mencetak angka recall kondisi hujan
        f.write(f"      - Kepekaan (Recall)     : {rec_1 * 100:.2f}%\n\n")
        
        # Menulis teks bagian 5 (Kesimpulan Grafik)
        f.write("5. KESIMPULAN DARI GRAFIK (Silakan cek gambar di folder):\n")
        # Panduan membaca gambar heatmap
        f.write("   - Pada 'matriks_korelasi.png', perhatikan angka pada baris/kolom 'rain_next_6h'.\n")
        # Penjelasan warna merah di grafik
        f.write("   - Nilai positif (merah) tertinggi menunjukkan faktor paling kuat pemicu hujan (biasanya kelembapan atau awan).\n")
        # Penjelasan warna biru di grafik
        f.write("   - Nilai negatif (biru) menunjukkan faktor penghambat hujan.\n")
        
    # Menampilkan pesan di terminal bahwa semua tugas telah selesai dan berhasil
    print(f"\n[+] PROSES BERHASIL! Seluruh hasil telah disimpan di folder '{output_dir}'.")

# Mengecek apakah file ini dieksekusi secara langsung (bukan di-import dari file lain)
if __name__ == "__main__":
    # Jika ya, maka jalankan fungsi main() di atas
    main()