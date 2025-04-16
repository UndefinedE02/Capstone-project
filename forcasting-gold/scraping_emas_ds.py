import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Ticker untuk emas
ticker_emas = 'GC=F'  # Gold Futures

# Rentang waktu data yang diambil
start_date = "2023-01-01"
end_date = "2025-04-01"
interval = "1h"  # Data per 1 jam

# Direktori penyimpanan dataset
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/gold_data_1d.csv"

# Fungsi untuk memproses data
def preprocess_data(data, ticker):
    if data.empty:
        return None

    # Menyaring hanya kolom yang diperlukan
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Menghapus duplikat
    data = data.drop_duplicates()

    # Menambahkan return harian
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)  # Menghapus nilai NaN

    # Menambahkan ticker
    data['Ticker'] = ticker

    return data

# Fungsi untuk mengambil data per 60 hari secara bertahap
def fetch_gold_data():
    all_data = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while start < end:
        next_end = start + timedelta(days=60)  # Maksimum rentang data 60 hari
        if next_end > end:
            next_end = end
        
        print(f"📊 Mengambil data dari {start.date()} hingga {next_end.date()}...")
        try:
            data_emas = yf.download(ticker_emas, start=start.strftime("%Y-%m-%d"), 
                                    end=next_end.strftime("%Y-%m-%d"), interval=interval, auto_adjust=False)
            
            if not data_emas.empty:
                cleaned_data = preprocess_data(data_emas, ticker_emas)
                all_data.append(cleaned_data)
        except Exception as e:
            print(f"❌ Error saat mengambil data emas dari {start.date()} hingga {next_end.date()}: {e}")

        start = next_end  # Pindah ke rentang waktu berikutnya

    # Menggabungkan semua data dan menyimpannya ke file CSV
    if all_data:
        final_data = pd.concat(all_data)
        final_data.to_csv(output_file, index=True)
        print(f"✅ Semua data berhasil disimpan dalam '{output_file}'.")
    else:
        print("⚠️ Tidak ada data yang berhasil diambil.")

# Jalankan proses pengambilan data
fetch_gold_data()