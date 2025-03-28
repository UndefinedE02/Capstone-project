import yfinance as yf
import pandas as pd

# Ticker untuk emas
ticker_emas = 'GC=F'  # Gold Futures (Harga emas dunia)

# Rentang waktu data yang diambil
start_date = "2000-01-01"
end_date = "2025-01-01"

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
    
    # Menstandarisasi format tanggal
    data.index = pd.to_datetime(data.index)
    
    # Menambahkan ticker instrumen
    data['Ticker'] = ticker
    
    return data

# Mengambil data emas
try:
    data_emas = yf.download(ticker_emas, start=start_date, end=end_date, auto_adjust=False)
    cleaned_data_emas = preprocess_data(data_emas, ticker_emas)
    
    if cleaned_data_emas is not None:
        cleaned_data_emas.to_csv("dataset/gold_data_cleaned.csv", index=True)
        print("Data emas berhasil disimpan dalam 'dataset/gold_data_cleaned.csv'.")
    else:
        print("Data emas kosong atau tidak tersedia.")
except Exception as e:
    print(f"Error saat mengambil data emas: {e}")
