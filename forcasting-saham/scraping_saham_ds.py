import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

ticker_saham = ['AAPL', 'MSFT', 'GOOGL', 'META', 'BBCA.JK', 'TLKM.JK']
start_date = "2023-01-01"
end_date = "2025-04-01"
interval = "1h"
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/saham_data_1h.csv"

def preprocess_data(data, ticker):
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)  # <<< tambahkan baris ini
    data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]  # pastikan 'Datetime' disimpan
    data['Return'] = data['Close'].pct_change()
    data['Ticker'] = ticker
    data.dropna(inplace=True)

    return data


def fetch_data():
    all_data = []
    for ticker in ticker_saham:
        print(f"\n📥 Mengambil data untuk {ticker}...")
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while start < end:
            next_end = start + timedelta(days=60)
            if next_end > end:
                next_end = end

            try:
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=next_end.strftime("%Y-%m-%d"),
                    interval=interval,
                    group_by="column",
                    progress=False
                )

                if not df.empty:
                    df_clean = preprocess_data(df, ticker)
                    if df_clean is not None:
                        all_data.append(df_clean)

            except Exception as e:
                print(f"❌ Gagal ambil {ticker} dari {start} sampai {next_end}: {e}")
            start = next_end

    if all_data:
        final_data = pd.concat(all_data)
        final_data.to_csv(output_file, index=False)
        print(f"\n✅ Data bersih disimpan di: {output_file}")
    else:
        print("⚠️ Tidak ada data yang berhasil diambil.")

# Jalankan
fetch_data()
