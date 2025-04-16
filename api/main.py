from flask import Flask, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

MODEL_DIR = "models"
ALLOWED_INSTRUMENTS = ["gold", "saham"]

def load_model_and_scaler(instrument):
    try:
        if instrument == "gold":
            model_path = os.path.join(MODEL_DIR, "model_gold.h5")
            scaler_path = os.path.join(MODEL_DIR, "scaler_gold.pkl")
            sequence_path = os.path.join(MODEL_DIR, "last_sequence_gold.npy")
        elif instrument == "saham":
            model_path = os.path.join(MODEL_DIR, "model_saham.h5")
            scaler_path = os.path.join(MODEL_DIR, "scaler_ticker_saham.pkl")
            sequence_path = os.path.join(MODEL_DIR, "last_sequence_saham.npy")

        model = load_model(model_path)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        last_sequence = np.load(sequence_path).astype(np.float32)
        last_sequence = last_sequence[-30:]  # pastikan panjang 30

        if instrument == "saham":
            return model, scaler, last_sequence
        else:
            return model, {"gold": scaler}, last_sequence

    except Exception as e:
        raise FileNotFoundError(f"Failed to load model, scaler, or sequence for '{instrument}': {str(e)}")

def predict_future_price(model, last_sequence, scaler_obj, instrument, ticker=None, future_days=180):
    predictions = []

    if instrument == "saham":
        if ticker is None:
            raise ValueError("Ticker must be provided for saham instrument.")
        scaler = scaler_obj.get(ticker)
        if scaler is None:
            raise ValueError(f"Scaler untuk ticker {ticker} tidak ditemukan.")

        n_teknikal = scaler.n_features_in_

        # Jika last_sequence 2D, tambahkan dimensi ke-3
        if len(last_sequence.shape) == 2:
            last_sequence = last_sequence.reshape((last_sequence.shape[0], last_sequence.shape[1]))

        n_total = last_sequence.shape[1]
        n_ticker = n_total - n_teknikal

        for _ in range(future_days):
            input_teknikal = last_sequence[:, :n_teknikal]
            input_onehot = last_sequence[:, n_teknikal:]

            input_teknikal_scaled = scaler.transform(input_teknikal)
            model_input = np.concatenate([input_teknikal_scaled, input_onehot], axis=1)
            model_input = model_input.reshape(1, 30, n_total)

            pred = model.predict(model_input, verbose=0)
            pred = pred.reshape(n_total,).astype(np.float32)

            predictions.append(pred)
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = pred

    else:  # GOLD
        scaler = scaler_obj["gold"]
        n_features = scaler.n_features_in_

        if len(last_sequence.shape) == 2:
            last_sequence = last_sequence.reshape((30, n_features))

        for _ in range(future_days):
            input_scaled = last_sequence.reshape(1, 30, n_features)

            pred = model.predict(input_scaled, verbose=0).reshape(n_features,).astype(np.float32)

            if np.isnan(pred).any():
                raise ValueError("Prediksi model mengandung NaN sebelum inverse_transform.")

            predictions.append(pred)
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = pred

    predictions_arr = np.array(predictions)

    try:
        if instrument == "saham":
            pred_teknikal_only = predictions_arr[:, :n_teknikal]
            pred_teknikal_only = scaler.inverse_transform(pred_teknikal_only)
            predictions_arr[:, :n_teknikal] = pred_teknikal_only
        else:
            predictions_arr = scaler.inverse_transform(predictions_arr)
    except Exception as e:
        raise ValueError(f"Scaler tidak cocok dengan output model: {e}")

    if np.isnan(predictions_arr).any():
        raise ValueError("Hasil prediksi mengandung NaN setelah inverse_transform.")

    harga_close = predictions_arr[:, 3]  # Kolom close, asumsinya kolom ke-4
    return harga_close


def plot_price(predicted_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_prices, label="Prediksi Harga", color="blue")
    plt.title("Prediksi Harga Masa Depan")
    plt.xlabel("Hari ke-")
    plt.ylabel("Harga")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        instrument = data.get("instrument", "").lower()
        modal = float(data.get("modal", 0))
        target_return = float(data.get("target_return", 0))
        duration = int(data.get("duration", 180))
        ticker = data.get("ticker", "")

        if instrument not in ALLOWED_INSTRUMENTS:
            return jsonify({"error": f"Instrumen '{instrument}' tidak dikenali. Gunakan salah satu: {ALLOWED_INSTRUMENTS}"}), 400
        if modal <= 0 or target_return <= 0 or duration <= 0:
            return jsonify({"error": "Modal, target_return, dan duration harus lebih besar dari 0."}), 400
        if instrument == "saham" and not ticker:
            return jsonify({"error": "Ticker harus disediakan untuk instrumen saham."}), 400

        model, scaler_obj, last_sequence = load_model_and_scaler(instrument)

        predicted_prices = predict_future_price(
            model=model,
            last_sequence=last_sequence,
            scaler_obj=scaler_obj,
            instrument=instrument,
            ticker=ticker,
            future_days=duration
        )

        harga_awal = float(predicted_prices[0])
        harga_akhir = float(predicted_prices[-1])

# Pastikan tidak ada pembagian dengan nol atau NaN
        if harga_awal == 0 or np.isnan(harga_awal) or np.isnan(harga_akhir):
            raise ValueError("Harga awal atau akhir tidak valid (NaN atau 0).")
        persentase_return = float(((harga_akhir - harga_awal) / harga_awal) * 100)
        nominal_return = float(modal * (persentase_return / 100))
        total_uang = float(modal + nominal_return)
        meets_target = persentase_return >= target_return

        grafik = plot_price(predicted_prices)

        return jsonify({
            "instrument": instrument,
            "harga_awal": round(harga_awal, 2),
            "harga_akhir": round(harga_akhir, 2),
            "persentase_return": round(persentase_return, 2),
            "nominal_return": round(nominal_return, 2),
            "total_uang": round(total_uang, 2),
            "rekomendasi": "Layak" if meets_target else "Tidak layak",
            "grafik_base64": grafik
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
