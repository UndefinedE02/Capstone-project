import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "instrument": "gold",
    "modal": 10000000,
    "target_return": 10,
    "duration": 180,
    "ticker": ""
}

# jika intrument gold makan ticker tidak perlu di isi
# list ticker = 'AAPL', 'MSFT', 'GOOGL', 'META', 'BBCA.JK', 'TLKM.JK'

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Return:", data['persentase_return'], "%")
    print("Rekomendasi:", data['rekomendasi'])
else:
    print("Error:", response.text)
