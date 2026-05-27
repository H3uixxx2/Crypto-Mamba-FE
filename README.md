# CryptoMamba Streamlit UI

Local Streamlit dashboard for the CryptoMamba project. The UI runs on your MacBook and calls a remote Colab model API for live prediction.

## Architecture

```text
MacBook Streamlit UI -> HTTP POST /predict -> Colab FastAPI -> CryptoMamba checkpoint on GPU
```

The Mac does not load the Mamba model and does not need CUDA.

## Setup

```bash
git clone https://github.com/H3uixxx2/Crypto-Mamba-FE.git
cd Crypto-Mamba-FE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```text
CRYPTO_MAMBA_API_URL=https://YOUR-NGROK-URL.ngrok-free.app
```

Optional, if you have the core CryptoMamba repo locally and want the UI to read evaluation artifacts:

```text
CRYPTO_MAMBA_CORE_ROOT=/path/to/CryptoMamba
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open: http://127.0.0.1:8501

## Expected Colab API contract

### `GET /health`

Returns model/server status.

### `POST /predict`

Request:

```json
{
  "prediction_date": "2024-09-17",
  "risk": 2,
  "candles": [
    {"date":"2024-09-03", "open":59000, "high":60000, "low":58000, "close":59500, "volume":30000000000}
  ]
}
```

`candles` must contain at least 14 daily OHLCV rows sorted or sortable by date.
