import os
import re
import sys
 
import joblib
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, send_from_directory
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
 
TRAINED_TICKER = "AAPL"  # the stock the models were trained on
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Prev_Close", "MA_5", "MA_10", "MA_20",
    "Daily_Change", "Daily_Pct_Change",
]
 
app = Flask(__name__)
 
# ---- Load trained models once at startup ----
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
except FileNotFoundError as e:
    sys.exit(f"Model file missing: {e.filename}\n"
             "Copy the models/ folder saved by the notebook next to app.py.")
 
 
def build_latest_features(ticker):
    """Download recent data and build the same features used in training.
    Returns (latest_row, recent_closes). Only past data is used."""
    data = yf.download(ticker, period="6mo", auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")
 
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
 
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop_duplicates().sort_values("Date").reset_index(drop=True)
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].ffill().dropna()
 
    data["Prev_Close"] = data["Close"].shift(1)
    for window in (5, 10, 20):
        data[f"MA_{window}"] = data["Close"].rolling(window=window).mean()
    data["Daily_Change"] = data["Close"] - data["Prev_Close"]
    data["Daily_Pct_Change"] = data["Close"].pct_change() * 100
 
    latest = data.iloc[[-1]]
    if latest[FEATURES].isnull().any(axis=None):
        raise ValueError("Not enough history to build the features for this ticker.")
 
    recent = data.tail(30)
    return latest, recent
 
 
def predict_with(model, latest, use_scaling):
    X_new = latest[FEATURES]
    if use_scaling:
        X_new = scaler.transform(X_new)
    return float(model.predict(X_new)[0])
 
 
@app.route("/")
def index():
    return render_template("index.html", trained_ticker=TRAINED_TICKER)
 
 
@app.route("/api/results")
def api_results():
    path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if not os.path.exists(path):
        return jsonify(error="outputs/model_comparison.csv not found. Run the notebook first."), 404
    df = pd.read_csv(path)
    return jsonify(rows=df.to_dict(orient="records"))
 
 
@app.route("/api/predict")
def api_predict():
    ticker = request.args.get("ticker", TRAINED_TICKER).strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", ticker):
        return jsonify(error="Enter a valid ticker symbol, for example AAPL."), 400
 
    try:
        latest, recent = build_latest_features(ticker)
    except Exception as e:
        return jsonify(error=str(e)), 400
 
    last_close = float(latest["Close"].iloc[0])
    predictions = {}
    for key, model, scaled in (
        ("linear_regression", lr_model, True),
        ("random_forest", rf_model, False),
    ):
        pred = predict_with(model, latest, scaled)
        predictions[key] = {
            "predicted": pred,
            "diff": pred - last_close,
            "pct": (pred - last_close) / last_close * 100,
        }
 
    return jsonify(
        ticker=ticker,
        date=latest["Date"].iloc[0].strftime("%Y-%m-%d"),
        latest_close=last_close,
        predictions=predictions,
        recent_closes=[round(float(c), 2) for c in recent["Close"]],
        trained_ticker=TRAINED_TICKER,
        off_distribution=(ticker != TRAINED_TICKER),
    )
 
 
@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)
 
 
if __name__ == "__main__":
    app.run(debug=False)

