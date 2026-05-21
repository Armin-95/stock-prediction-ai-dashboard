from flask import Flask, render_template, request, jsonify
import yfinance as yf
from collections import OrderedDict
import joblib
import os
from pathlib import Path
import pandas as pd  
from database.db import get_prediction_daily_bars
from ml_pipeline.market_data import sync_prediction_daily_data
from ml_pipeline.features import build_features
from services.model_comparison_service import get_or_create_ai_model_comparison_explanation
from services.model_metrics_service import get_all_model_metrics_for_symbol
from services.prediction_explanation_service import get_or_create_ai_prediction_explanation
from services.prediction_service import get_or_create_next_close_predictions
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

app = Flask(__name__, template_folder="templates", static_folder="static")

COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc.'
}

# Load ML model once at startup
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# Load every .joblib in MODELS_DIR once at startup
MODELS = {} 
for fname in os.listdir(MODELS_DIR): 
    if fname.lower().endswith(".joblib") and "_" in fname:
        model_type, symbol = os.path.splitext(fname)[0].split("_", 1)
        symbol = symbol.upper()
        model_type = model_type.lower()
        MODELS.setdefault(symbol, {})[model_type] = joblib.load(os.path.join(MODELS_DIR, fname))   #Structure {"AAPL": {"xgboost": <loaded xgboost model object>,"ridge": <loaded ridge model object>},...
if not MODELS:
    raise RuntimeError(f"No .joblib models found in {MODELS_DIR}")

# analyse stored data for 10 tickers - fetch_data   s
CACHE = OrderedDict()
CACHE_MAXSIZE = 10

def fetch_data(symbol: str):
    """Return 1y of data with simple daily cache refresh (max size enforced)."""
    symbol = symbol.upper()
    #
    # What is the latest closed trading day?
    recent = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
    if recent.empty:
        raise RuntimeError(f"No data for {symbol}")
    latest_closed = recent.index[-1].date()

    # If cached and fresh, reuse
    if symbol in CACHE:
        df, last_date = CACHE[symbol]
        if last_date == latest_closed:
            # Move to end (most recently used)
            CACHE.move_to_end(symbol)
            return df

    # Otherwise fetch fresh
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    CACHE[symbol] = (df, df.index[-1].date())
    CACHE.move_to_end(symbol)

    # Enforce max size (drop oldest if too big)
    if len(CACHE) > CACHE_MAXSIZE:
        CACHE.popitem(last=False)

    return df

@app.route('/', methods=['GET'])
def index():
    """Show form to enter ticker symbol."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Fetch data, compute stats, and render chart page."""
    symbol = request.form['symbol'].upper()
    df = fetch_data(symbol)

    dates = df.index.strftime('%Y-%m-%d').tolist()
    closes = df['Close',symbol].tolist()
    volumes = df['Volume',symbol].tolist()
    ma20 = df['MA20'].fillna(method='bfill').tolist()
    ma50 = df['MA50'].fillna(method='bfill').tolist()

    mean_price = round(df['Close',symbol].mean(), 2)
    mean_price_month = round(df['Close',symbol].tail(30).mean(), 2)                                                                                                                                                                                                                                                                                                                                                                                     
    volatility = round(df['Close',symbol].pct_change().std() * 100, 2)
    volatility_month = round(df['Close',symbol].tail(30).pct_change().std() * 100, 2)

    return render_template(
        'analysis.html',
        symbol=symbol,
        dates=dates, closes=closes,
        volumes=volumes, ma20=ma20, ma50=ma50,
        mean_price=mean_price, mean_price_month=mean_price_month, 
        volatility=volatility, volatility_month=volatility_month
    )

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol'].upper()
    company_name = COMPANY_NAMES.get(symbol, symbol)

    #check and if needed sync data for prediction (checks/populates: stock_daily_bars_prediction, market_calendar tables in db)
    sync_prediction_daily_data(symbol)

    # get data from DB after is up to date, then do feature engineering and model prediction
    df = get_prediction_daily_bars(symbol)
    
    #all features for prediction for all models (xgboost, ridge...) 
    df_features = build_features(df,symbol)


    ## Model select and model prediction

    #data for predict.html page
    prices = df_features['close'].ffill().values.tolist()
    times = pd.to_datetime(df_features["trading_date"]).dt.strftime('%Y-%m-%d').tolist()    

    # get all models for this ticker
    models_for_symbol = MODELS.get(symbol, {})
    if not models_for_symbol:
        return f"No models for {symbol}. Available tickers: {list(MODELS.keys())}", 400

    # create dict result (each model type for selected symbol) from db or create and insert values in db,  results= {"xgboost": predicted_return, predicted_close,...}
    results , predict_trading_date= get_or_create_next_close_predictions(symbol, models_for_symbol, prices, df_features)

    if not results or not predict_trading_date:
        logging.error(f"Failed to get predictions for {symbol}.")

    #get metrics for the last trained model of this ticker and model type (xgboost, ridge, lstm...) from DB
    model_metrics = get_all_model_metrics_for_symbol(symbol, MODELS)
    #structure {"xgboost": {"model_metrics_quality": {"mae": ..., "rmse": ..., "hit_ratio": ..., "corrcoef": ...}, "model_strategy_quality": {"strategy_mean": ..., "strategy_std": ..., "sharpe": ..., "total_return": ..., "max_loss": ..., "max_drawdown": ...}}}
    if not model_metrics:
        logging.error(f"No model metrics found for {symbol}.")

    # Append prediction
    prices_last_month = prices[-30:]  
    times_last_month = times[-30:]


    return render_template(
        'predict.html',
        company_name=company_name,
        symbol=symbol,
        times=times_last_month,
        prices=prices_last_month,
        results=results,
        model_metrics=model_metrics,
        predict_trading_date=predict_trading_date
    )

@app.route("/api/stocks/<symbol>/ai-prediction-explanation", methods=["POST"])
def ai_prediction_explanation(symbol):
    symbol =symbol.upper()

    ai_explanation, ai_model, ai_provider = get_or_create_ai_prediction_explanation(symbol, MODELS)

    if not ai_explanation:
        return jsonify({"error": f"No AI prediction explanation found for {symbol}."}), 404
    
    return jsonify({
        "symbol": symbol,
        "explanation": ai_explanation,
        "response_model": ai_model,
        "response_provider": ai_provider
    }) 


@app.route("/api/stocks/<symbol>/ai-model-comparison", methods=["POST"])
def ai_model_comparison(symbol):
    symbol =symbol.upper()

    ai_model_comparison_explanation, ai_model, ai_provider = get_or_create_ai_model_comparison_explanation(symbol, MODELS)
    
    if not ai_model_comparison_explanation:
        return jsonify({"error": f"No AI model explanation found for {symbol}."}), 404
    
    return jsonify({
        "symbol": symbol,
        "explanation": ai_model_comparison_explanation,
        "response_model": ai_model,
        "response_provider": ai_provider
    }) 


@app.route('/api/stock_data')
def stock_data():
    symbol = request.args.get('symbol')
    data = yf.download(symbol, period='2d', interval='5m', auto_adjust=True)
    # time zone Convert to Central European Summer Time
    idx = data.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    # Convert to UTC (standard)
    idx = idx.tz_convert("UTC")

    data = data.set_index(idx)
    close_series = data[('Close', symbol)]
    prices = close_series.ffill().values.tolist()

    # Send ISO format timestamps 
    times = data.index.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
    return jsonify({'times': times, 'prices': prices})


if __name__ == '__main__':
    pass
