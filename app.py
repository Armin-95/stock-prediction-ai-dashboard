from datetime import timedelta
from flask import Flask, render_template, request, jsonify, session
import yfinance as yf
from collections import OrderedDict
import joblib
import os
from pathlib import Path
import pandas as pd
import logging
from flask_limiter import Limiter
from google.genai.errors import ServerError

from database.db import get_prediction_daily_bars
from database.ai_chat_db import ChatConversationLimitExceeded
from ml_pipeline.market_data import sync_prediction_daily_data
from ml_pipeline.features import build_features
from services.model_comparison_service import get_or_create_ai_model_comparison_explanation
from services.model_metrics_service import get_all_model_metrics_for_symbol
from services.prediction_explanation_service import get_or_create_ai_prediction_explanation
from services.prediction_service import get_or_create_next_close_predictions
from services.stock_chat_service import (get_or_create_stock_ai_chat_conversation, 
    get_stock_ai_chat_messages_for_display,
    send_stock_ai_chat_message,
    delete_stock_ai_chat_conversation,
    create_new_stock_ai_chat_conversation,
    set_active_stock_ai_chat_conversation,
    list_available_stock_ai_chat_conversations
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

app = Flask(__name__, template_folder="templates", static_folder="static")

secret_key = os.getenv("SECRET_KEY")

if not secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set.")

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
app.config["SECRET_KEY"] = secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True


def get_client_ip():
    if request.access_route:
        return request.access_route[0]
    return request.remote_addr


@app.before_request
def log_request():
    app.logger.info(
        "CLIENT_IP=%s METHOD=%s PATH=%s",
        get_client_ip(),
        request.method,
        request.path
    )
    

@app.before_request
def maintenance_mode():
    if os.environ.get("MAINTENANCE_MODE") == "true":
        return "StockScopeAI is temporarily offline.", 503


limiter = Limiter(
    key_func=get_client_ip,
    app=app,
    default_limits=["40 per day", "20 per hour"]
)


def has_cookie_consent():
    return request.cookies.get("cookie_consent") == "accepted"


COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc.'
}

ALLOWED_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"}

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

@app.route("/privacy-policy")
def privacy_policy():

    return render_template("privacy_policy.html")


@app.route('/', methods=['GET'])
def index():
    """Show form to enter ticker symbol."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per hour; 40 per day")
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
@limiter.limit("20 per hour; 50 per day")
def predict():
    symbol = request.form.get('symbol',"").strip().upper()
    if symbol not in ALLOWED_SYMBOLS:
        return jsonify({"error":"unsupported symbol."}),400
    
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
@limiter.limit("20 per minute; 200 per day")
def ai_prediction_explanation(symbol):
    symbol = symbol.strip().upper()

    if symbol not in ALLOWED_SYMBOLS:
        return jsonify({"error": "Unsupported symbol."}), 400

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
@limiter.limit("20 per minute; 200 per day")
def ai_model_comparison(symbol):
    symbol = symbol.strip().upper()

    if symbol not in ALLOWED_SYMBOLS:
        return jsonify({"error": "Unsupported symbol."}), 400

    ai_model_comparison_explanation, ai_model, ai_provider = get_or_create_ai_model_comparison_explanation(symbol, MODELS)
    
    if not ai_model_comparison_explanation:
        return jsonify({"error": f"No AI model explanation found for {symbol}."}), 404
    
    return jsonify({
        "symbol": symbol,
        "explanation": ai_model_comparison_explanation,
        "response_model": ai_model,
        "response_provider": ai_provider
    }) 


@app.route("/api/stocks/<symbol>/ai-chat/open", methods=["POST"])
@limiter.limit("5 per minute; 50 per day")
def open_stock_ai_chat_conversation(symbol):
    symbol = symbol.strip().upper()
    if symbol not in ALLOWED_SYMBOLS:
        
        return jsonify({"error":"unsupported symbol."}),400
    
    if not has_cookie_consent():

        return jsonify({"error": "Please accept optional cookies to use saved AI chat history."}), 403
    
    try:
        get_or_create_stock_ai_chat_conversation(symbol)
        conversation_messages = get_stock_ai_chat_messages_for_display(symbol, limit=20)

        return jsonify (conversation_messages),200
    
    except Exception as e:
        app.logger.exception("Failed to open stock AI chat for %s.",symbol)

        return jsonify({"error":"Cound not open AI chat"}),500


@app.route("/api/stocks/<symbol>/ai-chat/send-message", methods=["POST"])
@limiter.limit("5 per minute; 50 per day")
def send_message_stock_ai_chat_conversation(symbol):
    symbol = symbol.strip().upper()
    if symbol not in ALLOWED_SYMBOLS:
        
        return jsonify({"error":"unsupported symbol."}),400
    
    if not has_cookie_consent():

        return jsonify({"error": "Please accept optional cookies to use saved AI chat history."}), 403
    
    try:
        data = request.get_json(silent=True) or {}
        user_question = data.get("question")
        ai_response = send_stock_ai_chat_message(symbol = symbol, question = user_question, models = MODELS)
        
        return jsonify(ai_response),200
    
    except ValueError as e:
        return jsonify({"error":str(e)}),400 #return exception defined in send_stock_ai_chat_message for bad input
    
    except ServerError:
        app.logger.warning( "AI provider temporarily unavailable for %s.",symbol)
        return jsonify({"error": "ai_service_unavailable",
            "error_message": "The AI service is temporarily busy. Please try again later."}), 503


    except Exception:
        app.logger.exception("Failed to create AI chat response for %s", symbol) #others exception
        return jsonify({"error":"Could not send AI chat message."}), 500


@app.route("/api/stocks/<symbol>/ai-chat/delete-conversation", methods=["DELETE"])
@limiter.limit("5 per minute; 20 per day")
def delete_user_stock_ai_chat_conversation(symbol):
    symbol = symbol.strip().upper()

    if symbol not in ALLOWED_SYMBOLS:

        return jsonify({"error":"unsupported symbol."}),400
    
    if not has_cookie_consent():

        return jsonify({"error": "Please accept optional cookies to use saved AI chat history."}), 403
    
    try:
        deleted_count = delete_stock_ai_chat_conversation(symbol) # in future add selected_conversation_id if user want to delete other than active conversation
        
        if not deleted_count:
            app.logger.warning("No active AI chat conversation found to delete for %s.", symbol )
            return jsonify({"error":"No active AI chat conversation found."}), 404


        app.logger.info("Deleted conversation for symbol %s.", symbol)
        return jsonify ({"message":"Conversation successfully deleted."}),200


    except Exception:
        app.logger.exception("Failed to delete conversation for %s .", symbol )
        return jsonify({"error":"Could not delete selected AI conversation." }), 500


@app.route("/api/stocks/<symbol>/ai-chat/new-conversation", methods=["POST"])
@limiter.limit("5 per minute; 50 per day")
def create_new_stock_ai_chat_conversation_route(symbol):
    symbol = symbol.strip().upper()

    if symbol not in ALLOWED_SYMBOLS:

        return jsonify({"error":"unsupported symbol."}),400
    
    if not has_cookie_consent():

        return jsonify({"error": "Please accept optional cookies to use saved AI chat history."}), 403

    try:
        new_conversation = create_new_stock_ai_chat_conversation(symbol)

        if new_conversation.get("error")== "conversation_limit_exceeded":
            app.logger.warning("Number of available conversation for symbol: %s exceeded limit.",symbol)
            return jsonify(new_conversation),429 
        
        app.logger.info("Created new AI chat conversation for %s.", symbol)
        return jsonify(new_conversation),201
    
    except Exception:
        app.logger.warning("Failed to create new stock AI conversation for: %s",symbol)
        return jsonify({"error":"Could not create new AI conversation."}),500


@app.route("/api/stocks/<symbol>/ai-chat/conversations/<uuid:conversation_id>/select", methods=["POST"])
@limiter.limit("5 per minute; 50 per day")
def set_active_stock_ai_chat_conversation_route(symbol, conversation_id):
    symbol = symbol.strip().upper()

    if symbol not in ALLOWED_SYMBOLS:
        
        return jsonify({"error":"unsupported symbol."}),400
    
    if not has_cookie_consent():

        return jsonify({"error": "cookie_consent_required", "error_message": "Please acept optional cookies to change AI conversation ."}), 403
    
    selected_conversation_id = str(conversation_id).strip()
    if not selected_conversation_id:

        return jsonify({"error": "invalid_request", "error_message": "A conversation ID is required."}), 400
    
    try:
        new_active_conversation_id = set_active_stock_ai_chat_conversation(symbol,selected_conversation_id)
        
        if not new_active_conversation_id:
            app.logger.warning("Conversation ID %s did not validate, could not be set as active for %s",selected_conversation_id, symbol)

            return jsonify({"error": "conversation_not_found","error_message":"Conversation could not be found for this session and symbol."}),404
        
        return jsonify({"message":" conversation successfully set up as active."}),200

    except Exception:
        app.logger.exception("Failed to set up choosen active stock AI conversation for: %s",symbol)

        return jsonify({"error":"Could not set up as active AI conversation."}),500


@app.route("/api/stocks/<symbol>/ai-chat/conversations", methods=["GET"])
@limiter.limit("5 per minute; 75 per day") 
def list_available_stock_ai_chat_conversations_route(symbol):
    symbol = symbol.strip().upper()

    if symbol not in ALLOWED_SYMBOLS:
        
        return jsonify({"error":"unsupported symbol."}),400
    
    if not has_cookie_consent():

        return jsonify({"error": "cookie_consent_required", "error_message": "Please acept optional cookies to change AI conversation ."}), 403
    
    try:
        available_conversations = list_available_stock_ai_chat_conversations (symbol)
        active_conversation_id = session.get(f"chat_conversation_id_{symbol}")

        return jsonify({"symbol":symbol, "active_conversation_id": active_conversation_id, "conversations":available_conversations})
    
    except Exception:
        app.logger.exception("Failed to list available conversations for: %s", symbol)

        return jsonify({"error": "Could not retrieve available conversations."}),500


@app.route('/api/stock_data')
@limiter.limit("15 per minute")
def stock_data():
    symbol = request.args.get('symbol')
    data = yf.download(symbol, period='2d', interval='5m', auto_adjust=True)
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
