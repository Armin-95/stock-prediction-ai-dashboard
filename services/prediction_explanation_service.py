import hashlib
import json
import logging
from database.ai_db import get_ai_prediction_explanations, insert_ai_prediction_explanations
from database.db import get_next_close_date, get_next_close_prediction, get_prediction_features
from services.ai.prediction_explanation_ai import explain_prediction_gemini
from services.model_metrics_service import get_all_model_metrics_for_symbol 

logger = logging.getLogger(__name__)

def _create_prediction_hash(model_predictions: dict, model_prediction_features: dict, model_metrics: dict):
    hash_data = {
        'model_predictions' :model_predictions,
        'model_prediction_features' : model_prediction_features,
        'model_metrics' : model_metrics
    }

    metrics_json = json.dumps(
        hash_data,
        sort_keys=True,
        separators=(",", ":"),
        default=str )
    return hashlib.sha256(metrics_json.encode("utf-8")).hexdigest()
    


def get_or_create_ai_prediction_explanation(symbol: str, models: dict):
    
    prediction_date, close_date_time = get_next_close_date(symbol)
    if not prediction_date:
        logger.error(f'No date for current prediction for {symbol}.')
        return None, None, None 

    ai_prediction_explanation = get_ai_prediction_explanations(symbol, prediction_date)

    if ai_prediction_explanation and ai_prediction_explanation.get("explanation"):
        ai_explanation = ai_prediction_explanation.get("explanation")
        ai_model = ai_prediction_explanation.get("ai_model")
        ai_provider = ai_prediction_explanation.get("ai_provider")
        return ai_explanation, ai_model, ai_provider

    model_predictions = {}
    model_prediction_features = {}
    models_for_symbol = models.get(symbol, {})

    if not models_for_symbol:
        logger.error("No models for %s. Available tickers: %s", symbol, list(models.keys()))
        return None, None, None
    
    for model_type in models_for_symbol.keys():

        predicted_return, predicted_close = get_next_close_prediction(symbol, prediction_date, model_type)
        model_predictions[model_type]= {
            "predicted_return" : predicted_return, 
            "predicted_close": predicted_close}

        prediction_features = get_prediction_features(symbol, prediction_date, model_type)
        model_prediction_features[model_type] = prediction_features

    model_metrics = get_all_model_metrics_for_symbol(symbol, models)

    ai_prediction_explanation = explain_prediction_gemini(symbol, model_predictions, model_prediction_features, model_metrics) #{"explanation": ...,"ai_model":..., "ai_provider":... }

    if not ai_prediction_explanation or not ai_prediction_explanation.get("ai_response"):
        logger.error(f'Failed to create AI prediction explanation for {symbol} and {prediction_date}.')
        return None, None, None
    
    prediction_hash = _create_prediction_hash(model_predictions, model_prediction_features, model_metrics)

    ai_explanation = ai_prediction_explanation.get("ai_response")
    ai_model = ai_prediction_explanation.get("ai_model")
    ai_provider = ai_prediction_explanation.get("ai_provider")
    ai_response_id = ai_prediction_explanation.get("response_id")

    insert_ai_prediction_explanations (symbol, prediction_date, prediction_hash, ai_explanation, ai_model, ai_provider, ai_response_id)

    
    return ai_explanation, ai_model, ai_provider