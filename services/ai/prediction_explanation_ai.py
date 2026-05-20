from services.ai.ai_client import get_gemini_client
import json
import logging

logger = logging.getLogger(__name__)

def _format_model_predictions_for_prompt(model_predictions: dict):
    prompt_lines = []

    for model_type, prediction in model_predictions.items():
        predicted_close = prediction.get("predicted_close")
        predicted_return = prediction.get("predicted_return")

        if predicted_close is None or predicted_return is None:
            prompt_lines.append(f"{model_type}: prediction is missing.")
            continue

        prompt_lines.append(f"- {model_type}: predicted next close = {predicted_close}, predicted return = {predicted_return}")

    return "\n".join(prompt_lines)




def explain_prediction_gemini(symbol:str, model_prediction:dict, latest_features: dict[dict], model_metrics: dict | None):
    client = get_gemini_client()

    predictions_text = _format_model_predictions_for_prompt(model_prediction)
    features_text = json.dumps(latest_features, indent=2, default=str)
    metrics_text = json.dumps(model_metrics, indent=2, default=str) if model_metrics else "No metrics provided."

    prompt = f"""
    You are explaining stock price predictions from multiple machine learning models to a beginner user.

    Stock symbol:
    {symbol}

    Model predictions (predicted close and predicted return):
    {predictions_text}

    Important meaning:
    - predicted next close = estimated next closing price.
    - predicted return = expected percentage/log return predicted by the model.
    - Positive predicted return means the model predicts an increase.
    - Negative predicted return means the model predicts a decrease.
    - If predicted return is close to zero, the signal is weak.

    Latest feature values used by each model:
    {features_text}

    Model quality metrics:
    {metrics_text}

    Task:
    Explain the model predictions in simple language.

    Rules:
    - Use 4 to 5 sentences maximum.
    - Use beginner-friendly language.
    - Do not give financial advice.
    - Do not tell the user to buy, sell, or hold the stock.
    - Mention the predicted next close and predicted return for each model.
    - Explain whether each model predicts an increase or decrease.
    - If models disagree, say the signal is mixed.
    - If returns are close to zero, say the signal is small.
    - Mention 1 to 3 important feature signals, such as RSI, moving average ratio, volatility, price range, volume z-score, or day of week.
    - If model metrics are provided, briefly mention reliability using RMSE, MAE, hit ratio, or Sharpe ratio.
    - Highlight the most important features or metrics with bold text.
    - End by saying this is a model-based estimate and can be wrong.
    """

    response =client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt)
    
    if response is None:
        logger.error("Gemini response object is None for symbol %s", symbol)
        return None

    if not response.text or not response.text.strip():
        logger.error("Gemini response text is empty for symbol %s", symbol)
        return None
    
    return {"ai_response":response.text.strip(),
             "response_id":response.response_id,
             "ai_model":response.model_version,
             "ai_provider": "google_gemini"}