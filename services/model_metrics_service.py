from database.db import get_model_metrics
import logging

logger = logging.getLogger(__name__)



def get_all_model_metrics_for_symbol(symbol: str, models: dict):
    models_for_symbol = models.get(symbol, {})

    if not models_for_symbol:
        logger.error("No models for %s. Available tickers: %s", symbol, list(models.keys()))
        return None
    
    model_metrics ={model_type: get_model_metrics(symbol, model_type)
                    for model_type in models_for_symbol}
    
    if not model_metrics:
        logger.error("No model metrics for symbol: %s and model types: %s", symbol, list(models_for_symbol.keys()))
        return None
    
    return model_metrics


#clear None values for AI explanation
def _filter_available_metrics(model_metrics: dict | None):
    if not model_metrics:
        logger.error("No model metrics has been sent.")
        return {}
    
    cleaned_model_metrics = {}
    for model_type, metric_type in model_metrics.items():
        if not model_type:
            continue

        cleaned_metrics={}
        if metric_type:

            for metric, value in metric_type.items():
                if value is not None:
                  cleaned_metrics[metric]  = value

        if cleaned_metrics:
            cleaned_model_metrics[model_type] = cleaned_metrics 

    return cleaned_model_metrics















