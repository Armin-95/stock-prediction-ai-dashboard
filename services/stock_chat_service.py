from database.ai_chat_db import (create_ai_chat_session, 
    touch_ai_chat_session, 
    create_ai_chat_conversation,
    get_valid_ai_chat_conversation_id,
    insert_ai_chat_message,
    get_ai_chat_messages_for_display,
    get_recent_ai_chat_messages_for_prompt,
    delete_ai_chat_conversation,
    touch_ai_chat_conversation,
    get_last_stock_ai_chat_conversation_id,
    ChatConversationLimitExceeded
)
from database.db import get_next_close_date, get_next_close_prediction,get_prediction_features
from services.model_metrics_service import _filter_available_metrics, get_all_model_metrics_for_symbol
from services.ai.stock_chat_ai import (ask_stock_chat, GEMINI_STOCK_CHAT_MODEL, AI_PROVIDER)

import logging
from flask import session


logger = logging.getLogger(__name__)


def _get_or_create_stock_ai_chat_session():
    session_id = session.get("ai_chat_session_id")

    if session_id is not None:
        valid_session_id = touch_ai_chat_session(session_id) # update last_active_at and expires_at

        if valid_session_id is not None:
            return str(valid_session_id)
        
        else: 
            logger.warning("Session ID did exist locally but cannot be verified in DB, new session will be created. ")
            session.pop("ai_chat_session_id", None)
    
    session_id = create_ai_chat_session()
    session["ai_chat_session_id"]=str(session_id)

    return str(session_id)


def _get_stock_session_conversation_key(symbol): #pointing to active ai_chat_conversation in Flask session
    return f"chat_conversation_id_{symbol.upper()}"

def get_or_create_stock_ai_chat_conversation(symbol):
    symbol = symbol.upper()
    session_id = _get_or_create_stock_ai_chat_session()

    active_conversation_session_key = _get_stock_session_conversation_key(symbol) # active conver. key per symbol 
    existing_conversation_id = session.get(active_conversation_session_key) #active conversation_id from local flask session

    if existing_conversation_id:
        conversation_id = get_valid_ai_chat_conversation_id(existing_conversation_id, session_id, symbol) # checking if realy conversation_id belongs to session_id in DB 

        if conversation_id: #after verified via DB (match: conversation_id, session_id, symbol) 
            
            return str(conversation_id), str(session_id)
        
    #check if exist any conversation_id (select newest) in DB for active session_id
    conversation_id = get_last_stock_ai_chat_conversation_id(symbol, session_id)

    if not conversation_id:
        conversation_id = create_ai_chat_conversation(session_id, symbol)
        
    session[active_conversation_session_key] = str(conversation_id) #store new conversation_id for symbol for this session

    return str(conversation_id), str(session_id)


def set_active_stock_ai_chat_conversation(symbol,selected_conversation_id):
    symbol = symbol.upper()
    session_id = _get_or_create_stock_ai_chat_session()
    active_conversation_session_key = _get_stock_session_conversation_key(symbol)
    conversation_id = get_valid_ai_chat_conversation_id(selected_conversation_id, session_id, symbol)
    
    if conversation_id is None: #doesnt exist or wrong selected_conversation_id 

        return None

    session[active_conversation_session_key] = str(conversation_id)

    return str(conversation_id)

def delete_stock_ai_chat_conversation(symbol, selected_conversation_id = None):
    symbol = symbol.upper()
    session_id = session.get("ai_chat_session_id") #dont call func because if none I dont want to create new
    
    if session_id is None: #no session user cannot delete 
        return None
    
    active_conversation_session_key = _get_stock_session_conversation_key(symbol) # active conver. key per symbol 
    active_conversation_id  = session.get(active_conversation_session_key) 

    
    # Currently delete only open/active chat
    #in the future, selected_conversation_id can come from chat sidebar
    if selected_conversation_id is None:
        selected_conversation_id = session.get(active_conversation_session_key)
        
        if not selected_conversation_id:
            return None
        
    conversation_id = get_valid_ai_chat_conversation_id(selected_conversation_id, session_id, symbol)

    if conversation_id is None: #doesnt exist or wrong selected_conversation_id 

        if active_conversation_id == str(selected_conversation_id):
            session.pop(active_conversation_session_key,None) #check if false conver. ID is stored in session and then remore it drom session
            logger.warning("Invalid active AI chat conversation ID for %s was removed from Flask sesion  ",symbol)
        
        return None
    
    deleted_count = delete_ai_chat_conversation (conversation_id, session_id, symbol)

    if deleted_count > 0: #if succesfull, remove conversation_id also from session if is active

        if  active_conversation_id  == str(conversation_id):
            session.pop(active_conversation_session_key, None)

    return deleted_count


def get_stock_ai_chat_messages_for_display(symbol, limit=20):
    symbol = symbol.upper()
    session_id = session.get("ai_chat_session_id")

    if session_id is None:

        return {
            "conversation_id": None,
            "messages": []
        }
    
    active_conversation_session_key = _get_stock_session_conversation_key(symbol)
    active_conversation_id = session.get(active_conversation_session_key)

    if active_conversation_id is None:

        return {
            "conversation_id": None,
            "messages": []
        }
    
    valid_conversation_id  = get_valid_ai_chat_conversation_id(active_conversation_id, session_id, symbol)

    if valid_conversation_id is None:
        session.pop(active_conversation_session_key, None) #remove invalid conversation from session is is not valid

        return {
            "conversation_id": None,
            "messages": []
        }
    
    messages = get_ai_chat_messages_for_display(valid_conversation_id, limit)

    return {
            "conversation_id": str(valid_conversation_id),
            "messages": messages
    }


def get_stock_ai_chat_messages_for_prompt(symbol, limit=5):
    symbol = symbol.upper()
    session_id = session.get("ai_chat_session_id")

    if session_id is None:

        return {
            "conversation_id": None,
            "messages": []
        }
    
    active_conversation_session_key = _get_stock_session_conversation_key(symbol)
    active_conversation_id = session.get(active_conversation_session_key)

    if active_conversation_id is None:

        return {
            "conversation_id": None,
            "messages": []
        }

    valid_conversation_id  = get_valid_ai_chat_conversation_id(active_conversation_id, session_id, symbol)

    if valid_conversation_id is None:
        session.pop(active_conversation_session_key, None) #remove invalid conversation from session is is not valid

        return {
            "conversation_id": None,
            "messages": []
        }

    messages = get_recent_ai_chat_messages_for_prompt(valid_conversation_id, limit)

    return {
        "conversation_id": str(valid_conversation_id),
        "messages": messages
    }

def create_new_stock_ai_chat_conversation(symbol):
    symbol = symbol.upper()
    session_id = _get_or_create_stock_ai_chat_session()

    try:
        new_conversation_id = create_ai_chat_conversation( session_id=session_id, symbol=symbol)

    except ChatConversationLimitExceeded as e:
        return {
            "conversation_id": None,
            "error": "conversation_limit_exceeded",
            "error_message": str(e) #for user
        }

    active_conversation_session_key = _get_stock_session_conversation_key(symbol)

    session[active_conversation_session_key] = str(new_conversation_id) #set up new conver. as active in session

    return {
        "conversation_id": str(new_conversation_id),
        "messages": [],
        "error": None
    }


def send_stock_ai_chat_message(symbol:str, question:str, models: dict):
    symbol = symbol.upper()

    if not isinstance(question,str): #only text as str allowed
        raise ValueError("Question must be text.")

    question = question.strip() #safety for " " case

    if not question: #dont send empty question from user to DB and AI client
        raise ValueError("Question is required.")
        
    active_conversation_id, active_session_id = get_or_create_stock_ai_chat_conversation(symbol) #points to active conversation

    prompt_history  = get_stock_ai_chat_messages_for_prompt(symbol, limit=5)
    messages_for_prompt = prompt_history.get("messages", [])

    stock_context = build_stock_ai_chat_context(symbol, models)
    
    insert_ai_chat_message(conversation_id = active_conversation_id, role = 'user', content = question, status ="completed", model_name=None , provider=None, error_message=None) #saved user message in db
    
    touch_ai_chat_conversation(active_conversation_id, symbol, active_session_id)
    
    if stock_context is None:
        logger.error("Could not build stock AI chat context for %s.", symbol)
        stock_context = {"warning": "Stock prediction context is currently unavailable. Do not invent prediction values."}

    try:
        ai_response = ask_stock_chat(symbol = symbol,  question = question, stock_context = stock_context, recent_messages = messages_for_prompt )

    except Exception as e:
        #logger.exception("Stock chat AI request failed for %s.", symbol)
        insert_ai_chat_message(conversation_id = active_conversation_id, role = "assistant", content = None, status="failed", model_name= GEMINI_STOCK_CHAT_MODEL, provider = AI_PROVIDER, error_message=str(e))
        raise

    insert_ai_chat_message(conversation_id = active_conversation_id, role = "assistant", content = ai_response.get('content'), status = ai_response.get("status","completed") , model_name= ai_response.get("model_name") , provider = ai_response.get("provider"), error_message=None)

    return ai_response
    

def build_stock_ai_chat_context(symbol: str, models: dict):
    symbol = symbol.upper()

    prediction_date, _ = get_next_close_date(symbol)
    if not prediction_date:
        logger.error("No date for current prediction for %s.", symbol)
        return None  
    
    model_predictions = {}
    model_prediction_features = {}
    models_for_symbol = models.get(symbol, {})

    if not models_for_symbol:
        logger.error("No models for %s. Available tickers: %s", symbol, list(models.keys()))
        return None 
    
    for model_type in models_for_symbol.keys():

        predicted_return, predicted_close = get_next_close_prediction(symbol, prediction_date, model_type)
        model_predictions[model_type]=_clean_prompt_data({
            "predicted_return" : predicted_return, 
            "predicted_close": predicted_close}
        )

        prediction_features = get_prediction_features(symbol, prediction_date, model_type)
        model_prediction_features[model_type] = _clean_prompt_data(prediction_features)

    model_metrics = get_all_model_metrics_for_symbol(symbol, models) # for specific symbol get all models and their metrics 
    model_metrics_for_ai = _filter_available_metrics(model_metrics) # clear from None for prompt 

    
    return {
        "symbol":symbol,
        "prediction_date":str(prediction_date),
        "model_predictions": model_predictions,
        "model_prediction_features":model_prediction_features,
        "model_metrics": model_metrics_for_ai}


def _clean_prompt_data(data: dict | None): #clear None from data in creation for use in AI prompt 
    if not data:
        return {}

    return {
        key: value
        for key, value in data.items()
        if value is not None
    }