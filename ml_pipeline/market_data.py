from datetime import datetime, timedelta, timezone
import time
import pandas as pd
import yfinance as yf
from database.db import get_latest_available_close_datetime, get_latest_stored_prediction_bar_date, upsert_prediction_daily_bars
from database.populate_calendar import main as populate_calendar
import logging

logger = logging.getLogger(__name__)

def _get_available_close_datetimes(symbol, utc_now):
    latest_close_dt, older_25_close_dt  = get_latest_available_close_datetime(symbol, utc_now)
    if latest_close_dt is None: # Populate calendar if no data available for this ticker, then re-check available close datetimes after calendar is populated
        populate_calendar()
        latest_close_dt, older_25_close_dt  = get_latest_available_close_datetime(symbol, utc_now)

    return latest_close_dt, older_25_close_dt


def _determine_download_range(latest_close_dt, older_25_close_dt):
    if latest_close_dt is None or older_25_close_dt is None:
        return None, None
    
    start_date = older_25_close_dt.date() # I need only date to set up YF download start date (not datetime)
    end_date = latest_close_dt.date() + timedelta(days=1) #end= in YF download is exclusive, Add 1 day to include latest_close_dt date
    
    return start_date, end_date


def _download_daily_data_with_retry(symbol, start_date, end_date, retries = 3): #retry safety if yf for some reason (too many req) fetch invalid data 
    if start_date is None or end_date is None or start_date >= end_date:
        return None
    
    df = None

    for attempt in range (1, retries + 1):
        logger.info("Downloading %s from yahoo, attempt: %s, start date: %s, end date: %s", symbol, attempt, start_date, end_date)

        df = yf.download(symbol, start=start_date ,end=end_date, interval="1d", auto_adjust =True, progress=False, threads=False)
     
        if df is None or df.empty or df.isnull().values.any():
            logger.warning("Fetched invalid/incomplete data from YF for %s", symbol)
            time.sleep(attempt * 2)
            continue
        
        return df

    if df is not None and not df.empty and df.isnull().values.any():#work with df with Null values, clear the df
        df_incomplete = df.dropna()
        if not df_incomplete.empty: #if cleared df has still any rows
            logger.warning("Could not retrieve all existing data for %s, prediction is calculated to the last available close.",symbol)
            
            return df_incomplete 


    logger.error("Could not retrieve correct data for %s after %s attempts.", symbol, retries)
    logger.warning("Prediction is calculated from the last available close in DB.")

    return None


def _prepare_daily_data(df,symbol):
    if df is None or df.empty:
        return None
     #prepare data for upsert in db, remove multiIndex if there is 
    is_multi = isinstance(df.columns, pd.MultiIndex) #multiIndex check 
    cols = (df.columns.get_level_values(0) #multiIndex fix 
            if is_multi
            else df.columns)

    df = (
        df
        .rename_axis(index="trading_date")
        .set_axis(cols.str.lower(), axis=1)
        .assign(symbol=symbol)   
        .reset_index()
        .assign(
                trading_date=lambda x: x["trading_date"].dt.date,
                volume=lambda x: pd.to_numeric(x["volume"], errors="coerce"),
                )
        [["symbol", "trading_date", "open", "high", "low", "close", "volume"]]
        )
    
    return (df)


def sync_prediction_daily_data(symbol):
    try:
        utc_now = datetime.now(timezone.utc)

        latest_close_dt, older_25_close_dt = _get_available_close_datetimes(symbol, utc_now)

        if latest_close_dt is None:
            return False
        
        start_date, end_date = _determine_download_range(latest_close_dt, older_25_close_dt)

        if start_date is None or end_date is None:
            return False

        last_stored_close_date = get_latest_stored_prediction_bar_date(symbol)

        if last_stored_close_date is None:
            raw_df = _download_daily_data_with_retry(symbol, start_date, end_date)

        elif latest_close_dt.date() > last_stored_close_date: # some last entries of close symbol missing in db, fetch from YF 
            if last_stored_close_date > older_25_close_dt.date(): # when less than 25 entries of close symbol missing in db (complete last 25 available entries)
                start_date = last_stored_close_date + timedelta(days=1) 
            raw_df = _download_daily_data_with_retry(symbol, start_date, end_date)

        else:
            return False
        
        if raw_df is None or raw_df.empty:
            return False

        prepared_df = _prepare_daily_data(raw_df, symbol)
        older_25_close_date = older_25_close_dt.date() if older_25_close_dt else None #older dates than this can be deleted from db, because the are outside of calculating features window

        if prepared_df is not None and not prepared_df.empty:
            upsert_prediction_daily_bars(prepared_df, symbol, older_25_close_date)
            return True
        
        return False
    
    except Exception:
        logger.exception("sync_prediction_daily_data failed for %s", symbol)
        return False
        

     