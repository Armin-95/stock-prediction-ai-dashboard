from datetime import timedelta, timezone, datetime
import json
import os
import psycopg2
from psycopg2.extras import execute_values, Json
import pandas as pd
import numpy as np

from database.ai_chat_db import init_ai_chat_tables
#from dotenv import load_dotenv #test env


#load_dotenv(override=True)  # Load environment variables from .env file #test env
DATABASE_URL = os.getenv("DATABASE_URL") 
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL")


# SSL check 
if "sslmode" not in DATABASE_URL:
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"


def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Create tables if not exist."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        """)

        # 1) Daily price storage
        cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_daily_bars (
            symbol TEXT NOT NULL,
            trading_date DATE NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low  DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (symbol, trading_date)
        );
        """)

        # 2) stock metadata (timezone + calendar +close time + last update)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_symbols (
            symbol TEXT PRIMARY KEY,
            calendar_code TEXT NOT NULL,
            exchange_tz TEXT NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        # 3) Trading calendar (authoritative open days)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS market_calendar (
            calendar_code TEXT NOT NULL,
            trading_date DATE NOT NULL,
            close_date_time TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (calendar_code, trading_date)
        );
        """)


        # Index for faster lookup in market_calendar by calendar_code and close_date_time  
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_calendar_code_close_dt
            ON market_calendar(calendar_code, close_date_time);
        """)

        # Predict return of next close symbol (with time of the prediction)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_future_predictions (
            symbol TEXT NOT NULL,
            model_type TEXT NOT NULL,
            trading_date DATE NOT NULL,
            close_date_time TIMESTAMPTZ NOT NULL,
            predicted_return DOUBLE PRECISION NOT NULL,
            predicted_close DOUBLE PRECISION NOT NULL,
            prediction_features JSONB NOT NULL,
            time_of_prediction TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, model_type, trading_date)
        );
        """)

        # model metrics storage for monitoring and model comparison 
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol TEXT NOT NULL,
            model_type TEXT NOT NULL,
            time_of_training TIMESTAMPTZ DEFAULT NOW(),
            ridge_coefficients JSONB,
            ridge_best_alpha DOUBLE PRECISION,
            xgboost_best_iteration INTEGER,
            val_xgboost_rmse DOUBLE PRECISION,
            test_mae DOUBLE PRECISION,
            test_rmse DOUBLE PRECISION,
            test_hit_ratio DOUBLE PRECISION,
            test_corrcoef DOUBLE PRECISION,   
            test_strategy_mean DOUBLE PRECISION,
            test_strategy_std DOUBLE PRECISION,
            test_sharpe DOUBLE PRECISION,
            test_total_return DOUBLE PRECISION, 
            test_max_loss DOUBLE PRECISION, 
            test_max_drawdown DOUBLE PRECISION
        );
        """)

        # AI model comparison metrics explanations 
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_model_comparison_explanations  (
            symbol TEXT NOT NULL,
            metric_hash TEXT NOT NULL,
            explanation TEXT NOT NULL,
            ai_provider TEXT,
            ai_model TEXT,
            response_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, metric_hash)
        );
        """)

        # AI prediction explainations 
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_prediction_explanations  (
            symbol TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            prediction_hash TEXT NOT NULL,
            explanation TEXT NOT NULL,
            ai_provider TEXT,
            ai_model TEXT,
            response_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, prediction_date)
        );
        """)

        #stock AI chat tables 
        init_ai_chat_tables(cur) 
        conn.commit()


def seed_symbols(rows):
    
    #rows: list of dictionaries:
      #{"symbol":"AAPL","calendar_code":"XNYS","exchange_tz":"America/New_York"}
    with get_connection() as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute("""
            INSERT INTO stock_symbols(symbol, calendar_code, exchange_tz)
            VALUES (%s,%s,%s)
            ON CONFLICT(symbol) DO UPDATE SET
              calendar_code=EXCLUDED.calendar_code,
              exchange_tz=EXCLUDED.exchange_tz,
              updated_at = NOW();
            """, (r["symbol"], r["calendar_code"], r["exchange_tz"]))

        conn.commit()

def upsert_calendar(open_days:pd.DataFrame):
    """Open trading days for a calendar_code, with close_date_time.
        After conversion to tuples for ex.: ('2026-03-18 20:00:00+00:00')
    """
    rows = list(open_days.itertuples(index=False, name=None))
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO market_calendar (
                calendar_code,
                trading_date,
                close_date_time
            )
            VALUES %s
            ON CONFLICT (calendar_code, trading_date)
            DO UPDATE SET
                close_date_time = EXCLUDED.close_date_time
        """, rows)

        conn.commit()


def get_latest_available_close_datetime(symbol: str, datetime_utc_now):
    buffered_utc_now = datetime_utc_now - timedelta(minutes=10)  #safety 10min buffer for close time, it will be comapred to table market_calendar -> close_date_time 
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(""" 
            WITH filtered AS (
                SELECT mc.close_date_time
                FROM market_calendar mc
                JOIN stock_symbols ss
                  ON mc.calendar_code = ss.calendar_code
                WHERE ss.symbol = %s
                  AND mc.close_date_time <= %s
                ORDER BY mc.close_date_time DESC
            )
            SELECT
                (SELECT close_date_time FROM filtered LIMIT 1) AS latest,
                (SELECT close_date_time FROM filtered OFFSET 24 LIMIT 1) AS older_25
        """, (symbol, buffered_utc_now))

        row = cur.fetchone()
        return row if row else (None, None) 


def get_latest_stored_prediction_bar_date(symbol:str):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute(""" SELECT MAX(trading_date) 
                    FROM prediction_daily_bars 
                    WHERE symbol = %s
                    """, (symbol,)) 
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    

def upsert_prediction_daily_bars(df: pd.DataFrame, symbol:str, older_25_close_date):
    # start_date is date from download window range on YF, older than start_date can be deleted from db, because they are already in db 
    if df is None or df.empty:
        return
    rows = list(df.itertuples(index=False, name=None)) # converted to tuples for ex.: ('AAPL',datetime.date(2025, 12, 18),273.35420232069225,273.374203136776,266.7004552212896,271.935546875,51630700),...
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO prediction_daily_bars (
                symbol,
                trading_date,
                open,
                high,
                low,
                close,
                volume
            )
            VALUES %s
            ON CONFLICT (symbol, trading_date)
            DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,rows)

        cur.execute("""
            DELETE FROM prediction_daily_bars
            WHERE symbol = %s AND trading_date < %s
        """, (symbol, older_25_close_date))
        
        conn.commit()


def upsert_stock_future_prediction(symbol: str, model_type: str, trading_date, close_date_time, predicted_return: float, predicted_close: float | int, prediction_features_dict: dict):
    predicted_return = float(predicted_return)
    predicted_close = float(predicted_close)
    prediction_features = Json(prediction_features_dict)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO stock_future_predictions (
                symbol,
                model_type,
                trading_date,
                close_date_time,
                predicted_return,
                predicted_close,
                prediction_features
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, model_type, trading_date)
            DO UPDATE SET
                predicted_return  = EXCLUDED.predicted_return,
                predicted_close   = EXCLUDED.predicted_close,
                close_date_time   = EXCLUDED.close_date_time,
                time_of_prediction = NOW()
        """, (symbol, model_type, trading_date, close_date_time, predicted_return, predicted_close, prediction_features))

        conn.commit()





def insert_model_metrics(symbol, model_type, mae, rmse, hit_ratio, corrcoef,
    strategy_mean, strategy_std,sharpe, total_return, max_loss, max_drawdown,ridge_coefficients, ridge_best_alpha, xgboost_best_iteration, val_xgboost_rmse):
    ridge_coefficients = json.dumps(ridge_coefficients) if ridge_coefficients is not None else None
    ridge_best_alpha = float(ridge_best_alpha) if ridge_best_alpha is not None else None
    mae = float(mae)
    rmse = float(rmse)
    hit_ratio = float(hit_ratio)
    corrcoef = float(corrcoef)
    strategy_mean = float(strategy_mean)
    strategy_std = float(strategy_std)
    sharpe = float(sharpe) if not np.isnan(sharpe) else None
    total_return = float(total_return)
    max_loss = float(max_loss)
    max_drawdown = float(max_drawdown)

    with get_connection() as conn, conn.cursor() as cur:
            cur.execute( """
                INSERT INTO model_metrics (
                        symbol,
                        model_type,
                        test_mae,
                        test_rmse,
                        test_hit_ratio,
                        test_corrcoef,
                        test_strategy_mean,
                        test_strategy_std,
                        test_sharpe,
                        test_total_return,
                        test_max_loss, 
                        test_max_drawdown,
                        ridge_coefficients,
                        ridge_best_alpha,
                        xgboost_best_iteration,
                        val_xgboost_rmse
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (symbol, model_type, mae, rmse, hit_ratio,
                  corrcoef,strategy_mean, strategy_std, sharpe, total_return, max_loss, max_drawdown,ridge_coefficients, ridge_best_alpha, xgboost_best_iteration, val_xgboost_rmse
            ))

            conn.commit()

def insert_ai_model_comparison_explanation(symbol, metric_hash, explanation, ai_provider, ai_model, response_id):
    with get_connection() as conn, conn.cursor() as cur:
            cur.execute( """
                INSERT INTO ai_model_comparison_explanations (
                        symbol,
                        metric_hash,
                        explanation,
                        ai_provider,
                        ai_model,
                        response_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, metric_hash) DO UPDATE SET
                    explanation = EXCLUDED.explanation,
                    response_id = EXCLUDED.response_id,
                    ai_provider = EXCLUDED.ai_provider,
                    ai_model = EXCLUDED.ai_model,
                    created_at = NOW()
            """, (symbol, metric_hash, explanation,ai_provider, ai_model, response_id))

            conn.commit()

def get_ai_model_comparison_explanation(symbol, metric_hash):
    with get_connection() as conn, conn.cursor() as cur:
            cur.execute( """
                SELECT explanation, ai_model, ai_provider
                FROM ai_model_comparison_explanations
                WHERE symbol = %s AND metric_hash = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (symbol, metric_hash))

            row = cur.fetchone()

            if not row or not row[0]:
                return None

            return row[0], row[1], row[2]

def get_prediction_daily_bars(symbol:str):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute(""" SELECT trading_date, open, high, low, close, volume
                    FROM prediction_daily_bars 
                    WHERE symbol = %s
                    ORDER BY trading_date DESC
                    LIMIT 25
                    """, (symbol,)) 
        rows = cur.fetchall()

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["trading_date", "open", "high", "low", "close", "volume"]
    )

    return df.sort_values("trading_date").reset_index(drop=True)


def get_model_metrics(symbol, model_type):
    with get_connection() as conn, conn.cursor() as cur:
            cur.execute( """
                SELECT test_mae, test_rmse, test_hit_ratio, test_corrcoef,
                       test_strategy_mean, test_strategy_std, test_sharpe,
                       test_total_return, test_max_loss, test_max_drawdown
                FROM model_metrics
                WHERE symbol = %s AND model_type = %s
                ORDER BY time_of_training DESC
                LIMIT 1
            """, (symbol, model_type))

            row = cur.fetchone()

            if row is None:
                return None

            return {
                "model_metrics_quality": {
                    "mae": row[0],
                    "rmse": row[1],
                    "hit_ratio": row[2],
                    "corrcoef": row[3],
                },
                "model_strategy_quality": {
                    "strategy_mean": row[4],
                    "strategy_std": row[5],
                    "sharpe": row[6],
                    "total_return": row[7],
                    "max_loss": row[8],
                    "max_drawdown": row[9],
                }
            }

            



def get_next_close_date(symbol: str):
    datetime_utc_now = datetime.now(timezone.utc)
    buffered_utc_now = datetime_utc_now - timedelta(minutes=10)  #safety 10min buffer for close time, it will be comapred to table market_calendar -> close_date_time 
    with get_connection () as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT mc.trading_date,
                   mc.close_date_time
            FROM market_calendar mc
            JOIN stock_symbols ss
              ON mc.calendar_code = ss.calendar_code
            WHERE ss.symbol = %s
              AND mc.close_date_time > %s
            ORDER BY mc.close_date_time ASC
            LIMIT 1
            """,
            (symbol, buffered_utc_now)
        )
        row = cur.fetchone()

        if row is None:
            return None, None

        trading_date, close_date_time = row

        if trading_date is None or close_date_time is None:
            return None, None

        return trading_date, close_date_time

    


def get_next_close_prediction(symbol: str, next_close_date, model_type):
    with get_connection () as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT predicted_return, predicted_close
            FROM stock_future_predictions
            WHERE symbol = %s AND trading_date = %s AND model_type= %s
            """,
            (symbol, next_close_date, model_type)
        )
        row  = cur.fetchone()

        if not row:
            return None, None

        predicted_return, predicted_close = row

        if predicted_return is None or predicted_close is None:
            return None, None

        return predicted_return, predicted_close


def get_prediction_features(symbol: str, next_close_date, model_type):
    with get_connection () as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT prediction_features
            FROM stock_future_predictions
            WHERE symbol = %s AND trading_date = %s AND model_type= %s
            """,
            (symbol, next_close_date, model_type)
        )
        row  = cur.fetchone()

        if not row:
            return None

        prediction_features = row

        return prediction_features