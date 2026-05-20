from datetime import datetime
import os
import psycopg2
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
    

def get_ai_prediction_explanations(symbol:str , prediction_date: datetime):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT explanation, ai_model, ai_provider
            FROM ai_prediction_explanations
            WHERE symbol = %s AND prediction_date = %s
            """, (symbol, prediction_date))

        row = cur.fetchone()

        if not row or not row[0]:
            return None
    
        return {"explanation": row[0],
                "ai_model": row[1],
                "ai_provider": row[2]}
    

def insert_ai_prediction_explanations(symbol:str , prediction_date: datetime, prediction_hash: str, explanation: str, ai_model: str, ai_provider: str, response_id: str):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_prediction_explanations (
                symbol,
                prediction_date,
                prediction_hash,
                explanation,
                ai_model,
                ai_provider,
                response_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, prediction_date) DO UPDATE SET
                prediction_hash = EXCLUDED.prediction_hash,
                explanation = EXCLUDED.explanation,
                ai_model = EXCLUDED.ai_model,
                ai_provider = EXCLUDED.ai_provider,
                response_id = EXCLUDED.response_id,
                created_at = NOW()
        """, (symbol, prediction_date, prediction_hash, explanation, ai_model, ai_provider, response_id))

        conn.commit()