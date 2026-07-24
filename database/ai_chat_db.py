import os
import psycopg2
#from dotenv import load_dotenv #test env

MAX_CONVERSATIONS_PER_SESSION = 10

class ChatConversationLimitExceeded(Exception): #custom value error for conversatin limit
    """Raised when an AI chat sesion reaches the maximum number of conversations"""
    pass

#load_dotenv(override=True)  # Load environment variables from .env file #test env
DATABASE_URL = os.getenv("DATABASE_URL") 
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL")


# SSL check 
if "sslmode" not in DATABASE_URL:
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_ai_chat_tables(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_chat_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days')
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_chat_conversations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID NOT NULL REFERENCES ai_chat_sessions(id) ON DELETE CASCADE,
        user_id BIGINT NULL, --futute log users 
        symbol TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '10 days')
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_chat_messages (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        conversation_id UUID NOT NULL REFERENCES ai_chat_conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT,
        status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN('completed', 'failed')),
        model_name TEXT,
        provider TEXT,
        error_message TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_chat_conversations_session_id
    ON ai_chat_conversations(session_id);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_chat_conversations_symbol
    ON ai_chat_conversations(symbol);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_chat_messages_conversation_id
    ON ai_chat_messages(conversation_id);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_chat_messages_created_at
    ON ai_chat_messages(created_at);
    """)


def create_ai_chat_session():
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_chat_sessions DEFAULT VALUES
            RETURNING id;
        """)
        session_id = cur.fetchone()[0]
        conn.commit()

    return session_id


def touch_ai_chat_session(session_id):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ai_chat_sessions
            SET last_active_at = NOW(),
            expires_at = NOW() + INTERVAL '30 days'
            WHERE id = %s
            RETURNING id;
        """, (session_id,))
        row = cur.fetchone()
        conn.commit()

    return row[0] if row else None

def touch_ai_chat_conversation(conversation_id, symbol, session_id):
    symbol = symbol.upper()

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ai_chat_conversations
            SET updated_at = NOW(),
                expires_at = NOW() + INTERVAL '10 days'
            WHERE id = %s
              AND symbol =%s
              AND session_id =%s
            RETURNING id;
        """, (conversation_id, symbol , session_id))

        row = cur.fetchone()
        conn.commit()

    return row[0] if row else None


def count_ai_chat_conversations_for_session(session_id):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(id)
            FROM ai_chat_conversations
            WHERE session_id = %s
                AND expires_at > NOW();
        """, (session_id,))

        count = cur.fetchone()[0]

    return count


def create_ai_chat_conversation(session_id, symbol):
    symbol = symbol.upper()
    conversation_per_session = count_ai_chat_conversations_for_session(session_id)
    if conversation_per_session >= MAX_CONVERSATIONS_PER_SESSION:
        raise ChatConversationLimitExceeded(f"Max number of allowed chats conversations ({MAX_CONVERSATIONS_PER_SESSION}) exceeded, delete existing chat conversation.")
    
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_chat_conversations (session_id, symbol)
            VALUES (%s, %s)
            RETURNING id;

        """, (session_id,symbol))
        conversation_id = cur.fetchone()[0]
        conn.commit()

    return conversation_id

def get_last_stock_ai_chat_conversation_id(symbol, session_id):
    symbol = symbol.upper()

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id
            FROM ai_chat_conversations
            WHERE symbol = %s
                AND session_id = %s
                AND expires_at > NOW()
            ORDER BY updated_at DESC
            LIMIT 1;
        """,(symbol, session_id))
        conversations_id = cur.fetchone()

    return conversations_id[0] if conversations_id else None


def get_valid_ai_chat_conversation_id(conversation_id, session_id, symbol): #check and validate if this conversation belongs to session and symbol
    symbol = symbol.upper()
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id 
            FROM ai_chat_conversations
            WHERE id = %s
                AND session_id = %s
                AND symbol = %s
                AND expires_at > NOW ();

        """,(conversation_id, session_id, symbol))

        conversation_row = cur.fetchone()

    if conversation_row is None:
        return None
    
    return conversation_row[0]
        

def insert_ai_chat_message(conversation_id, role, content, status="completed", model_name=None , provider=None, error_message=None):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_chat_messages (
                conversation_id,
                role, 
                content, 
                status, 
                model_name, 
                provider, 
                error_message
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id

        """,(conversation_id, role, content, status, model_name, provider, error_message))

        ai_chat_message_row = cur.fetchone()
        conn.commit()

    if ai_chat_message_row is None:

        return None
    
    return ai_chat_message_row[0]


def get_ai_chat_messages_for_display(conversation_id, limit= 30): #user chat: get 30 (DESC) newest, change order ASC for frontend)
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT role, content, status, created_at
            FROM(
                SELECT role, content, status, created_at
                FROM ai_chat_messages
                WHERE conversation_id = %s
                        ORDER by created_at DESC
                        LIMIT %s
            ) recent_messages
            ORDER by created_at ASC

        """, (conversation_id, limit))

        message_rows = cur.fetchall()

    messages = [{
            "role": row[0], 
            "content": row[1], 
            "status": row[2], 
            "created_at": row[3].isoformat() if row[3] else None
            } 
        for row in message_rows
    ]        
  
    return messages #empty [] if there arent any msgs


def get_recent_ai_chat_messages_for_prompt(conversation_id, limit= 5):#AI feed
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT role, content
            FROM(
                SELECT role, content, created_at
                FROM ai_chat_messages
                WHERE conversation_id = %s
                    AND status = 'completed'
                    AND content IS NOT NULL
                ORDER BY created_at DESC
                LIMIT %s
            )recent_messages 
            ORDER BY created_at ASC
        """,(conversation_id, limit))

        message_rows = cur.fetchall()

    messages = [{ "role": row[0], "content": row[1]} for row in message_rows]        
    
    return messages #empty [] if there arent any msgs


def delete_ai_chat_conversation(conversation_id, session_id, symbol):
    symbol = symbol.upper()
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            DELETE FROM ai_chat_conversations
            WHERE id = %s
                AND session_id = %s
                AND symbol = %s 
        """,(conversation_id, session_id, symbol)) 
        deleted_count = cur.rowcount

        conn.commit() 

    return deleted_count  

def get_ai_chat_available_conversations(session_id, symbol):
    symbol = symbol.upper()
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, created_at, updated_at
            FROM ai_chat_conversations
            WHERE session_id = %s
                AND symbol = %s
                AND expires_at > NOW()
            ORDER by updated_at DESC
        """, (session_id, symbol))
    
        conversations_rows = cur.fetchall()

    conversations = [
        {"id": str(row[0]), 
        "created_at": row[1].isoformat() if row[1] else None, 
        "updated_at" : row[2].isoformat() if row[2] else None} for row in conversations_rows]

    return conversations #empty list [] if there aren't any messages










