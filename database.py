import sqlite3

DB_NAME = "nodin_ai.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            first_name TEXT,
            username TEXT,
            language TEXT DEFAULT 'ru',
            is_vip INTEGER DEFAULT 0,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_user_lang(user_id: int) -> str:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT language FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "ru"

def set_user_lang(user_id: int, first_name: str, lang: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (user_id, first_name, language) 
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET language = ?, first_name = ?
    """, (user_id, first_name, lang, lang, first_name))
    conn.commit()
    conn.close()

def add_unique_user(user_id: int, first_name: str, username: str = None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (user_id, first_name, username) 
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET first_name = ?, username = COALESCE(?, username)
    """, (user_id, first_name, username, first_name, username))
    conn.commit()
    conn.close()

def get_total_users_count() -> int:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_vip_users_count() -> int:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_vip = 1")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_last_users(limit: int = 10):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, first_name, username, is_vip, joined_at 
        FROM users 
        ORDER BY joined_at DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def set_vip_status(user_id: int, is_vip: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_vip = ? WHERE user_id = ?", (is_vip, user_id))
    conn.commit()
    conn.close()

def is_user_vip(user_id: int) -> bool:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT is_vip FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return bool(row and row[0] == 1)