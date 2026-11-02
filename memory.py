
import sqlite3

DB_NAME = "chatbot_memory.db"


def create_memory_database():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_key TEXT UNIQUE,
            fact_value TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_memory(key, value):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO user_memory (fact_key, fact_value)
        VALUES (?, ?)
        ON CONFLICT(fact_key)
        DO UPDATE SET fact_value = excluded.fact_value
    """, (key, value))

    conn.commit()
    conn.close()


def get_all_memories():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT fact_key, fact_value
        FROM user_memory
    """)

    memories = cursor.fetchall()

    conn.close()

    return memories


def add_interest(new_interest):

    memories = dict(get_all_memories())

    old_interests = memories.get("interests", "")

    interests = [
        item.strip()
        for item in old_interests.split(",")
        if item.strip()
    ]

    if new_interest.lower() not in [
        item.lower() for item in interests
    ]:
        interests.append(new_interest)

    save_memory("interests", ", ".join(interests))


def delete_memory(key):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM user_memory WHERE fact_key = ?",
        (key,)
    )

    conn.commit()
    conn.close()


create_memory_database()
