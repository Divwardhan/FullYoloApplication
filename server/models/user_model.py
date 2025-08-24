import psycopg2
from psycopg2 import sql
from .db import get_connection


def init_db():
    """Create the users table if it doesn't exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def create_user(email: str, password: str) -> int:
    """Insert a new user and return the user id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, password) VALUES (%s, %s) RETURNING id;",
        (email, password),
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return user_id


def get_user_by_id(user_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password FROM users WHERE id = %s;", (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def get_user_by_email(email: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password FROM users WHERE email = %s;", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def update_user(user_id: int, email: str = None, password: str = None) -> bool:
    """Update email and/or password for the specified user."""
    if email is None and password is None:
        return False
    fields = []
    values = []
    if email is not None:
        fields.append("email = %s")
        values.append(email)
    if password is not None:
        fields.append("password = %s")
        values.append(password)
    values.append(user_id)
    conn = get_connection()
    cur = conn.cursor()
    query = "UPDATE users SET " + ", ".join(fields) + " WHERE id = %s;"
    cur.execute(query, values)
    updated = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return updated


def delete_user(user_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted
