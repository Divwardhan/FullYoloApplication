import os
import psycopg2


def get_connection():
    """Create and return a new PostgreSQL connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "yolo_api_calling"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "k6pnog6g5y"),
    )


def test_connection() -> bool:
    """Attempt a simple query to ensure the database connection works."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        cur.close()
        conn.close()
        return True
    except Exception as exc:
        print(f"Database connection failed: {exc}")
        return False
