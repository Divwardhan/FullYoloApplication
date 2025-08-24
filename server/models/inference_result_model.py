import psycopg2
from .db import get_connection


def init_inference_table():
    """Create the inference_results table if it doesn't exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_results (
            id SERIAL PRIMARY KEY,
            model_id INTEGER,
            model_name TEXT,
            user_email TEXT,
            input_image_url TEXT,
            output_image_url TEXT,
            json_response TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def create_inference_result(model_id: int, model_name: str, user_email: str,
                             input_image_url: str, output_image_url: str,
                             json_response: str) -> dict:
    """Insert a new inference result and return its ID as JSON."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO inference_results (
            model_id, model_name, user_email, input_image_url,
            output_image_url, json_response
        ) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
        """,
        (model_id, model_name, user_email, input_image_url,
         output_image_url, json_response)
    )
    
    
    result_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "success", "result_id": result_id}




def get_inference_result_by_id(result_id: int) -> dict:
    """Fetch inference result by ID and return as JSON."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, model_id, model_name, user_email, input_image_url,
               output_image_url, json_response
        FROM inference_results WHERE id = %s;
        """,
        (result_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {
            "id": row[0],
            "model_id": row[1],
            "model_name": row[2],
            "user_email": row[3],
            "input_image_url": row[4],
            "output_image_url": row[5],
            "json_response": row[6]
        }
    return {"error": "Result not found"}


def get_all_inference_results() -> list:
    """Fetch all inference results as a list of JSON dicts."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, model_id, model_name, user_email, input_image_url,
               output_image_url, json_response
        FROM inference_results ORDER BY id;
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "model_id": row[1],
            "model_name": row[2],
            "user_email": row[3],
            "input_image_url": row[4],
            "output_image_url": row[5],
            "json_response": row[6]
        })
    return results
