import psycopg2
from psycopg2 import sql
from .db import get_connection


def init_model_table():
    """Create the ml_models table if it doesn't exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
    """
    CREATE TABLE IF NOT EXISTS ml_models (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        image TEXT,
        weight_file TEXT,
        type TEXT,
        description TEXT  -- ✅ New field
    );
    """
)
    conn.commit()
    cur.close()
    conn.close()


def get_weight_file(model_id: int):
    model = get_model_by_id(model_id)
    if not model:
        return 404, {"message": "Model not found"}
    
    print(model)
    _, name, _, _, weight_file ,_ = model
    if not weight_file: 
        return 404, {"message": "Weight file not found"}
    return 200, {"weight_file": weight_file , "name": name}


def create_model(name: str, image: str, model_type: str, weight_file: str, description: str) -> int:
    """Insert a new ML model and return its id."""
    conn = get_connection()
    cur = conn.cursor()
    print("***************",name, image, model_type, weight_file, description, "***************") 
    cur.execute(
        "INSERT INTO ml_models (name, image, type, weight_file, description) VALUES (%s, %s, %s, %s, %s) RETURNING id;",
        (name, image, model_type, weight_file, description),
    )
    model_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return model_id


def get_model_by_id(model_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id , name , image, type , weight_file,description FROM ml_models WHERE id = %s;",
        (model_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def update_model(model_id: int, name: str = None, image: str = None, model_type: str = None, weight_file: str = None, description: str = None) -> bool:
    """Update model fields."""
    if not any([name, image, model_type, weight_file, description]):
        return False

    fields = []
    values = []
    if name is not None:
        fields.append("name = %s")
        values.append(name)
    if image is not None:
        fields.append("image = %s")
        values.append(image)
    if model_type is not None:
        fields.append("type = %s")
        values.append(model_type)
    if weight_file is not None:
        fields.append("weight_file = %s")
        values.append(weight_file)
    if description is not None:
        fields.append("description = %s")
        values.append(description)

    values.append(model_id)

    conn = get_connection()
    cur = conn.cursor()
    query = "UPDATE ml_models SET " + ", ".join(fields) + " WHERE id = %s;"
    cur.execute(query, values)
    updated = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return updated


def delete_model(model_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM ml_models WHERE id = %s;", (model_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def get_all_models():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ml_models ORDER BY id;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
