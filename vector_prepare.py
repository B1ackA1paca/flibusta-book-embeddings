import lancedb
from lancedb.pydantic import LanceModel, Vector
import sqlite3
import numpy as np
from typing import Generator, List, Dict, Any
from tqdm import tqdm
from lancedb.index import IvfPq

class BookEmbedding(LanceModel):
    id: str
    vector: Vector(768)

def get_lancedb_table(db_path: str, table_name: str, schema: LanceModel):
    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        return db.open_table(table_name)
    return db.create_table(table_name, schema=schema)

def read_sqlite_in_batches(sqlite_path: str, table_name: str = "embeddings", batch_size: int = 10000) -> Generator[List[Dict[str, Any]], None, None]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(f"SELECT flibusta_id, vector FROM {table_name}")
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        batch = []
        for row in rows:
            vector_array = np.frombuffer(row["vector"], dtype=np.float32)
            batch.append({
                "id": str(row["flibusta_id"]),
                "vector": vector_array.tolist()
            })
        yield batch
    conn.close()

def insert_to_lancedb(table, batch_data: List[Dict[str, Any]]):
    table.add(batch_data)

def build_lancedb_index(table, metric: str = "cosine"):
    table.create_index("vector", config=IvfPq(distance_type=metric))

table = get_lancedb_table("./vector_database", "books", BookEmbedding)

for batch in tqdm(read_sqlite_in_batches("books_embeddings.db")):
    insert_to_lancedb(table, batch)
build_lancedb_index(table)