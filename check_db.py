import sqlite3
import numpy as np

db_name = "books_embeddings.db"

conn = sqlite3.connect(db_name)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM embeddings")
total_rows = cursor.fetchone()[0]
print(f"=== Проверка базы данных: {db_name} ===")
print(f"Всего записей в таблице: {total_rows}")

if total_rows == 0:
    print("Ошибка: База данных пуста!")
    conn.close()
    exit()

cursor.execute("SELECT flibusta_id, vector FROM embeddings LIMIT 5")
samples = cursor.fetchall()

print("\n=== Проверка целостности первых векторов ===")
for i, row in enumerate(samples, 1):
    flibusta_id = row[0]
    blob_data = row[1]
    
    vector = np.frombuffer(blob_data, dtype=np.float32)
    
    print(f"Запись #{i}:")
    print(f"  Flibusta ID: {flibusta_id}")
    print(f"  Размер BLOB в БД: {len(blob_data)} байт")
    print(f"  Восстановленная длина вектора: {len(vector)}")
    print(f"  Тип данных массива: {vector.dtype}")
    
    has_nan = np.isnan(vector).any()
    has_inf = np.isinf(vector).any()
    is_all_zeros = np.all(vector == 0)
    
    print(f"  Содержит NaN/Inf: {has_nan or has_inf}")
    print(f"  Состоит только из нулей: {is_all_zeros}")
    print(f"  Первые 3 числа: {vector[:3]}")
    print("-" * 40)

conn.close()