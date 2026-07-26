import os

os.environ["CUDA_HOME"] = "/opt/cuda"
os.environ["PATH"] = "/opt/cuda/bin:" + os.environ.get("PATH", "")

import json
import pandas as pd
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from tqdm import tqdm
from src.data_prep import get_data_df, load_book_text
from json_repair import repair_json

class BookAnalysis(BaseModel):
    main_character: str = Field(description="Подробный психотип, внешность, мотивация и поведение ключевого персонажа на РУССКОМ ЯЗЫКЕ (минимум 4 предложения). СТРОГО ЗАПРЕЩЕНО указывать любые имена, фамилии, прозвища или титулы.")
    setting: str = Field(description="Детальное описание мира, локаций, эпохи и атмосферы на РУССКОМ ЯЗЫКЕ (минимум 4 предложения). СТРОГО ЗАПРЕЩЕНО использовать любые собственные имена, названия городов, стран, планет или организаций.")
    plot_line: str = Field(description="Подробная сюжетная линия с ключевыми событиями на РУССКОМ ЯЗЫКЕ (минимум 4 предложения). СТРОГО ЗАПРЕЩЕНО упоминать любые имена персонажей и собственные названия.")
    reflection_depth: int = Field(description="Глубина размышлений от 0 до 10.")
    attitude_toward_mc: str = Field(description="Развернутое описание отношения окружающих к персонажу на РУССКОМ ЯЗЫКЕ (минимум 3 предложения). СТРОГО ЗАПРЕЩЕНО использовать имена и названия.")

llm = LLM(
    model="Qwen/Qwen3-4B-AWQ",
    gpu_memory_utilization=0.9,
    max_model_len=6400,
    enforce_eager=True,
    max_num_seqs=64
)

structured_params = StructuredOutputsParams(json=BookAnalysis.model_json_schema())

sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=1536,
    structured_outputs=structured_params
)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def safe_json_loads(text: str) -> dict:
    text_clean = text.strip()
    if text_clean.startswith("```"):
        text_clean = text_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text_clean)
    except Exception:
        repaired = repair_json(text_clean)
        return json.loads(repaired)

def process_books_batch(batch_rows: list[pd.Series], chunk_size: int = 7000, merge_batch_size: int = 4):
    valid_books = []
    for row in batch_rows:
        flibusta_id = str(row['flibusta_id'])
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{flibusta_id}.json")
        
        if os.path.exists(checkpoint_path):
            continue 
        
        try:
            text = load_book_text(f"data/{row['archive']}/{row['flibusta_id']}.fb2")
            if not text or len(text) < 100:
                continue
            valid_books.append((flibusta_id, row, text))
        except Exception as e:
            print(f"[ERROR] Не удалось прочитать книгу {flibusta_id}: {e}")
            continue

    if not valid_books:
        return

    all_level1_prompts = []
    prompt_book_map = []

    for flibusta_id, row, text in valid_books:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for c in chunks:
            prompt = (
                "Ты аналитик текстов для обучения семантического поиска. Сделай подробный разбор фрагмента.\n"
                "КРИТИЧЕСКИЕ ПРАВИЛА ОБЕЗЛИЧИВАНИЯ (ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ):\n"
                "1. СТРОГО ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ ЛЮБЫЕ ИМЕНА И СОБСТВЕННЫЕ НАЗВАНИЯ во всех полях! Заменяй их обобщенными понятиями (например: 'главный герой', 'его соратник', 'столица империи', 'сопредельное государство', 'космический корабль').\n"
                "2. Пиши ИСКЛЮЧИТЕЛЬНО на русском языке.\n"
                "3. Каждое текстовое поле должно быть максимально подробным (НЕ МЕНЕЕ 4-5 ПОЛНОЦЕННЫХ ПРЕДЛОЖЕНИЙ).\n\n"
                f"Фрагмент:\n{c}"
            )
            all_level1_prompts.append(prompt)
            prompt_book_map.append(flibusta_id)

    print(f"--> Генерация 1-го уровня: {len(all_level1_prompts)} чанков для {len(valid_books)} книг...")
    
    outputs = llm.generate(all_level1_prompts, sampling_params)

    book_jsons = {f_id: [] for f_id, _, _ in valid_books}
    for out, f_id in zip(outputs, prompt_book_map):
        parsed = safe_json_loads(out.outputs[0].text)
        book_jsons[f_id].append(parsed)

    for flibusta_id, row, _ in valid_books:
        current_jsons = book_jsons[flibusta_id]
        count = 0
        
        while len(current_jsons) > 1 and count < 5:
            batches = [
                current_jsons[i:i + merge_batch_size]
                for i in range(0, len(current_jsons), merge_batch_size)
            ]

            merge_prompts = [
                "Объедини несколько JSON-суммаризаций фрагментов книги в один единый детальный JSON.\n"
                "КРИТИЧЕСКИЕ ПРАВИЛА ОБЕЗЛИЧИВАНИЯ:\n"
                "1. СТРОГО ЗАПРЕЩЕНО указывать любые имена персонажей, названия городов, локаций, фракций, организаций или стран.\n"
                "2. Заменяй любые встреченные имена и названия на общие категории ('главная героиня', 'магический орден', 'древний город').\n"
                "3. Пиши ИСКЛЮЧИТЕЛЬНО на русском языке и сохраняй максимальную детализацию контекста.\n\n"
                f"Входные данные:\n{json.dumps(b, ensure_ascii=False)}"
                for b in batches
            ]

            outputs = llm.generate(merge_prompts, sampling_params)
            current_jsons = [safe_json_loads(out.outputs[0].text) for out in outputs]
            count += 1

        result_payload = {
            "flibusta_id": flibusta_id,
            "title": str(row.get('name', '')),
            "author": str(row.get('author', '')),
            "genre": str(row.get('genre', '')),
            "analysis": current_jsons[0]
        }

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{flibusta_id}.json")
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] Чекпоинт сохранен: {checkpoint_path} ({row.get('name')})")


if __name__ == "__main__":
    data_df = get_data_df().sample(frac=1, random_state=42).reset_index(drop=True)

    BOOK_BATCH_SIZE = 8

    rows = [row for _, row in data_df.iterrows()]

    for i in range(0, len(rows), BOOK_BATCH_SIZE):
        batch = rows[i:i + BOOK_BATCH_SIZE]
        process_books_batch(batch)