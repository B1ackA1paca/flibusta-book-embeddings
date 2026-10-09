import os

os.environ["CUDA_HOME"] = "/opt/cuda"
os.environ["PATH"] = "/opt/cuda/bin:" + os.environ.get("PATH", "")

import numpy as np
import torch
import lancedb
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.models import FullBookEncoder
from src.data_prep import get_data_df

model_name = "Qwen/Qwen3-8B-AWQ"
embedding_model = "Alibaba-NLP/gte-multilingual-base"

checkpoint_path = "model_checkpoints/model_27.pth"
database_path = "./vector_database"
inpx_path = "/data/flibusta/data/fb2.flibusta.lib.rus.ec.7z.inpx"

top_k = 10
device = "cpu"  # GPU остаётся для vLLM.

def encode_story(story, tokenizer, encoder):
    token_ids = tokenizer.encode(story, add_special_tokens=False)
    if not token_ids:
        raise ValueError("LLM вернула пустую историю")

    chunk_size = 448 - tokenizer.num_special_tokens_to_add(pair=False)
    chunks = []

    for start in range(0, len(token_ids), chunk_size - 32):
        chunks.append(token_ids[start:start + chunk_size])
        if start + chunk_size >= len(token_ids):
            break

    max_chunks = min(64, encoder.summarizer.pos_embedding.shape[1])
    if len(chunks) > max_chunks:
        indices = np.linspace(0, len(chunks) - 1, max_chunks, dtype=int)
        chunks = [chunks[i] for i in indices]

    prepared = [
        tokenizer.prepare_for_model(
            chunk,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        for chunk in chunks
    ]

    inputs = tokenizer.pad(
        prepared,
        padding="max_length",
        max_length=448,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        vector = encoder(
            chunk_input_ids=inputs["input_ids"],
            chunk_attention_mask=inputs["attention_mask"],
        )[0].float().cpu().numpy()

    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Энкодер вернул некорректный вектор")

    return (vector / norm).astype(np.float32)

if __name__ == "__main__":
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    for key in ("state_dict", "model_state_dict"):
        if key in state_dict:
            state_dict = state_dict[key]
            break

    # Убираем префиксы torch.compile / DataParallel.
    weights = {}
    for key, value in state_dict.items():
        while key.startswith(("_orig_mod.", "module.")):
            key = key.split(".", 1)[1]
        if key in weights:
            raise ValueError(f"Повторяющийся ключ чекпоинта: {key}")
        weights[key] = value

    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model,
        trust_remote_code=True,
    )

    encoder = FullBookEncoder(model_name=embedding_model)
    encoder.load_state_dict(weights, strict=True)
    encoder.gte_lora.gradient_checkpointing_disable()
    encoder.gte_lora.disable_input_require_grads()
    encoder = encoder.to(device).eval()

    del state_dict, weights

    if not os.path.isdir(database_path):
        raise FileNotFoundError(f"Не найдена база: {database_path}")

    db = lancedb.connect(database_path)
    table = db.open_table("books")

    if table.count_rows() == 0:
        raise ValueError("Таблица books пуста")

    data_df = get_data_df(inpx_path)
    books = {
        str(row.flibusta_id).strip(): (str(row.name), str(row.author))
        for row in data_df.itertuples(index=False)
    }

    llm = LLM(
        model=model_name,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        max_num_seqs=1,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=3072,
    )

    while True:
        try:
            query = input(
                "\nКакую книгу ищешь? (exit — выход): "
            ).strip()

            if query.lower() in {"exit", "quit", "выход"}:
                break
            if not query:
                continue

            conversations = [[
                {
                    "role": "system",
                    "content": (
                        "Напиши оригинальную связную историю на русском языке, "
                        "примерно 400–600 слов, по пожеланиям читателя. "
                        "Сохрани жанр, мир, тип героя, настроение и ограничения запроса. "
                        "Покажи характеры через поступки, диалоги и реакции окружающих. "
                        "Дай истории начало, конфликт и завершение. "
                        "Верни только художественный текст, без рассуждений, "
                        "перечня книг и рекомендаций."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ]]

            output = llm.chat(
                messages=conversations,
                sampling_params=sampling_params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )[0].outputs[0]

            if output.finish_reason != "stop":
                raise ValueError(
                    "История не завершена; увеличь max_tokens или контекст LLM"
                )

            vector = encode_story(
                output.text.strip(),
                tokenizer,
                encoder,
            )

            if vector.size != table.schema.field("vector").type.list_size:
                raise ValueError("Размер вектора не совпадает с базой")

            results = (
                table.search(vector, vector_column_name="vector")
                .distance_type("cosine")
                .limit(top_k)
                .refine_factor(10)
                .select(["id", "_distance"])
                .to_list()
            )

            print("\nПохожие книги:")
            for i, result in enumerate(results, 1):
                book_id = str(result["id"])
                title, author = books.get(
                    book_id,
                    (f"Книга ID {book_id}", ""),
                )
                print(f"{i}. {title} — {author}")

        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break
        except (ValueError, RuntimeError) as exc:
            print(f"Ошибка: {exc}")