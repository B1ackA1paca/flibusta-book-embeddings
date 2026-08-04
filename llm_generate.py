import os

os.environ["CUDA_HOME"] = "/opt/cuda"
os.environ["PATH"] = "/opt/cuda/bin:" + os.environ.get("PATH", "")

import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from typing import List, Optional

from langchain_text_splitters import TokenTextSplitter

from src.data_prep import get_data_df, load_book_text

class Character(BaseModel):
    name: str = Field(
        description="Имя персонажа, прозвище, если таких нет, то роль. если есть несколько возможных укажи через запятую"
    )
    character: str = Field(
        description="Описание характера, внешности персонажа. максимально кратко не придумывай от себя. если персонаж использует магию, или какие-то способности опиши их"
    )
    impression: str = Field(
        description="Впечатление, которое он оказывает на окружающих. важно отметить, если все в шоке от персонажа. не более 2 предложений. важно именно через запятую написать какие эмоции он вызывет у всех(шок, радость, удивление)"
    )

class BookSummary(BaseModel):
    world: str = Field(
        description="Описание мира, которое можно получить из этого фрагмента(<= 4 предложений)"
    )
    places: str = Field(
        description="Основные места, где происходят события(перечисли через запятую)"
    )
    reasoning: str = Field(
        description="Рассмотри реакции окружающих на слова и действия каждого персонажа, но не больше 4-5 предложений"
    )
    characters: List[Character] = Field(
        description="Опиши каждого персонажа, который участвует в событиях во фрагменте"
    )

model_name = "Qwen/Qwen3-4B-AWQ"

if __name__ == '__main__':
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        max_num_seqs=16
    )

    os.makedirs("checkpoints", exist_ok=True)

    structured_outputs = StructuredOutputsParams(json=BookSummary.model_json_schema())

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=2048,
        structured_outputs=structured_outputs
    )

    batch_size = 4

    data_df = get_data_df().sample(frac=1).reset_index(drop=True)

    chunk_size = 4096
    chunk_overlap = 50

    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for i in range(0, len(data_df), batch_size):
        sample_data = data_df.iloc[i: i + batch_size, :].reset_index(drop=True)
        conversations = []
        books = [] # [book_id, book_size_in_chunks]
        for j in range(0, len(sample_data)):
            text = load_book_text(f"data/{sample_data.loc[j, 'archive']}/{sample_data.loc[j, 'file_number']}.fb2")
            text_split = text_splitter.split_text(text)
            books.append([sample_data.loc[j, "flibusta_id"], len(text_split)])
            conversations_i = [
                [{
                    "role": "system",
                    "content": "Ты - литературный критик, выдели из фрагмента информацию которую из него можно получить. тщательно анализируй поведение героев при определении характера. если персонажей можно отнести к массовке не описывай их, или объедини в одну группу. для меня очень важно находить героев, действий которых никто не ожидал и которые шокируют всех"
                },
                {
                    "role": "user",
                    "content": f"Проанализируй следующий фрагмент: \n\n{text_split[k]}"
                }] for k in range(len(text_split))
            ]
            conversations += conversations_i

        output = llm.chat(
            messages=conversations,
            sampling_params=sampling_params
        )

        while len(books):
            conversations = []
            p = 0
            books_1 = []
            for j in range(len(books)):
                if books[j][1] == 1:
                    with open(f"checkpoints/{books[j][0]}.json", mode="w", encoding="utf-8") as f:
                        f.write(output[p].outputs[0].text)
                    p += 1
                else:
                    text_i = ""
                    b_1, b_2 = books[j][0], 0
                    for k in range(books[j][1]):
                        if len(text_i + output[p].outputs[0].text) < chunk_size:
                            text_i += output[p].outputs[0].text
                        else:
                            conversation_i = [
                                {
                                    "role": "system",
                                    "content": "Ты - литературный критик, я обработал разные отрывки книги и получил из них суммаризации с такими же полями как дал тебе. для меня очень важно находить героев, действий которых никто не ожидал и которые шокируют всех"
                                },
                                {
                                    "role": "user",
                                    "content": f"Проанализируй следующие суммаризации: \n\n {text_i}"
                                }
                            ]
                            text_i = output[p].outputs[0].text
                            conversations.append(conversation_i)
                            b_2 += 1
                        p += 1
                    if len(text_i) > 0:
                        conversation_i = [
                            {
                                "role": "system",
                                "content": "Ты - литературный критик, я обработал разные отрывки книги и получил из них суммаризации с такими же полями как дал тебе. для меня очень важно находить героев, действий которых никто не ожидал и которые шокируют всех"
                            },
                            {
                                "role": "user",
                                "content": f"Проанализируй следующие суммаризации: \n\n {text_i}"
                            }
                        ]
                        conversations.append(conversation_i)
                        b_2 += 1
                    books_1.append([b_1, b_2])
            if conversations:
                output = llm.chat(
                    messages=conversations,
                    sampling_params=sampling_params
                )
            books = books_1
