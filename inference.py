import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import numpy as np

import chromadb
from tqdm import tqdm
import sqlite3

from concurrent.futures import ThreadPoolExecutor

import time
from src.models import FullBookEncoder
from src.dataset import TestDataset, test_collate_fn

from src.data_prep import get_genres, get_data_df

db_name = "books_embeddings.db"
conn = sqlite3.connect(db_name, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (flibusta_id TEXT PRIMARY KEY, vector BLOB)")
conn.commit()

data_df = get_data_df()

genres = get_genres(data_df, "genre")

fantastic_genres = ['adventure_fantasy', 'asian_fantasy', 'child_sf', 'child_sf_fantasy', 'child_sf_horror',
'child_sf_hronoopera', 'child_sf_space', 'dark_fantasy', 'everyday_fantasy', 'fairy_fantasy',
'fantasy_alt_hist', 'fantasy_det', 'foreign_sf', 'historical_fantasy', 'hronoopera',
'magic_school', 'nsf', 'popadancy', 'russian_fantasy', 'sf', 'sf_action', 'sf_cyberpunk',
'sf_detective', 'sf_epic', 'sf_etc', 'sf_fantasy', 'sf_fantasy_city', 'sf_heroic',
'sf_history', 'sf_horror', 'sf_humor', 'sf_industrial_magic', 'sf_litrpg', 'sf_mystic',
'sf_postapocalyptic', 'sf_realrpg', 'sf_social', 'sf_space', 'sf_stimpank', 'sf_su',
'sf_technofantasy', 'slavic_fantasy', 'Боевое фэнтези', 'ЛитРПГ', 'Эпическое фэнтези',
'adventure', 'det_artifact', 'gothic_novel', 'thriller', 'thriller_mystery']

mask = data_df['genre'].str.contains(f'(?:^|:)({"|".join(fantastic_genres)})(?=:|$)')
filtered_df = data_df[mask].reset_index(drop=True)

device = ("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

test_loader = DataLoader(TestDataset(filtered_df, tokenizer), batch_size=1, shuffle=False, collate_fn=test_collate_fn, pin_memory=True)

# executor = ThreadPoolExecutor(max_workers=1)

def save_batch_to_db(flibusta_ids, vectors):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    data = [(id, vec.tobytes()) for id, vec in zip(flibusta_ids, vectors)]
    cursor.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", data)
    conn.commit()
    conn.close()
    def forward(self, chunk_input_ids, chunk_attention_mask):
        outputs = self.gte_lora(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
        token_embs = outputs.last_hidden_state
        
        mask = chunk_attention_mask.unsqueeze(-1)
        chunk_embeddings = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
        
        chunk_embeddings = chunk_embeddings.unsqueeze(0)
        book_embedding = self.summarizer(chunk_embeddings)
        return book_embedding


model = FullBookEncoder()
state_dict = torch.load("model_checkpoints/model_27.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model = torch.compile(model)
model.eval()
with torch.inference_mode():
    for input_ids, attention_mask, flibusta_ids in tqdm(test_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.amp.autocast("cuda"):
            pred_emb = model(chunk_input_ids=input_ids, chunk_attention_mask=attention_mask)
        embeddings_np = pred_emb.cpu().numpy()
        save_batch_to_db(flibusta_ids, embeddings_np)
        # executor.submit(save_batch_to_db, flibusta_ids, embeddings_np)

        # if executor._work_queue.qsize() > 300:
        #     time.sleep(0.1)
# executor.shutdown(wait=True)
print("all good, stop")