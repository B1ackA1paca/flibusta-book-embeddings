import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from src.data_prep import get_checkpoints_df, clean_text

device = ("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем собранный checkpoints_df
checkpoints_df = get_checkpoints_df()

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)
model.eval()

os.makedirs("book_embeddings", exist_ok=True)
df = checkpoints_df[~checkpoints_df["file_number"].apply(lambda x: os.path.exists(f"book_embeddings/book_{x}.npy"))].reset_index(drop=True)

batch_size = 32

for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i : i + batch_size]
    texts = [clean_text(row["semantic_passport"]) for _, row in batch_df.iterrows()]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    token_embs = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
    embs_np = embs.cpu().numpy()
    
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        np.save(f"book_embeddings/book_{row['file_number']}.npy", embs_np[idx])