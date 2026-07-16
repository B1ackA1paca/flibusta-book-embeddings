import torch
import numpy as np
from tqdm import tqdm
from src.data_prep import read_fb2

class TrainDataset():
    def __init__(self, checkpoints_df, tokenizer, k_chunks=64):
        self.checkpoints_df = checkpoints_df.reset_index(drop=True)
        self.k_chunks = k_chunks
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng()
        
        self.cached_books = {}
        for i, row in tqdm(self.checkpoints_df.iterrows(), total=len(self.checkpoints_df), desc="Caching books"):
            path = f"data/{row['archive']}/{row['file_number']}.fb2"
            self.cached_books[i] = np.array(read_fb2(path), dtype=object)

    def __len__(self):
        return self.checkpoints_df.shape[0]

    def __getitem__(self, i):
        embedding = np.load(f"book_embeddings/book_{self.checkpoints_df.loc[i, 'file_number']}.npy")
        emb_tensor = torch.tensor(embedding, dtype=torch.float16)

        text_list = self.cached_books[i]
        if len(text_list) < self.k_chunks:
            sampled_text = text_list.tolist()
        else:
            indices = self.rng.choice(len(text_list), size=self.k_chunks, replace=False, shuffle=False)
            sampled_text = text_list[indices].tolist()
        
        tokens = self.tokenizer(
            sampled_text,
            truncation=True,
            max_length=448,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )
        return emb_tensor, tokens["input_ids"], tokens["attention_mask"]

def collate_fn(batch):
    emb, input_ids, attention_mask = batch[0]
    return emb.unsqueeze(0), input_ids, attention_mask