import torch
import numpy as np

class TrainDataset():
    def __init__(self, checkpoints_df):
        self.checkpoints_df = checkpoints_df
    def __len__(self):
        return self.checkpoints_df.shape[0]
    def __getitem__(self, i):
        embedding = np.load(f"book_embeddings/book_{self.checkpoints_df.loc[i, 'file_number']}.npy")
        emb_tensor = torch.tensor(embedding, dtype=torch.float16)

        return emb_tensor, self.checkpoints_df.loc[i, 'file_number'], self.checkpoints_df.loc[i, 'archive']