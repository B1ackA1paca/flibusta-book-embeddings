import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data_prep import get_checkpoints_df, read_fb2
from src.dataset import TrainDataset
from src.models import FullBookEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if torch.cuda.is_available() else "cpu"

def recall_at_k(preds, targets, k=5):
    p = F.normalize(preds.float(), dim=-1)
    t = F.normalize(targets.float(), dim=-1)
    top_idx = torch.topk(p @ t.T, k=min(k, t.size(0)), dim=-1).indices
    true_idx = torch.arange(len(p), device=p.device).unsqueeze(-1)
    return (top_idx == true_idx).any(dim=-1).float().mean().item()

class SmartLoss(nn.Module):
    def __init__(self, device="cuda", queue_size=1024, embedding_dim=768, temperature=0.05, delta=1e-5):
        super().__init__()
        embeddings = torch.rand(queue_size, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        self.register_buffer("queue", embeddings)
        self.pointer = 0
        self.alpha = 1.0
        self.delta = delta
        self.mse_loss = nn.MSELoss().to(device)
        self.temperature = temperature

    def forward(self, pred_emb, true_emb):
        pred_norm = F.normalize(pred_emb, p=2, dim=-1) 
        true_norm = F.normalize(true_emb, p=2, dim=-1) 
        mse_loss = self.mse_loss(pred_emb, true_emb)
        
        with torch.no_grad():
            self.queue[self.pointer] = true_norm.squeeze().detach()
            
        infonnce_loss = -1 * F.log_softmax(pred_norm @ self.queue.T / self.temperature, dim=-1)[0, self.pointer]
        self.pointer = (self.pointer + 1) % self.queue.size(0)

        total_loss = self.alpha * mse_loss + (1.0 - self.alpha) * infonnce_loss
        self.alpha = max(0.0, self.alpha - self.delta)

        return total_loss

def main():
    torch.cuda.empty_cache()
    
    print("Loading data...")
    checkpoints_df = get_checkpoints_df()
    
    train_part, val_part = train_test_split(checkpoints_df, test_size=0.2, shuffle=True, random_state=42)
    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)
    
    train_loader = DataLoader(TrainDataset(train_part), shuffle=True, batch_size=1)
    val_loader = DataLoader(TrainDataset(val_part), shuffle=False, batch_size=1)
    
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    model = FullBookEncoder().to(device)
    
    optimizer = torch.optim.AdamW([
        {
            "params": filter(lambda p: p.requires_grad, model.gte_lora.parameters()),
            "lr": 1e-5,
            "weight_decay": 0.01
        }, 
        {
            "params": filter(lambda p: p.requires_grad, model.summarizer.parameters()),
            "lr": 2e-4,
            "weight_decay": 0.01
        }
    ])
    
    loss_fn = SmartLoss(device=device).to(device)
    
    epochs = 50
    best_metric = -1
    lag = 0
    max_lag = 7
    min_epoch = 20
    accumulation_steps = 16
    best_parameters = None

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        for step, (true_emb, file_number, archive) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            true_emb = true_emb.to(device)
            
            path = f"./data/{archive[0]}/{file_number[0]}.fb2"
            try:
                text_list = read_fb2(path)
                if not text_list:
                    continue
                N = 4
                indices = np.linspace(0, len(text_list) - 1, N, dtype=int)
                text_list = [text_list[i] for i in indices]
            except Exception:
                continue
            
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == "cuda" else torch.float32):
                inputs = tokenizer(text_list, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(device)    
                pred_emb = model(chunk_input_ids=inputs["input_ids"], chunk_attention_mask=inputs["attention_mask"])
                
                loss = loss_fn(pred_emb, true_emb)
                loss = loss / accumulation_steps
                
            loss.backward()
        
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
        model.eval()
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for true_emb, file_number, archive in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                true_emb = true_emb.to(device)
                path = f"./data/{archive[0]}/{file_number[0]}.fb2"
                try:
                    text_list = read_fb2(path)
                    if not text_list:
                        continue
                    N = 4
                    indices = np.linspace(0, len(text_list) - 1, N, dtype=int)
                    text_list = [text_list[i] for i in indices]
                except Exception:
                    continue
                
                with torch.amp.autocast(device_type=device_type):
                    inputs = tokenizer(text_list, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(device)    
                    pred_emb = model(chunk_input_ids=inputs["input_ids"], chunk_attention_mask=inputs["attention_mask"])
                    
                val_preds.append(pred_emb)
                val_trues.append(true_emb)
                
        if val_preds and val_trues:
            preds_tensor = torch.cat(val_preds, dim=0)  
            targets_tensor = torch.cat(val_trues, dim=0)
            recall = recall_at_k(preds_tensor, targets_tensor, k=5)
            print(f"Epoch {epoch} - Recall@5: {recall:.4f}")
            
            with open("ans.txt", "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch} - Recall@5: {recall:.4f}\n")
                
            if recall > best_metric:
                best_metric = recall
                lag = 0
                best_parameters = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                lag += 1
                if lag >= max_lag and epoch >= min_epoch:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch} - Validation skipped")

    print("Training finished. Loading best weights...")
    if best_parameters is not None:
        model.load_state_dict(best_parameters)

    print("Saving model to model.pth...")
    torch.save(model.to("cpu"), "model.pth")
    print("Done!")

if __name__ == "__main__":
    main()