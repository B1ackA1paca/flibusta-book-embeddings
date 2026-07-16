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
from src.dataset import TrainDataset, collate_fn
from src.models import FullBookEncoder

import warnings 
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if torch.cuda.is_available() else "cpu"

def recall_at_k(preds, targets, k=10):
    p = F.normalize(preds.float(), dim=-1)
    t = F.normalize(targets.float(), dim=-1)
    top_idx = torch.topk(p @ t.T, k=min(k, t.size(0)), dim=-1).indices
    true_idx = torch.arange(len(p), device=p.device).unsqueeze(-1)
    return (top_idx == true_idx).any(dim=-1).float().mean().item()

class SmartLoss(nn.Module):
    def __init__(self, device="cuda", queue_size=1024, embedding_dim=768, temperature=0.05, delta=3e-5, alpha=0.7):
        super().__init__()
        embeddings = torch.rand(queue_size, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        self.register_buffer("queue", embeddings)
        self.pointer = 0
        self.alpha = alpha
        self.delta = delta
        self.mse_loss = nn.MSELoss().to(device)
        self.temperature = temperature
    def forward(self, pred_emb, true_emb):
        pred_norm = F.normalize(pred_emb, p=2, dim=-1).float()
        true_norm = F.normalize(true_emb, p=2, dim=-1).float()
        mse_loss = self.mse_loss(pred_emb, true_emb)
        with torch.no_grad():
            self.queue[self.pointer] = true_norm.squeeze().detach()
        infonnce_loss = -1 * F.log_softmax(pred_norm @ self.queue.T / self.temperature, dim=-1)[0, self.pointer]
        self.pointer = (self.pointer + 1) % self.queue.size(0)

        total_loss = self.alpha * mse_loss + (1.0 - self.alpha) * infonnce_loss
        self.alpha = max(0.0, self.alpha - self.delta)

        return total_loss

def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("model_checkpoints", exist_ok=True) 

    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    print("Loading data...")
    checkpoints_df = get_checkpoints_df()
    
    train_part, val_part = train_test_split(checkpoints_df, test_size=0.2, shuffle=True, random_state=42)
    train_part, val_part = train_part.reset_index(drop=True), val_part.reset_index(drop=True)
    
    train_loader = DataLoader(TrainDataset(train_part, tokenizer), batch_size=1, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(TrainDataset(val_part, tokenizer), batch_size=1, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
    print("Initializing model and tokenizer...")

    epochs = 50
    first_epoch = 1

    loss_fn = SmartLoss().to(device)

    model = FullBookEncoder()
    # model.load_state_dict(torch.load(f"model_checkpoints/model_{first_epoch - 1}.pth", map_location=device))
    model = model.to(device)
    model = torch.compile(model)
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

    best_metric, lag, max_lag, min_epoch, min_delta, last_change_metric = -1, 0, 7, 20, 0.02, -1

    accumulation_steps = 16

    print("Start train...")

    for epoch in range(first_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        
        for step, (true_emb, chunk_input_ids, chunk_attention_mask) in enumerate(tqdm(train_loader, desc=f"train epoch: {epoch}")):
            true_emb = true_emb.to(device)
            chunk_input_ids = chunk_input_ids.to(device)
            chunk_attention_mask = chunk_attention_mask.to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_emb = model(chunk_input_ids=chunk_input_ids, chunk_attention_mask=chunk_attention_mask)
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
            for true_emb, chunk_input_ids, chunk_attention_mask in tqdm(val_loader, desc=f"val epoch: {epoch}"):
                true_emb = true_emb.to(device)
                chunk_input_ids = chunk_input_ids.to(device)
                chunk_attention_mask = chunk_attention_mask.to(device)
                
                with torch.amp.autocast("cuda"):
                    pred_emb = model(chunk_input_ids=chunk_input_ids, chunk_attention_mask=chunk_attention_mask)
                val_preds.append(pred_emb.detach().cpu())
                val_trues.append(true_emb.detach().cpu())
                
        preds_tensor = torch.cat(val_preds, dim=0)  
        targets_tensor = torch.cat(val_trues, dim=0)
        recall = recall_at_k(preds_tensor, targets_tensor, k=5)
        print(f"recall: {recall}")
        with open("ans.txt", "a", encoding="utf-8") as f:
            f.write(f"epoch: {epoch}\n recall: {recall}\n")
            
        torch.save(model.state_dict(), f"model_checkpoints/model_{epoch}.pth")
        if recall > best_metric:
            if recall > (1 + min_delta) * last_change_metric:
                lag = 0
                last_change_metric = recall
            best_metric = recall
            best_parameters = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            lag += 1
            if lag >= max_lag and epoch >= min_epoch:
                break
            
    print("end, load weights")
    if best_metric > -1:
        model.load_state_dict(best_parameters)

if __name__ == "__main__":
    main()