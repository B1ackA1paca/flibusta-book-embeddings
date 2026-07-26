import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

class Router(nn.Module):
    def __init__(self, hidden_dim=768, rang=8, k=5, total_budget=32):
        super().__init__()
        self.k = k
        self.total_budget = total_budget

        self.proj_q = nn.Linear(hidden_dim, rang)
        self.proj_k = nn.Linear(hidden_dim, rang)

        self.v1 = nn.Parameter(torch.randn(rang, 1))
        self.v2 = nn.Parameter(torch.randn(rang, 1))

        nn.init.kaiming_uniform_(self.v1)
        nn.init.kaiming_uniform_(self.v2)

        self.out_layer = nn.Softmax(dim=-1)
    def forward(self, x, mask):
        # x = [n_batchs. max_seq_len, 768]
        # mask = [n_batchs, max_seq_len]
        q = self.proj_q(x)
        k = self.proj_k(x)

        mask = mask.unsqueeze(-1).to(x.dtype) # [n_batchs, max_seq_len, 1]

        q = q * mask
        k = k * mask

        matrix_p = torch.bmm(q.transpose(1, 2), k)
        real_len = mask.sum(dim=1, keepdim=True).clamp(min=1) # [N, 1, 1]

        matrix_p /= real_len

        out = torch.matmul(matrix_p, self.v1).squeeze(-1) # [n_batchs, rang]
        scores = torch.matmul(out, self.v2).squeeze(-1) # [n_batchs]

        sorted_idx = torch.argsort(scores, descending=True)
        top_k_idx = sorted_idx[:self.k]
        other_idx = sorted_idx[self.k:]

        rand_idx = other_idx[torch.randperm(len(other_idx), device=x.device)[:self.total_budget - self.k]]

        indices, _ = torch.sort(torch.cat([top_k_idx, rand_idx]))

        weights = torch.softmax(scores[indices], dim=-1).unsqueeze(-1).unsqueeze(-1)
        filtered_x = x[indices] * weights

        return filtered_x, indices

class BookSummarizer(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=2, nhead=8, max_chunks=100):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_chunks, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, chunk_embeddings):
        batch_size = chunk_embeddings.size(0)
        seq_len = chunk_embeddings.size(1)
        chunk_embeddings = chunk_embeddings + self.pos_embedding[:, :seq_len, :]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, chunk_embeddings), dim=1)
        
        output = self.transformer(x)
        return output[:, 0]

class FullBookEncoder(nn.Module):
    def __init__(self, model_name="Alibaba-NLP/gte-multilingual-base", hidden_dim=768):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa", trust_remote_code=True)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["qkv_proj", "o_proj", "up_gate_proj", "down_proj"],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.gte_lora = get_peft_model(base_model, peft_config)
        self.gte_lora.enable_input_require_grads()
        self.gte_lora.gradient_checkpointing_enable() 
        self.router = Router(hidden_dim=hidden_dim)
        self.summarizer = BookSummarizer(hidden_dim=hidden_dim)

    def forward(self, chunk_input_ids, chunk_attention_mask):
        outputs = self.gte_lora(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
        token_embs = outputs.last_hidden_state
        
        mask = chunk_attention_mask.unsqueeze(-1)
        chunk_embeddings = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
        
        chunk_embeddings = chunk_embeddings.unsqueeze(0)
        book_embedding = self.summarizer(chunk_embeddings)
        return book_embedding