import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

class BookSummarizer(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=2, nhead=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
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
        self.gte_lora.gradient_checkpointing_enable() 
        self.summarizer = BookSummarizer(hidden_dim=hidden_dim)

    def forward(self, chunk_input_ids, chunk_attention_mask):
        outputs = self.gte_lora(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
        token_embs = outputs.last_hidden_state
        
        mask = chunk_attention_mask.unsqueeze(-1)
        chunk_embeddings = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
        
        chunk_embeddings = chunk_embeddings.unsqueeze(0)
        book_embedding = self.summarizer(chunk_embeddings)
        return book_embedding