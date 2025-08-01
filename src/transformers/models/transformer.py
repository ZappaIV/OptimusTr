import torch
import torch.nn as nn

class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))  # Semplificato
        self.transformer = nn.Transformer(d_model, nhead)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    @staticmethod    
    def generate_square_subsequent_mask(sz):
        """Genera causal mask per il decoder (triangolare superiore = True)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod    
    def create_padding_mask(seq, pad_idx=0):
        """Crea mask per token di padding"""
        return (seq == pad_idx)

    def forward(self, src, tgt, pad_idx=0):
        # Embedding + positional encoding
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        
        src_len, tgt_len = src.shape[0], tgt.shape[0]
        src_emb += self.pos_encoding[:src_len, :].unsqueeze(1)
        tgt_emb += self.pos_encoding[:tgt_len, :].unsqueeze(1)
        
        # Masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_len)
        src_padding_mask = self.create_padding_mask(src.transpose(0, 1), pad_idx)
        tgt_padding_mask = self.create_padding_mask(tgt.transpose(0, 1), pad_idx)
        
        # Transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.output_projection(output)

