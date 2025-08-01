import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from src.transformers.models.attentions import MultiHeadAttention, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout=0.1,
                 ):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[Tensor] = False
    ) -> Tensor:
        # Self-attention con connessione residua
        # x = x.transpose(0,1)
        attn_output = self.self_attention(
            x,
            x,
            x,
            attn_mask = src_mask, # None
            padding_mask = src_key_padding_mask, # Filled
            is_causal = src_is_causal
            )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward con connessione residua
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout=0.1,
                 ):
        super(DecoderLayer, self).__init__()
                
        self.self_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, # Target x Target
        memory_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None, # Source x Target
        tgt_is_causal: Optional[Tensor] = None,
        memory_is_causal: Optional[Tensor] = None
    ):
        # Self-attention mascherata
        attn_output = self.self_attention(
            x,
            x,
            x,
            attn_mask = tgt_mask, # Causal
            padding_mask = tgt_key_padding_mask,
            is_causal = tgt_is_causal # None
            )
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention con l'encoder
        cross_attn_output = self.cross_attention(
            x,
            memory, 
            memory, 
            attn_mask = memory_mask, # None
            padding_mask = memory_key_padding_mask, # FIlled
            is_causal = memory_is_causal
            )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
