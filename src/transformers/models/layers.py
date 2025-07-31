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
        mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        self_is_causal: Optional[Tensor] = False
    ) -> Tensor:
        # Self-attention con connessione residua
        # x = x.transpose(0,1)
        attn_output = self.self_attention(
            x,
            x,
            x,
            mask,
            padding_mask = src_padding_mask,
            is_causal = self_is_causal
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
        mask: Optional[Tensor] = None,
        self_padding_mask: Optional[Tensor] = None, # Target x Target
        cross_padding_mask: Optional[Tensor] = None, # Source x Target
        self_is_causal: Optional[Tensor] = None,
        cross_is_causal: Optional[Tensor] = None
    ):
        # Self-attention mascherata
        attn_output = self.self_attention(
            x,
            x,
            x,
            mask = mask,
            padding_mask = self_padding_mask,
            is_causal = self_is_causal
            )
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention con l'encoder
        cross_attn_output = self.cross_attention(
            x,
            memory, 
            memory, 
            padding_mask = cross_padding_mask,
            is_causal = cross_is_causal
            )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
