import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math
from typing import Callable, Optional, TYPE_CHECKING, Union

####################
#                  #
# MASKING FUNCTION #
#                  #
####################

class AttentionBlock(nn.Module):
    
    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 num_heads, 
                 dropout=0.0
                 ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        embed_dim:int, 
        num_heads: int
    ):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim) # Query
        self.w_k = nn.Linear(embed_dim, embed_dim) # Key
        self.w_v = nn.Linear(embed_dim, embed_dim) # Value
        self.w_o = nn.Linear(embed_dim, embed_dim) # Norm Scaled-dot-Product

        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def scaled_dot_product_attention(
        query: Tensor,
        key: Tensor, 
        value: Tensor,
        attn_mask: Optional[Tensor] = None, 
        padding_mask: Optional[Tensor] = None,
        dropout_p: Optional[float] = 0.0,
        is_causal: Optional[bool] = False,
        scale: Optional[float] = None,
        enable_gqa: Optional[bool]=False
    ) -> tuple[Tensor, Tensor]:

        from src.transformers.models.functionals import clear_nan, mask_fill_combined 
        
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), -1e9)
            attn_bias.to(query.dtype)

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = mask_fill_combined(
            attention_scores=attn_weight,
            attn_mask=attn_mask,
            padding_mask=padding_mask
        )        
        attn_weight += attn_bias
        attn_weight = clear_nan(torch.softmax(attn_weight, dim=-1)) + 1e-12
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        
        return attn_weight @ value, attn_weight

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor], 
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = False
    ):
        """
        Forward pass del Multi-Head Attention
        
        Args:
            query: Tensor query [batch_size, seq_len_q, d_model]
            key: Tensor key [batch_size, seq_len_k, d_model] 
            value: Tensor value [batch_size, seq_len_v, d_model]
            mask: Maschera [batch_size, seq_len_q, seq_len_k] o [batch_size, 1, seq_len_q, seq_len_k]
        
        Returns:
            output: Output dell'attention [batch_size, seq_len_q, d_model]
            attention_weights: Pesi di attenzione [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        
        batch_size, seq_len_q, embed_dim = query.shape

        if key is None:
            key = query
        if value is None:
            value = query

        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Trasformazioni lineari e reshape per multi-head
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)

        # Applica l'attenzione
        context, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask, padding_mask=padding_mask,
            is_causal=is_causal)

        # Concatena le teste
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
            )

        # Proiezione finale
        output = self.w_o(context)

        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        return self.norm(x + self.linear2(self.dropout(F.relu(x))))
    
if __name__ == '__main__':

    from src.transformers.models.functionals import ( generate_square_subsequent_mask, create_cross_attention_mask )
    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8

    src = torch.tensor(
        [[ 1,3,4,2,0,0,0],
        [ 1,3,4,4,2,0,0],
        [ 1,2,0,0,0,0,0]]
    )
        
    # en_tensor = torch.randint(1, 10**4,[batch_size, seq_len], dtype=int)
    tgt = torch.tensor(
        [[ 1,3,2,0,0,0,0,0],
        [ 1,3,3,3,2,0,0,0],
        [ 1,3,2,0,0,0,0,0]]
    )

    print(src.shape, tgt.shape)

    embedding = nn.Embedding(10**4, embedding_dim=embed_dim)
    src_embedding = embedding(src)
    tgt_embedding = embedding(tgt)

    print(src_embedding.shape, tgt_embedding.shape)
    memory_padding_mask = (src == 0)

    multihead_block = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_head   
    )

    print(
        f"\n{src_embedding.shape=}"
        f"\n{tgt_embedding.shape=}",
        f"\n{multihead_block(tgt_embedding, src_embedding, src_embedding, is_causal=False).shape=}",
        f"\n{multihead_block(tgt_embedding, src_embedding, src_embedding, padding_mask=memory_padding_mask).shape=}",

    )

