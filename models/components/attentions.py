import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    def __init__(self, 
                 embed_dim, 
                 num_heads
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

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Calcola i punteggi di attenzione
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Applica la maschera se fornita
        if mask is not None:
            _MASKING_VALUE = -1e+30 if scores.dtype == torch.float32 else -1e+4
            scores = scores.masked_fill(mask == 0, _MASKING_VALUE)

        # Applica softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Applica l'attenzione ai valori
        context = torch.matmul(attention_weights, V)

        return context, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Trasformazioni lineari e reshape per multi-head
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Applica l'attenzione
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatena le teste
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

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

    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8

    la_tensor = torch.ones(batch_size, embed_dim, dtype=int)
    print(la_tensor.shape)
    embedding = nn.Embedding(10**4, embedding_dim=embed_dim)

    attention_block = AttentionBlock(
        embed_dim=embed_dim,
        hidden_dim=200,
        num_heads=num_head
        )

    multihead_block = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_head   
    )

    ff_block = FeedForward(
        embed_dim=embed_dim  
    )

    la_embedding = embedding(la_tensor)
    print(
        f"\n{la_tensor.shape=}"
        f"\n{la_embedding.shape=}",
        f"\n{attention_block(la_embedding).shape=}", 
        f"\n{multihead_block(la_embedding, la_embedding, la_embedding).shape=}",
        f"\n{ff_block(la_embedding).shape=}"
    )

    del la_embedding, multihead_block, attention_block, embedding, ff_block    