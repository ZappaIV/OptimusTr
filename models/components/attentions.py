import torch
import torch.nn as nn
import torch.nn.functional as F
import math

####################
#                  #
# MASKING FUNCTION #
#                  #
####################

def create_padding_mask(seq, pad_token_id=0):
    """
    Crea una maschera per il padding
    
    Args:
        seq: Sequenza di token [batch_size, seq_len]
        pad_token_id: ID del token di padding
    
    Returns:
        mask: Maschera [batch_size, 1, seq_len, seq_len]
    """
    # Identifica le posizioni non-padding
    padding_mask = (seq != pad_token_id) # [batch_size, 1, 1, seq_len]
    
    # Opzione 1: Maschera per una singola sequenza del batch (es. la prima)
    single_seq_mask = padding_mask[0]  # Shape: (7,)
    # Per self-attention, espandi a (L, S)
    mask = single_seq_mask.unsqueeze(0) & single_seq_mask.unsqueeze(1)
    # Espandi per creare maschera bidimensionale
    # seq_len = seq.size(1)
    # mask = mask.expand(-1, -1, seq_len, -1)  # [batch_size, 1, seq_len, seq_len]
    
    return mask

def create_causal_mask(seq_len, device):
    """
    Crea una maschera causale (triangolare inferiore) per prevenire l'attention su token futuri
    
    Args:
        seq_len: Lunghezza della sequenza
        device: Device su cui creare il tensore
    
    Returns:
        mask: Maschera causale [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)

def create_cross_attention_mask(query_seq, key_seq, pad_token_id=0):
    """
    Crea una maschera per cross-attention considerando il padding
    
    Args:
        query_seq: Sequenza query [batch_size, seq_len_q]
        key_seq: Sequenza key [batch_size, seq_len_k]
        pad_token_id: ID del token di padding
    
    Returns:
        mask: Maschera [batch_size, 1, seq_len_q, seq_len_k]
    """
    # Maschera per le posizioni non-padding nelle key
    key_mask = (key_seq != pad_token_id).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len_k]
    
    # Espandi per tutte le posizioni query
    seq_len_q = query_seq.size(1)
    mask = key_mask.expand(-1, -1, seq_len_q, -1)  # [batch_size, 1, seq_len_q, seq_len_k]
    
    return mask


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

    @staticmethod
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tuple[torch.Tensor, torch.Tensor]:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight

    def forward(self, query, key=None, value=None, mask=None, is_causal = False):
        """
        Forward pass del Multi-Head Attention
        
        Args:
            query: Tensor query [batch_size, seq_len_q, d_model]
            key: Tensor key [batch_size, seq_len_k, d_model] (se None, usa query per self-attention)
            value: Tensor value [batch_size, seq_len_v, d_model] (se None, usa query per self-attention)
            mask: Maschera [batch_size, seq_len_q, seq_len_k] o [batch_size, 1, seq_len_q, seq_len_k]
        
        Returns:
            output: Output dell'attention [batch_size, seq_len_q, d_model]
            attention_weights: Pesi di attenzione [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        
        batch_size = query.size(0)

        if key is None:
            key = query
        if value is None:
            value = query

        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Trasformazioni lineari e reshape per multi-head
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)

        # Applica l'attenzione
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask, is_causal=is_causal)

        # Concatena le teste
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)

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