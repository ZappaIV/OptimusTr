import torch
from torch import Tensor
from typing import Optional

def generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), -1e9, dtype=dtype, device=device),
        diagonal=1,
    )
    
def create_padding_mask(
    seq: Tensor,
    pad_token_id: Optional[int] = 0,
) -> Tensor:
    """
    Crea una maschera per il padding
    
    Args:
        seq: Sequenza di token [batch_size, seq_len]
        pad_token_id: ID del token di padding
    
    Returns:
        mask: Maschera [batch_size, 1, seq_len, seq_len]
    """
    # Identifica le posizioni non-padding
    padding_mask = (seq == pad_token_id) # [batch_size, 1, 1, seq_len]
    
    # Opzione 1: Maschera per una singola sequenza del batch (es. la prima)
    single_seq_mask = padding_mask[0]  # Shape: (7,)
    # Per self-attention, espandi a (L, S)
    mask = single_seq_mask.unsqueeze(0) & single_seq_mask.unsqueeze(1)
    # Espandi per creare maschera bidimensionale
    # seq_len = seq.size(1)
    # mask = mask.expand(-1, -1, seq_len, -1)  # [batch_size, 1, seq_len, seq_len]
    
    return torch.zeros_like(padding_mask, dtype=torch.float).masked_fill_(
              padding_mask, -1e9
            )

def create_causal_mask(
    seq_len: int, 
    device: Optional[torch.device] = None, 
) -> Tensor:
    """
    Crea una maschera causale (triangolare inferiore) per prevenire l'attention su token futuri
    
    Args:
        seq_len: Lunghezza della sequenza
        device: Device su cui creare il tensore
    
    Returns:
        mask: Maschera causale [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask # .unsqueeze(0).unsqueeze(0)

def create_cross_attention_mask(
    query_seq: Tensor,
    key_seq: Tensor,
    q_pad_token_id: Optional[int] = 0,
    k_pad_token_id: Optional[int] = 0,
) -> Tensor:
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
    key_mask = (key_seq == k_pad_token_id).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len_k]
    query_mask = (query_seq == q_pad_token_id).unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1, seq_len_v]
    # Espandi per tutte le posizioni query
    seq_len_q = query_seq.size(1)
    seq_len_k = key_seq.size(1)
    key_mask = key_mask.expand(-1, -1, seq_len_q, -1)  # [batch_size, 1, seq_len_q, seq_len_k]
    query_mask = query_mask.expand(-1, -1, seq_len_k, -1)  # [batch_size, 1, seq_len_k, seq_len_q]
    
    key = key_mask + query_mask.transpose(-2,-1) # [batch_size, 1, seq_len_q, seq_len_k]
    
    return torch.zeros_like(key, dtype=torch.float).masked_fill_(
              key, -1e9
            )
    
def clear_nan(tensor: torch.Tensor):
    return torch.where(torch.isnan(tensor), 
                            torch.zeros_like(tensor), 
                            tensor)
    
def mask_fill_combined(
    attention_scores: Tensor,
    padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None
):
    # attention_scores: [batch, num_heads, tgt_len, tgt_len]
    # tgt_mask: [tgt_len, tgt_len] - causale
    # tgt_key_padding_mask: [batch, tgt_len] - padding

    batch_size, num_heads, tgt_len, _ = attention_scores.shape
    
    # 1. APPLICA CAUSAL MASK (stessa per tutto il batch e tutte le teste)
    if attn_mask is not None:
        # Espandi per batch e heads: [1, 1, tgt_len, tgt_len]
        expanded_tgt_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(expanded_tgt_mask, -1e9)
    
    # 2. APPLICA PADDING MASK (diversa per ogni batch item)
    if padding_mask is not None:
        # Espandi per heads e query positions: [batch, 1, 1, tgt_len] 
        expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores.masked_fill(expanded_padding_mask, -1e9)
    
    return attention_scores
    