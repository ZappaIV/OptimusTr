import torch
import torch.nn as nn

from models.components.attentions import MultiHeadAttention, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 d_ff, 
                 dropout=0.1,
                 use_nn_mha=False,
                 ):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads) if not use_nn_mha else nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self-attention con connessione residua
        if isinstance(self.self_attention, MultiHeadAttention):
            attn_output = self.self_attention(x, x, x, mask)
        elif isinstance(self.self_attention, nn.MultiheadAttention):
            attn_output, _ = self.self_attention(x, x, x,  key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward con connessione residua
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 d_ff, 
                 dropout=0.1,
                 use_nn_mha=False,
                 ):
        super(DecoderLayer, self).__init__()
                
        self.self_attention = MultiHeadAttention(embed_dim, num_heads) if not use_nn_mha else nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads) if not use_nn_mha else nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Self-attention mascherata
        if isinstance(self.self_attention, MultiHeadAttention):
            attn_output = self.self_attention(x, x, x, tgt_mask)
        elif isinstance(self.self_attention, nn.MultiheadAttention):
            attn_output, _ = self.self_attention(x, x, x, attn_mask = tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention con l'encoder
        if isinstance(self.cross_attention, MultiHeadAttention):
            cross_attn_output = self.cross_attention(
                x,
                encoder_output, 
                encoder_output, 
                src_mask
                )
        elif isinstance(self.cross_attention, nn.MultiheadAttention):
            cross_attn_output, _ = self.cross_attention(
                x,
                encoder_output, 
                encoder_output, 
                src_mask
            )    
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
    
if __name__ == '__main__':

    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8
    
    la_tensor = torch.ones(batch_size, embed_dim, dtype=int)
    en_tensor = torch.ones(batch_size, embed_dim, dtype=int)
    print(la_tensor.shape)
    
    embedding = nn.Embedding(10**4, embedding_dim=embed_dim)
    la_embedding = embedding(la_tensor)
    en_embedding = embedding(en_tensor)
    
    encoder_layer = EncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_head,
        d_ff=d_ff,
        use_nn_mha=True
    )    
    decoder_layer = DecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_head,
        d_ff=d_ff,
        use_nn_mha=True
    )  
    
    print(
        f"\n{la_tensor.shape=}"
        f"\n{la_embedding.shape=}",
        f"\n{encoder_layer(la_embedding).shape=}",
        f"\n{decoder_layer(en_embedding, encoder_layer(la_embedding)).shape=}"
    )
    del la_embedding, en_embedding, encoder_layer, decoder_layer, embedding