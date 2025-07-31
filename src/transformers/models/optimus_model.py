import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional

from src.transformers.models.encoders import Encoder
from src.transformers.models.decoders import Decoder


class OptimusTransformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size,
                 embed_dim=256, 
                 num_heads=8,
                 n_layers=6,
                 max_len=5000,
                 dropout=0.1,
                 ):
        super(OptimusTransformer, self).__init__()
        print(f"{src_vocab_size=}, {tgt_vocab_size=}")
        
        
        self.encoder = Encoder(
            src_vocab_size, 
            embed_dim, 
            num_heads, 
            n_layers, 
            max_len, 
            dropout,
            )
        
        self.decoder = Decoder(
            tgt_vocab_size, 
            embed_dim, 
            num_heads, 
            n_layers, 
            max_len, 
            dropout,
            )
        
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

        # Inizializzazione dei pesi
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(
        self, 
        src: Tensor, 
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        cross_padding_mask: Optional[Tensor] = None,
        cross_is_causal: Optional[bool] = False,
        tgt_is_causal: Optional[bool] = False,
        memory_is_causal: bool = False,
    ):
        
        # pre-checks
        is_batched = src.dim() == 3
        if (src.size(0) != tgt.size(0)) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

              
        # Encoding
        memory = self.encoder(
            src,
            mask = memory_mask,
            src_padding_mask=memory_padding_mask,
            self_is_causal=memory_is_causal
        )

        # Decoding
        decoder_output = self.decoder(
            tgt,
            memory,
            mask = tgt_mask,
            self_padding_mask = tgt_padding_mask,
            cross_padding_mask = cross_padding_mask,
            self_is_causal = tgt_is_causal,
            cross_is_causal = cross_is_causal
            )

        # Proiezione finale
        logits = self.output_projection(decoder_output)

        logits = F.softmax(logits,-1)

        return logits

    def generate(
        self,
        src: Tensor,
        max_len: int = 100,
        start_token: int = 1,
        end_token: int = 2
    ):
        """
        Genera una traduzione dato un input source
        """
        from src.transformers.models.functionals import create_cross_attention_mask
    
        self.eval()
        with torch.no_grad():
            # Encoding del source
            memory = self.encoder(
                src,
                src_padding_mask = None,
                self_is_causal = True
            )

            # Inizializza il target con il token di start
            batch_size = src.size(0)
            tgt = torch.full((batch_size, 1), start_token, device=src.device)

            for _ in range(max_len):
                # TODO: fix masking for generation
                # Crea la maschera per il target corrente
                cross_padding_mask = create_cross_attention_mask(tgt, src)

                # Decodifica
                decoder_output = self.decoder(
                    tgt,
                    memory,
                    self_is_causal=True,
                    cross_padding_mask=cross_padding_mask
                )

                # Predizione del prossimo token
                next_token_logits = F.softmax(
                    self.output_projection(decoder_output[:, -1, :]),
                    -1
                    )
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Aggiungi il token predetto
                tgt = torch.cat([tgt, next_token], dim=1)

                # Controlla se tutti i batch hanno terminato
                if torch.all(next_token.squeeze() == end_token):
                    break

            return tgt


if __name__ == '__main__':
    
    from src.transformers.models.functionals import create_cross_attention_mask
    
    config = {
        'src_vocab_size': 10**4,
        'tgt_vocab_size': 11**4,
        'embed_dim': 128,
        'num_heads': 16,
        'n_layers': 1,
        'hidden_dim': 256,
        'max_seq_length': 100,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 10,
        'warmup_steps': 4000,
        'label_smoothing': 0.1
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = OptimusTransformer(
    src_vocab_size=config['src_vocab_size'],
    tgt_vocab_size=config['tgt_vocab_size'],
    n_layers=config['n_layers'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    max_len=config['max_seq_length'],
    dropout=config['dropout'],
    ).to(device)
    
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainable: {trainable_params:,}")
    
    # src = torch.randint(1, config['src_vocab_size'], (config['batch_size'], 20))
    # tgt = torch.randint(1, config['tgt_vocab_size'], (config['batch_size'], 20))

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
    
    start_seq = torch.tensor([[1],[1],[1]])
    
    cross_padding_mask = create_cross_attention_mask(tgt, src)
    
    # Test forward pass
    print("=== Test Forward Pass ===")
    with torch.no_grad():
        output = model(
            src,
            tgt,
            cross_padding_mask = cross_padding_mask,
            tgt_is_causal = True,
            memory_is_causal = True
        )
        print(f"Input source shape: {src.shape}")
        print(f"Input target shape: {tgt.shape}")
        print(f"Output shape: {output.shape}")  # [batch_size, tgt_len, vocab_size]
    
    # Test generazione
    print("\n=== Test Generazione ===")
    with torch.no_grad():
        generated = model.generate(src, max_len=20, start_token=1, end_token=2)
        print(f"Generated sequence shape: {generated.shape}")
        print(f"First generated sequence: {generated[0].tolist()}")
    
