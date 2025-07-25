import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.components.encoders import Encoder
from models.components.decoders import Decoder

class TransformerTranslation(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size,
                 embed_dim=256, 
                 num_heads=8,
                 d_ff=256, 
                 n_layers=6,
                 max_len=5000,
                 dropout=0.1,
                 use_nn_mha = False
                 ):
        super(TransformerTranslation, self).__init__()
        print(f"{src_vocab_size=}, {tgt_vocab_size=}")
        
        self.use_nn_mha = use_nn_mha
        
        self.encoder = Encoder(
            src_vocab_size, 
            embed_dim, 
            num_heads, 
            d_ff, 
            n_layers, 
            max_len, 
            dropout,
            use_nn_mha
            )
        
        self.decoder = Decoder(
            tgt_vocab_size, 
            embed_dim, 
            num_heads, 
            d_ff, 
            n_layers, 
            max_len, 
            dropout,
            use_nn_mha
            )
        
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

        # Inizializzazione dei pesi
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Genera la maschera triangolare per il decoder (causal mask)"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def create_pad_mask(self, seq, pad_id):
        return (seq == pad_id)
    
    def make_src_mask(self, src):
        # Maschera per padding nel source
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        # Maschera per padding nel target
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        # Maschera causale per impedire di vedere token futuri
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()

        # Combina le maschere
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def encode(self, src, src_pad_idx=0):
        """Encoding della sequenza sorgente"""
        src_mask = self.create_padding_mask(src, src_pad_idx)
        return self.encoder(src, src_mask)
    
    def decode_step(self, tgt, encoder_output, src_mask=None):
        """Singolo step di decodifica (utile per inferenza)"""
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return self.output_projection(decoder_output)


    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        # Crea le maschere
        if not self.use_nn_mha:
            src_mask = self.make_src_mask(src)
            tgt_mask = self.make_tgt_mask(tgt)
        else:
            src_mask = self.create_pad_mask(src, src_pad_idx)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        # Encoding
        encoder_output = self.encoder(src, src_mask)

        # Decoding
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)

        # Proiezione finale
        output = self.output_projection(decoder_output)

        # soft_max_output = F.softmax(output,2)

        return output

    def generate(self, src, max_len=100, start_token=1, end_token=2):
        """
        Genera una traduzione dato un input source
        """
        self.eval()
        with torch.no_grad():
            # Encoding del source
            src_mask = self.make_src_mask(src)
            encoder_output = self.encoder(src, src_mask)

            # Inizializza il target con il token di start
            batch_size = src.size(0)
            tgt = torch.full((batch_size, 1), start_token, device=src.device)

            for _ in range(max_len):
                # Crea la maschera per il target corrente
                tgt_mask = self.make_tgt_mask(tgt)

                # Decodifica
                decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)

                # Predizione del prossimo token
                next_token_logits = F.softmax(
                    self.output_projection(decoder_output[:, -1, :]),
                    2
                    )
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Aggiungi il token predetto
                tgt = torch.cat([tgt, next_token], dim=1)

                # Controlla se tutti i batch hanno terminato
                if torch.all(next_token.squeeze() == end_token):
                    break

            return tgt


if __name__ == '__main__':
    config = {
        'src_vocab_size': 10**4,
        'tgt_vocab_size': 11**4,
        'embed_dim': 128,
        'num_heads': 16,
        'n_layers': 1,
        'hidden_dim': 256,
        'd_ff': 256,
        'max_seq_length': 100,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 10,
        'warmup_steps': 4000,
        'label_smoothing': 0.1
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TransformerTranslation(
    src_vocab_size=config['src_vocab_size'],
    tgt_vocab_size=config['tgt_vocab_size'],
    n_layers=config['n_layers'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff'],
    max_len=config['max_seq_length'],
    dropout=config['dropout'],
    use_nn_mha=True
    ).to(device)
    
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainable: {trainable_params:,}")