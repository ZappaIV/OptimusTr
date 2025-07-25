import torch
import os, json, re
import pandas as pd 
import random
import numpy as np 

# import training.training_func

from training.training_func import train_epoch, evaluate, save_checkpoint, load_checkpoint, LabelSmoothingLoss, NoamScheduler
from models.optimus_la_en import TransformerTranslation
from tokenizers_utl.dataloaders import L2NDataset, collate_fn
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_path = 'datasets/vocabulary.json'
train_dataset_path = 'datasets/corpus_la_en_train.json'
val_dataset_path = 'datasets/corpus_la_en_val.json'

config = {
        'src_vocab_size': None,
        'tgt_vocab_size': None,
        'embed_dim': 768,
        'num_heads': 16,
        'n_layers': 8,
        'hidden_dim': 1024,
        'd_ff': 1024,
        'max_seq_length': 10000,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 50,
        'warmup_steps': 4000,
        'label_smoothing': 0.1
    }


print('Carico il Dataset')

df_train = pd.read_json(train_dataset_path, orient='records')
df_val = pd.read_json(val_dataset_path, orient='records')

if os.path.exists(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as fp:
        vocab = json.load(fp)
    la_vocab = vocab['la_vocab']
    en_vocab = vocab['en_vocab']
else:
    raise ('No Vocab Found')

print('Caricati TrainSet, ValSet e Vocabulary')

train_dataset = L2NDataset(la_corpus=df_train['la_tok'].values.tolist(),
                           en_corpus=df_train['en_tok'].values.tolist(),
                           la_vocab=la_vocab,
                           en_vocab=en_vocab)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = L2NDataset(la_corpus=df_val['la_tok'].values.tolist(),
                           en_corpus=df_val['en_tok'].values.tolist(),
                           la_vocab=la_vocab,
                           en_vocab=en_vocab)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

config['src_vocab_size'] = len(la_vocab)
config['tgt_vocab_size'] = len(en_vocab)

print('Esempio di Elemento del Dataset tokenizzato')
la_tensor, en_tensor = next(iter(train_dataloader))
print(la_tensor.shape, en_tensor.shape, la_tensor[0], en_tensor[0])

print(f"Config: {json.dumps(config, indent=2)}")

print('inizializzo il modello')
model = TransformerTranslation(
    src_vocab_size=len(la_vocab),
    tgt_vocab_size=len(en_vocab),
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

print('Inizializzo  Ottimizzatore e LossFunc')
criterion = LabelSmoothingLoss(config['tgt_vocab_size'],
                                  config['label_smoothing'], ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, config['embed_dim'], config['warmup_steps'])

print(f"Modello con {sum(p.numel() for p in model.parameters()):,} parametri")


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Training loop
best_val_loss = float('inf')

model.to(device)

for epoch in range(config['num_epochs']):
    print(f"\n--- Epoca {epoch+1}/{config['num_epochs']} ---")

    # Training
    train_loss = train_epoch(model, train_dataloader, criterion, optimizer,
                            scheduler, device, epoch+1)

    # Validation
    val_loss, val_perplexity = evaluate(model, val_dataloader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, 
                        'best_transformer_model.pt')
        print("Nuovo miglior modello salvato!")
    
    # Salva checkpoint periodico
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, 
                        f'transformer_checkpoint_epoch_{epoch+1}.pt')

print("Training completato!")

