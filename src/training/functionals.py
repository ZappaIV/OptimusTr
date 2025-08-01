import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import time, math
from tqdm import tqdm

from src.transformers.models.functionals import create_causal_mask

class LabelSmoothingLoss(nn.Module):
    """Label smoothing per migliorare la generalizzazione"""

    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        pred: [batch_size * seq_len, vocab_size]
        target: [batch_size * seq_len]
        """
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_index] = 0
            mask = torch.nonzero(target == self.ignore_index, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(torch.sum(-true_dist * pred, dim=1))


class NoamScheduler:
    """Learning rate scheduler come nel paper originale"""

    def __init__(
        self,
        optimizer: torch.optim.Adam, 
        d_model: int, 
        warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5),
                                          self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Valutazione del modello"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, tuple):
                la_input, en_input = batch_data[0], batch_data[1]
                la_input, en_input = la_input.to(device), en_input.to(device)
            elif isinstance(batch_data, dict):
                la_input = batch_data['src'].to(device)
                en_input = batch_data['tgt'].to(device)


            src = la_input
            tgt = en_input
            
            tgt_input = tgt[:-1]
            tgt_output = tgt[1:]


            src_key_padding_mask = (src == 0)
            memory_key_padding_mask = src_key_padding_mask
            tgt_key_padding_mask = (tgt_input == 0)
            tgt_mask = create_causal_mask(tgt_input.size(-1))

            # Teacher forcing: input del decoder senza l'ultimo token
            # Target: output atteso senza il primo token (SOS)
            tgt_output = en_input[:, 1:]

            # Forward pass
            logits = model(
                src, 
                tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask 
            )

            output_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss = criterion(output_flat, tgt_flat)

            num_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: LabelSmoothingLoss, 
    optimizer: torch.optim.Adam, 
    scheduler: NoamScheduler,
    device: torch.device,
    epoch: int
):
    """Training per una singola epoca"""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    # batch_idx = 0
    # for la_input, en_input in tqdm(dataloader, desc="Training", unit="batch"):
    for batch_idx, batch_data in enumerate(dataloader):
        if isinstance(batch_data, tuple):
            la_input, en_input = batch_data[0], batch_data[1]
            la_input, en_input = la_input.to(device), en_input.to(device)
        elif isinstance(batch_data, dict):
            la_input = batch_data['src'].to(device)
            en_input = batch_data['tgt'].to(device)

        src = la_input
        tgt = en_input
        
        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        src_key_padding_mask = (src == 0)
        memory_key_padding_mask = src_key_padding_mask
        tgt_key_padding_mask = (tgt_input == 0)
        tgt_mask = create_causal_mask(tgt_input.size(-1))

        # Teacher forcing: input del decoder senza l'ultimo token
        # Target: output atteso senza il primo token (SOS)
        tgt_output = en_input[:, 1:]

        # Forward pass
        logits = model(
            src, 
            tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask 
        )

        # Calcola loss
        output_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_output.reshape(-1)

        # Maschera i padding tokens
        loss = criterion(output_flat, tgt_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Statistiche
        num_tokens = (tgt_output != 0).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        if batch_idx % 250 == 0:
            elapsed_time = time.time() - start_time
            lr = scheduler.get_lr()
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'LR: {lr:.2e}, Tokens/sec: {total_tokens/elapsed_time:.0f}')
        # batch_idx += 1


    avg_loss = total_loss / total_tokens
    return avg_loss


def evaluate_transformers(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Valutazione del modello"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, tuple):
                la_input, en_input = batch_data[0], batch_data[1]
                la_input, en_input = la_input.to(device), en_input.to(device)
            elif isinstance(batch_data, dict):
                la_input = batch_data['src'].to(device)
                en_input = batch_data['tgt'].to(device)


            src = la_input
            tgt = en_input
            
            tgt_input = tgt[:-1]
            tgt_output = tgt[1:]

            # Teacher forcing: input del decoder senza l'ultimo token
            # Target: output atteso senza il primo token (SOS)
            tgt_output = en_input[:, 1:]

            # Forward pass
            logits = model(
                src, 
                tgt_input
            )


            output_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss = criterion(output_flat, tgt_flat)

            num_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity



def train_epoch_transformers(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.CrossEntropyLoss, 
    optimizer: torch.optim.Adam, 
    scheduler: NoamScheduler,
    device: torch.device,
    epoch: int
):
    """Training per una singola epoca"""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    # batch_idx = 0
    # for la_input, en_input in tqdm(dataloader, desc="Training", unit="batch"):
    for batch_idx, batch_data in enumerate(dataloader):
        if isinstance(batch_data, tuple):
            la_input, en_input = batch_data[0], batch_data[1]
            la_input, en_input = la_input.to(device), en_input.to(device)
        elif isinstance(batch_data, dict):
            la_input = batch_data['src'].to(device)
            en_input = batch_data['tgt'].to(device)

        src = la_input
        tgt = en_input
        
        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        # Teacher forcing: input del decoder senza l'ultimo token
        # Target: output atteso senza il primo token (SOS)
        tgt_output = en_input[:, 1:]

        # Forward pass
        logits = model(
            src, 
            tgt_input
        )

        # Calcola loss
        output_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_output.reshape(-1)

        # Maschera i padding tokens
        loss = criterion(output_flat, tgt_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Statistiche
        num_tokens = (tgt_output != 0).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        if batch_idx % 250 == 0:
            elapsed_time = time.time() - start_time
            lr = scheduler.get_lr()
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'LR: {lr:.2e}, Tokens/sec: {total_tokens/elapsed_time:.0f}')
        # batch_idx += 1


    avg_loss = total_loss / total_tokens
    return avg_loss



def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    scheduler: NoamScheduler,
    epoch: int,
    loss: float,
    path: str
):
    """Salva checkpoint del modello"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler.step_num,
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Adam, 
    scheduler: NoamScheduler, 
    path: str
):
    """Carica checkpoint del modello"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.step_num = checkpoint['scheduler_step']
    return checkpoint['epoch'], checkpoint['loss']

