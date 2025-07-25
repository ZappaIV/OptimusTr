import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def embed(sent: list[str], vocab: dict):
    return [ vocab[word] for word in sent ]

class L2NDataset(Dataset):
    def __init__(self, la_corpus, en_corpus , la_vocab, en_vocab):
        self.la_corpus = la_corpus
        self.en_corpus = en_corpus
        self.la_vocab = la_vocab; self.en_vocab = en_vocab
        self.la_embeded_corpus = [ [self.la_vocab[word] for word in sent] for sent in self.la_corpus ]
        self.en_embeded_corpus = [ [self.en_vocab[word] for word in sent] for sent in self.en_corpus ]
    def __len__(self):
        return len(self.la_corpus)

    def __getitem__(self, idx):
        return torch.tensor(self.la_embeded_corpus[idx]), torch.tensor(self.en_embeded_corpus[idx])

def collate_fn(batch):
    sos_token = 1  # Start of sequence
    eos_token = 2  # End of sequence

    la_tensor, en_tensor = zip(*batch)
    # la_tensor = torch.stack(la_tensor)
    # en_tensor = torch.stack(en_tensor)


    sequence_lens = [len(label) for label in la_tensor+en_tensor]
    max_len = max(sequence_lens) + 1 # +1 for SOS or EOS
    max_len = (max_len + 8 - max_len%8)  # (1023 + 8 - 1023%8)%8

    la_input = torch.full((len(la_tensor), max_len), fill_value=0, dtype=torch.long)  # For decoder input
    en_target = torch.full((len(la_tensor), max_len), fill_value=0, dtype=torch.long)  # For decoder output

    for i, label in enumerate(la_tensor):
        la_input[i, 0] = sos_token
        la_input[i, 1:len(label)+1] = label
        la_input[i, len(label)] = eos_token

    for i, label in enumerate(en_tensor):
        en_target[i, 0] = sos_token
        en_target[i, 1:len(label)+1] = label
        en_target[i, len(label)] = eos_token

    return la_input, en_target

class TranslationDataset(Dataset):
    """Dataset per coppie di frasi parallele per traduzione"""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Token speciali
        self.src_pad_idx = src_vocab.get('<PAD>', 0)
        self.tgt_pad_idx = tgt_vocab.get('<PAD>', 0)
        self.tgt_sos_idx = tgt_vocab.get('<SOS>', 1)
        self.tgt_eos_idx = tgt_vocab.get('<EOS>', 2)
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Converti in indici
        src_indices = [self.src_vocab.get(token, self.src_vocab.get('<UNK>', -1)) 
                      for token in src_sentence]
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab.get('<UNK>', -1)) 
                      for token in tgt_sentence]
        
        # Tronca se troppo lungo
        src_indices = src_indices[:self.max_length-1]
        tgt_indices = tgt_indices[:self.max_length-2]
        
        # Aggiungi token speciali
        tgt_indices = [self.tgt_sos_idx] + tgt_indices + [self.tgt_eos_idx]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_len': len(src_indices),
            'tgt_len': len(tgt_indices)
        }


def trl_nn_collate_fn(batch):
    """Funzione per il padding delle sequenze nel batch"""
    src_sequences = [item['src'] for item in batch]
    tgt_sequences = [item['tgt'] for item in batch]
    
    # Padding
    src_padded = nn.utils.rnn.pad_sequence(src_sequences, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_sequences, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded
    }


