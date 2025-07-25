import os, re, json
from transformers import BertTokenizer

def build_vocab(corpus_tokenized: list[list[str]]):

    idx_to_char, char_to_idx = ["<pad>", "<sos>", "<eos>"], {"<pad>":0, "<sos>":1, "<eos>":2}

    for sentence in corpus_tokenized:
        for token in sentence:
            if token not in idx_to_char:
                idx_to_char.append(token)
                char_to_idx[token] = len(idx_to_char)-1

    return char_to_idx

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# tokenizer_func = word_tokenize
tokenizer_func = bert_tokenizer.tokenize

if False:
  tokenizer_func = lambda x: tokenizer.tokenize_text(x, syl_padding = '')

# device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_path = 'datasets/vocabulary.json'
train_dataset_path = 'datasets/corpus_la_en_train.json'
val_dataset_path = 'datasets/corpus_la_en_val.json'

import pandas as pd

splits = {'train': 'data/train-00000-of-00001-9b65fddb561aafc9.parquet', 'test': 'data/test-00000-of-00001-da9124f738a58dfa.parquet', 'valid': 'data/valid-00000-of-00001-cf90f0473c819b17.parquet'}


if os.path.exists(train_dataset_path):  
    print('TrainSet Presente')
    df_train = pd.read_json(train_dataset_path, orient='records')
else:
    df_train_og = pd.read_parquet("hf://datasets/grosenthal/latin_english_parallel/" + splits["train"])
    print(len(df_train_og))
    nu_records = []
    for i, row in df_train_og.iterrows():
        la_text , en_text= row['la'], row['en']
        for la_sent, en_sent in zip(re.split(r'[\.\;\:\!\?\-]', la_text), re.split(r'[\.\;\:\!\?\-]', en_text)):
            if (len(la_sent.split())> 4) and (len(la_sent.split()) < 20):
                nu_records.append({'la': la_sent, 'en': en_sent})
    df_train = pd.DataFrame(nu_records)
    df_train.to_json(train_dataset_path, orient='records')

print(len(df_train))
print(df_train.head())


if os.path.exists(val_dataset_path):    
    print('ValSet Presente')
    df_val = pd.read_json(val_dataset_path, orient='records')
else:
    df_val_og = pd.read_parquet("hf://datasets/grosenthal/latin_english_parallel/" + splits["valid"])
    print(len(df_train_og))
    nu_records = []
    for i, row in df_val_og.iterrows():
        la_text , en_text= row['la'], row['en']
        for la_sent, en_sent in zip(re.split(r'[\.\;\:\!\?\-]', la_text), re.split(r'[\.\;\:\!\?\-]', en_text)):
            if (len(la_sent.split())> 4) and (len(la_sent.split()) < 20):
                nu_records.append({'la': la_sent, 'en': en_sent})
    df_val = pd.DataFrame(nu_records)
    df_val.to_json(val_dataset_path, orient='records')

print(len(df_val))
print(df_val.head())


if not 'la_tok' in df_train.columns.to_list():
    df_train['la_tok'] = df_train['la'].apply(lambda x: tokenizer_func(x))
if not 'en_tok' in df_train.columns.to_list():
    df_train['en_tok'] = df_train['en'].apply(lambda x: tokenizer_func(x))
df_train.to_json(train_dataset_path, orient='records', indent=2)
df_train.head()

if not 'la_tok' in df_val.columns.to_list():
    df_val['la_tok'] = df_val['la'].apply(lambda x: tokenizer_func(x))
if not 'en_tok' in df_val.columns.to_list():
    df_val['en_tok'] = df_val['en'].apply(lambda x: tokenizer_func(x))
df_val.to_json(val_dataset_path, orient='records', indent=2)
df_val.head()


# if False:
if os.path.exists(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as fp:
        vocab = json.load(fp)
    la_vocab = vocab['la_vocab']
    en_vocab = vocab['en_vocab']
else:
    la_vocab = build_vocab(df_train[:]['la_tok'].to_list()+df_val[:]['la_tok'].to_list())
    en_vocab = build_vocab(df_train[:]['en_tok'].to_list()+df_val[:]['en_tok'].to_list())

    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    with open(vocab_path, 'w', encoding='utf-8') as fp:
        json.dump({'la_vocab': la_vocab, 'en_vocab': en_vocab}, fp, indent=1)

df_la_voc = pd.DataFrame({'id': la_vocab.values(), 'token': la_vocab.keys()})
df_en_voc = pd.DataFrame({'id': en_vocab.values(), 'token': en_vocab.keys()})
df_la_voc.sort_values(by=['token'])


