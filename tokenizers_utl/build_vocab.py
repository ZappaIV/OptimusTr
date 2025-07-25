def build_vocab(corpus_tokenized: list[list[str]]):

    idx_to_char, char_to_idx = ["<pad>", "<sos>", "<eos>"], {"<pad>":0, "<sos>":1, "<eos>":2}

    for sentence in corpus_tokenized:
        for token in sentence:
            if token not in idx_to_char:
                idx_to_char.append(token)
                char_to_idx[token] = len(idx_to_char)-1

    return char_to_idx