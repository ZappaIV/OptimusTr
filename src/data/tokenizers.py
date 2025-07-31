import re
from typing import List, Dict, Set
import json
# Importing required modules
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')



class LatinSyllabicTokenizer:
    def __init__(self):
        # Vocali latine (incluse le lunghe)
        self.vowels = set('aeiouāēīōūy')
        self.consonants = set('bcdfghjklmnpqrstvwxz')

        # Dittonghi comuni in latino
        self.diphthongs = {'ae', 'au', 'ei', 'eu', 'oe', 'ui'}

        # Gruppi consonantici che non si dividono
        self.consonant_clusters = {
            'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr',
            'pl', 'pr', 'sc', 'sp', 'st', 'tr', 'qu', 'ch', 'th', 'ph'
        }

        # Vocabolario per mapping token-to-id
        self.vocab = {}
        self.id_to_token = {}
        self.next_id = 0

        # Token speciali
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,  # Start of sentence
            '<EOS>': 3,  # End of sentence
            '<SEP>': 4   # Separator
        }

        # Inizializza il vocabolario con token speciali
        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id
            self.id_to_token[token_id] = token
            self.next_id = max(self.next_id, token_id + 1)

    def is_vowel(self, char: str) -> bool:
        """Verifica se un carattere è una vocale"""
        return char.lower() in self.vowels

    def is_consonant(self, char: str) -> bool:
        """Verifica se un carattere è una consonante"""
        return char.lower() in self.consonants

    def find_diphthong(self, text: str, pos: int) -> str:
        """Trova un dittongo a partire dalla posizione data"""
        for length in [2, 3]:  # Dittonghi di 2-3 caratteri
            if pos + length <= len(text):
                candidate = text[pos:pos + length].lower()
                if candidate in self.diphthongs:
                    return text[pos:pos + length]
        return ""

    def find_consonant_cluster(self, text: str, pos: int) -> str:
        """Trova un gruppo consonantico che non si divide"""
        for length in [2, 3]:
            if pos + length <= len(text):
                candidate = text[pos:pos + length].lower()
                if candidate in self.consonant_clusters:
                    return text[pos:pos + length]
        return ""

    def syllabify_word(self, word: str) -> List[str]:
        """Divide una parola latina in sillabe"""
        word = word.strip()
        if not word:
            return []

        syllables = []
        current_syllable = ""
        i = 0
        if len(set(word) & self.vowels) < 2:
          return [word]

        while i < len(word):
            char = word[i]

            # Gestione dittonghi
            diphthong = self.find_diphthong(word, i)
            if diphthong:
                current_syllable += diphthong
                i += len(diphthong)
                continue

            # Gestione vocali
            if self.is_vowel(char):
                current_syllable += char
                i += 1

                # Controlla se la sillaba è completa
                if i < len(word):
                    next_char = word[i]

                    # Se la prossima è una vocale, chiudi la sillaba
                    if self.is_vowel(next_char):
                        syllables.append(current_syllable)
                        current_syllable = ""
                    # Se seguono consonanti, applica le regole di divisione
                    elif self.is_consonant(next_char):
                        consonant_run = ""
                        j = i

                        # Raccogli tutte le consonanti consecutive
                        while j < len(word) and self.is_consonant(word[j]):
                            consonant_run += word[j]
                            j += 1

                        # Applica le regole di divisione sillabica
                        if len(consonant_run) == 1:
                            # Una sola consonante va con la sillaba successiva
                            syllables.append(current_syllable)
                            current_syllable = ""
                        elif len(consonant_run) >= 2:
                            # Controlla se c'è un cluster che non si divide
                            cluster = self.find_consonant_cluster(word, i)
                            if cluster:
                                # Il cluster va con la sillaba successiva
                                syllables.append(current_syllable)
                                current_syllable = ""
                            else:
                                # Divide dopo la prima consonante
                                current_syllable += consonant_run[0]
                                syllables.append(current_syllable)
                                current_syllable = ""
                                i += 1
                continue

            # Gestione consonanti
            if self.is_consonant(char):
                current_syllable += char
                i += 1
                continue

            # Altri caratteri (punteggiatura, spazi, etc.)
            current_syllable += char
            i += 1

        # Aggiungi l'ultima sillaba se non vuota
        if current_syllable:
            syllables.append(current_syllable)

        # out = [syl for syl in syllables if syl.strip()]

        final_out = []

        for syl in syllables:
          if len( set(syl) & self.vowels):
            final_out.extend([syl])
          else:
            final_out[-1] = final_out[-1] + syl

        return final_out

    def preprocess_text(self, text: str) -> str:
        """Preprocessa il testo normalizzando caratteri e punteggiatura"""
        # Converti in minuscolo
        text = text.lower()

        # Normalizza caratteri speciali latini
        char_map = {
            'v': 'u',  # V classico diventa U
            'j': 'i',  # J diventa I
        }

        for old_char, new_char in char_map.items():
            text = text.replace(old_char, new_char)

        # Rimuovi caratteri non latini eccetto punteggiatura base
        text = re.sub(r'[^a-zāēīōūæœ\s\.\,\;\:\!\?\-]', '', text)

        return text

    def tokenize_text(self, text: str, syl_padding='##') -> List[str]:
        """Tokenizza un testo in sillabe"""
        text = self.preprocess_text(text)

        word_token_list = word_tokenize(text)

        tokens = []

        for word in word_token_list:
            if word in '.!?;':
                tokens.append(word)
                continue
            if re.match(r'\w+', word):
                syllables = self.syllabify_word(word)
                if len(syllables) == 2:
                  syllables = [ syllables[0] , syl_padding + syllables[-1] ]

                elif len(syllables) > 2:
                  syllables = [syllables[0] ] + [f"{syl_padding}{syl}"  for syl in syllables[1:]]
                tokens.extend(syllables+[' '])
            else:
                tokens.extend([word, ' '])

        return tokens

    def build_vocab_from_texts(self, texts: List[str]):
        """Costruisce il vocabolario da una lista di testi"""
        all_tokens = []

        for text in texts:
            tokens = self.tokenize_text(text)
            all_tokens.extend(tokens)

        # Conta la frequenza dei token
        token_freq = {}
        for token in all_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

        # Aggiungi token al vocabolario (ordinati per frequenza)
        for token, freq in sorted(token_freq.items(), key=lambda x: x[1], reverse=True):
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Codifica un testo in sequenza di ID"""
        tokens = self.tokenize_text(text)

        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens['<SOS>'])

        for token in tokens:
            token_id = self.vocab.get(token, self.special_tokens['<UNK>'])
            ids.append(token_id)

        if add_special_tokens:
            ids.append(self.special_tokens['<EOS>'])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodifica una sequenza di ID in testo"""
        tokens = []

        for token_id in ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue

            token = self.id_to_token.get(token_id, '<UNK>')
            tokens.append(token)

        # Ricostruisci il testo
        text = ''.join(tokens)

        # Aggiungi spazi prima della punteggiatura dove appropriato
        text = re.sub(r'([a-zāēīōū])([.!?;,])', r'\1 \2', text)

        return text.strip()

    def save_vocab(self, filepath: str):
        """Salva il vocabolario su file"""
        vocab_data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath: str):
        """Carica il vocabolario da file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data['vocab']
        self.special_tokens = vocab_data['special_tokens']

        # Ricostruisci id_to_token
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_id = max(self.vocab.values()) + 1 if self.vocab else 0

    def get_vocab_size(self) -> int:
        """Restituisce la dimensione del vocabolario"""
        return len(self.vocab)


if __name__ == '__main__':
    tokenizer = LatinSyllabicTokenizer()

    # Testi di esempio latino-italiano
    latin_texts = [
        "Gallia est omnis divisa in partes tres.",
        "Veni, vidi, vici.",
        "Carpe diem, quam minimum credula postero.",
        "Alea iacta est.",
        "Et tu, Brute?"
    ]

    italian_texts = [
        "La Gallia è tutta divisa in tre parti.",
        "Arrivai, vidi, vinsi.",
        "Cogli l'attimo, confidando il meno possibile nel domani.",
        "Il dado è tratto.",
        "Anche tu, Bruto?"
    ]

    # Costruisci vocabolario
    all_texts = latin_texts + italian_texts
    tokenizer.build_vocab_from_texts(all_texts)

    print(f"Dimensione vocabolario: {tokenizer.get_vocab_size()}")

    # Test di tokenizzazione
    test_text = "Gallia est omnis divisa in partes tres."
    print(f"\nTesto originale: {test_text}")

    # Tokenizzazione sillabica
    syllables = tokenizer.tokenize_text(test_text)
    print(f"Sillabe: {syllables}")

    # Codifica e decodifica
    encoded = tokenizer.encode(test_text)
    print(f"Codificato: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decodificato: {decoded}")

    # Esempio con parola singola
    word = "divisa"
    word_syllables = tokenizer.syllabify_word(word)
    print(f"\n'{word}' -> sillabe: {word_syllables}")

    word = "credula"
    word_syllables = tokenizer.syllabify_word(word)
    print(f"'{word}' -> sillabe: {word_syllables}")
