import os
import json
from collections import Counter
from torch.utils.data import Dataset
from transformers import BertTokenizer

class WMT14ENDeDataset(Dataset):
    def __init__(self, data_dir, split='train', max_len=512, min_freq=2):
        self.data_dir = data_dir
        self.split = split
        self.max_len = max_len
        self.min_freq = min_freq

        self.src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

        self.src_vocab, self.tgt_vocab = self.build_vocabulary()
        self.samples = self.load_data()

    def build_vocabulary(self):
        src_tokens = []
        tgt_tokens = []

        for split in ['train', 'valid', 'test']:
            data = json.load(open(os.path.join(self.data_dir, f'{split}.json'), 'r'))
            for sample in data:
                src_tokens.extend(self.src_tokenizer.tokenize(sample['src']))
                tgt_tokens.extend(self.tgt_tokenizer.tokenize(sample['tgt']))

        src_vocab = self.create_vocabulary(src_tokens)
        tgt_vocab = self.create_vocabulary(tgt_tokens)

        return src_vocab, tgt_vocab

    def create_vocabulary(self, tokens):
        token_counts = Counter(tokens)
        vocab = {'<sos>': 0, '<eos>': 1, '<unk>': 2}
        for token, count in token_counts.most_common():
            if count >= self.min_freq:
                vocab[token] = len(vocab)
        return vocab

    def load_data(self):
        data = json.load(open(os.path.join(self.data_dir, f'{self.split}.json'), 'r'))
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        src_text, tgt_text = sample['src'], sample['tgt']

        src_ids = [self.src_vocab['<sos>']] + [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in self.src_tokenizer.tokenize(src_text)] + [self.src_vocab['<eos>']]
        tgt_ids = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in self.tgt_tokenizer.tokenize(tgt_text)] + [self.tgt_vocab['<eos>']]

        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)
