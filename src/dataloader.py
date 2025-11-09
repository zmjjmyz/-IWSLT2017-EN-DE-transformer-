import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
import os
import zipfile

class IWSLT2017Dataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=128):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        
        src_tokens = []
        for token in src.split()[:self.max_length-2]:
            if token in self.src_vocab:
                src_tokens.append(self.src_vocab[token])
            else:
                src_tokens.append(self.src_vocab['<unk>'])
        
        tgt_tokens = []
        for token in tgt.split()[:self.max_length-2]:
            if token in self.tgt_vocab:
                tgt_tokens.append(self.tgt_vocab[token])
            else:
                tgt_tokens.append(self.tgt_vocab['<unk>'])
        
        src_tokens = [self.src_vocab['<sos>']] + src_tokens + [self.src_vocab['<eos>']]
        tgt_tokens = [self.tgt_vocab['<sos>']] + tgt_tokens + [self.tgt_vocab['<eos>']]
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def build_vocab(sentences, min_freq=1): 
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.split())
    
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq: 
            vocab[word] = idx
            idx += 1
    
    return vocab

def load_iwslt_data(data_path, max_samples=10000):
    # zip_path = os.path.join(data_path, "en-de.zip")
    # extract_path = os.path.join(data_path, "iwslt_extracted")
    #
    # if not os.path.exists(extract_path):
    #     os.makedirs(extract_path, exist_ok=True)
    #     print(f"Extracting {zip_path} to {extract_path}")
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall(extract_path)
    # 直接指定train_dir为你的de-en目录（根据服务器实际路径修改）
    train_dir = os.path.join(data_path, "de-en")  # 假设data_path是./data，这里就是./data/de-en

    # 检查train_dir下是否有目标文件
    if not (os.path.exists(os.path.join(train_dir, "train.tags.de-en.en")) and
            os.path.exists(os.path.join(train_dir, "train.tags.de-en.de"))):
        raise FileNotFoundError(f"Could not find train files in {train_dir}")

    print(f"Found training data in: {train_dir}")

    # 后续读取文件的代码不变...
    src_file = os.path.join(train_dir, "train.tags.de-en.en")
    tgt_file = os.path.join(train_dir, "train.tags.de-en.de")
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('<'):
                src_lines.append(line)
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('<'):
                tgt_lines.append(line)
    
    min_len = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:min_len]
    tgt_lines = tgt_lines[:min_len]
    
    if max_samples > 0:
        src_lines = src_lines[:max_samples]
        tgt_lines = tgt_lines[:max_samples]
    
    src_vocab = build_vocab(src_lines, min_freq=1)
    tgt_vocab = build_vocab(tgt_lines, min_freq=1)
    
    src_vocab_inv = {v: k for k, v in src_vocab.items()}
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}
    
    return src_lines, tgt_lines, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv

def get_dataloaders(data_path, batch_size=16, max_samples=5000, max_length=64): # 调整默认值以适应更小的数据集
    src_lines, tgt_lines, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv = load_iwslt_data(data_path, max_samples)
    
    train_size = int(0.9 * len(src_lines))
    val_size = len(src_lines) - train_size
    
    train_src = src_lines[:train_size]
    train_tgt = tgt_lines[:train_size]
    val_src = src_lines[train_size:]
    val_tgt = tgt_lines[train_size:]
    
    train_dataset = IWSLT2017Dataset(train_src, train_tgt, src_vocab, tgt_vocab, max_length)
    val_dataset = IWSLT2017Dataset(val_src, val_tgt, src_vocab, tgt_vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    src_max_len = max([len(seq) for seq in src_batch])
    tgt_max_len = max([len(seq) for seq in tgt_batch])
    
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded_tensor = torch.zeros(src_max_len, dtype=torch.long)
        tgt_padded_tensor = torch.zeros(tgt_max_len, dtype=torch.long)
        
        src_padded_tensor[:len(src)] = src
        tgt_padded_tensor[:len(tgt)] = tgt
        
        src_padded.append(src_padded_tensor)
        tgt_padded.append(tgt_padded_tensor)
    
    return torch.stack(src_padded), torch.stack(tgt_padded)