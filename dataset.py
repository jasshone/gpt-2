import torch
from torch.utils.data import DataLoader
#!wget "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

import torch
from transformers import AutoTokenizer
class ShakespeareData(torch.utils.data.Dataset):
    def __init__(self, text, seqlen):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
        tokenized_data = []
        for i in text.split("\n"):
                tokenized_data.append(torch.tensor(tokenizer(i)['input_ids']))
        final_data = torch.cat(tokenized_data)
        self.tokenized_data = final_data
        self.seqlen = seqlen
    def __len__(self):
        return len(self.tokenized_data)//self.seqlen
    def __getitem__(self, idx):
        return self.tokenized_data[idx*self.seqlen:(idx+1)*self.seqlen]

def get_data(seqlen, batch_size, split, train_pct):
    with open("input.txt", "r") as f:
        text = f.read()
    dataset = ShakespeareData(text, seqlen)
    total_len = len(dataset)
    train_size = int(total_len * train_pct)
    if split == 'train':
        subset = torch.utils.data.Subset(dataset, range(train_size))
    elif split == 'test':
        subset = torch.utils.data.Subset(dataset, range(train_size, total_len))
    else:
        subset = dataset  
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=(split == 'train'))
    return dataloader
