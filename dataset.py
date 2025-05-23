import torch
from torch.utils.data import DataLoader
#!wget "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
import lightning.pytorch as pl
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

class ShakespeareDataset(pl.LightningDataModule):
    def __init__(self, seqlen, batch_size, train_pct):
        super().__init__()
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.train_pct = train_pct
    def setup(self, stage=None):
        with open("input.txt", "r") as f:
            text = f.read()
        dataset = ShakespeareData(text, self.seqlen)
        total_len = len(dataset)
        train_size = int(total_len * self.train_pct)
        train = torch.utils.data.Subset(dataset, range(train_size))
        val= torch.utils.data.Subset(dataset, range(train_size, total_len))
        
        self.train_data = DataLoader(train, batch_size=self.batch_size, shuffle=(self.split == 'train'))
        self.val_data = DataLoader(val, batch_size=self.batch_size, shuffle=(self.split == 'train'))
    def train_dataloader(self):
        return self.train_data
    def val_dataloader(self):
        return self.val_data
   
