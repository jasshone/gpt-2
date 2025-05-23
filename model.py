import torch.nn as nn
class feed_forward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim),
            nn.dropout(0.1)
        )
    def forward(self, x):
        return self.model(x)
class decoder(nn.Module):
    def __init__(self, B, seq_len, embedding_dim, dk, dv, n_heads, N_blocks, vocab_size):
        super().__init__()
        self.B = B
        self.seq_len = B
        self.embedding_dim = embedding_dim
        self.dk = dk
        self.dv = dv
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.N_blocks = N_blocks
        self.decoder_blocks = [decoder_block(B, seq_len, embedding_dim, dk, dv, n_heads) for i in range(N_blocks)]
        self.blocks = nn.Sequential(*self.decoder_blocks)
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x):
        out = self.blocks(x)
        out = self.blocks(out)
        out = self.norm(out)
        out = self.linear(out)
        out = torch.softmax(out, dim = -1)
        return out

class decoder_block(nn.Module):
    def __init__(self, B, seq_len, embedding_dim, dk, dv, n_heads):
        super().__init__()
        self.multi_at = multi_head_attention(B, seq_len, embedding_dim, dk, dv, n_heads, True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = feed_forward(embedding_dim)
    def forward(self, x):
        out = self.norm(x)
        out = x + self.multi_at(out)
        return self.feed_forward(out)

class multi_head_attention(nn.Module):
    def __init__(self, B, seq_len, embedding_dim, dk, dv, n_heads, mask):
        super().__init__()
        self.heads = [attention_head(B, seq_len, embedding_dim, dk, dv, mask) for i in range(n_heads)]
        self.Wo = nn.Parameter(torch.nn.init.normal_(torch.zeros(n_heads * dv,embedding_dim), 0, 0.02))
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.B = B
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        concat = []
        for head in self.heads:
            concat.append(head(x))
        concat = torch.concat(concat, dim = -1)
        out = concat @ self.Wo.unsqueeze(0).expand(self.B,-1, -1)
        out = self.linear(out)
        return self.dropout(out)

class attention_head(nn.Module):
    def __init__(self, B, seq_len, embedding_dim, dk, dv, mask):
        super().__init__()
        self.B = B
        self.seq_len = seq_len
        self.dk = dk
        self.dv = dv
        self.embedding_dim = embedding_dim
        self.Wq = nn.Parameter(nn.init.normal_(torch.rand(embedding_dim,dk).to(device), 0, 0.02))
        self.Wk = nn.Parameter(nn.init.normal_(torch.rand(embedding_dim,dk).to(device), 0, 0.02))
        self.mask = mask
        self.Wv = nn.Parameter(torch.nn.init.normal_(torch.rand(embedding_dim,dv).to(device), 0, 0.02))
    def forward(self, x):
        Q = x @ self.Wq.unsqueeze(0).expand(self.B,-1, -1)
        K = x @ self.Wk.unsqueeze(0).expand(self.B,-1, -1)
        V = x @ self.Wv.unsqueeze(0).expand(self.B,-1, -1)

        QKT = torch.bmm(Q,torch.transpose(K, 1, 2))
        if self.mask:
            QKT = torch.triu(QKT.transpose(1, 2), diagonal = 1).transpose(1,2)
            QKT = torch.masked_fill(QKT, QKT.bool(), -torch.inf)
            #diff = att(x,x,x)[0]- (torch.softmax(QKT/(self.dk**(1/2)), dim = -1)@ V)
            #print(torch.max(diff))
        return torch.softmax(QKT/(self.dk**(1/2)), dim = -1)@ V

class Transformer(pl.LightningModule):
    def __init__(self, B, seq_len, embedding_dim, dk, dv, n_heads, N_blocks, vocab_size, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.B = B
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.lookup = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        self.decoder = decoder(B, seq_len, embedding_dim, dk, dv, n_heads, N_blocks, vocab_size, device)
        self.pos_encoding = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.dropout = nn.Dropout(0.1)

    def get_pos_encoding(self, x):
        if self.pos_encoding is not None: return self.pos_encoding
        pos = torch.arange(0, self.seq_len).unsqueeze(0).unsqueeze(-1).expand(self.B, -1, self.embedding_dim)
        i_even = torch.arange(0, self.embedding_dim//2).unsqueeze(0).unsqueeze(0).expand(self.B, self.seq_len, -1)
        i_odd = torch.arange(0, math.ceil(self.embedding_dim/2)).unsqueeze(0).unsqueeze(0).expand(self.B, self.seq_len, -1)

        even = pos[:,:,::2]
        odd = pos[:,:,1::2]
        exp_even = i_even/self.vocab_size
        exp_odd = i_odd/self.vocab_size
        even = torch.sin(torch.pow((even/10000),exp_even ))
        odd = torch.cos(torch.pow(odd/10000,exp_odd))
        self.pos_encoding = torch.stack((even, odd), dim = -1).view(self.B, self.seq_len, self.embedding_dim)
        return self.pos_encoding.to(self.device)

    def forward(self, x):
        #print(self.lookup(x).shape, self.get_pos_encoding(x).shape)
        x = self.lookup(x) + self.get_pos_encoding(x).to(self.device)
        x = self.dropout(x)
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch.to(torch.int))
        target = F.one_hot(batch.to(torch.int64), num_classes = self.vocab_size)
        loss = self.loss_fn(out, target.to(torch.float))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch.to(torch.int))
        target = F.one_hot(batch.to(torch.int64), num_classes = self.vocab_size)
        loss = self.loss_fn(out, target.to(torch.float))
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)