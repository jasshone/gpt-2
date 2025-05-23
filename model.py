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