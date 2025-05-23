from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from dataset import ShakespeareDataset
from model import Transformer
import torch
import torch.optim as optim
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
if __name__ == "__main__":
    seed_everything(42)
    seq_len = 128
    loss_fn = nn.CrossEntropyLoss()

    split = 0.7
    seq_len = 64
    embedding_dim = 64
    dk = 32
    dv = 32
    n_heads = 1
    N_blocks = 1
    vocab_size = 50257
    max_epochs = 1000
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datamodule = ShakespeareDataset(seq_len, batch_size, train_pct=0.7)
    transformer = Transformer(32, seq_len, embedding_dim, dk, dv, n_heads, N_blocks, vocab_size, device).to(device)
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        logger=TensorBoardLogger('logs'),
        log_every_n_steps=10,
        enable_progress_bar=True,
        check_val_every_n_epoch=10
    )
    trainer.fit(transformer, datamodule=datamodule)
    