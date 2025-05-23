from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from dataset import get_data
from model import Transformer
import torch
import torch.optim as optim

if __name__ == "__main__":
    B = 64
    seq_len = 64
    embedding_dim = 64
    dk = 64
    dv = 64
    n_heads = 1
    N_blocks = 1
    vocab_size = 50257
    num_epochs = 10
    learning_rate = 0.001

    model = Transformer(B, seq_len, embedding_dim, dk, dv, n_heads, N_blocks, vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_data(seq_len, B, 'train', train_pct=0.8)
    test_loader = get_data(seq_len, B, 'test', train_pct=0.8)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

        torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pth")