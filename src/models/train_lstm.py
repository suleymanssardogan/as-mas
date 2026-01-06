import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from models.dataset import ResidualWindowDataset
from models.lstm_autoencoder import LSTMAutoencoder


def train(csv_path="data/residuals.csv", seq_len=60, epochs=10, batch_size=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ResidualWindowDataset(csv_path=csv_path, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMAutoencoder(input_dim=1, hidden_dim=32, latent_dim=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in dl:
            batch = batch.to(device)          # (B, T, 1)
            pred = model(batch)
            loss = loss_fn(pred, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / max(len(dl), 1)
        print(f"Epoch {epoch}/{epochs} - loss: {avg:.6f}")

    # save weights
    torch.save(model.state_dict(), "data/lstm_ae.pt")
    print("Saved: data/lstm_ae.pt")


if __name__ == "__main__":
    train()
