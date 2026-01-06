import torch
import numpy as np
import pandas as pd
from models.dataset import ResidualWindowDataset
from models.lstm_autoencoder import LSTMAutoencoder
import torch.nn as nn


def infer(
    csv_path="data/residuals.csv",
    model_path="data/lstm_ae.pt",
    seq_len=60
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ResidualWindowDataset(csv_path=csv_path, seq_len=seq_len)
    model = LSTMAutoencoder(input_dim=1, hidden_dim=32, latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn = nn.MSELoss(reduction="none")

    scores = []
    timestamps = []

    with torch.no_grad():
        for i in range(len(ds)):
            window = torch.tensor(ds[i]).unsqueeze(0).to(device)  # (1, T, 1)
            recon = model(window)
            mse = loss_fn(recon, window).mean().item()
            scores.append(mse)
            timestamps.append(i)

    return np.array(scores)


if __name__ == "__main__":
    scores = infer()
    print("Anomaly scores (first 10):")
    print(scores[:10])
    print("max score:", scores.max())
