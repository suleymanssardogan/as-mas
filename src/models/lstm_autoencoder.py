import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, latent_dim=16, num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T, 1)
        enc_out, (h, c) = self.encoder(x)      # enc_out: (B, T, H)
        h_last = enc_out[:, -1, :]             # (B, H)

        z = self.to_latent(h_last)             # (B, Z)

        dec_in = self.from_latent(z).unsqueeze(1)   # (B, 1, H)
        # repeat for full sequence length
        dec_in = dec_in.repeat(1, x.size(1), 1)     # (B, T, H)

        dec_out, _ = self.decoder(dec_in)           # (B, T, H)
        y = self.output_layer(dec_out)              # (B, T, 1)
        return y
