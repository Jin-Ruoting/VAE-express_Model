# models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)  # -> 60 → 30 → 15

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 15, 128)

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 30]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 15]
        x = self.flatten(x)
        x = F.relu(self.fc(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, output_channels=1, latent_dim=16):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 15)

        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)  # -> 30
        self.deconv2 = nn.ConvTranspose1d(32, output_channels, kernel_size=4, stride=2, padding=1)  # -> 60

    def forward(self, z):
        x = F.relu(self.fc(z))             # [B, 64*15]
        x = x.view(-1, 64, 15)             # [B, 64, 15]
        x = F.relu(self.deconv1(x))       # [B, 32, 30]
        x = self.deconv2(x)               # [B, 1, 60]
        return x


class ExpressionRegressor(nn.Module):
    def __init__(self, latent_dim=16):
        super(ExpressionRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.regressor(z)


class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(output_channels=input_channels, latent_dim=latent_dim)
        self.regressor = ExpressionRegressor(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        expr_pred = self.regressor(z)
        return x_hat, expr_pred, mu, logvar