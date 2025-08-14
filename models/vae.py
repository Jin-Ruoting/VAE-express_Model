# models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=7, latent_dim=64, sequence_length=2000):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length

        # 适配7个组蛋白修饰通道的CNN层
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)  # -> L/4
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)  # -> L/16
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # 固定输出长度
        )

        self.flatten = nn.Flatten()

        # 全连接层
        self.fc_common = nn.Sequential(
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # VAE参数输出
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.flatten(x)
        x = self.fc_common(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, output_channels=7, latent_dim=64, sequence_length=2000):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.output_channels = output_channels

        # 从潜在空间重建到特征图
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 16)
        )

        # 转置卷积层重建序列
        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose1d(64, output_channels, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.Sigmoid()  # 输出范围[0,1]
        )

    def forward(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 16)

        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = self.deconv_block3(x)

        # 调整到目标长度
        if x.size(-1) != self.sequence_length:
            x = F.interpolate(x, size=self.sequence_length, mode='linear', align_corners=False)

        return x


class ExpressionRegressor(nn.Module):
    def __init__(self, latent_dim=64, hidden_dims=[128, 64]):
        super(ExpressionRegressor, self).__init__()

        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, z):
        return self.regressor(z).squeeze(-1)


class VAE(nn.Module):
    def __init__(self, input_channels=7, latent_dim=64, sequence_length=2000):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, sequence_length)
        self.decoder = Decoder(input_channels, latent_dim, sequence_length)
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