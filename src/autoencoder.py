from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch import nn

from losses import EigenLoss, MSELoss, MultiLoss


class AutoencoderWrapper(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.autoencoder = Autoencoder(args)

        self.lr = args.lr
        self.batch_size = args.batch_size

        loss_modules = [MSELoss('reals', 'decoded', 'rec_loss', args.w_rec),
                        MSELoss('reals', 'pred_decoded', 'pred_ambient_loss', args.w_pred / 2),
                        MSELoss('latents', 'pred_latents', 'pred_latent_loss', args.w_pred / 2),
                        EigenLoss(args.static_size, args.dynamic_thresh, 'k_mats', 'eigen_loss', args.w_eigs)]
        self.losses = MultiLoss(loss_modules)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5623413)

        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'monitor': 'test/loss'}}

    def training_step(self, batch, batch_idx):
        loss, losses = self.step(batch)
        log_dict = {'train/loss': loss.detach(), 'train/lr': self.optimizers().param_groups[0]['lr']}
        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()
        log_dict = {k: v for k, v in log_dict.items() if torch.isfinite(torch.tensor([v])).item()}
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses = self.step(batch)
        log_dict = {'test/loss': loss.detach()}
        for loss_name, loss_value in losses.items():
            log_dict[f'test/{loss_name}'] = loss_value.detach()
        log_dict = {k: v for k, v in log_dict.items() if torch.isfinite(torch.tensor([v])).item()}
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

    def step(self, batch):
        reals = batch['data']
        latents, encoder_info = self.autoencoder.encode(reals)
        latents_after_dropout = torch.nn.functional.dropout(latents, 0.2)
        pred_latents = encoder_info['pred_latents']

        decoded = self.autoencoder.decode(latents_after_dropout)
        pred_decoded = self.autoencoder.decode(pred_latents)

        loss_info = {'reals': reals, 'latents': latents, 'decoded': decoded, 'pred_decoded': pred_decoded, **encoder_info}

        return self.losses(loss_info)


class Autoencoder(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        if args.dataset in ('Sprites', 'dSprites', 'Moving dSprites', '3D Shapes'):
            self.encoder = Encoder(3, args.k_dim, True)
            self.decoder = Decoder(args.k_dim, 3, True, args.hidden_dim)
        else:
            raise NotImplementedError

        self.bottleneck = KoopmanBottleneck(args.k_dim)

    def encode(self, x):
        latents = self.encoder(x)
        latents, bottleneck_info = self.bottleneck(latents)

        return latents, bottleneck_info

    def decode(self, latents):
        return self.decoder(latents)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, with_cnn):
        super().__init__()

        self.with_cnn = with_cnn

        if self.with_cnn:
            self.cnn = nn.Sequential(Conv(in_dim, 32, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(32, 64, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(64, 128, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(128, 256, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(256, out_dim, 4, 1, 0, True, nn.Tanh()))

            in_dim = out_dim

        self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True)

    def forward(self, x):
        if self.with_cnn:
            shape = x.shape
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            x = self.cnn(x)
            x = x.reshape(shape[0], shape[1], -1)

        return self.lstm(x)[0]


class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, bn, activation):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
                                  activation)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, with_cnn, hidden_dim=None):
        super().__init__()

        self.with_cnn = with_cnn

        self.lstm = nn.LSTM(in_dim, hidden_dim if self.with_cnn else out_dim, batch_first=True)

        if self.with_cnn:
            self.cnn = nn.Sequential(ConvTranspose(hidden_dim, 256, 4, 1, 0, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(256, 128, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(128, 64, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(64, 32, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(32, out_dim, 4, 2, 1, False, nn.Sigmoid()))

    def forward(self, x):
        x = self.lstm(x)[0]

        if self.with_cnn:
            shape = x.shape
            x = x.reshape(shape[0] * shape[1], shape[2], 1, 1)
            x = self.cnn(x)
            x = x.reshape(shape[0], shape[1], *x.shape[1:])

        return x


class ConvTranspose(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, bn, activation):
        super().__init__()

        self.conv_t = nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
                                    activation)

    def forward(self, x):
        return self.conv_t(x)


class KoopmanBottleneck(nn.Module):
    def __init__(self, k_dim):
        super().__init__()

        self.k_dim = k_dim

    # noinspection PyMethodMayBeStatic
    def forward(self, z):
        x, y = z[:, :-1].type(torch.float64), z[:, 1:].type(torch.float64)

        k_mats = torch.linalg.lstsq(x, y).solution

        pred_y = x @ k_mats
        pred = torch.cat((x[:, 0].unsqueeze(dim=1).type(torch.float32), pred_y.type(torch.float32)), dim=1)

        return z, {'k_mats': k_mats, 'pred_latents': pred}
