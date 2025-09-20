import pytorch_lightning as pl
import torch
import torch.nn as nn
# noinspection PyPackageRequirements
from lightning_fabric import seed_everything
from prefigure import get_all_args

import print_filter  # noqa
from autoencoder import Conv
from data import make_dataloaders
from io_utils import CustomNeptuneLogger as Logger


class ClassifierWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.classifier = Classifier()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=0)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3162, patience=5, threshold=0.001)

        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'monitor': 'test/loss'}}

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch)
        log_dict = {'train/loss': loss.detach()}
        for metric_name, metric_value in metrics.items():
            log_dict[f'train/{metric_name}'] = metric_value.detach()
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch)
        log_dict = {'test/loss': loss.detach()}
        for metric_name, metric_value in metrics.items():
            log_dict[f'test/{metric_name}'] = metric_value.detach()
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

    def step(self, batch):
        data, static_labels, dynamic_labels = batch['data'], batch['static_labels'], batch['dynamic_labels']

        color, shape, scale, position_x, position_y = self.classifier(data)

        metrics = {'loss_color': self.cross_entropy_loss(color, static_labels[:, 0]),
                   'loss_shape': self.cross_entropy_loss(shape, static_labels[:, 1]),
                   'loss_scale': self.cross_entropy_loss(scale, dynamic_labels[:, 0]),
                   'loss_position_x': self.cross_entropy_loss(position_x, dynamic_labels[:, 1]),
                   'loss_position_y': self.cross_entropy_loss(position_y, dynamic_labels[:, 2]),
                   'accuracy_color': (torch.argmax(color, 1) == static_labels[:, 0]).float().mean(),
                   'accuracy_shape': (torch.argmax(shape, 1) == static_labels[:, 1]).float().mean(),
                   'accuracy_scale': (torch.argmax(scale, 1) == dynamic_labels[:, 0]).float().mean(),
                   'accuracy_position_x': (torch.argmax(position_x, 1) == dynamic_labels[:, 1]).float().mean(),
                   'accuracy_position_y': (torch.argmax(position_y, 1) == dynamic_labels[:, 2]).float().mean()}

        loss = (metrics['loss_color'] + metrics['loss_shape'] +
                metrics['loss_scale'] + metrics['loss_position_x'] + metrics['loss_position_y'])

        return loss, metrics


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(Conv(3, 32, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                 Conv(32, 64, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                 Conv(64, 128, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                 Conv(128, 256, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                 Conv(256, 40, 4, 1, 0, True, nn.Tanh()))

        self.lstm = nn.LSTM(40, 128, bidirectional=True, batch_first=True)

        self.color = Feature(6)
        self.shape = Feature(3)
        self.scale = Feature(10)
        self.position_x = Feature(8)
        self.position_y = Feature(8)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], *shape[2:])
        x = self.cnn(x)
        x = x.reshape(shape[0], shape[1], -1)
        x = self.lstm(x)[0]
        x = torch.cat((x[:, 7, 0:128], x[:, 0, 128:256]), dim=1)

        return self.color(x), self.shape(x), self.scale(x), self.position_x(x), self.position_y(x)


class Feature(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.feature = nn.Sequential(nn.Linear(256, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, num_classes),
                                     nn.Softmax(1))

    def forward(self, x):
        return self.feature(x)


def main():
    args = get_all_args()

    seed_everything(args.seed)

    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='test/loss', dirpath='./checkpoints/moving_dsprites_classifier/',
                                                 save_last=True, filename='best')

    logger = Logger(api_key=args.api_key, project=args.project, log_model_checkpoints=False)

    trainer = pl.Trainer(callbacks=[ckpt_callback], max_epochs=30, logger=logger,
                         gradient_clip_val=1, log_every_n_steps=10)

    model = ClassifierWrapper()

    training_loader, test_loader = make_dataloaders(args)[:2]

    trainer.fit(model, training_loader, test_loader)


if __name__ == '__main__':
    main()
