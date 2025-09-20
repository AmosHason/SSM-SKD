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
        data, static_labels, dynamic_labels = batch['data'], batch['static_labels'][:, 0], batch['dynamic_labels'][:, 0]

        skin, pants, top, hairstyle, action = self.classifier(data)

        metrics = {'loss_skin': self.cross_entropy_loss(skin, static_labels[:, 0].float()),
                   'loss_pants': self.cross_entropy_loss(pants, static_labels[:, 1].float()),
                   'loss_top': self.cross_entropy_loss(top, static_labels[:, 2].float()),
                   'loss_hairstyle': self.cross_entropy_loss(hairstyle, static_labels[:, 3].float()),
                   'loss_action': self.cross_entropy_loss(action, dynamic_labels.float()),
                   'accuracy_skin': (torch.argmax(skin, 1) == torch.argmax(static_labels[:, 0], 1)).float().mean(),
                   'accuracy_pants': (torch.argmax(pants, 1) == torch.argmax(static_labels[:, 1], 1)).float().mean(),
                   'accuracy_top': (torch.argmax(top, 1) == torch.argmax(static_labels[:, 2], 1)).float().mean(),
                   'accuracy_hairstyle': (torch.argmax(hairstyle, 1) == torch.argmax(static_labels[:, 3], 1)).float().mean(),
                   'accuracy_action': (torch.argmax(action, 1) == torch.argmax(dynamic_labels, 1)).float().mean()}

        loss = (metrics['loss_skin'] + metrics['loss_pants'] + metrics['loss_top'] + metrics['loss_hairstyle'] +
                metrics['loss_action'])

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

        self.skin = Feature(6)
        self.pants = Feature(6)
        self.top = Feature(6)
        self.hairstyle = Feature(6)
        self.action = Feature(9)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], *shape[2:])
        x = self.cnn(x)
        x = x.reshape(shape[0], shape[1], -1)
        x = self.lstm(x)[0]
        x = torch.cat((x[:, 7, 0:128], x[:, 0, 128:256]), dim=1)

        return self.skin(x), self.pants(x), self.top(x), self.hairstyle(x), self.action(x)


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

    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='test/loss', dirpath='./checkpoints/sprites_classifier/',
                                                 save_last=True, filename='best')

    logger = Logger(api_key=args.api_key, project=args.project, log_model_checkpoints=False)

    trainer = pl.Trainer(callbacks=[ckpt_callback], max_epochs=30, logger=logger,
                         gradient_clip_val=1, log_every_n_steps=10)

    model = ClassifierWrapper()

    training_loader, test_loader = make_dataloaders(args)[:2]

    trainer.fit(model, training_loader, test_loader)


if __name__ == '__main__':
    main()
