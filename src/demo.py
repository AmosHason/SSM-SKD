import pytorch_lightning as pl
import torch

from inference import swap


class AutoencoderDemoCallback(pl.Callback):
    def __init__(self, loader, every_n_epochs, static_size, evaluator):
        super().__init__()

        self.batch = next(iter(loader))

        self.every_n_epochs = every_n_epochs
        self.static_size = static_size
        self.evaluator = evaluator

    def on_train_start(self, trainer, module):
        if trainer.current_epoch == 0 and not trainer.sanity_checking:
            module.eval()
            with torch.no_grad():
                self.demo(trainer, module, 0)
            module.train()

    def on_validation_epoch_end(self, trainer, module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0 and not trainer.sanity_checking:
            duofactor_score = self.demo(trainer, module, trainer.current_epoch + 1)

            if duofactor_score is not None:
                module.log('duofactor_score', duofactor_score, logger=False)

    def demo(self, trainer, module, epoch):
        model = module.autoencoder
        reals = self.batch['data'].to(module.device)
        dataset_indices = tuple(x.item() for x in self.batch['index'][:2])
        latents, encoder_info = model.encode(reals)
        k_mats = encoder_info['k_mats']
        decoded = model.decode(latents)

        out_filenames = swap(model, reals, decoded, latents, k_mats, (0, 1), dataset_indices, self.static_size)

        for x in out_filenames:
            trainer.logger.experiment[f'artifacts/epoch_{epoch}/{x.split("/")[-1].split(".")[0]}'].upload(x)

        if self.evaluator is not None:
            return self.evaluator.evaluate(model)
