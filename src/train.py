import pytorch_lightning as pl
# noinspection PyPackageRequirements
from lightning_fabric import seed_everything
from prefigure.prefigure import get_all_args
from pytorch_lightning.utilities.model_summary import ModelSummary

import print_filter  # noqa
from autoencoder import AutoencoderWrapper
from data import make_dataloaders
from demo import AutoencoderDemoCallback
from evaluation.factory import build_evaluator
from io_utils import CustomNeptuneLogger as Logger, get_checkpoint_dir, get_environment_details


def main():
    args = get_all_args()

    seed_everything(args.seed, workers=True)

    logger = Logger(api_key=args.api_key, project=args.project, log_model_checkpoints=False, capture_stdout=False)

    training_loader, test_loader, evaluation_loader = make_dataloaders(args)

    evaluator = build_evaluator(args, logger, evaluation_loader) if args.classifier_path != '' else None

    demo_callback = AutoencoderDemoCallback(evaluation_loader, args.eval_every_n_epochs, args.static_size, evaluator)

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=args.eval_every_n_epochs if evaluator is not None else 1,
                                                 monitor='duofactor_score' if evaluator is not None else 'test/loss',
                                                 save_on_train_epoch_end=False, save_last=True,
                                                 dirpath=get_checkpoint_dir(args), filename='best')

    trainer = pl.Trainer(callbacks=[demo_callback, ckpt_callback], max_epochs=args.epochs, logger=logger,
                         deterministic=True, gradient_clip_val=args.gradient_clip_val)

    model = AutoencoderWrapper(args)

    logger.log_hyperparams(vars(args))
    logger.log_metrics(get_environment_details(), 0)
    logger.log_metrics({'trainable_parameters': ModelSummary(model).trainable_parameters}, 0)

    if args.train == 1:
        trainer.fit(model, training_loader, test_loader, ckpt_path='last')

    if evaluator is not None:
        evaluator.evaluate()
        if args.evaluate_multifactor == 1:
            evaluator.evaluate_multifactor()


if __name__ == '__main__':
    main()
