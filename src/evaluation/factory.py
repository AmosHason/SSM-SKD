import evaluation.dsprites.evaluator
import evaluation.moving_dsprites.evaluator
import evaluation.sprites.evaluator
import evaluation.three_dimensional_shapes.evaluator


def build_evaluator(args, logger, loader):
    if args.dataset == 'Sprites':
        return evaluation.sprites.evaluator.Evaluator(args, logger, loader)
    elif args.dataset == 'dSprites':
        return evaluation.dsprites.evaluator.Evaluator(args, logger, loader)
    elif args.dataset == 'Moving dSprites':
        return evaluation.moving_dsprites.evaluator.Evaluator(args, logger, loader)
    elif args.dataset == '3D Shapes':
        return evaluation.three_dimensional_shapes.evaluator.Evaluator(args, logger, loader)
    else:
        raise NotImplementedError
