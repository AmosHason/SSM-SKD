# noinspection PyPackageRequirements
import lightning_utilities.core.rank_zero as rz

to_filter = ['(ckpt_path="last") is set, but there is no last checkpoint available.',
             'you should set `torch.set_float32_matmul_precision(\'medium\' | \'high\')`',
             'The `srun` command is available on your system',
             'exists and is not empty']


def _info(*args, stacklevel=2, **kwargs):
    if all([x not in args[0] for x in to_filter]):
        if rz.python_version() >= '3.8.0':
            kwargs['stacklevel'] = stacklevel
        rz.log.info(*args, **kwargs)


def _warn(message, stacklevel=2, **kwargs):
    if all([x not in message for x in to_filter]):
        rz.warnings.warn(message, stacklevel=stacklevel, **kwargs)


rz._info = _info
rz._warn = _warn
