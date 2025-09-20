import os
import platform
import subprocess
import uuid

import matplotlib.pyplot as plt
import numpy as np
# noinspection PyPackageRequirements
import pip
import pytorch_lightning
import torch
from matplotlib.patches import Circle
from pytorch_lightning.loggers.neptune import NeptuneLogger, rank_zero_only

UUID = uuid.uuid4()


class CustomNeptuneLogger(NeptuneLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)

        return super().log_metrics(metrics, step)


def get_environment_details():
    lscpu = subprocess.check_output(['lscpu']).decode().split('\n')
    line_idx = [i for i, _ in enumerate(lscpu) if 'Model name' in lscpu[i]][0]
    cpu = lscpu[line_idx].split(':')[1].lstrip()

    lspci = subprocess.check_output(['lspci', '-vnn']).decode().split('\n')
    line_idx = [i for i, _ in enumerate(lspci) if torch.cuda.get_device_name(0).removeprefix('NVIDIA ') in lspci[i]][0]
    gpu = lspci[line_idx].split(': ')[1] + ' (' + lspci[line_idx + 1].split(': ')[1] + ')'

    return {'sys/job': os.environ.get('SLURM_JOBID', -1),
            'sys/node': platform.node(),
            'sys/cpu': cpu,
            'sys/architecture': platform.processor(),
            'sys/kernel': platform.release(),
            'sys/glibc': platform.libc_ver()[1],
            'sys/gpu': gpu,
            'sys/nvidia': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                   '--format=csv,noheader']).decode().removesuffix('\n'),
            'sys/vbios': subprocess.check_output(['nvidia-smi', '--query-gpu=vbios_version',
                                                  '--format=csv,noheader']).decode().removesuffix('\n'),
            'sys/cuda': torch.version.cuda,
            'sys/cudnn': torch.backends.cudnn.version(),
            'sys/python': platform.python_version(),
            'sys/pip': pip.__version__,
            'sys/numpy': np.__version__,
            'sys/pytorch': torch.__version__,
            'sys/lightning': pytorch_lightning.__version__,
            **{f'sys/packages/{x.split("==")[0]}': x.split('==')[1]
               for x in subprocess.check_output(['pip', 'list', '--format=freeze']).decode().split('\n')[:-1]}}


def get_checkpoint_dir(args):
    path = ('./checkpoints/'
            f'seed={args.seed}'
            f'_lr={args.lr}'
            f'_bsz={args.batch_size}'
            f'_k={args.k_dim}'
            f'_h={args.hidden_dim}'
            f'_rec={args.w_rec}'
            f'_pred={args.w_pred}'
            f'_eigs={args.w_eigs}'
            f'_static={args.static_size}'
            f'_dynamic={args.dynamic_thresh}'
            f'_data={args.dataset}/')

    os.makedirs(path, exist_ok=True)

    return path


def plot_seqeunces(sequences, titles):
    os.makedirs(f'./figures/{UUID}/', exist_ok=True)

    plt.close('all')
    rc = len(sequences) // 2
    fig, axs = plt.subplots(rc, 2, figsize=(50, 30))

    for i, seq in enumerate(sequences):
        tsz, csz, hsz, wsz = seq.shape
        seq = seq.permute(2, 0, 3, 1).reshape(hsz, tsz * wsz, -1)
        ri, ci = i // 2, i % 2
        axs[ri][ci].imshow(seq)
        axs[ri][ci].set_axis_off()
        axs[ri][ci].set_title(titles[i], fontsize=50)

    plt.subplots_adjust(wspace=0.05, hspace=0)

    plt.savefig(f'./figures/{UUID}/seqeunces.png')

    return f'./figures/{UUID}/seqeunces.png'


def plot_spectrum(eigvals_static, eigvals_dynamic, idx):
    os.makedirs(f'./figures/{UUID}/', exist_ok=True)

    plt.close('all')
    plt.plot(eigvals_static.real, eigvals_static.imag, 'o', color='#ff4500', alpha=0.45)
    plt.plot(eigvals_dynamic.real, eigvals_dynamic.imag, 'o', color='#0036ff', alpha=0.45)

    ax = plt.gca()

    circle = Circle((0.0, 0.0), 1.0, fill=False)
    ax.add_artist(circle)

    ax.set_aspect('equal')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.xlabel('Real component')
    plt.ylabel('Imaginary component')
    plt.savefig(f'./figures/{UUID}/spectrum{idx}.png')

    return f'./figures/{UUID}/spectrum{idx}.png'
