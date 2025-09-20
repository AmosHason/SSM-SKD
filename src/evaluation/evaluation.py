import math
from abc import ABC, abstractmethod

import torch
from matplotlib import pyplot as plt


class EvaluatorBase(ABC):
    def __init__(self, args, logger, loader, classifier):
        self.args = args
        self.logger = logger
        self.loader = loader
        self.classifier = classifier

    @abstractmethod
    def evaluate(self, autoencoder=None):
        ...

    @abstractmethod
    def evaluate_multifactor(self):
        ...

    @abstractmethod
    def evaluate_factorial_swap(self, autoencoder, epochs, batch_size, static_coordinates_to_retain, retain_dynamic=False):
        ...

    def coordinate_mutual_information(self, autoencoder, coordinate_indices):
        data = next(iter(self.loader))['data'].cuda()

        with torch.no_grad():
            latents, bottleneck_info = autoencoder.encode(data)

        k_mats = bottleneck_info['k_mats']
        eigvals = torch.linalg.eigvals(k_mats)
        eigvals_distance_from_one = torch.sqrt((torch.real(eigvals) - 1) ** 2 + torch.imag(eigvals) ** 2)
        indices = torch.argsort(eigvals_distance_from_one, dim=1)
        indices_static = indices[:, 0]
        static_eigvals = eigvals[torch.arange(latents.shape[0]), indices_static]
        eigvecs = shifted_inverse_power_iteration(k_mats, static_eigvals.real)[1]
        i_eigvecs = shifted_inverse_power_iteration(k_mats.transpose(1, 2), static_eigvals.real)[1]

        static = (latents[:, -1:].type(torch.float64) @ eigvecs.unsqueeze(2)
                  @ i_eigvecs.unsqueeze(1)).squeeze().real.type(torch.float32)[:, coordinate_indices]

        return torch.tensor([[float('nan') if j >= i else
                              mutual_information(static[:, i], static[:, j]) for j in range(static.shape[1])]
                             for i in range(static.shape[1])])

    def plot(self, data, swap_type):
        rc = len(data[0])
        fig, axs = plt.subplots(rc, 2, figsize=(50, 10))

        for i, col in enumerate(data):
            for j, img in enumerate(col):
                img = img.cpu()
                tsz, csz, hsz, wsz = img.shape
                img = img.permute(2, 0, 3, 1).reshape(hsz, tsz * wsz, -1)
                axs[j][i].imshow(img)
                axs[j][i].set_axis_off()

        plt.subplots_adjust(wspace=0.005, hspace=0)

        self.logger.run[f'artifacts/{swap_type}_swap'].append(fig)

        plt.close()

    def plot_mutual_information_matrix(self, data, titles):
        data = data.cpu()

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(data, cmap='coolwarm', vmin=0, vmax=data.nan_to_num().max())

        ax.set_xticks(torch.arange(len(titles)), labels=titles)
        ax.set_yticks(torch.arange(len(titles)), labels=titles)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for i in range(len(titles)):
            for j in range(len(titles)):
                if j < i:
                    ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                            color='white' if data[i, j] < 0.25 * data.max() or data[i, j] > 0.75 * data.max() else 'black')

        self.logger.run['eval/coordinate_mutual_information'].upload(fig)

        plt.close()

    def report_metrics(self, swap_type, factor, labels_original: torch.Tensor,
                       labels_after_swap, probs_after_swap=None, unique_classes=None):
        accuracy = (labels_original == labels_after_swap).float().mean().item()

        self.logger.run[f'eval/{swap_type}_swap/{factor}/accuracy'].append(accuracy)
        if probs_after_swap is not None:
            self.logger.run[f'eval/{swap_type}_swap/{factor}/inception_score'].append(inception_score(probs_after_swap))
            self.logger.run[f'eval/{swap_type}_swap/{factor}/intra_entropy'].append(intra_entropy(probs_after_swap))
            self.logger.run[f'eval/{swap_type}_swap/{factor}/inter_entropy'].append(inter_entropy(probs_after_swap))

        if unique_classes is None:
            return 1 - accuracy
        return abs(1 / unique_classes - accuracy)


def shifted_inverse_power_iteration(mats, eigval_approximations, iterations=20):
    mats = mats.to(eigval_approximations)
    b = mats.shape[0]
    n = mats.shape[1]
    eigval_approximations = eigval_approximations.reshape(b, 1, 1)
    identity = torch.eye(n).expand(b, n, n).to(eigval_approximations)
    eigvecs = torch.randn(n, b, dtype=eigval_approximations.dtype, device=eigval_approximations.device)
    eigvecs = eigvecs / torch.linalg.norm(eigvecs, dim=0)

    for _ in range(iterations):
        eigvecs = torch.linalg.solve(mats - eigval_approximations * identity, eigvecs.T).T
        eigvecs = eigvecs / torch.linalg.norm(eigvecs, dim=0)

    eigvals = (eigvecs.T.reshape(b, 1, n) @ (mats @ eigvecs.T.unsqueeze(2))).squeeze()

    return eigvals, eigvecs.T


def mutual_information(x_data, y_data, num_kernels=128, epsilon=1e-10, normalized=True):
    def k(u):
        return torch.exp(- u ** 2 / 2) / math.sqrt(2 * torch.pi)

    padding = (x_data.max() - x_data.min()) * 0.025
    x = torch.linspace(x_data.min() - padding, x_data.max() + padding, num_kernels, device=x_data.device)
    h_x = (x[-1] - x[0]) / (2 * (1 + math.log(x_data.shape[0])))
    x_kernels = k((x.unsqueeze(1) - x_data.unsqueeze(0)) / h_x)
    p_x = x_kernels.mean(dim=1) / h_x
    p_x = torch.cat((torch.tensor([0], device=p_x.device), p_x))
    p_x = (x[1] - x[0]) * (p_x[0:-1] + p_x[1:]) / 2

    padding = (y_data.max() - y_data.min()) * 0.025
    y = torch.linspace(y_data.min() - padding, y_data.max() + padding, num_kernels, device=y_data.device)
    h_y = (y[-1] - y[0]) / (2 * (1 + math.log(y_data.shape[0])))
    y_kernels = k((y.unsqueeze(1) - y_data.unsqueeze(0)) / h_y)
    p_y = y_kernels.mean(dim=1) / h_y
    p_y = torch.cat((torch.tensor([0], device=p_y.device), p_y))
    p_y = (y[1] - y[0]) * (p_y[0:-1] + p_y[1:]) / 2

    def k(h, u):
        return torch.exp(- u.transpose(2, 3) @ h.inverse() @ u / 2) / torch.det(h).sqrt() / (2 * torch.pi)

    xy_data = torch.stack((x_data, y_data), dim=1)
    xy = torch.cartesian_prod(x, y)
    h_xy = torch.tensor([[h_x / num_kernels, 0], [0, h_y / num_kernels]], device=x_data.device)
    p_xy = k(h_xy, torch.stack(((xy[:, 0].unsqueeze(1) - xy_data[:, 0].unsqueeze(0)),
                                (xy[:, 1].unsqueeze(1) - xy_data[:, 1].unsqueeze(0))), 2).unsqueeze(3)).squeeze().mean(dim=1)
    p_xy = p_xy.reshape(num_kernels, num_kernels)
    p_xy = torch.cat((torch.tensor([[0] * num_kernels], device=p_xy.device), p_xy))
    p_xy = torch.cat((torch.tensor([[0] * (num_kernels + 1)], device=p_xy.device).T, p_xy), dim=1)
    p_xy = ((x[1] - x[0]) * (y[1] - y[0]) * (p_xy[0:-1, 0:-1] + p_xy[0:-1, 1:] + p_xy[1:, 0:-1] + p_xy[1:, 1:]) / 4).flatten()

    p_x_times_p_y = torch.cartesian_prod(p_x, p_y)
    p_x_times_p_y = p_x_times_p_y[:, 0] * p_x_times_p_y[:, 1]

    mi = (p_xy * torch.log(p_xy / (p_x_times_p_y + epsilon) + epsilon)).sum()

    if not normalized:
        return mi
    else:
        return 2 * mi / (- (p_x * torch.log(p_x + epsilon)).sum() - (p_y * torch.log(p_y + epsilon)).sum())


def inception_score(p_y_given_x, eps=1e-16):
    p_y = p_y_given_x.mean(dim=0, keepdim=True)
    kl_div = p_y_given_x * ((p_y_given_x + eps).log() - (p_y + eps).log())
    mean_kl_div = kl_div.sum(dim=1).mean()

    return mean_kl_div.exp().item()


def intra_entropy(p_y_given_x, eps=1e-16):
    return (-1) * (p_y_given_x * (p_y_given_x + eps).log()).sum(dim=1).mean().item()


def inter_entropy(p_y_given_x, eps=1e-16):
    p_y = p_y_given_x.mean(dim=0)

    return (-1) * (p_y * (p_y + eps).log()).sum().item()
