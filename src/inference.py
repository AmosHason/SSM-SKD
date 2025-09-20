import torch

from io_utils import plot_seqeunces, plot_spectrum


def swap(model, x_orig, x, z, k_mats, x_indices, dataset_indices, static_size):
    x1_idx, x2_idx = x_indices[0], x_indices[1]

    x1_orig, x1, z1, k_mat1 = x_orig[x1_idx], x[x1_idx], z[x1_idx], k_mats[x1_idx]
    x2_orig, x2, z2, k_mat2 = x_orig[x2_idx], x[x2_idx], z[x2_idx], k_mats[x2_idx]

    z1_pred = torch.cat((z1[0].reshape((1, -1)).type(torch.float64), z1[:-1].type(torch.float64) @ k_mat1))
    z2_pred = torch.cat((z2[0].reshape((1, -1)).type(torch.float64), z2[:-1].type(torch.float64) @ k_mat2))

    x1_pred = model.decode(z1_pred.unsqueeze(0).type(torch.float32)).squeeze(0)
    x2_pred = model.decode(z2_pred.unsqueeze(0).type(torch.float32)).squeeze(0)

    eigvals1, eigvecs1 = torch.linalg.eig(k_mat1)
    i_eigvecs1 = torch.linalg.inv(eigvecs1)
    eigvals2, eigvecs2 = torch.linalg.eig(k_mat2)
    i_eigvecs2 = torch.linalg.inv(eigvecs2)

    z1_proj, z2_proj = z1.type(torch.complex128) @ eigvecs1, z2.type(torch.complex128) @ eigvecs2

    eigvals_distance_from_one = torch.sqrt((torch.real(eigvals1) - 1) ** 2 + torch.imag(eigvals1) ** 2)
    indices = torch.argsort(eigvals_distance_from_one)
    indices_static1 = indices[:static_size]
    indices_dynamic1 = indices[static_size:]
    eigvals_distance_from_one = torch.sqrt((torch.real(eigvals2) - 1) ** 2 + torch.imag(eigvals2) ** 2)
    indices = torch.argsort(eigvals_distance_from_one)
    indices_static2 = indices[:static_size]
    indices_dynamic2 = indices[static_size:]

    z1_static, z1_dynamic = (z1_proj[:, indices_static1].type(torch.complex128) @ i_eigvecs1[indices_static1],
                             z1_proj[:, indices_dynamic1].type(torch.complex128) @ i_eigvecs1[indices_dynamic1])
    z2_static, z2_dynamic = (z2_proj[:, indices_static2].type(torch.complex128) @ i_eigvecs2[indices_static2],
                             z2_proj[:, indices_dynamic2].type(torch.complex128) @ i_eigvecs2[indices_dynamic2])

    x1_static = model.decode(torch.real(z1_static).unsqueeze(0).type(torch.float32)).squeeze(0)
    x1_dynamic = model.decode(torch.real(z1_dynamic).unsqueeze(0).type(torch.float32)).squeeze(0)
    x2_static = model.decode(torch.real(z2_static).unsqueeze(0).type(torch.float32)).squeeze(0)
    x2_dynamic = model.decode(torch.real(z2_dynamic).unsqueeze(0).type(torch.float32)).squeeze(0)

    z1_static_z2_dynamic = torch.real(z1_static + z2_dynamic)
    z2_static_z1_dynamic = torch.real(z2_static + z1_dynamic)

    x1_static_x2_dynamic = model.decode(z1_static_z2_dynamic.unsqueeze(0).type(torch.float32)).squeeze(0)
    x2_static_x1_dynamic = model.decode(z2_static_z1_dynamic.unsqueeze(0).type(torch.float32)).squeeze(0)

    sequences = [x1_orig.cpu(), x2_orig.cpu(), x1.cpu(), x2.cpu(), x1_pred.cpu(), x2_pred.cpu(),
                 x1_static.cpu(), x2_static.cpu(), x1_dynamic.cpu(), x2_dynamic.cpu(),
                 x1_static_x2_dynamic.cpu(), x2_static_x1_dynamic.cpu()]
    x1_idx, x2_idx = dataset_indices[0], dataset_indices[1]
    titles = [f'S{x1_idx}', f'S{x2_idx}', f'S{x1_idx}rec', f'S{x2_idx}rec', f'S{x1_idx}pred', f'S{x2_idx}pred',
              f'S{x1_idx}s', f'S{x2_idx}s', f'S{x1_idx}d', f'S{x2_idx}d',
              f'S{x1_idx}s{x2_idx}d', f'S{x2_idx}s{x1_idx}d']

    return (plot_seqeunces(sequences, titles),
            plot_spectrum(eigvals1[indices_static1].cpu(), eigvals1[indices_dynamic1].cpu(), x1_idx),
            plot_spectrum(eigvals2[indices_static2].cpu(), eigvals2[indices_dynamic2].cpu(), x2_idx))
