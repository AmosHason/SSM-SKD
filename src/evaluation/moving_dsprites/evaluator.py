import csv
import os

import torch

from autoencoder import AutoencoderWrapper
from evaluation.evaluation import EvaluatorBase
from evaluation.moving_dsprites.classifier import ClassifierWrapper
from io_utils import UUID, get_checkpoint_dir


class Evaluator(EvaluatorBase):
    def __init__(self, args, logger, loader):
        super().__init__(args, logger, loader,
                         ClassifierWrapper.load_from_checkpoint(args.classifier_path).classifier.cuda().eval())

    def evaluate(self, autoencoder=None):  # noqa: C901
        if autoencoder is None:
            autoencoder = AutoencoderWrapper.load_from_checkpoint(f'{get_checkpoint_dir(self.args)}best.ckpt').autoencoder
            epochs = self.args.final_eval_epochs
        else:
            epochs = self.args.mid_training_eval_epochs
        batch_size = self.args.eval_batch_size

        autoencoder = autoencoder.cuda().eval()

        color_labels_original = []
        shape_labels_original = []
        scale_labels_original = []
        position_x_labels_original = []
        position_y_labels_original = []

        color_probs_after_dynamic_swap, color_labels_after_dynamic_swap = [], []
        shape_probs_after_dynamic_swap, shape_labels_after_dynamic_swap = [], []
        scale_labels_after_dynamic_swap = []
        position_x_labels_after_dynamic_swap = []
        position_y_labels_after_dynamic_swap = []

        color_labels_after_static_swap = []
        shape_labels_after_static_swap = []
        scale_probs_after_static_swap, scale_labels_after_static_swap = [], []
        position_x_probs_after_static_swap, position_x_labels_after_static_swap = [], []
        position_y_probs_after_static_swap, position_y_labels_after_static_swap = [], []

        for epoch in range(epochs):
            print(f'Running evaluation epoch #{epoch}.')
            for batch_idx, batch in enumerate(self.loader):
                data, static_labels, dynamic_labels = (batch['data'].cuda(),
                                                       batch['static_labels'].cuda(),
                                                       batch['dynamic_labels'].cuda())

                with torch.no_grad():
                    latents, bottleneck_info = autoencoder.encode(data)

                k_mats = bottleneck_info['k_mats']
                eigvals, eigvecs = torch.linalg.eig(k_mats)
                i_eigvecs = torch.linalg.inv(eigvecs)
                eigvals_distance_from_one = torch.sqrt((torch.real(eigvals) - 1) ** 2 + torch.imag(eigvals) ** 2)
                indices = torch.argsort(eigvals_distance_from_one, dim=1)
                indices_static, indices_dynamic = indices[:, :self.args.static_size], indices[:, self.args.static_size:]

                z_proj = latents.type(torch.complex128) @ eigvecs
                z_static = (z_proj.gather(2, indices_static.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                            i_eigvecs.gather(1, indices_static.unsqueeze(2).expand(-1, -1, latents.shape[2])))
                z_dynamic = (z_proj.gather(2, indices_dynamic.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                             i_eigvecs.gather(1, indices_dynamic.unsqueeze(2).expand(-1, -1, latents.shape[2])))

                permutation1 = torch.randperm(batch_size, device=latents.device)
                permutation2 = torch.randperm(batch_size, device=latents.device)
                permutations_static = torch.stack((z_static[permutation1], z_static[permutation2]))
                permutations_dynamic = torch.stack((z_dynamic[permutation1], z_dynamic[permutation2]))
                coefficients = torch.rand((2, batch_size), device=latents.device)
                coefficients = coefficients / torch.sum(coefficients, dim=0, keepdim=True)
                coefficients = coefficients.unsqueeze(2).unsqueeze(3).expand(-1, -1, latents.shape[1], latents.shape[2])
                mix_static = (coefficients * permutations_static).sum(dim=0)
                mix_dynamic = (coefficients * permutations_dynamic).sum(dim=0)

                swapped_dynamic = torch.real(z_static + mix_dynamic).type(torch.float32)
                swapped_static = torch.real(z_dynamic + mix_static).type(torch.float32)

                with torch.no_grad():
                    swapped_dynamic = autoencoder.decode(swapped_dynamic)
                    color, shape, scale_, position_x_, position_y_ = self.classifier(swapped_dynamic)

                    swapped_static = autoencoder.decode(swapped_static)
                    color_, shape_, scale, position_x, position_y = self.classifier(swapped_static)

                if epoch == 0 and batch_idx == 0:
                    self.plot([data[:10], swapped_dynamic[:10]], 'dynamic')
                    self.plot([data[10:20], swapped_static[10:20]], 'static')

                color_labels_original.append(static_labels[:, 0])
                shape_labels_original.append(static_labels[:, 1])
                scale_labels_original.append(dynamic_labels[:, 0])
                position_x_labels_original.append(dynamic_labels[:, 1])
                position_y_labels_original.append(dynamic_labels[:, 2])

                color_probs_after_dynamic_swap.append(color)
                color_labels_after_dynamic_swap.append(torch.argmax(color, 1))
                shape_probs_after_dynamic_swap.append(shape)
                shape_labels_after_dynamic_swap.append(torch.argmax(shape, 1))
                scale_labels_after_dynamic_swap.append(torch.argmax(scale_, 1))
                position_x_labels_after_dynamic_swap.append(torch.argmax(position_x_, 1))
                position_y_labels_after_dynamic_swap.append(torch.argmax(position_y_, 1))

                color_labels_after_static_swap.append(torch.argmax(color_, 1))
                shape_labels_after_static_swap.append(torch.argmax(shape_, 1))
                scale_probs_after_static_swap.append(scale)
                scale_labels_after_static_swap.append(torch.argmax(scale, 1))
                position_x_probs_after_static_swap.append(position_x)
                position_x_labels_after_static_swap.append(torch.argmax(position_x, 1))
                position_y_probs_after_static_swap.append(position_y)
                position_y_labels_after_static_swap.append(torch.argmax(position_y, 1))

        color_labels_original = torch.cat(color_labels_original)
        shape_labels_original = torch.cat(shape_labels_original)
        scale_labels_original = torch.cat(scale_labels_original)
        position_x_labels_original = torch.cat(position_x_labels_original)
        position_y_labels_original = torch.cat(position_y_labels_original)

        color_probs_after_dynamic_swap = torch.cat(color_probs_after_dynamic_swap)
        color_labels_after_dynamic_swap = torch.cat(color_labels_after_dynamic_swap)
        shape_probs_after_dynamic_swap = torch.cat(shape_probs_after_dynamic_swap)
        shape_labels_after_dynamic_swap = torch.cat(shape_labels_after_dynamic_swap)
        scale_labels_after_dynamic_swap = torch.cat(scale_labels_after_dynamic_swap)
        position_x_labels_after_dynamic_swap = torch.cat(position_x_labels_after_dynamic_swap)
        position_y_labels_after_dynamic_swap = torch.cat(position_y_labels_after_dynamic_swap)

        color_labels_after_static_swap = torch.cat(color_labels_after_static_swap)
        shape_labels_after_static_swap = torch.cat(shape_labels_after_static_swap)
        scale_probs_after_static_swap = torch.cat(scale_probs_after_static_swap)
        scale_labels_after_static_swap = torch.cat(scale_labels_after_static_swap)
        position_x_probs_after_static_swap = torch.cat(position_x_probs_after_static_swap)
        position_x_labels_after_static_swap = torch.cat(position_x_labels_after_static_swap)
        position_y_probs_after_static_swap = torch.cat(position_y_probs_after_static_swap)
        position_y_labels_after_static_swap = torch.cat(position_y_labels_after_static_swap)

        n_per_class = min([(color_labels_original == x).int().sum().item()
                           for x in torch.unique(color_labels_original)])
        indices = torch.cat([torch.nonzero(color_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(color_labels_original)])
        color_probs_after_dynamic_swap = color_probs_after_dynamic_swap[indices]
        n_per_class = min([(shape_labels_original == x).int().sum().item()
                           for x in torch.unique(shape_labels_original)])
        indices = torch.cat([torch.nonzero(shape_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(shape_labels_original)])
        shape_probs_after_dynamic_swap = shape_probs_after_dynamic_swap[indices]
        n_per_class = min([(scale_labels_original == x).int().sum().item()
                           for x in torch.unique(scale_labels_original)])
        indices = torch.cat([torch.nonzero(scale_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(scale_labels_original)])
        scale_probs_after_static_swap = scale_probs_after_static_swap[indices]
        n_per_class = min([(position_x_labels_original == x).int().sum().item()
                           for x in torch.unique(position_x_labels_original)])
        indices = torch.cat([torch.nonzero(position_x_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(position_x_labels_original)])
        position_x_probs_after_static_swap = position_x_probs_after_static_swap[indices]
        n_per_class = min([(position_y_labels_original == x).int().sum().item()
                           for x in torch.unique(position_y_labels_original)])
        indices = torch.cat([torch.nonzero(position_y_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(position_y_labels_original)])
        position_y_probs_after_static_swap = position_y_probs_after_static_swap[indices]

        preservation_score = \
            (self.report_metrics('dynamic', 'color', color_labels_original, color_labels_after_dynamic_swap,
                                 color_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'shape', shape_labels_original, shape_labels_after_dynamic_swap,
                                 shape_probs_after_dynamic_swap) +

             self.report_metrics('static', 'scale', scale_labels_original, scale_labels_after_static_swap,
                                 scale_probs_after_static_swap) +
             self.report_metrics('static', 'position_x', position_x_labels_original, position_x_labels_after_static_swap,
                                 position_x_probs_after_static_swap) +
             self.report_metrics('static', 'position_y', position_y_labels_original, position_y_labels_after_static_swap,
                                 position_y_probs_after_static_swap))

        self.logger.run['eval/preservation_score'].append(preservation_score)

        sampling_score = \
            (self.report_metrics('static', 'color', color_labels_original, color_labels_after_static_swap, unique_classes=6) +
             self.report_metrics('static', 'shape', shape_labels_original, shape_labels_after_static_swap, unique_classes=3) +

             self.report_metrics('dynamic', 'scale', scale_labels_original, scale_labels_after_dynamic_swap,
                                 unique_classes=10) +
             self.report_metrics('dynamic', 'position_x', position_x_labels_original, position_x_labels_after_dynamic_swap,
                                 unique_classes=8) +
             self.report_metrics('dynamic', 'position_y', position_y_labels_original, position_y_labels_after_dynamic_swap,
                                 unique_classes=8))

        self.logger.run['eval/sampling_score'].append(sampling_score)

        duofactor_score = preservation_score + sampling_score

        self.logger.run['eval/duofactor_score'].append(duofactor_score)

        return duofactor_score

    def evaluate_multifactor(self):  # noqa: C901
        autoencoder = (AutoencoderWrapper
                       .load_from_checkpoint(f'{get_checkpoint_dir(self.args)}best.ckpt').autoencoder.cuda().eval())
        epochs = self.args.final_eval_epochs
        batch_size = self.args.eval_batch_size

        factorial_swaps = []

        color_coordinates, shape_coordinates, rest_coordinates = [], [], []
        coordinate_titles = []

        print('Retaining nothing.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [])
        factorial_swaps.append({'Retaining': 'Nothing', 'Color Acc.': acc[0], 'Shape Acc.': acc[1],
                                'Scale Acc.': acc[2], 'Position X Acc.': acc[3], 'Position Y Acc.': acc[4]})
        no_swap_color_acc = acc[0]
        no_swap_shape_acc = acc[1]

        for c in range(self.args.k_dim):
            print(f'Retaining static coordinate #{c}.')
            acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [c])
            factorial_swaps.append({'Retaining': f'Static coordinate {c}', 'Color Acc.': acc[0], 'Shape Acc.': acc[1],
                                    'Scale Acc.': acc[2], 'Position X Acc.': acc[3], 'Position Y Acc.': acc[4]})
            color_acc = acc[0]
            shape_acc = acc[1]

            max_diff = max(color_acc - no_swap_color_acc, shape_acc - no_swap_shape_acc)
            if max_diff <= 0:
                rest_coordinates.append(c)
                coordinate_titles.append('Rest')
                self.logger.run[f'eval/coordinates/{c}'] = 'Rest'
            elif max_diff == color_acc - no_swap_color_acc:
                color_coordinates.append(c)
                coordinate_titles.append('Color')
                self.logger.run[f'eval/coordinates/{c}'] = 'Color'
            else:
                shape_coordinates.append(c)
                coordinate_titles.append('Shape')
                self.logger.run[f'eval/coordinates/{c}'] = 'Shape'

        multifactor_score = 0
        print('Retaining color.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, color_coordinates)
        factorial_swaps.append({'Retaining': 'Color', 'Color Acc.': acc[0], 'Shape Acc.': acc[1],
                                'Scale Acc.': acc[2], 'Position X Acc.': acc[3], 'Position Y Acc.': acc[4]})
        multifactor_score += (1 - acc[0] +
                              abs(1 / 3 - acc[1]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 8 - acc[3]) +
                              abs(1 / 8 - acc[4]))
        print('Retaining shape.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, shape_coordinates)
        factorial_swaps.append({'Retaining': 'Shape', 'Color Acc.': acc[0], 'Shape Acc.': acc[1],
                                'Scale Acc.': acc[2], 'Position X Acc.': acc[3], 'Position Y Acc.': acc[4]})
        multifactor_score += (1 - acc[1] +
                              abs(1 / 6 - acc[0]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 8 - acc[3]) +
                              abs(1 / 8 - acc[4]))
        print('Retaining dynamic.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [], True)
        factorial_swaps.append({'Retaining': 'Dynamic', 'Color Acc.': acc[0], 'Shape Acc.': acc[1],
                                'Scale Acc.': acc[2], 'Position X Acc.': acc[3], 'Position Y Acc.': acc[4]})
        multifactor_score += (1 - acc[2] +
                              1 - acc[3] +
                              1 - acc[4] +
                              abs(1 / 6 - acc[0]) +
                              abs(1 / 3 - acc[1]))
        print('Retaining rest.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, rest_coordinates)
        factorial_swaps.append({'Retaining': 'Rest', 'Color Acc.': acc[0], 'Shape Acc.': acc[1],
                                'Scale Acc.': acc[2], 'Position X Acc.': acc[3], 'Position Y Acc.': acc[4]})

        self.logger.run['eval/multifactor_score'] = multifactor_score

        os.makedirs(f'./output/{UUID}/', exist_ok=True)

        with open(f'./output/{UUID}/factorial_swaps.csv', 'w') as f:
            header = ['Retaining', 'Color Acc.', 'Shape Acc.', 'Scale Acc.', 'Position X Acc.', 'Position Y Acc.']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in factorial_swaps:
                writer.writerow(row)
        self.logger.run['eval/factorial_swaps'].upload(f'./output/{UUID}/factorial_swaps.csv')

        titles_nums = torch.tensor([0 if t == 'Color' else
                                    1 if t == 'Shape' else
                                    2 for t in coordinate_titles], dtype=torch.int)
        title_indices = torch.argsort(titles_nums)
        titles = [titles_nums[i] for i in title_indices]
        coordinate_titles = ['Color' if t == 0 else
                             'Shape' if t == 1 else
                             'Rest' for t in titles]

        self.plot_mutual_information_matrix(self.coordinate_mutual_information(autoencoder, title_indices), coordinate_titles)

    def evaluate_factorial_swap(self, autoencoder, epochs, batch_size, static_coordinates_to_retain, retain_dynamic=False):
        color_labels_original, color_labels_after_static_swap = [], []
        shape_labels_original, shape_labels_after_static_swap = [], []
        scale_labels_original, scale_labels_after_static_swap = [], []
        position_x_labels_original, position_x_labels_after_static_swap = [], []
        position_y_labels_original, position_y_labels_after_static_swap = [], []

        for epoch in range(epochs):
            print(f'Running evaluation epoch #{epoch}.')
            for batch_idx, batch in enumerate(self.loader):
                data, static_labels, dynamic_labels = (batch['data'].cuda(),
                                                       batch['static_labels'].cuda(),
                                                       batch['dynamic_labels'].cuda())

                with torch.no_grad():
                    latents, bottleneck_info = autoencoder.encode(data)

                k_mats = bottleneck_info['k_mats']
                eigvals, eigvecs = torch.linalg.eig(k_mats)
                i_eigvecs = torch.linalg.inv(eigvecs)
                eigvals_distance_from_one = torch.sqrt((torch.real(eigvals) - 1) ** 2 + torch.imag(eigvals) ** 2)
                indices = torch.argsort(eigvals_distance_from_one, dim=1)
                indices_static, indices_dynamic = indices[:, :self.args.static_size], indices[:, self.args.static_size:]

                z_proj = latents.type(torch.complex128) @ eigvecs
                z_static = (z_proj.gather(2, indices_static.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                            i_eigvecs.gather(1, indices_static.unsqueeze(2).expand(-1, -1, latents.shape[2])))
                z_dynamic = (z_proj.gather(2, indices_dynamic.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                             i_eigvecs.gather(1, indices_dynamic.unsqueeze(2).expand(-1, -1, latents.shape[2])))

                permutation1 = torch.randperm(batch_size, device=latents.device)
                permutation2 = torch.randperm(batch_size, device=latents.device)
                permutations_static = torch.stack((z_static[permutation1], z_static[permutation2]))
                permutations_dynamic = torch.stack((z_dynamic[permutation1], z_dynamic[permutation2]))
                coefficients = torch.rand((2, batch_size), device=latents.device)
                coefficients = coefficients / torch.sum(coefficients, dim=0, keepdim=True)
                coefficients = coefficients.unsqueeze(2).unsqueeze(3).expand(-1, -1, latents.shape[1], latents.shape[2])
                mix_static = (coefficients * permutations_static).sum(dim=0)
                mix_dynamic = (coefficients * permutations_dynamic).sum(dim=0)

                mask_coordinates_to_retain = torch.zeros(self.args.k_dim, dtype=torch.complex128).cuda()
                mask_coordinates_to_retain[static_coordinates_to_retain] = 1
                mask_coordinates_to_retain = mask_coordinates_to_retain.unsqueeze(0).unsqueeze(0)

                z_static = ((z_proj.gather(2, indices_static.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                             i_eigvecs.gather(1, indices_static.unsqueeze(2).
                                              expand(-1, -1, latents.shape[2]))) * mask_coordinates_to_retain +
                            mix_static * (1 - mask_coordinates_to_retain))

                if not retain_dynamic:
                    swapped = torch.real(z_static + mix_dynamic).type(torch.float32)
                else:
                    swapped = torch.real(z_static + z_dynamic).type(torch.float32)

                with torch.no_grad():
                    swapped = autoencoder.decode(swapped)
                    color, shape, scale, position_x, position_y = self.classifier(swapped)

                color_labels_original.append(static_labels[:, 0])
                shape_labels_original.append(static_labels[:, 1])
                scale_labels_original.append(dynamic_labels[:, 0])
                position_x_labels_original.append(dynamic_labels[:, 1])
                position_y_labels_original.append(dynamic_labels[:, 2])

                color_labels_after_static_swap.append(torch.argmax(color, 1))
                shape_labels_after_static_swap.append(torch.argmax(shape, 1))
                scale_labels_after_static_swap.append(torch.argmax(scale, 1))
                position_x_labels_after_static_swap.append(torch.argmax(position_x, 1))
                position_y_labels_after_static_swap.append(torch.argmax(position_y, 1))

        color_labels_original = torch.cat(color_labels_original)
        shape_labels_original = torch.cat(shape_labels_original)
        scale_labels_original = torch.cat(scale_labels_original)
        position_x_labels_original = torch.cat(position_x_labels_original)
        position_y_labels_original = torch.cat(position_y_labels_original)

        color_labels_after_static_swap = torch.cat(color_labels_after_static_swap)
        shape_labels_after_static_swap = torch.cat(shape_labels_after_static_swap)
        scale_labels_after_static_swap = torch.cat(scale_labels_after_static_swap)
        position_x_labels_after_static_swap = torch.cat(position_x_labels_after_static_swap)
        position_y_labels_after_static_swap = torch.cat(position_y_labels_after_static_swap)

        color_accuracy = (color_labels_original == color_labels_after_static_swap).float().mean().item()
        shape_accuracy = (shape_labels_original == shape_labels_after_static_swap).float().mean().item()
        scale_accuracy = (scale_labels_original == scale_labels_after_static_swap).float().mean().item()
        position_x_accuracy = (position_x_labels_original == position_x_labels_after_static_swap).float().mean().item()
        position_y_accuracy = (position_y_labels_original == position_y_labels_after_static_swap).float().mean().item()

        return color_accuracy, shape_accuracy, scale_accuracy, position_x_accuracy, position_y_accuracy
