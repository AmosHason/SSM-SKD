import csv
import os

import torch

from autoencoder import AutoencoderWrapper
from evaluation.evaluation import EvaluatorBase
from evaluation.sprites.classifier import ClassifierWrapper
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

        skin_labels_original = []
        pants_labels_original = []
        top_labels_original = []
        hairstyle_labels_original = []
        action_labels_original = []

        skin_probs_after_dynamic_swap, skin_labels_after_dynamic_swap = [], []
        pants_probs_after_dynamic_swap, pants_labels_after_dynamic_swap = [], []
        top_probs_after_dynamic_swap, top_labels_after_dynamic_swap = [], []
        hairstyle_probs_after_dynamic_swap, hairstyle_labels_after_dynamic_swap = [], []
        action_labels_after_dynamic_swap = []

        skin_labels_after_static_swap = []
        pants_labels_after_static_swap = []
        top_labels_after_static_swap = []
        hairstyle_labels_after_static_swap = []
        action_probs_after_static_swap, action_labels_after_static_swap = [], []

        for epoch in range(epochs):
            print(f'Running evaluation epoch #{epoch}.')
            for batch_idx, batch in enumerate(self.loader):
                data, static_labels, dynamic_labels = (batch['data'].cuda(),
                                                       batch['static_labels'][:, 0].cuda(),
                                                       batch['dynamic_labels'][:, 0].cuda())

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
                    skin, pants, top, hairstyle, action_ = self.classifier(swapped_dynamic)

                    swapped_static = autoencoder.decode(swapped_static)
                    skin_, pants_, top_, hairstyle_, action = self.classifier(swapped_static)

                if epoch == 0 and batch_idx == 0:
                    self.plot([data[:10], swapped_dynamic[:10]], 'dynamic')
                    self.plot([data[10:20], swapped_static[10:20]], 'static')

                skin_labels_original.append(torch.argmax(static_labels[:, 0], 1))
                pants_labels_original.append(torch.argmax(static_labels[:, 1], 1))
                top_labels_original.append(torch.argmax(static_labels[:, 2], 1))
                hairstyle_labels_original.append(torch.argmax(static_labels[:, 3], 1))
                action_labels_original.append(torch.argmax(dynamic_labels, 1))

                skin_probs_after_dynamic_swap.append(skin)
                skin_labels_after_dynamic_swap.append(torch.argmax(skin, 1))
                pants_probs_after_dynamic_swap.append(pants)
                pants_labels_after_dynamic_swap.append(torch.argmax(pants, 1))
                top_probs_after_dynamic_swap.append(top)
                top_labels_after_dynamic_swap.append(torch.argmax(top, 1))
                hairstyle_probs_after_dynamic_swap.append(hairstyle)
                hairstyle_labels_after_dynamic_swap.append(torch.argmax(hairstyle, 1))
                action_labels_after_dynamic_swap.append(torch.argmax(action_, 1))

                skin_labels_after_static_swap.append(torch.argmax(skin_, 1))
                pants_labels_after_static_swap.append(torch.argmax(pants_, 1))
                top_labels_after_static_swap.append(torch.argmax(top_, 1))
                hairstyle_labels_after_static_swap.append(torch.argmax(hairstyle_, 1))
                action_probs_after_static_swap.append(action)
                action_labels_after_static_swap.append(torch.argmax(action, 1))

        skin_labels_original = torch.cat(skin_labels_original)
        pants_labels_original = torch.cat(pants_labels_original)
        top_labels_original = torch.cat(top_labels_original)
        hairstyle_labels_original = torch.cat(hairstyle_labels_original)
        action_labels_original = torch.cat(action_labels_original)

        skin_probs_after_dynamic_swap = torch.cat(skin_probs_after_dynamic_swap)
        skin_labels_after_dynamic_swap = torch.cat(skin_labels_after_dynamic_swap)
        pants_probs_after_dynamic_swap = torch.cat(pants_probs_after_dynamic_swap)
        pants_labels_after_dynamic_swap = torch.cat(pants_labels_after_dynamic_swap)
        top_probs_after_dynamic_swap = torch.cat(top_probs_after_dynamic_swap)
        top_labels_after_dynamic_swap = torch.cat(top_labels_after_dynamic_swap)
        hairstyle_probs_after_dynamic_swap = torch.cat(hairstyle_probs_after_dynamic_swap)
        hairstyle_labels_after_dynamic_swap = torch.cat(hairstyle_labels_after_dynamic_swap)
        action_labels_after_dynamic_swap = torch.cat(action_labels_after_dynamic_swap)

        skin_labels_after_static_swap = torch.cat(skin_labels_after_static_swap)
        pants_labels_after_static_swap = torch.cat(pants_labels_after_static_swap)
        top_labels_after_static_swap = torch.cat(top_labels_after_static_swap)
        hairstyle_labels_after_static_swap = torch.cat(hairstyle_labels_after_static_swap)
        action_probs_after_static_swap = torch.cat(action_probs_after_static_swap)
        action_labels_after_static_swap = torch.cat(action_labels_after_static_swap)

        n_per_class = min([(skin_labels_original == x).int().sum().item()
                           for x in torch.unique(skin_labels_original)])
        indices = torch.cat([torch.nonzero(skin_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(skin_labels_original)])
        skin_probs_after_dynamic_swap = skin_probs_after_dynamic_swap[indices]
        n_per_class = min([(pants_labels_original == x).int().sum().item()
                           for x in torch.unique(pants_labels_original)])
        indices = torch.cat([torch.nonzero(pants_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(pants_labels_original)])
        pants_probs_after_dynamic_swap = pants_probs_after_dynamic_swap[indices]
        n_per_class = min([(top_labels_original == x).int().sum().item()
                           for x in torch.unique(top_labels_original)])
        indices = torch.cat([torch.nonzero(top_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(top_labels_original)])
        top_probs_after_dynamic_swap = top_probs_after_dynamic_swap[indices]
        n_per_class = min([(hairstyle_labels_original == x).int().sum().item()
                           for x in torch.unique(hairstyle_labels_original)])
        indices = torch.cat([torch.nonzero(hairstyle_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(hairstyle_labels_original)])
        hairstyle_probs_after_dynamic_swap = hairstyle_probs_after_dynamic_swap[indices]
        n_per_class = min([(action_labels_original == x).int().sum().item()
                           for x in torch.unique(action_labels_original)])
        indices = torch.cat([torch.nonzero(action_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(action_labels_original)])
        action_probs_after_static_swap = action_probs_after_static_swap[indices]

        preservation_score = \
            (self.report_metrics('dynamic', 'skin', skin_labels_original, skin_labels_after_dynamic_swap,
                                 skin_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'pants', pants_labels_original, pants_labels_after_dynamic_swap,
                                 pants_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'top', top_labels_original, top_labels_after_dynamic_swap,
                                 top_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'hairstyle', hairstyle_labels_original, hairstyle_labels_after_dynamic_swap,
                                 hairstyle_probs_after_dynamic_swap) +

             self.report_metrics('static', 'action', action_labels_original, action_labels_after_static_swap,
                                 action_probs_after_static_swap))

        self.logger.run['eval/preservation_score'].append(preservation_score)

        sampling_score = \
            (self.report_metrics('static', 'skin', skin_labels_original, skin_labels_after_static_swap, unique_classes=6) +
             self.report_metrics('static', 'pants', pants_labels_original, pants_labels_after_static_swap, unique_classes=6) +
             self.report_metrics('static', 'top', top_labels_original, top_labels_after_static_swap, unique_classes=6) +
             self.report_metrics('static', 'hairstyle', hairstyle_labels_original, hairstyle_labels_after_static_swap,
                                 unique_classes=6) +

             self.report_metrics('dynamic', 'action', action_labels_original, action_labels_after_dynamic_swap,
                                 unique_classes=9))

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

        skin_coordinates, pants_coordinates, top_coordinates, hairstyle_coordinates, rest_coordinates = [], [], [], [], []
        coordinate_titles = []

        print('Retaining nothing.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [])
        factorial_swaps.append({'Retaining': 'Nothing', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
        no_swap_skin_acc = acc[0]
        no_swap_pants_acc = acc[1]
        no_swap_top_acc = acc[2]
        no_swap_hairstyle_acc = acc[3]

        for c in range(self.args.k_dim):
            print(f'Retaining static coordinate #{c}.')
            acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [c])
            factorial_swaps.append({'Retaining': f'Static coordinate {c}', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                    'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
            skin_acc = acc[0]
            pants_acc = acc[1]
            top_acc = acc[2]
            hairstyle_acc = acc[3]

            max_diff = max(skin_acc - no_swap_skin_acc, pants_acc - no_swap_pants_acc,
                           top_acc - no_swap_top_acc, hairstyle_acc - no_swap_hairstyle_acc)
            if max_diff <= 0:
                rest_coordinates.append(c)
                coordinate_titles.append('Rest')
                self.logger.run[f'eval/coordinates/{c}'] = 'Rest'
            elif max_diff == skin_acc - no_swap_skin_acc:
                skin_coordinates.append(c)
                coordinate_titles.append('Skin')
                self.logger.run[f'eval/coordinates/{c}'] = 'Skin'
            elif max_diff == pants_acc - no_swap_pants_acc:
                pants_coordinates.append(c)
                coordinate_titles.append('Pants')
                self.logger.run[f'eval/coordinates/{c}'] = 'Pants'
            elif max_diff == top_acc - no_swap_top_acc:
                top_coordinates.append(c)
                coordinate_titles.append('Top')
                self.logger.run[f'eval/coordinates/{c}'] = 'Top'
            else:
                hairstyle_coordinates.append(c)
                coordinate_titles.append('Hairstyle')
                self.logger.run[f'eval/coordinates/{c}'] = 'Hairstyle'

        multifactor_score = 0
        print('Retaining skin.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, skin_coordinates)
        factorial_swaps.append({'Retaining': 'Skin', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
        multifactor_score += (1 - acc[0] +
                              abs(1 / 6 - acc[1]) +
                              abs(1 / 6 - acc[2]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 9 - acc[4]))
        print('Retaining pants.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, pants_coordinates)
        factorial_swaps.append({'Retaining': 'Pants', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
        multifactor_score += (1 - acc[1] +
                              abs(1 / 6 - acc[0]) +
                              abs(1 / 6 - acc[2]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 9 - acc[4]))
        print('Retaining top.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, top_coordinates)
        factorial_swaps.append({'Retaining': 'Top', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
        multifactor_score += (1 - acc[2] +
                              abs(1 / 6 - acc[0]) +
                              abs(1 / 6 - acc[1]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 9 - acc[4]))
        print('Retaining hairstyle.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, hairstyle_coordinates)
        factorial_swaps.append({'Retaining': 'Hairstyle', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
        multifactor_score += (1 - acc[3] +
                              abs(1 / 6 - acc[0]) +
                              abs(1 / 6 - acc[1]) +
                              abs(1 / 6 - acc[2]) +
                              abs(1 / 9 - acc[4]))
        print('Retaining action.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [], True)
        factorial_swaps.append({'Retaining': 'Action', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})
        multifactor_score += (1 - acc[4] +
                              abs(1 / 6 - acc[0]) +
                              abs(1 / 6 - acc[1]) +
                              abs(1 / 6 - acc[2]) +
                              abs(1 / 6 - acc[3]))
        print('Retaining rest.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, rest_coordinates)
        factorial_swaps.append({'Retaining': 'Rest', 'Skin Acc.': acc[0], 'Pants Acc.': acc[1],
                                'Top Acc.': acc[2], 'Hairstyle Acc.': acc[3], 'Action Acc.': acc[4]})

        self.logger.run['eval/multifactor_score'] = multifactor_score

        os.makedirs(f'./output/{UUID}/', exist_ok=True)

        with open(f'./output/{UUID}/factorial_swaps.csv', 'w') as f:
            header = ['Retaining', 'Skin Acc.', 'Pants Acc.', 'Top Acc.', 'Hairstyle Acc.', 'Action Acc.']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in factorial_swaps:
                writer.writerow(row)
        self.logger.run['eval/factorial_swaps'].upload(f'./output/{UUID}/factorial_swaps.csv')

        titles_nums = torch.tensor([0 if t == 'Skin' else
                                    1 if t == 'Pants' else
                                    2 if t == 'Top' else
                                    3 if t == 'Hairstyle' else
                                    4 for t in coordinate_titles], dtype=torch.int)
        title_indices = torch.argsort(titles_nums)
        titles = [titles_nums[i] for i in title_indices]
        coordinate_titles = ['Skin' if t == 0 else
                             'Pants' if t == 1 else
                             'Top' if t == 2 else
                             'Hairstyle' if t == 3 else
                             'Rest' for t in titles]

        self.plot_mutual_information_matrix(self.coordinate_mutual_information(autoencoder, title_indices), coordinate_titles)

    def evaluate_factorial_swap(self, autoencoder, epochs, batch_size, static_coordinates_to_retain, retain_dynamic=False):
        skin_labels_original, skin_labels_after_static_swap = [], []
        pants_labels_original, pants_labels_after_static_swap = [], []
        top_labels_original, top_labels_after_static_swap = [], []
        hairstyle_labels_original, hairstyle_labels_after_static_swap = [], []
        action_labels_original, action_labels_after_static_swap = [], []

        for epoch in range(epochs):
            print(f'Running evaluation epoch #{epoch}.')
            for batch_idx, batch in enumerate(self.loader):
                data, static_labels, dynamic_labels = (batch['data'].cuda(),
                                                       batch['static_labels'][:, 0].cuda(),
                                                       batch['dynamic_labels'][:, 0].cuda())

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
                    skin, pants, top, hairstyle, action = self.classifier(swapped)

                skin_labels_original.append(torch.argmax(static_labels[:, 0], 1))
                pants_labels_original.append(torch.argmax(static_labels[:, 1], 1))
                top_labels_original.append(torch.argmax(static_labels[:, 2], 1))
                hairstyle_labels_original.append(torch.argmax(static_labels[:, 3], 1))
                action_labels_original.append(torch.argmax(dynamic_labels, 1))

                skin_labels_after_static_swap.append(torch.argmax(skin, 1))
                pants_labels_after_static_swap.append(torch.argmax(pants, 1))
                top_labels_after_static_swap.append(torch.argmax(top, 1))
                hairstyle_labels_after_static_swap.append(torch.argmax(hairstyle, 1))
                action_labels_after_static_swap.append(torch.argmax(action, 1))

        skin_labels_original = torch.cat(skin_labels_original)
        pants_labels_original = torch.cat(pants_labels_original)
        top_labels_original = torch.cat(top_labels_original)
        hairstyle_labels_original = torch.cat(hairstyle_labels_original)
        action_labels_original = torch.cat(action_labels_original)

        skin_labels_after_static_swap = torch.cat(skin_labels_after_static_swap)
        pants_labels_after_static_swap = torch.cat(pants_labels_after_static_swap)
        top_labels_after_static_swap = torch.cat(top_labels_after_static_swap)
        hairstyle_labels_after_static_swap = torch.cat(hairstyle_labels_after_static_swap)
        action_labels_after_static_swap = torch.cat(action_labels_after_static_swap)

        skin_accuracy = (skin_labels_original == skin_labels_after_static_swap).float().mean().item()
        pants_accuracy = (pants_labels_original == pants_labels_after_static_swap).float().mean().item()
        top_accuracy = (top_labels_original == top_labels_after_static_swap).float().mean().item()
        hairstyle_accuracy = (hairstyle_labels_original == hairstyle_labels_after_static_swap).float().mean().item()
        action_accuracy = (action_labels_original == action_labels_after_static_swap).float().mean().item()

        return skin_accuracy, pants_accuracy, top_accuracy, hairstyle_accuracy, action_accuracy
