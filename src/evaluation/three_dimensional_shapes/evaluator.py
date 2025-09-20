import csv
import os

import torch

from autoencoder import AutoencoderWrapper
from evaluation.evaluation import EvaluatorBase
from evaluation.three_dimensional_shapes.classifier import ClassifierWrapper
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

        floor_hue_labels_original = []
        wall_hue_labels_original = []
        object_hue_labels_original = []
        initial_size_labels_original = []
        shape_labels_original = []
        size_change_labels_original = []
        camera_rotation_labels_original = []

        floor_hue_probs_after_dynamic_swap, floor_hue_labels_after_dynamic_swap = [], []
        wall_hue_probs_after_dynamic_swap, wall_hue_labels_after_dynamic_swap = [], []
        object_hue_probs_after_dynamic_swap, object_hue_labels_after_dynamic_swap = [], []
        initial_size_probs_after_dynamic_swap, initial_size_labels_after_dynamic_swap = [], []
        shape_probs_after_dynamic_swap, shape_labels_after_dynamic_swap = [], []
        size_change_labels_after_dynamic_swap = []
        camera_rotation_labels_after_dynamic_swap = []

        floor_hue_labels_after_static_swap = []
        wall_hue_labels_after_static_swap = []
        object_hue_labels_after_static_swap = []
        initial_size_labels_after_static_swap = []
        shape_labels_after_static_swap = []
        size_change_probs_after_static_swap, size_change_labels_after_static_swap = [], []
        camera_rotation_probs_after_static_swap, camera_rotation_labels_after_static_swap = [], []

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

                with (torch.no_grad()):
                    swapped_dynamic = autoencoder.decode(swapped_dynamic)
                    (floor_hue, wall_hue, object_hue, initial_size, shape,
                     size_change_, camera_rotation_) = self.classifier(swapped_dynamic)

                    swapped_static = autoencoder.decode(swapped_static)
                    (floor_hue_, wall_hue_, object_hue_, initial_size_, shape_,
                     size_change, camera_rotation) = self.classifier(swapped_static)

                if epoch == 0 and batch_idx == 0:
                    self.plot([data[:10], swapped_dynamic[:10]], 'dynamic')
                    self.plot([data[10:20], swapped_static[10:20]], 'static')

                floor_hue_labels_original.append(static_labels[:, 0])
                wall_hue_labels_original.append(static_labels[:, 1])
                object_hue_labels_original.append(static_labels[:, 2])
                initial_size_labels_original.append(static_labels[:, 3])
                shape_labels_original.append(static_labels[:, 4])
                size_change_labels_original.append(dynamic_labels[:, 0])
                camera_rotation_labels_original.append(dynamic_labels[:, 1])

                floor_hue_probs_after_dynamic_swap.append(floor_hue)
                floor_hue_labels_after_dynamic_swap.append(torch.argmax(floor_hue, 1))
                wall_hue_probs_after_dynamic_swap.append(wall_hue)
                wall_hue_labels_after_dynamic_swap.append(torch.argmax(wall_hue, 1))
                object_hue_probs_after_dynamic_swap.append(object_hue)
                object_hue_labels_after_dynamic_swap.append(torch.argmax(object_hue, 1))
                initial_size_probs_after_dynamic_swap.append(initial_size)
                initial_size_labels_after_dynamic_swap.append(torch.argmax(initial_size, 1))
                shape_probs_after_dynamic_swap.append(shape)
                shape_labels_after_dynamic_swap.append(torch.argmax(shape, 1))
                size_change_labels_after_dynamic_swap.append(torch.argmax(size_change_, 1))
                camera_rotation_labels_after_dynamic_swap.append(torch.argmax(camera_rotation_, 1))

                floor_hue_labels_after_static_swap.append(torch.argmax(floor_hue_, 1))
                wall_hue_labels_after_static_swap.append(torch.argmax(wall_hue_, 1))
                object_hue_labels_after_static_swap.append(torch.argmax(object_hue_, 1))
                initial_size_labels_after_static_swap.append(torch.argmax(initial_size_, 1))
                shape_labels_after_static_swap.append(torch.argmax(shape_, 1))
                size_change_probs_after_static_swap.append(size_change)
                size_change_labels_after_static_swap.append(torch.argmax(size_change, 1))
                camera_rotation_probs_after_static_swap.append(camera_rotation)
                camera_rotation_labels_after_static_swap.append(torch.argmax(camera_rotation, 1))

        floor_hue_labels_original = torch.cat(floor_hue_labels_original)
        wall_hue_labels_original = torch.cat(wall_hue_labels_original)
        object_hue_labels_original = torch.cat(object_hue_labels_original)
        initial_size_labels_original = torch.cat(initial_size_labels_original)
        shape_labels_original = torch.cat(shape_labels_original)
        size_change_labels_original = torch.cat(size_change_labels_original)
        camera_rotation_labels_original = torch.cat(camera_rotation_labels_original)

        floor_hue_probs_after_dynamic_swap = torch.cat(floor_hue_probs_after_dynamic_swap)
        floor_hue_labels_after_dynamic_swap = torch.cat(floor_hue_labels_after_dynamic_swap)
        wall_hue_probs_after_dynamic_swap = torch.cat(wall_hue_probs_after_dynamic_swap)
        wall_hue_labels_after_dynamic_swap = torch.cat(wall_hue_labels_after_dynamic_swap)
        object_hue_probs_after_dynamic_swap = torch.cat(object_hue_probs_after_dynamic_swap)
        object_hue_labels_after_dynamic_swap = torch.cat(object_hue_labels_after_dynamic_swap)
        initial_size_probs_after_dynamic_swap = torch.cat(initial_size_probs_after_dynamic_swap)
        initial_size_labels_after_dynamic_swap = torch.cat(initial_size_labels_after_dynamic_swap)
        shape_probs_after_dynamic_swap = torch.cat(shape_probs_after_dynamic_swap)
        shape_labels_after_dynamic_swap = torch.cat(shape_labels_after_dynamic_swap)
        size_change_labels_after_dynamic_swap = torch.cat(size_change_labels_after_dynamic_swap)
        camera_rotation_labels_after_dynamic_swap = torch.cat(camera_rotation_labels_after_dynamic_swap)

        floor_hue_labels_after_static_swap = torch.cat(floor_hue_labels_after_static_swap)
        wall_hue_labels_after_static_swap = torch.cat(wall_hue_labels_after_static_swap)
        object_hue_labels_after_static_swap = torch.cat(object_hue_labels_after_static_swap)
        initial_size_labels_after_static_swap = torch.cat(initial_size_labels_after_static_swap)
        shape_labels_after_static_swap = torch.cat(shape_labels_after_static_swap)
        size_change_probs_after_static_swap = torch.cat(size_change_probs_after_static_swap)
        size_change_labels_after_static_swap = torch.cat(size_change_labels_after_static_swap)
        camera_rotation_probs_after_static_swap = torch.cat(camera_rotation_probs_after_static_swap)
        camera_rotation_labels_after_static_swap = torch.cat(camera_rotation_labels_after_static_swap)

        n_per_class = min([(floor_hue_labels_original == x).int().sum().item()
                           for x in torch.unique(floor_hue_labels_original)])
        indices = torch.cat([torch.nonzero(floor_hue_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(floor_hue_labels_original)])
        floor_hue_probs_after_dynamic_swap = floor_hue_probs_after_dynamic_swap[indices]
        n_per_class = min([(wall_hue_labels_original == x).int().sum().item()
                           for x in torch.unique(wall_hue_labels_original)])
        indices = torch.cat([torch.nonzero(wall_hue_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(wall_hue_labels_original)])
        wall_hue_probs_after_dynamic_swap = wall_hue_probs_after_dynamic_swap[indices]
        n_per_class = min([(object_hue_labels_original == x).int().sum().item()
                           for x in torch.unique(object_hue_labels_original)])
        indices = torch.cat([torch.nonzero(object_hue_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(object_hue_labels_original)])
        object_hue_probs_after_dynamic_swap = object_hue_probs_after_dynamic_swap[indices]
        n_per_class = min([(initial_size_labels_original == x).int().sum().item()
                           for x in torch.unique(initial_size_labels_original)])
        indices = torch.cat([torch.nonzero(initial_size_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(initial_size_labels_original)])
        initial_size_probs_after_dynamic_swap = initial_size_probs_after_dynamic_swap[indices]
        n_per_class = min([(shape_labels_original == x).int().sum().item()
                           for x in torch.unique(shape_labels_original)])
        indices = torch.cat([torch.nonzero(shape_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(shape_labels_original)])
        shape_probs_after_dynamic_swap = shape_probs_after_dynamic_swap[indices]
        n_per_class = min([(size_change_labels_original == x).int().sum().item()
                           for x in torch.unique(size_change_labels_original)])
        indices = torch.cat([torch.nonzero(size_change_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(size_change_labels_original)])
        size_change_probs_after_static_swap = size_change_probs_after_static_swap[indices]
        n_per_class = min([(camera_rotation_labels_original == x).int().sum().item()
                           for x in torch.unique(camera_rotation_labels_original)])
        indices = torch.cat([torch.nonzero(camera_rotation_labels_original == x)[0, :n_per_class]
                             for x in torch.unique(camera_rotation_labels_original)])
        camera_rotation_probs_after_static_swap = camera_rotation_probs_after_static_swap[indices]

        preservation_score = \
            (self.report_metrics('dynamic', 'floor_hue', floor_hue_labels_original,
                                 floor_hue_labels_after_dynamic_swap, floor_hue_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'wall_hue', wall_hue_labels_original,
                                 wall_hue_labels_after_dynamic_swap, wall_hue_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'object_hue', object_hue_labels_original,
                                 object_hue_labels_after_dynamic_swap, object_hue_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'initial_size', initial_size_labels_original,
                                 initial_size_labels_after_dynamic_swap, initial_size_probs_after_dynamic_swap) +
             self.report_metrics('dynamic', 'shape', shape_labels_original,
                                 shape_labels_after_dynamic_swap, shape_probs_after_dynamic_swap) +

             self.report_metrics('static', 'size_change', size_change_labels_original,
                                 size_change_labels_after_static_swap, size_change_probs_after_static_swap) +
             self.report_metrics('static', 'camera_rotation', camera_rotation_labels_original,
                                 camera_rotation_labels_after_static_swap, camera_rotation_probs_after_static_swap))

        self.logger.run['eval/preservation_score'].append(preservation_score)

        sampling_score = \
            (self.report_metrics('static', 'floor_hue', floor_hue_labels_original,
                                 floor_hue_labels_after_static_swap, unique_classes=10) +
             self.report_metrics('static', 'wall_hue', wall_hue_labels_original,
                                 wall_hue_labels_after_static_swap, unique_classes=10) +
             self.report_metrics('static', 'object_hue', object_hue_labels_original,
                                 object_hue_labels_after_static_swap, unique_classes=10) +
             self.report_metrics('static', 'initial_size', initial_size_labels_original,
                                 initial_size_labels_after_static_swap, unique_classes=6) +
             self.report_metrics('static', 'shape', shape_labels_original,
                                 shape_labels_after_static_swap, unique_classes=4) +

             self.report_metrics('dynamic', 'size_change', size_change_labels_original,
                                 size_change_labels_after_dynamic_swap, unique_classes=2) +
             self.report_metrics('dynamic', 'camera_rotation', camera_rotation_labels_original,
                                 camera_rotation_labels_after_dynamic_swap, unique_classes=3))

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

        (floor_hue_coordinates, wall_hue_coordinates, object_hue_coordinates,
         initial_size_coordinates, shape_coordinates, rest_coordinates) = [], [], [], [], [], []
        coordinate_titles = []

        print('Retaining nothing.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [])
        factorial_swaps.append({'Retaining': 'Nothing', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        no_swap_floor_hue_acc = acc[0]
        no_swap_wall_hue_acc = acc[1]
        no_swap_object_hue_acc = acc[2]
        no_swap_initial_size_acc = acc[3]
        no_swap_shape_acc = acc[4]

        for c in range(self.args.k_dim):
            print(f'Retaining static coordinate #{c}.')
            acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [c])
            factorial_swaps.append({'Retaining': f'Static coordinate {c}', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                    'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                    'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
            floor_hue_acc = acc[0]
            wall_hue_acc = acc[1]
            object_hue_acc = acc[2]
            initial_size_acc = acc[3]
            shape_acc = acc[4]

            max_diff = max(floor_hue_acc - no_swap_floor_hue_acc, wall_hue_acc - no_swap_wall_hue_acc,
                           object_hue_acc - no_swap_object_hue_acc, initial_size_acc - no_swap_initial_size_acc,
                           shape_acc - no_swap_shape_acc)
            if max_diff <= 0:
                rest_coordinates.append(c)
                coordinate_titles.append('Rest')
                self.logger.run[f'eval/coordinates/{c}'] = 'Rest'
            elif max_diff == floor_hue_acc - no_swap_floor_hue_acc:
                floor_hue_coordinates.append(c)
                coordinate_titles.append('Floor Hue')
                self.logger.run[f'eval/coordinates/{c}'] = 'Floor Hue'
            elif max_diff == wall_hue_acc - no_swap_wall_hue_acc:
                wall_hue_coordinates.append(c)
                coordinate_titles.append('Wall Hue')
                self.logger.run[f'eval/coordinates/{c}'] = 'Wall Hue'
            elif max_diff == object_hue_acc - no_swap_object_hue_acc:
                object_hue_coordinates.append(c)
                coordinate_titles.append('Object Hue')
                self.logger.run[f'eval/coordinates/{c}'] = 'Object Hue'
            elif max_diff == initial_size_acc - no_swap_initial_size_acc:
                initial_size_coordinates.append(c)
                coordinate_titles.append('Initial Size')
                self.logger.run[f'eval/coordinates/{c}'] = 'Initial Size'
            else:
                shape_coordinates.append(c)
                coordinate_titles.append('Shape')
                self.logger.run[f'eval/coordinates/{c}'] = 'Shape'

        multifactor_score = 0
        print('Retaining floor hue.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, floor_hue_coordinates)
        factorial_swaps.append({'Retaining': 'Floor Hue', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        multifactor_score += (1 - acc[0] +
                              abs(1 / 10 - acc[1]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 4 - acc[4]) +
                              abs(1 / 2 - acc[5]) +
                              abs(1 / 3 - acc[6]))
        print('Retaining wall hue.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, wall_hue_coordinates)
        factorial_swaps.append({'Retaining': 'Wall Hue', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        multifactor_score += (1 - acc[1] +
                              abs(1 / 10 - acc[0]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 4 - acc[4]) +
                              abs(1 / 2 - acc[5]) +
                              abs(1 / 3 - acc[6]))
        print('Retaining object hue.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, object_hue_coordinates)
        factorial_swaps.append({'Retaining': 'Object Hue', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        multifactor_score += (1 - acc[2] +
                              abs(1 / 10 - acc[0]) +
                              abs(1 / 10 - acc[1]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 4 - acc[4]) +
                              abs(1 / 2 - acc[5]) +
                              abs(1 / 3 - acc[6]))
        print('Retaining initial size.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, initial_size_coordinates)
        factorial_swaps.append({'Retaining': 'Initial Size', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        multifactor_score += (1 - acc[3] +
                              abs(1 / 10 - acc[0]) +
                              abs(1 / 10 - acc[1]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 4 - acc[4]) +
                              abs(1 / 2 - acc[5]) +
                              abs(1 / 3 - acc[6]))
        print('Retaining shape.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, shape_coordinates)
        factorial_swaps.append({'Retaining': 'Shape', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        multifactor_score += (1 - acc[4] +
                              abs(1 / 10 - acc[0]) +
                              abs(1 / 10 - acc[1]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 2 - acc[5]) +
                              abs(1 / 3 - acc[6]))
        print('Retaining dynamic.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, [], True)
        factorial_swaps.append({'Retaining': 'Dynamic', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})
        multifactor_score += (1 - acc[5] +
                              1 - acc[6] +
                              abs(1 / 10 - acc[0]) +
                              abs(1 / 10 - acc[1]) +
                              abs(1 / 10 - acc[2]) +
                              abs(1 / 6 - acc[3]) +
                              abs(1 / 4 - acc[4]))
        print('Retaining rest.')
        acc = self.evaluate_factorial_swap(autoencoder, epochs, batch_size, rest_coordinates)
        factorial_swaps.append({'Retaining': 'Rest', 'Floor Hue Acc.': acc[0], 'Wall Hue Acc.': acc[1],
                                'Object Hue Acc.': acc[2], 'Initial Size Acc.': acc[3], 'Shape Acc.': acc[4],
                                'Size Change Acc.': acc[5], 'Camera Rotation Acc.': acc[6]})

        self.logger.run['eval/multifactor_score'] = multifactor_score

        os.makedirs(f'./output/{UUID}/', exist_ok=True)

        with open(f'./output/{UUID}/factorial_swaps.csv', 'w') as f:
            header = ['Retaining', 'Floor Hue Acc.', 'Wall Hue Acc.', 'Object Hue Acc.', 'Initial Size Acc.', 'Shape Acc.',
                      'Size Change Acc.', 'Camera Rotation Acc.']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in factorial_swaps:
                writer.writerow(row)
        self.logger.run['eval/factorial_swaps'].upload(f'./output/{UUID}/factorial_swaps.csv')

        titles_nums = torch.tensor([0 if t == 'Floor Hue' else
                                    1 if t == 'Wall Hue' else
                                    2 if t == 'Object Hue' else
                                    3 if t == 'Initial Size' else
                                    4 if t == 'Shape' else
                                    5 for t in coordinate_titles], dtype=torch.int)
        title_indices = torch.argsort(titles_nums)
        titles = [titles_nums[i] for i in title_indices]
        coordinate_titles = ['Floor Hue' if t == 0 else
                             'Wall Hue' if t == 1 else
                             'Object Hue' if t == 2 else
                             'Initial Size' if t == 3 else
                             'Shape' if t == 4 else
                             'Rest' for t in titles]

        self.plot_mutual_information_matrix(self.coordinate_mutual_information(autoencoder, title_indices), coordinate_titles)

    def evaluate_factorial_swap(self, autoencoder, epochs, batch_size, static_coordinates_to_retain, retain_dynamic=False):
        floor_hue_labels_original, floor_hue_labels_after_static_swap = [], []
        wall_hue_labels_original, wall_hue_labels_after_static_swap = [], []
        object_hue_labels_original, object_hue_labels_after_static_swap = [], []
        initial_size_labels_original, initial_size_labels_after_static_swap = [], []
        shape_labels_original, shape_labels_after_static_swap = [], []
        size_change_labels_original, size_change_labels_after_static_swap = [], []
        camera_rotation_labels_original, camera_rotation_labels_after_static_swap = [], []

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
                    (floor_hue, wall_hue, object_hue, initial_size, shape,
                     size_change, camera_rotation) = self.classifier(swapped)

                floor_hue_labels_original.append(static_labels[:, 0])
                wall_hue_labels_original.append(static_labels[:, 1])
                object_hue_labels_original.append(static_labels[:, 2])
                initial_size_labels_original.append(static_labels[:, 3])
                shape_labels_original.append(static_labels[:, 4])
                size_change_labels_original.append(dynamic_labels[:, 0])
                camera_rotation_labels_original.append(dynamic_labels[:, 1])

                floor_hue_labels_after_static_swap.append(torch.argmax(floor_hue, 1))
                wall_hue_labels_after_static_swap.append(torch.argmax(wall_hue, 1))
                object_hue_labels_after_static_swap.append(torch.argmax(object_hue, 1))
                initial_size_labels_after_static_swap.append(torch.argmax(initial_size, 1))
                shape_labels_after_static_swap.append(torch.argmax(shape, 1))
                size_change_labels_after_static_swap.append(torch.argmax(size_change, 1))
                camera_rotation_labels_after_static_swap.append(torch.argmax(camera_rotation, 1))

        floor_hue_labels_original = torch.cat(floor_hue_labels_original)
        wall_hue_labels_original = torch.cat(wall_hue_labels_original)
        object_hue_labels_original = torch.cat(object_hue_labels_original)
        initial_size_labels_original = torch.cat(initial_size_labels_original)
        shape_labels_original = torch.cat(shape_labels_original)
        size_change_labels_original = torch.cat(size_change_labels_original)
        camera_rotation_labels_original = torch.cat(camera_rotation_labels_original)

        floor_hue_labels_after_static_swap = torch.cat(floor_hue_labels_after_static_swap)
        wall_hue_labels_after_static_swap = torch.cat(wall_hue_labels_after_static_swap)
        object_hue_labels_after_static_swap = torch.cat(object_hue_labels_after_static_swap)
        initial_size_labels_after_static_swap = torch.cat(initial_size_labels_after_static_swap)
        shape_labels_after_static_swap = torch.cat(shape_labels_after_static_swap)
        size_change_labels_after_static_swap = torch.cat(size_change_labels_after_static_swap)
        camera_rotation_labels_after_static_swap = torch.cat(camera_rotation_labels_after_static_swap)

        floor_hue_accuracy = (floor_hue_labels_original == floor_hue_labels_after_static_swap).float().mean().item()
        wall_hue_accuracy = (wall_hue_labels_original == wall_hue_labels_after_static_swap).float().mean().item()
        object_hue_accuracy = (object_hue_labels_original == object_hue_labels_after_static_swap).float().mean().item()
        initial_size_accuracy = (initial_size_labels_original == initial_size_labels_after_static_swap).float().mean().item()
        shape_accuracy = (shape_labels_original == shape_labels_after_static_swap).float().mean().item()
        size_change_accuracy = (size_change_labels_original == size_change_labels_after_static_swap).float().mean().item()
        camera_rotation_accuracy = ((camera_rotation_labels_original == camera_rotation_labels_after_static_swap)
                                    .float().mean().item())

        return (floor_hue_accuracy, wall_hue_accuracy, object_hue_accuracy, initial_size_accuracy, shape_accuracy,
                size_change_accuracy, camera_rotation_accuracy)
