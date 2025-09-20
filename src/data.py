import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset


def make_dataloaders(args):
    if args.dataset == 'Sprites':
        training_data, test_data = load_sprites(args)
    elif args.dataset == 'dSprites':
        training_data, test_data = load_dsprites(args)
    elif args.dataset == 'Moving dSprites':
        training_data, test_data = load_moving_dsprites(args)
    elif args.dataset == '3D Shapes':
        training_data, test_data = load_3d_shapes(args)
    else:
        raise NotImplementedError

    return (DataLoader(training_data, num_workers=4, batch_size=args.batch_size,
                       shuffle=True, drop_last=True, pin_memory=True),
            DataLoader(test_data, num_workers=4, batch_size=args.batch_size,
                       shuffle=False, drop_last=True, pin_memory=True),
            DataLoader(test_data, num_workers=4, batch_size=args.eval_batch_size,
                       shuffle=False, drop_last=True, pin_memory=True))


def load_sprites(args):
    with open(args.dataset_path + 'sprites_X_train.npy', 'rb') as f:
        training_images = np.load(f)
    with open(args.dataset_path + 'sprites_X_test.npy', 'rb') as f:
        test_images = np.load(f)
    with open(args.dataset_path + 'sprites_A_train.npy', 'rb') as f:
        training_static_labels = np.load(f)
    with open(args.dataset_path + 'sprites_A_test.npy', 'rb') as f:
        test_static_labels = np.load(f)
    with open(args.dataset_path + 'sprites_D_train.npy', 'rb') as f:
        training_dynamic_labels = np.load(f)
    with open(args.dataset_path + 'sprites_D_test.npy', 'rb') as f:
        test_dynamic_labels = np.load(f)

    return (Sprites(training_images, training_static_labels, training_dynamic_labels),
            Sprites(test_images, test_static_labels, test_dynamic_labels))


class Sprites(Dataset):
    def __init__(self, images, static_labels, dynamic_labels):
        self.images = images
        self.static_labels = static_labels
        self.dynamic_labels = dynamic_labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return {'index': index,
                'data': self.images[index].transpose((0, 3, 1, 2)),
                'static_labels': self.static_labels[index],
                'dynamic_labels': self.dynamic_labels[index]}


def load_dsprites(args):
    with h5py.File(args.dataset_path + 'train.h5', 'r') as f:
        training_images = f['data'][:]
        training_labels = f['labels'][:]

    with h5py.File(args.dataset_path + 'test.h5', 'r') as f:
        test_images = f['data'][:]
        test_labels = f['labels'][:]

    return (DSprites(training_images, training_labels),
            DSprites(test_images, test_labels))


class DSprites(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'index': index,
                'data': self.images[index].astype(np.float32) / 255,
                'static_labels': self.labels[index, [0, 1, 2, 3]].astype(np.long),
                'dynamic_labels': self.labels[index, [4]].astype(np.long)}


def load_moving_dsprites(args):
    with h5py.File(args.dataset_path + 'train.h5', 'r') as f:
        training_images = f['data'][:]
        training_labels = f['labels'][:]

    with h5py.File(args.dataset_path + 'test.h5', 'r') as f:
        test_images = f['data'][:]
        test_labels = f['labels'][:]

    return (MovingDSprites(training_images, training_labels),
            MovingDSprites(test_images, test_labels))


class MovingDSprites(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'index': index,
                'data': self.images[index].astype(np.float32) / 255,
                'static_labels': self.labels[index, [0, 1]].astype(np.long),
                'dynamic_labels': self.labels[index, [2, 3, 4]].astype(np.long)}


def load_3d_shapes(args):
    with h5py.File(args.dataset_path + 'shapes3d_dataset.h5', 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        training_images = data[f['train_indices'][:]]
        training_labels = labels[f['train_indices'][:]]

        test_images = data[f['test_indices'][:]]
        test_labels = labels[f['test_indices'][:]]

    return (ThreeDimensionalShapes(training_images, training_labels),
            ThreeDimensionalShapes(test_images, test_labels))


class ThreeDimensionalShapes(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'index': index,
                'data': self.images[index].astype(np.float32) / 255,
                'static_labels': self.labels[index, [0, 1, 2, 3, 4]].astype(np.long),
                'dynamic_labels': self.labels[index, [6, 7]].astype(np.long)}
