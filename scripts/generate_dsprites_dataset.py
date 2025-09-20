import itertools
import random

import h5py
import numpy as np

DATA_IN = ''
DATA_OUT = ''


def get_scales():
    scale_dynamics = []
    for _ in range(10):
        roll_pos = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.long)
        roll_pos = np.roll(roll_pos, random.randint(0, 5))
        pos_speed = np.roll(np.array([2, 1, 0, -1, -2, -1, 0, 1]), random.randint(0, 4))

        scales = []
        for _ in range(8):
            scales.append(roll_pos[0])
            roll_pos = np.roll(roll_pos, pos_speed[0])
            pos_speed = np.roll(pos_speed, random.choice([0, 0, 1, -1]))
        scale_dynamics.append(scales)

    return np.array(scale_dynamics, dtype=np.long)


def get_labels():
    idx_color = list(range(6))
    idx_shape = list(range(3))
    idx_position_x = list(range(8))
    idx_position_y = list(range(8))
    idx_static = list(itertools.product(idx_color, idx_shape, idx_position_x, idx_position_y))

    idx_scale = [(x,) for x in range(10)]

    random.shuffle(idx_static)
    split = int(0.8 * len(idx_static))
    idx_train = list(itertools.product(idx_static[:split], idx_scale))
    idx_train = [[*i[0], *i[1]] for i in idx_train]
    idx_test = list(itertools.product(idx_static[split:], idx_scale))
    idx_test = [[*i[0], *i[1]] for i in idx_test]
    random.shuffle(idx_train)
    random.shuffle(idx_test)

    return np.array(idx_train, dtype=np.long), np.array(idx_test, dtype=np.long)


def make_sequence(images, colors, scales, labels):
    video = np.zeros((8, 3, 64, 64), dtype=np.uint8)

    idx_color = labels[0]
    idx_shape = labels[1]
    idx_posx = labels[2] * 4 + 2
    idx_posy = labels[3] * 4 + 2
    idx_scale = scales[labels[4]]
    idx_tot = idx_shape * 32 * 32 * 40 * 6 + idx_scale * 32 * 32 * 40 + idx_posx * 32 + idx_posy

    video[:, 0] = images[idx_tot] * colors[idx_color, 0]
    video[:, 1] = images[idx_tot] * colors[idx_color, 1]
    video[:, 2] = images[idx_tot] * colors[idx_color, 2]

    return video


random.seed(42)
imgs = np.load(DATA_IN, allow_pickle=True)['imgs']
color_possib = np.array([(102, 205, 170), (255, 165, 0), (0, 255, 0),
                         (0, 0, 255), (30, 144, 255), (255, 20, 147)], dtype=np.uint8)
scale_dynamics_possib = get_scales()
labels_train, labels_test = get_labels()

frames_train = [make_sequence(imgs, color_possib, scale_dynamics_possib, lbls) for lbls in labels_train]
frames_test = [make_sequence(imgs, color_possib, scale_dynamics_possib, lbls) for lbls in labels_test]

with h5py.File(DATA_OUT + 'train.h5', 'w') as hf:
    hf.create_dataset('data', data=np.stack(frames_train))
    hf.create_dataset('labels', data=np.stack(labels_train))

with h5py.File(DATA_OUT + 'test.h5', 'w') as hf:
    hf.create_dataset('data', data=np.stack(frames_test))
    hf.create_dataset('labels', data=np.stack(labels_test))
