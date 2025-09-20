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


def get_positions():
    position_dynamics = [[31, 0, 31, 0, 31, 0, 31, 0],
                         [19, 17, 15, 13, 11, 9, 7, 5],
                         [0, 3, 6, 9, 12, 15, 18, 21],
                         [2, 6, 10, 14, 18, 22, 26, 30],
                         [27, 27, 19, 19, 11, 11, 3, 3],
                         [30, 20, 20, 10, 10, 20, 20, 30],
                         [1, 1, 2, 3, 5, 8, 13, 21],
                         [31, 29, 23, 19, 17, 13, 11, 7]]

    return np.array(position_dynamics, dtype=np.long)


def get_labels():
    idx_color = list(range(6))
    idx_shape = list(range(3))
    idx_scale = list(range(10))
    idx_position_x = list(range(8))
    idx_position_y = list(range(8))
    idx = list(itertools.product(idx_color, idx_shape, idx_scale, idx_position_x, idx_position_y))

    random.shuffle(idx)
    split = int(0.8 * len(idx))

    return np.array(idx[:split], dtype=np.long), np.array(idx[split:], dtype=np.long)


def make_sequence(images, colors, scales, positions, labels):
    video = np.zeros((8, 3, 64, 64), dtype=np.uint8)

    idx_color = labels[0]
    idx_shape = labels[1]
    idx_scale = scales[labels[2]]
    idx_posx = positions[labels[3]]
    idx_posy = positions[labels[4]]
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
position_dynamics_possib = get_positions()
labels_train, labels_test = get_labels()

frames_train = [make_sequence(imgs, color_possib, scale_dynamics_possib,
                              position_dynamics_possib, lbls) for lbls in labels_train]
frames_test = [make_sequence(imgs, color_possib, scale_dynamics_possib,
                             position_dynamics_possib, lbls) for lbls in labels_test]

with h5py.File(DATA_OUT + 'train.h5', 'w') as hf:
    hf.create_dataset('data', data=np.stack(frames_train))
    hf.create_dataset('labels', data=np.stack(labels_train))

with h5py.File(DATA_OUT + 'test.h5', 'w') as hf:
    hf.create_dataset('data', data=np.stack(frames_test))
    hf.create_dataset('labels', data=np.stack(labels_test))
