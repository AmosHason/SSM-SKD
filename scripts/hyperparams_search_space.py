import os
import random

space = {'--k-dim': (8, 12, 16, 20, 24, 28, 32),
         '--hidden-dim': (80, 120, 160, 200),
         '--w-pred': (0.25, 1.0, 4.0),
         '--w-eigs': (0.25, 1.0, 4.0),
         '--dynamic-thresh': (0.125, 0.275, 0.425, 0.575, 0.725, 0.875)}

combinations = []


def create_combinations(combination=None):
    if combination is None:
        combination = {}

    i = len(combination.items())

    if i == len(space.items()):
        combinations.append(combination)
        return

    for v in list(space.items())[i][1]:
        create_combinations({**combination, list(space.items())[i][0]: v})


def main():
    create_combinations()
    random.shuffle(combinations)

    with open('hyperparams_search_space.txt', 'w') as f:
        for c in combinations:
            for i, (k, v) in enumerate(c.items()):
                if i != 0:
                    f.write(' ')
                f.write(f'{k} {v}')
            f.write(os.linesep)


main()
