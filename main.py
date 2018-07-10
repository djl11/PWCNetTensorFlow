#!/usr/bin/env python3
import faulthandler
import numpy as np
from task import Task

def main():

    seed = 1
    np.random.seed(seed)

    faulthandler.enable()

    mode = input('\n\ntrain (t), visualise (v) ?\n')
    while mode != 't' and mode != 'v':
        mode = input('\n\ntrain (t), visualise (v) ?\n')

    task = Task()

    if mode == 't':
        task.run(train=True)
    else:
        task.run(train=False)


if __name__ == '__main__':
    main()
