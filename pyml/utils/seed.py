import time
import random


def set_seed(seed=None):
    if seed is None:
        seed = time.time()

    # set random seed
    random.seed(seed)

    return seed