import random


class dmcr_random:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args, **kwargs):
        random.setstate(self.old_state)
