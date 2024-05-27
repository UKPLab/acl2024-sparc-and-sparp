import numpy as np


class ListExemplars:
    def __init__(self, exemplars, seed=42, init_shuffle=False):
        self.exemplars = exemplars
        self.rng = np.random.default_rng(seed=seed)
        if init_shuffle:
            self.dataset = self.rng.permuted(dataset)

    def sample(self, k=1, random=True):
        if random:
            samples = self.rng.choice(self.exemplars, size=k, replace=False)
            return samples.tolist()
        return self.exemplars[:k]
