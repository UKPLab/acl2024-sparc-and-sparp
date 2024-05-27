import numpy as np


class HFDatasetExemplars:
    def __init__(self, dataset, seed=42, init_shuffle=False):
        self.dataset = dataset
        if init_shuffle:
            self.dataset = dataset.shuffle(seed=42)
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, k=1, random=True):
        if random:
            self.dataset = self.dataset.shuffle(generator=self.rng)
        samples = self.dataset.select(range(k)).to_list()
        return samples
