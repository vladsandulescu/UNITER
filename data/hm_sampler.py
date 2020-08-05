import random
import numpy as np
import itertools
from torch.utils.data import Sampler


class HMSampler(Sampler):
    def __init__(self, labels, batch_size, replacement=True, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.replacement = replacement
        self.shuffle = shuffle

    def __iter__(self):
        pos_ids = np.where(self.labels == 1)[0]
        neg_ids = np.where(self.labels == 0)[0]
        h_bs = int(self.batch_size/2)
        if self.shuffle:
            random.shuffle(pos_ids)
            random.shuffle(neg_ids)

        if len(neg_ids) > len(pos_ids):
            batches = list(zip(neg_ids[:len(pos_ids)], pos_ids))
            if self.replacement:
                unmatched_neg_ids = neg_ids[len(pos_ids):]
                resampled_positives = random.sample(list(pos_ids), len(unmatched_neg_ids))
                batches.extend(list(zip(unmatched_neg_ids, resampled_positives)))
            batches = [batches[i:i + h_bs] for i in range(0, len(batches), h_bs)]
            batches = list(list(itertools.chain(*batch)) for batch in batches)

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported.")
