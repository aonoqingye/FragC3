"""Complete, balanced batches without BatchNorm singleton batches."""

import math

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Partition a unique sampler sequence into ceil(N/B) near-equal batches."""

    def __init__(self, sampler, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 3:
            raise ValueError("Balanced BatchNorm training requires an integer batch_size >= 3")
        self.sampler = sampler
        self.batch_size = batch_size
        self.sample_count = len(sampler)
        if self.sample_count < 2:
            raise ValueError("Balanced BatchNorm training requires at least two distinct rows")
        self.step_count = math.ceil(self.sample_count / self.batch_size)
        self.small_size, self.large_count = divmod(self.sample_count, self.step_count)
        if self.small_size < 2:
            raise ValueError("The requested row/step limits cannot avoid singleton batches")

    def __len__(self):
        return self.step_count

    def __iter__(self):
        values = list(self.sampler)
        if len(values) != self.sample_count:
            raise ValueError("Sampler length changed after training batch construction")
        if len(set(values)) != self.sample_count:
            raise ValueError("Complete training batches require unique sampler indices")
        offset = 0
        for step in range(self.step_count):
            size = self.small_size + int(step < self.large_count)
            yield values[offset : offset + size]
            offset += size
        assert offset == self.sample_count
