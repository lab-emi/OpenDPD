"""Vectorized indexing for the in-memory IQ datasets.

The original DataLoader still owns shuffling, RNG consumption, batching and
pinning. Only fetching a list of rows is vectorized, eliminating one Python
call and two tiny tensor allocations per training sequence.
"""
import os

from torch.utils.data import DataLoader, Dataset


class _BatchedIQ(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __getitems__(self, indices):
        return self.dataset.features[indices], self.dataset.targets[indices]


def _already_batched(batch):
    return batch


def iq_loader(dataset, *, batch_size, shuffle, pin_memory):
    if os.getenv("OPENDPD_DISABLE_BATCHED_LOADER", "0") == "1":
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return DataLoader(_BatchedIQ(dataset), batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, collate_fn=_already_batched)
