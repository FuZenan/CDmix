# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from random import sample
import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            # sampler = torch.utils.data.RandomSampler(dataset,
            #     replacement=True)
            sampler = torch.utils.data.RandomSampler(dataset, 
                replacement=False)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
                # value = next(self._infinite_iterator)
                # print("Iterator value:", value)
                x, y, z = next(self._infinite_iterator)
                if y is None:
                    raise ValueError("Received None label in the iterator")
                yield x, y, z

    def __len__(self):
        raise ValueError

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=1,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

    # InfiniteDataLoader: 设计用于训练阶段，允许数据加载器在数据集上无限循环。它的特点是可以一直从数据集中获取样本，且样本的选择可以是加权采样（使用 WeightedRandomSampler）或者随机采样。这种设计适用于需要在训练过程中连续从数据集中抽取批次的场景，尤其是在数据增强或域自适应等任务中。
    # FastDataLoader: 设计用于评估阶段，旨在提高数据加载的速度。它通过优化 DataLoader 的工作方式，使得每个 epoch 不需要重新生成 worker 进程。FastDataLoader 不支持无限循环，适合评估任务，因为在评估时通常只需要在数据集上遍历一遍