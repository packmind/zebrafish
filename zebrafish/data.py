import os
import random
import pathlib
import io
import functools
from typing import Callable

import torch
import torch.utils.data
import skimage
import numpy as np
import albumentations 
import albumentations.pytorch

import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import tqdm


def equalize(image, **kwargs):
    return skimage.exposure.equalize_hist(image)

def to_one_channel(image, **kwargs):
    return image[:,:,1]

def image_decoder(rawbytes: bytes):
    """
    take raw encoded bytes, e.g. png, and return a numpy array
    """
    with io.BytesIO(rawbytes) as filelike:
        img = skimage.io.imread(filelike)
    return img

NO_AUGS = albumentations.Compose([
    albumentations.ToFloat(),
    albumentations.pytorch.transforms.ToTensorV2(),
])

FMNIST_AUGS = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(always_apply=True),
    albumentations.ToFloat(),
    albumentations.GaussNoise((0.02, 0.05), per_channel=False, p=0.25),
    albumentations.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.4, always_apply=True),
    albumentations.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5),    
    albumentations.pytorch.transforms.ToTensorV2(),
])

UCMERCED_AUGS = albumentations.Compose([
    albumentations.RandomCrop(224, 224, always_apply=True),  
    albumentations.ToGray(always_apply=True),
    albumentations.Lambda(image=to_one_channel),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(always_apply=True),
    albumentations.RandomResizedCrop(224, 224, (0.2, 1.0), ratio=(1.0, 1.0)),
    albumentations.ToFloat(),
    albumentations.GaussNoise((0.02, 0.05), per_channel=False, p=0.25),
    albumentations.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.4, always_apply=True),
    albumentations.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5),
    albumentations.CoarseDropout(max_holes=1, max_height=0.5, max_width=0.5, min_width=0.05, min_height=0.05, fill_value=0.5, p=0.25),
    albumentations.pytorch.transforms.ToTensorV2(),
])

def augviz(images: list[np.ndarray], transforms, preproc=None):
    N = min(len(images), 5)
    fig, axs = plt.subplots(N, 2, width_ratios=(1,3))
    for rowidx, img in enumerate(images[:N]):
        if preproc:
            img = preproc(img)
        augs = np.concatenate([transforms(image=img)['image'].permute((1,2,0)) for _ in range(4)], axis=1)
        axs[rowidx,0].imshow(img)
        axs[rowidx,0].axis('off')
        axs[rowidx,1].imshow(augs)
        axs[rowidx,1].axis('off')
    plt.tight_layout()
    plt.show()

def pair_augment(rec, transforms1, transforms2):
    assert 'image' in rec, "pair_augment expects an 'image' field"
    #img = np.asarray(rec['image'])
    img = image_decoder(rec['image'])
    rec['pixels'] = img
    rec['aug1'] = transforms1(image=img)['image']
    rec['aug2'] = transforms2(image=img)['image']
    return rec

class ContrastivePairDataset(torch.utils.data.Dataset):
    def __init__(self, basedir: str, transform1, transform2, preproc=None):
        with open(os.path.join(basedir, 'filelist.txt'), 'r') as f:
            self.img_paths = [os.path.join(basedir, line.strip()) for line in f]
            self.transform1 = transform1
            self.transform2 = transform2
            self.preproc = preproc
            random.shuffle(self.img_paths)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = skimage.io.imread(self.img_paths[idx])
        if self.preproc:
            img = self.preproc(img)
        aug1 = self.transform1(image=img)['image']
        aug2 = self.transform2(image=img)['image']
        return (aug1, aug2)

class ShardedParquetDataset(torch.utils.data.IterableDataset):
    """
    data is sharded across many parquet files
    files will be shuffled and split among dataloader workers, including 
    DDP processes if ddp_rank and ddp_world_size are set
    records from each file will come out in order, and need to be
    shuffled further

    - all parquet files should have the same number of records
    - if the number of parquet files is not evenly divisible by
      num_workers * ddp_world_size, some files will be skipped
      each epoch. this ensures all ddp processes have the same
      epoch sizes
    - only file order is shuffled, __iter__ yields records
      in order from each file
    """
    def __init__(self, 
                 basedir: str, 
                 columns: list[str]|None = None, 
                 transform: Callable[[dict], dict]|None = None, 
                 seed: int = 500, 
                 ddp_rank: int = 0, 
                 ddp_world_size: int = 1,
                 include_source: bool = False
                 ):
        """
        leave ddp_rank and ddp_world_size at 0 and 1 if not running DDP
        """
        self.columns = columns
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.transform = transform
        self.include_source = include_source
        # using python lists can lead to large memory consumption with DataLoader multiprocessing
        # see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.parquet_files = np.array(list(pathlib.Path(basedir).rglob('*.parquet')), dtype='U')
        self.parquet_files.sort()
        self.num_shards = len(self.parquet_files)
        assert self.num_shards > 0, f"no parquet files found under {basedir}"
        with pq.ParquetFile(self.parquet_files[0]) as pf:
            # all files *should* have the same number of shards
            self.records_per_shard = pf.metadata.num_rows

        self.rng = np.random.default_rng(seed)

    def samples_per_epoch(self, num_workers):
        """
        assuming num_workers dl workers, return # of samples per epoch
        (depends on num_workers because of even divisibility issues)
        """
        world_size = self.ddp_world_size * num_workers
        max_divisible = (self.num_shards // world_size) * world_size
        return self.records_per_shard * max_divisible
        

    def __iter__(self):
        # how many dataloader workers in this DDP process and which am I?
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            dl_rank = worker_info.id
            dl_world_size = worker_info.num_workers
        else:
            dl_rank = 0
            dl_world_size = 1
        # across all DDP processes, how many dataloader processes and which am I?
        world_size = self.ddp_world_size * dl_world_size
        rank = dl_world_size * self.ddp_rank + dl_rank
        # shuffle parquet files (shared seed -> all workers get the same order)
        epoch_permutation = self.rng.permutation(self.parquet_files)
        # discard some files each epoch to get even numbers across workers
        max_divisible = (self.num_shards // world_size) * world_size
        this_worker_shards = epoch_permutation[rank:max_divisible:world_size]
        for fn in this_worker_shards:
            with pq.ParquetFile(fn) as pf:
                assert pf.metadata.num_rows == self.records_per_shard, "parquet shards appear to have varying numbers of rows"
                batches = pf.iter_batches(columns=self.columns)
                records = (record for batch in batches for record in batch.to_pylist())
                for recordnum, record in enumerate(records):
                    if self.transform:
                        record = self.transform(record)
                    if self.include_source:
                        record['parquet_filename'] = fn
                        record['parquet_recordnum'] = recordnum
                    yield record

def shuffling_dataloader(ds, batch_size:int, num_workers:int, buffer_size:int=100_000):
    """
    approximately shuffle iterable datasets w/ multiple processes for e.g. augmentation
    - DataLoader #1: spawn multiple workers and iterate one record at a time
    - ShuffleIterDataPipe: shuffle the output of DL1 with a buffer
    - DataLoader #2: batch the output otput of the shuffler

    notes:
    - data should be sharded and shards should be shuffed in Dataset.__iter__
    - Dataset should handle ensuring workers and DDP processes get disjoint sets of records
    - see ShardedParquetDataset
    - relies on persistent_workers to have different shuffles every epoch
    """
    dl1 = torch.utils.data.DataLoader(ds, batch_size=None, sampler=None, shuffle=False, num_workers=num_workers, persistent_workers=True, multiprocessing_context='spawn')
    shuffled = torch.utils.data.datapipes.iter.combinatorics.ShufflerIterDataPipe(dl1, buffer_size=buffer_size)
    # note that DataLoader shuffle parameter does control ShufflerIterDataPipe
    dl2 = torch.utils.data.DataLoader(shuffled, batch_size=batch_size, drop_last=True, sampler=None, shuffle=True, pin_memory=True)
    return dl2

if __name__ == '__main__':
    transform = functools.partial(pair_augment, transforms1=FMNIST_AUGS, transforms2=FMNIST_AUGS)
    ds = ShardedParquetDataset('ds/alzpq', transform=transform)
    dl = shuffling_dataloader(ds, 32, 4, buffer_size=10)
    for batchnum, batch in tqdm.tqdm(enumerate(dl)):
        #print(batch['pixels'].sum())
        pass