import os
import random
import functools
import pathlib

import torch
import torch.distributed as dist
import skimage
import numpy as np
import albumentations 
import albumentations.pytorch
import datasets
import datasets.distributed

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

def huggingface_pil_decoder(batch_dict):
    batch_dict['image'] = [NO_AUGS(image=np.asarray(img))['image'] for img in batch_dict['image']]
    return batch_dict

def huggingface_pair_augmenter(batch_dict, transforms1, transforms2):
    aug1 = [transforms1(image=np.asarray(img))['image'] for img in batch_dict['image']]
    aug2 = [transforms2(image=np.asarray(img))['image'] for img in batch_dict['image']]
    return {'aug1': aug1, 'aug2': aug2}

def huggingface_contrastive_pair(name: str, split: str, transforms, rank: int, world_size: int):
    hg_ds = datasets.load_dataset(name, split=split)
    ds = hg_ds.with_transform(functools.partial(huggingface_pair_augmenter, transforms1=transforms, transforms2=transforms))
    ds = datasets.distributed.split_dataset_by_node(ds, rank, world_size)
    return ds

def pair_augment(rec, transforms1, transforms2):
    assert 'image' in rec, "pair_augment expects an 'image' field"
    #img = np.asarray(rec['image'])
    img = image_decoder(rec['image'])
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
    todo: 
     optional transform function dict -> dict
    """
    def __init__(self, basedir: str, columns: list[str]|None = None, seed: int = 500, ddp_rank: int = 0, ddp_world_size: int = 1):
        """
        leave ddp_rank and ddp_world_size at 0 and 1 if not running DDP
        """
        self.columns = columns
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        # using python lists can lead to large memory consumption with DataLoader multiprocessing
        # see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.parquet_files = np.array(list(pathlib.Path(basedir).rglob('*.parquet')), dtype='U')
        self.parquet_files.sort()
        self.num_shards = len(self.parquet_files)
        self.rng = np.random.default_rng(seed)

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
                for batch in pf.iter_batches(columns=self.columns):
                    for record in batch.to_pylist():
                        record['fn'] = fn
                        record['loader_rank'] = rank
                        yield record

def shuffling_dataloader(ds, batch_size, num_workers, buffer_size=100_000):
    dl1 = torch.utils.data.DataLoader(ds, batch_size=None, sampler=None, shuffle=False, num_workers=num_workers, persistent_workers=True, multiprocessing_context='spawn')
    shuffled = torch.utils.data.datapipes.iter.combinatorics.ShufflerIterDataPipe(dl1, buffer_size=buffer_size)
    dl2 = torch.utils.data.DataLoader(shuffled, batch_size=batch_size)
    return dl2

if __name__ == '__main__':
    #ds = huggingface_contrastive_pair('fashion_mnist', 'train', FMNIST_AUGS, 0, 1)
    #dl = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=32, drop_last=True, num_workers=4)
    #for batchnum, batch in tqdm.tqdm(enumerate(dl)):
    #    pass
    ds = ShardedParquetDataset('ds/seqpq')
    #dl = torch.utils.data.DataLoader(ds, batch_size=None)
    dl = shuffling_dataloader(ds, 1, 2, buffer_size=100_000)
    for batchnum, batch in tqdm.tqdm(enumerate(dl)):
        print(batch)
    