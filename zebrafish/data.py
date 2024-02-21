import os
import random
import functools

import torch
import skimage
import numpy as np
import albumentations 
import albumentations.pytorch
import datasets
import datasets.distributed

import matplotlib.pyplot as plt
import tqdm


def equalize(image, **kwargs):
    return skimage.exposure.equalize_hist(image)

def to_one_channel(image, **kwargs):
    return image[:,:,1]


FMNIST_AUGS = albumentations.Sequential([
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


def huggingface_pair_augmenter(batch_dict, transforms1, transforms2):
    aug1 = [transforms1(image=np.asarray(img))['image'] for img in batch_dict['image']]
    aug2 = [transforms2(image=np.asarray(img))['image'] for img in batch_dict['image']]
    return {'aug1': aug1, 'aug2': aug2}

def huggingface_contrastive_pair(name: str, split: str, transforms, rank: int, world_size: int):
    hg_ds = datasets.load_dataset(name, split=split)
    ds = hg_ds.with_transform(functools.partial(huggingface_pair_augmenter, transforms1=transforms, transforms2=transforms))
    ds = datasets.distributed.split_dataset_by_node(ds, rank, world_size)
    return ds


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


if __name__ == '__main__':
    ds = huggingface_contrastive_pair('fashion_mnist', 'train', FMNIST_AUGS, 0, 1)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=32, drop_last=True, num_workers=4)
    for batchnum, batch in tqdm.tqdm(enumerate(dl)):
        pass
    