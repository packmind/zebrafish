import os
import collections
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import timm
import timm.optim
import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm
import datasets
import datasets.distributed
import albumentations

import zebrafish as zf

def train(rank, world_size, cfg):
    dist.init_process_group(cfg.distributed.backend, init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)

    local_batch_size = cfg.global_batch_size // world_size
    local_device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
    
    transform = functools.partial(zf.data.pair_augment, transforms1=zf.data.FMNIST_AUGS, transforms2=zf.data.FMNIST_AUGS)
    ds = zf.data.ShardedParquetDataset('ds/alzpq', transform=transform, ddp_rank=rank, ddp_world_size=world_size)
    dl = zf.data.shuffling_dataloader(ds, local_batch_size, 2, buffer_size=100)

    # model
    fcn = timm.create_model(cfg.fcn.model, pretrained=False, global_pool='avg', in_chans=cfg.fcn.nchannels, num_classes=0)
    projector = zf.contrastive.build_projector(fcn.num_features, cfg.projector.layers)
    model = nn.Sequential(collections.OrderedDict([('fcn', fcn), ('projector', projector)]))
    model.to(local_device)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False)
    
    # loss and optimizer
    opt = timm.optim.create_optimizer_v2(model, opt=cfg.optimizer.algo, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    loss_fn = zf.contrastive.NTXentLoss(cfg.global_batch_size, cfg.ntxent.temperature, cfg.ntxent.eps, local_device)
    #loss_fn = zf.contrastive.VicRegLoss(25, 25, 1)
    #torch.autograd.set_detect_anomaly(True)

    #for v1,v2 in tqdm.tqdm(dl):
    for epoch in range(40):
        print(f'EPOOOOOOOOCCCCHHHHH {epoch}')
        #ds.set_epoch(epoch)
        for batch, ex in enumerate(dl):
            opt.zero_grad()
            aug1 = ex['aug1'].to(local_device)
            aug2 = ex['aug2'].to(local_device)
            e1 = model(aug1)
            e2 = model(aug2)
            
            loss = loss_fn(e1,e2)
            loss.backward()
            opt.step()
            print(f'i am process {rank} out of {world_size} @{epoch}.{batch} with loss {loss}')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.fcn.model)
    print('cpus', os.cpu_count())
    print('cuda?', torch.cuda.is_available())
    #world_size = torch.cuda.device_count()
    world_size = 2 #os.cpu_count()
    assert cfg.global_batch_size % world_size == 0, "global batch size must be even multiple of world_size"
    mp.spawn(train,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    main()