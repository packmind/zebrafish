import os
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import scipy.stats
import numpy as np

import zebrafish as zf

def shuffle(rank, world_size):
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)
    ds = zf.data.ShardedParquetDataset('ds/seqpq', ddp_rank=rank, ddp_world_size=world_size, include_source=True)
    dl = zf.data.shuffling_dataloader(ds, 8, 2, buffer_size=100_000)
    observed_epochs = list()
    for epoch in range(2):
        observed = list()
        print('epoch ', epoch)
        for batchnum, batch in enumerate(dl):
            observed.extend(list(zip(batch['parquet_filename'], batch['parquet_recordnum'].tolist())))
        observed_epochs.append(observed)
        
    observed_all = [None for _ in range(world_size)]
    dist.gather_object(observed_epochs, observed_all if rank == 0 else None, dst=0)
    if rank == 0:
        observed_array = np.array(observed_all)
        # nsplits x nepochs x splitsize
        records = np.char.add(observed_array[:,:,:,0], observed_array[:,:,:,1])
        # interleave splits for each epoch -> nepochs x epochsize
        interleaved = records.transpose(1, 2, 0).reshape((records.shape[1],-1))
        print('rank correlation between epoch 1 and epoch 2:')
        print('\t', scipy.stats.spearmanr(interleaved[0,:], interleaved[1,:]))
        for epoch in range(2):
            print(f'epoch{epoch}')
            # exhaustive filenames
            all_filenames = set(observed_array[:,epoch,:,0].ravel())
            print(f'{len(all_filenames)}/{ds.num_shards} filenames observed')

            # overlapping filenames
            pooled = len(set(observed_array[:,epoch,:,0].ravel()))
            per_split = [len(set(observed_array[split,epoch,:,0])) for split in range(observed_array.shape[0])]
            if pooled == sum(per_split):
                print('no overlapping filenames across DDP')
            else:
                print('ATTENTION: some overlap across DDP')
            # duplicate records
            if len(interleaved[epoch,:]) == len(set(interleaved[epoch,:])):
                print('no duplicate records')
            else:
                print('ATTENTION: duplicate records')

def main() -> None:
    print('cpus', os.cpu_count())
    print('cuda?', torch.cuda.is_available())
    #world_size = torch.cuda.device_count()    
    world_size = 3
    mp.spawn(shuffle,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')    
    main()