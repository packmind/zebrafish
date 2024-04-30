import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import zebrafish as zf

def shuffle(rank, world_size):
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)
    ds = zf.data.ShardedParquetDataset('ds/seqpq', ddp_rank=rank, ddp_world_size=world_size)
    dl = zf.data.shuffling_dataloader(ds, 1, 2, buffer_size=100_000)

    for epoch in range(2):
        observed = set()
        for batchnum, batch in tqdm.tqdm(enumerate(dl)):
            observed.update(batch['fn'])
        dist.barrier()
        print(epoch, rank, sorted(observed))

def main() -> None:
    print('cpus', os.cpu_count())
    print('cuda?', torch.cuda.is_available())
    #world_size = torch.cuda.device_count()    
    world_size = 2
    mp.spawn(shuffle,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')    
    main()