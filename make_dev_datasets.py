
import io
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import numpy as np
import datasets


def batchify(n: int, it: Iterable, return_partial: bool = True):
    batch = []
    for record in it:
        batch.append(record)
        if len(batch) == n:
            yield batch
            batch = []
    if batch and return_partial:
        yield batch


def hf_adapter(ds):
    for sample in ds:
        imgbuffer = io.BytesIO()
        sample['image'].save(imgbuffer, format='PNG')
        yield {'image': imgbuffer.getvalue(), 'label': sample['label']}

if __name__ == '__main__':
    # # mri images to webdataset
    # for split, ds in datasets.load_dataset('Falah/Alzheimer_MRI').items():
    #     ds = ds.shuffle(42)
    #     with wds.ShardWriter(f'ds/alzheimer_mri/shard-{split}-%06d.tar', maxcount=500, encoder=False) as sink:
    #         for idx,example in tqdm.tqdm(enumerate(ds)):
    #             img = np.asarray(example['image']).astype(np.uint16) * 257 # expand to uint16 range
    #             is_success, imgbytes = cv2.imencode(".png", img)
    #             meta = {'label': example['label']}
    #             sample = {
    #                 '__key__': f"{idx:011}",
    #                 'image.png': imgbytes.tobytes(),
    #                 'meta.json': json.dumps(meta).encode('utf8')
    #             }
    #             sink.write(sample)
    
    # mri images to parquet
    #datasets = {'alzmri': 'Falah/Alzheimer_MRI', 'fmnist': 'fashion_mnist'}
    schema = pa.schema({'image': pa.binary(), 'label':pa.int64()}) 
    records_per_file = 10
    for split, ds in datasets.load_dataset('Falah/Alzheimer_MRI').items():
        ds = ds.shuffle(42)
        for batch_idx, batch in tqdm.tqdm(enumerate(batchify(records_per_file,hf_adapter(ds),return_partial=False))):
            table = pa.Table.from_pylist(batch, schema=schema)
            pq.write_table(table, f'ds/alzpq/{split}-{batch_idx:08}.parquet', compression='zstd')    

    # simple sequential shards
    schema = pa.schema({'id': pa.int64()})
    for batch_idx, batch in tqdm.tqdm(enumerate(batchify(10,[{'id': idx} for idx in range(2000)],return_partial=False))):
        table = pa.Table.from_pylist(batch, schema=schema)
        pq.write_table(table, f'ds/seqpq/{batch_idx:08}.parquet', compression='zstd')

