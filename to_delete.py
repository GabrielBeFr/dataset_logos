import h5py
import numpy as np
from utils import read_jsonl
import settings
import tqdm


def get_not_embedded_logos():

    with h5py.File("datasets/test_dataset.hdf5", 'r') as f:
        logos = f['embedding']
        ids = f['external_id']
        classes = f['class']
        arr_ids = np.array(ids)
        arr_classes = np.array(classes)
        embedded_logos = set(int(id) for id in ids)
    breakpoint()
    res = {}
    for i in tqdm.tqdm(range(len(arr_classes))):
        classe = arr_classes[i]
        if arr_ids[i] == 0: break
        try:
            res[classe] += 1
        except:
            res[classe] = 0

    for key in res.keys():
        if res[key]<100:
            print(key)

        
    breakpoint()
    print("got the embedded one")

diff = get_not_embedded_logos()
breakpoint()
    