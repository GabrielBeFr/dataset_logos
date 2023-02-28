import pathlib
from typing import Iterable
from utils import get_offset, read_jsonl

import h5py

def generate_embedding_from_hdf5(
    data_gen: Iterable, output_path: pathlib.Path, output_dim: int, count: int
):

    """Save the embedding and the external id of each logo (data in data_gen) in an hdf5 file (the output_path).

    - data_gen: yielded embeddings and external ids of each logo from generate_embeddings_iter
    - output_path: path of the output hdf5 file
    - output_dim: dimension of the embeddings (depends on the computer vision model used)
    - count: amount of embeddings you want to save 
    """

    file_exists = output_path.is_file()

    with h5py.File(str(output_path), "a") as f:
        if not file_exists:
            embedding_dset = f.create_dataset(
                "embedding", (count, output_dim), dtype="f", chunks=True
            )
            external_id_dset = f.create_dataset(
                "external_id", (count,), dtype="i", chunks=True
            )
            class_dset = f.create_dataset(
                "class", (count,), dtype="i", chunks=True
            )
            weight_dset = f.create_dataset(
                "weight", (count,), dtype="f", chunks=True
            )
            offset = 0
        else:
            offset = get_offset(f)
            embedding_dset = f["embedding"]
            external_id_dset = f["external_id"]
            class_dset = f["class"]
            weight_dset = f["weight"]

        print("Offset: {}".format(offset))

        for dict in data_gen:
            slicing = slice(offset, offset + 1)
            embedding_dset[slicing] = dict["embedding"]
            external_id_dset[slicing] = dict["id"]
            class_dset[slicing] = dict["class"]
            weight_dset[slicing] = dict["weight"]
            offset += 1

data_gen = read_jsonl("Torch_test/data_file.jsonl")
generate_embedding_from_hdf5(data_gen, pathlib.Path("Torch_test/data_file.hdf5"), 512, 10000)

