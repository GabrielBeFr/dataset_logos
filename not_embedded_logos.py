import h5py
from utils import read_jsonl, append_dict_to_jsonl
import settings
import numpy as np
import tqdm
import torch
from transformers import CLIPModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from logos_loader import ImageDataset, custom_collate_fn

def count_cropped_logos():
    with h5py.File("cropped_logos.hdf5", 'r') as f:
        external_id_dset = f["external_id"]
        print(len(external_id_dset))

def create_hdf5_dataset():
    data_gen = read_jsonl(settings.jsonl_dataset)
    dataset_logos = set(int(dict["logo_id"]) for dict in data_gen)

    count = len(dataset_logos)

    with h5py.File(settings.hdf5_dataset, 'a') as f_dataset:
        embedding_dset = f_dataset.create_dataset(
            "embedding", (count, 512), dtype="f", chunks=True
        )
        external_id_dset = f_dataset.create_dataset(
            "external_id", (count,), dtype="i", chunks=True
        )
        f_dataset.create_dataset(
            "class", (count,), dtype="i", chunks=True
        )
        offset = 0

        with h5py.File(settings.logos_embeddings, 'r') as f_embeddings:
            logos = f_embeddings['embedding']
            ids = f_embeddings['external_id']
            for i in tqdm.tqdm(range(len(logos))):
                id = ids[i]
                if id in dataset_logos:
                    external_id_dset[offset] = id
                    embedding_dset[offset] = logos[i]
                    offset += 1
        


def embed_logos(logos, device, processor, model):
    with torch.inference_mode():
        inputs = processor(images=[logo for logo in logos], return_tensors="pt", padding=True).pixel_values
        outputs = model(**{'pixel_values':inputs.to(device),'attention_mask':torch.from_numpy(np.ones((len(logos),2), dtype=int)).to(device), 'input_ids':torch.from_numpy(np.ones((len(logos),2),dtype=int)*[49406,49407]).to(device)})
        return outputs.image_embeds.cpu().detach().numpy()


def add_embedding_to_not_embed_logos():
    with h5py.File(settings.hdf5_dataset, 'a') as f: 
        embeddings = f['embedding']
        ids = f['external_id']  

        offset = 0
        while ids[offset] != 0:
            offset += 1

        array = ids[:]
        non_zero_indexes = np.flatnonzero(array)
        offset = int(non_zero_indexes[-1]) + 1
        assert ids[offset] == 0

        for embedding_batch, id_batch in tqdm.tqdm(generate_logos_batch()):
            slicing = slice(offset, offset + len(embedding_batch))
            assert all(id == 0 for id in ids[slicing])
            ids[slicing] = id_batch
            embeddings[slicing] = embedding_batch
            offset += len(embedding_batch)


def generate_logos_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(f"openai/clip-vit-base-patch32").to(device)
    processor = CLIPImageProcessor()

    dataset = ImageDataset()
    dataloader = DataLoader(dataset, batch_size=16, num_workers=1, collate_fn=custom_collate_fn)

    for logos in dataloader:
      id_batch = np.array([logo[0] for logo in logos]) 
      image_batch = np.array([logo[1] for logo in logos]) 
      embedding_batch = embed_logos(image_batch, device, processor, model)
      yield embedding_batch, id_batch

def get_classes_id():
    data_gen = read_jsonl(settings.class_infos)
    res = {}
    for data in data_gen:
        res[data["class"]] = data["id"]
    return res


def add_classes():
    data_gen = read_jsonl(settings.jsonl_dataset)
    classes_id = get_classes_id()
    res = {}
    for dict in data_gen:
        res[str(dict["logo_id"])]=dict["class"]
    with h5py.File(settings.test_dataset, 'a') as f:
        ids = f['external_id']  
        classes = f['class']
        for i in tqdm.tqdm(range(len(ids))):
            str_classe = res[str(ids[i])]
            if str_classe is None:
                str_classe = "no_class"
            classes[i] = classes_id[str_classe]

def new_ids_classes():
    data_gen = read_jsonl("class_infos.jsonl")
    res = []
    id = 0
    for dict in data_gen:
        id +=1
        res.append({"class": dict["class"], "id": id, "amount": dict["amount"]})
    res[-1]["id"] = 0
    append_dict_to_jsonl(res, "new_class_infos.jsonl")


if __name__ == '__main__':
    print("add_classes()")
    add_classes()
