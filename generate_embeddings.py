import h5py
from utils import read_jsonl
import settings
import numpy as np
import tqdm
import torch
from transformers import CLIPModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from logos_loader import ImageDataset, custom_collate_fn
import logging


def embed_logos(logos, device, processor, model):
    with torch.inference_mode():
        inputs = processor(images=[logo for logo in logos], return_tensors="pt", padding=True).pixel_values
        outputs = model(**{'pixel_values':inputs.to(device),'attention_mask':torch.from_numpy(np.ones((len(logos),2), dtype=int)).to(device), 'input_ids':torch.from_numpy(np.ones((len(logos),2),dtype=int)*[49406,49407]).to(device)})
        return outputs.image_embeds.cpu().numpy()


def add_embedding_to_not_embed_logos():
    with h5py.File(settings.hdf5_dataset, 'a') as f: 
        embeddings = f['embedding']
        ids = f['external_id']  

        array = ids[:]
        non_zero_indexes = np.flatnonzero(array)
        offset = int(non_zero_indexes[-1]) + 1
        assert ids[offset] == 0
        print(f"offset is {offset}")

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
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, num_workers=settings.num_workers, collate_fn=custom_collate_fn)

    for logos in dataloader:
        id_batch = np.array([logo[0] for logo in logos]) 
        image_batch = np.array([logo[1] for logo in logos])
        embedding_batch = embed_logos(image_batch, device, processor, model)
        yield embedding_batch, id_batch


if __name__ == '__main__':
    print("add_embedding_to_not_embed_logos()")

    logging.basicConfig(filename='logs.log', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

    add_embedding_to_not_embed_logos()
