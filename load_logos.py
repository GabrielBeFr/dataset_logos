from torch.utils.data import Dataset
from utils import append_dict_to_jsonl, read_jsonl, crop_image, convert_image_to_array
import settings
import tqdm
import h5py
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

# Define a custom dataset class that downloads the logos
class ImageDataset(Dataset):
  def __init__(self):

    logo_id_list = []
    source_image_list = []
    bounding_box_list = []

    not_embedded_logos = get_not_embedded_logos()
    data_gen = read_jsonl(settings.jsonl_dataset)
    error_logos = get_error_logos()

    print(f"Len diff is {len(not_embedded_logos)}")

    print("Getting infos on not_embedded_logos")
    for dict in tqdm.tqdm(data_gen):
        logo_id = dict["logo_id"]
        if dict["logo_id"] not in not_embedded_logos or dict["logo_id"] in error_logos:  # Continue if logo has already been embedded
            continue
        logo_id_list.append(logo_id)
        source_image_list.append(dict["source_img"])
        bounding_box_list.append(dict["bounding_box"])
    
    self.logo_ids = logo_id_list
    self.source_images = source_image_list
    self.bounding_boxes = bounding_box_list

  def __len__(self):
    return len(self.logo_ids)

  def __getitem__(self, idx):
    # Download the logo and return it
    logo = generate_logo(self.source_images[idx], self.bounding_boxes[idx]) 
    return (self.logo_ids[idx], logo)

def custom_collate_fn(batch):
  batch_res = []

  for item in batch:
    if item[1] is None:
      append_dict_to_jsonl([{"logo_id":item[0]}], settings.error)
      continue
    batch_res.append((item[0], item[1]))

  return batch_res

def get_error_logos():
  res = set([])
  datagen = read_jsonl(settings.error)
  for id in datagen:
    res.add(id["logo_id"])
  return res

def get_not_embedded_logos():

    with h5py.File(settings.hdf5_dataset, 'r') as f:
        logos = f['embedding']
        ids = f['external_id']

        embedded_logos = set(int(id) for id in ids)

    print("got the embedded one")

    data_gen = read_jsonl(settings.jsonl_dataset)
    dataset_logos = set(int(dict["logo_id"]) for dict in data_gen)

    print("got the dataset ones")

    diff = dataset_logos-embedded_logos

    return diff

def generate_logo(source_img: str, bounding_box: list):
    try:
      r = requests.get("https://images.openfoodfacts.org/images/products"+source_img)
      image = Image.open(BytesIO(r.content))
      array_image = convert_image_to_array(image) 
    except:
      return None
    return process_image(array_image, bounding_box)

def process_image(array_image: np.array, bounding_box: list):
    cropped_img = crop_image(array_image, bounding_box)
    b, g, r = cv2.split(cropped_img)
    b = cv2.resize(b, (200, 200), interpolation=cv2.INTER_CUBIC)
    g = cv2.resize(g, (200, 200), interpolation=cv2.INTER_CUBIC)
    r = cv2.resize(r, (200, 200), interpolation=cv2.INTER_CUBIC)
    cropped_resized_img = cv2.merge((b, g, r))
    return cropped_resized_img


if __name__ == '__main__':
  error_logos = get_error_logos()
  breakpoint()