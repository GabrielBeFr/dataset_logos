import json
from pathlib import Path
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import settings
from typing import Tuple
from math import floor
import h5py
import tqdm

def append_dict_to_jsonl(list_dict_to_append: list, file_path: str):
    Path(file_path).touch(exist_ok=True)        
    with open(file_path, 'a+') as f:
        for dict in list_dict_to_append:
            json.dump(dict, f)
            f.write('\n')

def read_jsonl(file_path: str):
    with open(file_path, 'rb') as f:
        for row in f:
            yield json.loads(row)

def count_jsonl_lines(file_path: str):
    with open(file_path, 'r') as f:
        res = sum(1 for line in f)
    return res
    
def get_list_classes(nb_logos_min: int, file_path: str):
    classes_list = []
    annotation_types = []
    amounts_list = []
    data_gen = read_jsonl(file_path)
    for dict in data_gen:
        if dict["amount"]<nb_logos_min:
            break
        classes_list.append(dict["class"])
        amounts_list.append(dict["amount"])
        annotation_types.append(dict["annotation_type"])
    return classes_list, annotation_types, amounts_list

def get_random_dict_from_jsonl(file_path: str, max: int):
    data_gen = read_jsonl(file_path)
    alea = np.random.randint(0,max)
    count = 0
    for dict in data_gen:
        if count == alea:
            return dict
        count +=1

def resize_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    width, height = image.size
    max_width, max_height = max_size

    if width > max_width or height > max_height:
        new_image = image.copy()
        new_image.thumbnail((max_width, max_height))
        return new_image

    return image

def convert_image_to_array(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def get_image_from_url(url: str):
    r = requests.get(url)
    image = Image.open(BytesIO(r.content))
    resized_image = resize_image(image, settings.image_size)
    image_array = convert_image_to_array(resized_image)
    return image_array

def crop_image(
    image: np.ndarray, bounding_box: Tuple[float, float, float, float]
) -> np.ndarray:

    """Return the cropped logo as an array extracted from the array of the image"""

    ymin, xmin, ymax, xmax = bounding_box
    height, width = image.shape[:2]
    (left, right, top, bottom) = (
        floor(xmin * width),
        floor(xmax * width),
        floor(ymin * height),
        floor(ymax * height),
    )
    return image[top:bottom, left:right]

def get_offset(f: h5py.File) -> int:
    external_id_dset = f["external_id"]
    array = external_id_dset[:]
    non_zero_indexes = np.flatnonzero(array)
    return int(non_zero_indexes[-1]) + 1

def create_json_data():
    res = []
    id = -1
    amounts = [0 for i in range(10)]
    for i in tqdm.tqdm(range(10000)):
        classe = int(np.random.choice(10, 1, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.3]))
        embedding = list(np.random.rand(512))
        id += 1
        res.append({"embedding": embedding, "class": classe, "id": id})
        amounts[classe]=amounts[classe]+1
        assert type(res[0]["embedding"])==list

    for i in tqdm.tqdm(range(10000)):
        logo_dict = res[i]
        logo_dict["weight"] = 1/amounts[logo_dict["class"]]

    append_dict_to_jsonl(res, "Torch_test/data_file.jsonl")

def matching_classes(classe:str):
    if classe == "en:max-havelaar":
        classe = "en:fairtrade-international"
    elif classe == "en:european-vegetarian-union-vegan":
        classe = "en:european-vegetarian-union"
    elif classe == "Cocacola":
        classe = "Coca-Cola"
    elif classe == "Leclerc":
        classe = "E.Leclerc"
    elif classe == "Lindt":
        classe = "Lindt & Spr\u00fcngli"
    elif classe == "en:no-gluten":
        classe = None
    elif "-gold-medal-of-the-german-agricultural-society" in classe:
        classe = "en:gold-medal-of-the-german-agricultural-society"
    return classe

def get_product_from_source(source_img: str):
    numbers = source_img.split("/")[1:-1]
    product = ""
    for number in numbers:
        product += number
    return product


if __name__ == "__main__":
    get_product_from_source("/426/054/168/7191/5.jpg")
    breakpoint()