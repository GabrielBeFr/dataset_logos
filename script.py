import numpy as np
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import torch
from transformers import CLIPModel, CLIPImageProcessor
from utils import crop_image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import json

def save_logo_embeddings(bounding_boxes: list, image: Image.Image):
    """Generate logo embeddings using CLIP model and save them in
    logo_embedding table."""
    resized_cropped_images = []
    for i in range(len(bounding_boxes)):
        y_min, x_min, y_max, x_max = bounding_boxes[i]
        (left, right, top, bottom) = (
            x_min * image.width,
            x_max * image.width,
            y_min * image.height,
            y_max * image.height,
        )
        cropped_image = image.crop((left, top, right, bottom))
        resized_cropped_images.append(cropped_image.resize((224,224)))
    embeddings = generate_clip_embedding(resized_cropped_images)
    return embeddings

def generate_clip_embedding(images: list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(f"openai/clip-vit-base-patch32").to(device)
    #model.eval()
    processor = CLIPImageProcessor()
    with torch.inference_mode():
        inputs = processor(images=[logo for logo in images], return_tensors="pt", padding=True).pixel_values
        outputs = model(**{'pixel_values':inputs.to(device),'attention_mask':torch.from_numpy(np.ones((len(images),2), dtype=int)).to(device), 'input_ids':torch.from_numpy(np.ones((len(images),2),dtype=int)*[49406,49407]).to(device)})
        return outputs.image_embeds.cpu().numpy()


def embed_logos(logos, device, processor, model):
    with torch.inference_mode():
        inputs = processor(images=[logo for logo in logos], return_tensors="pt", padding=True).pixel_values
        outputs = model(**{'pixel_values':inputs.to(device),'attention_mask':torch.from_numpy(np.ones((len(logos),2), dtype=int)).to(device), 'input_ids':torch.from_numpy(np.ones((len(logos),2),dtype=int)*[49406,49407]).to(device)})
        return outputs.image_embeds.cpu().numpy()

def generate_logo(bounding_boxes: list):
    r = requests.get("https://openfoodfacts.org/images/products/501/653/365/5155/1.jpg")
    image_array = np.asarray(bytearray(r.content), dtype=np.uint8)
    cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    res = []
    for bb in bounding_boxes:
        res.append(process_image(cv_image, bb))
    return res

def process_image(array_image: np.array, bounding_box: list):
    cropped_img = crop_image(array_image, bounding_box)
    cropped_resized_img = resize_image(cropped_img)
    return cropped_resized_img

def resize_image(image):
  size = 224
  b, g, r = cv2.split(image)
  b = cv2.resize(b, (size, size), interpolation=cv2.INTER_CUBIC)
  g = cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
  r = cv2.resize(r, (size, size), interpolation=cv2.INTER_CUBIC)
  return cv2.merge((b, g, r))

def check_embeddings(bounding_boxes, image):
    embeddings_new = save_logo_embeddings(bounding_boxes, image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(f"openai/clip-vit-base-patch32").to(device)
    processor = CLIPImageProcessor()
    logos = generate_logo(bounding_boxes)
    embeddings_old = embed_logos(logos, device, processor, model) 

    return embeddings_new, embeddings_old

def get_logos_infos():
    res = {}
    count = 0
    with open("jsonl_dataset.jsonl", 'rb') as f:
        for row in f:
            dict=json.loads(row)
            res[str(dict["logo_id"])] = [dict["bounding_box"], dict["source_img"]]
            count +=1
            if count > 3: break
    
    with h5py.File("hdf5_dataset.hdf5") as f:
        logos = f['embedding']
        ids = f['external_id']
        for i in range(len(ids)):
            if str(ids[i]) in res.keys():
                print(f"found you at index {i}")
                res[str(ids[i])].append(logos[i])

    return res               


if __name__ == "__main__":
    logos_infos = get_logos_infos()
    old_embeddings_list = []
    PIL_embeddings_list = []
    cv_embeddings_list = []
    for id in logos_infos.keys():
        bounding_box, source_img, old_embedding = logos_infos[id]
        r = requests.get("https://openfoodfacts.org/images/products" + source_img)
        image = Image.open(BytesIO(r.content))

        bounding_boxes = [bounding_box]

        PIL_embedding, cv_embedding = check_embeddings(bounding_boxes, image)
        old_embeddings_list.append(old_embedding)
        PIL_embeddings_list.append(PIL_embedding)
        cv_embeddings_list.append(cv_embedding)
    breakpoint()