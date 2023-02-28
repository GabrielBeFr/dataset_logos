from utils import append_dict_to_jsonl, get_list_classes, matching_classes
import settings
import tqdm
import os
import numpy as np

def get_classes_logos(cur):
    res = []
    cur.execute(f"SELECT COUNT(id), taxonomy_value, annotation_type FROM logo_annotation WHERE (annotation_type = 'label' OR annotation_type = 'brand') AND taxonomy_value IS NOT NULL GROUP BY taxonomy_value, annotation_type ORDER BY COUNT(id) DESC")
    for column in cur.fetchall():
        res.append({"class": column[1], "annotation_type": column[2], "amount": column[0]})

    append_dict_to_jsonl(res, settings.file_logos_amount)

def get_logos_classes(cur):
    classes, annotation_types, amounts = get_list_classes(settings.nb_logos_min, settings.file_logos_amount)

    for classe in tqdm.tqdm(classes):
        file_name = settings.repo_dataset + classe + ".jsonl"
        if os.path.exists(file_name): continue
        res = []
        if "'" in classe : classe = classe.replace("'","''")
        cur.execute(f"SELECT DISTINCT id, source_image, taxonomy_value, bounding_box FROM logo_annotation WHERE (annotation_type = 'label' OR annotation_type = 'brand') AND taxonomy_value = \'{classe}\'")
        for column in tqdm.tqdm(cur.fetchall()):
            res.append({"class": column[2], "logo_id": column[0], "source_img": column[1], "bounding_box": column[3]})
        append_dict_to_jsonl(res, file_name)

def find_source_img(cur):
    logo_id = '4791210'
    cur.execute(f"SELECT * FROM logo_annotation JOIN annotation_vote ON WHERE id = {logo_id}")
    for column in cur.fetchall():
        print(column)

def find_date_logo_extraction(cur):
    barcode = '20260330'
    image_id = '3'
    print((f"SELECT timestamp FROM image_prediction AS i_p JOIN image as im ON i_p.image_id = im.id  WHERE im.barcode = {barcode} AND im.image_id = {image_id} AND i_p.model_name = 'universal-logo-detector'"))
    cur.execute(f"SELECT timestamp FROM image_prediction AS i_p JOIN image as im ON i_p.image_id = im.id  WHERE im.barcode = '{barcode}' AND im.image_id = '{image_id}' AND i_p.model_name = 'universal-logo-detector'")
    for column in cur.fetchall():
        print(column)

def find_not_annotated_logos(cur):
    file_path = settings.not_annotated_logos
    res = []
    cur.execute(f"SELECT DISTINCT id FROM logo_annotation WHERE annotation_type is NULL LIMIT 100")
    for column in tqdm.tqdm(cur.fetchall()):
        res.append({"logo_id": column[0]})
    append_dict_to_jsonl(res, file_path)

def average_logos_images(cur):
    cur.execute(f"SELECT COUNT(id) FROM logo_annotation GROUP BY barcode")

    res = np.array([column[0] for column in tqdm.tqdm(cur.fetchall()) if column[0]>0])
    print(res.mean())

def get_schemas(cur):
    res = []
    cur.execute(f"SELECT schema_name FROM information_schema.schemata")
    for column in tqdm.tqdm(cur.fetchall()):
        res.append(column)
    breakpoint()

def non_right_embedded(cur):
    res = []
    cur.execute(f"SELECT * FROM logo_embedding LIMIT 0")
    for column in tqdm.tqdm(cur.fetchall()):
        res.append(column)
    return res

def create_json_dataset(cur):
    file_path = settings.jsonl_dataset
    res = []
    classes, annotation_types, amounts = get_list_classes(settings.nb_logos_min, settings.file_logos_amount)
    for i in tqdm.tqdm(range(len(classes))):
        classe = classes[i]
        if "'" in classe : classe = classe.replace("'","''")
        annotation_type = annotation_types[i]
        cur.execute(f"SELECT DISTINCT id, taxonomy_value, source_image, bounding_box FROM logo_annotation WHERE annotation_type = \'{annotation_type}\' AND taxonomy_value = \'{classe}\'")
        for column in tqdm.tqdm(cur.fetchall()):
            classe = matching_classes(column[1])
            res.append({"logo_id": column[0], "class": classe, "source_img": column[2], "bounding_box": column[3]})     
    
    cur.execute(f"SELECT id, source_image, bounding_box FROM logo_annotation WHERE annotation_type is NULL LIMIT 500000")
    for column in tqdm.tqdm(cur.fetchall()):
        res.append({"logo_id": column[0], "class": "no_class", "source_img": column[1], "bounding_box": column[2]})   

    append_dict_to_jsonl(res,file_path) 

def get_logo_infos(cur, id1, id2):
    cur.execute(f"SELECT source_image, bounding_box, id FROM logo_annotation WHERE id = {id1} OR id = {id2}")
    return cur.fetchall()





