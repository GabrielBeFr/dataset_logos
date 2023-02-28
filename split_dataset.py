from utils import read_jsonl, get_product_from_source, append_dict_to_jsonl
import settings
import tqdm
import numpy as np
import h5py

def first_split_dataset():
    '''
    Creates a dict where keys are logo ids and values are the dataset they fit in id.
    0 is for not dataset, meaning the logos can be removed.
    1 for the training dataset.
    2 for the validation dataset.
    3 for the test dataset.
    '''
    res = {}

    logos = read_jsonl(settings.jsonl_dataset)

    current_class = -1
    logos_to_func = []
    product = {}
    for logo in logos:
        classe = logo["class"]
        if classe != current_class:
            if current_class != -1:
                print(f"Current class is {current_class}")
                split_product(product)
                if current_class == None:
                    res.update({str(logo["logo_id"]):0 for logo in logos_to_func})
                else:
                    res.update(split_dataset_class(current_class, product, logos_to_func))         
            current_class = classe    
            count = 0
            logos_to_func = []
            product = {}
        try:
            product[str(get_product_from_source(logo["source_img"]))][0] += 1
        except:
            product[str(get_product_from_source(logo["source_img"]))] = [1]

        logos_to_func.append(logo)
        count +=1
    
    split_product(product)
    res.update(split_dataset_class(current_class, product, logos_to_func))  
    append_dict_to_jsonl([res], settings.split_ids_dataset)

def split_product(products: dict):
    train = 1
    validation = 2
    test = 3

    product_array = np.array([a for a in products.keys()])
    amount_array = np.array([products[a][0] for a in products.keys()])
    total = np.sum(amount_array)

    sorted_indices = np.argsort(-amount_array)

    product_array = product_array[sorted_indices]
    amount_array = amount_array[sorted_indices]

    train_count = 0
    validation_count = 0
    test_count = 0
    for i in range(len(amount_array)):
        amount = amount_array[i]
        product = product_array[i]
        if train_count + amount <= 0.8*total:
            train_count += amount
            products[product].append(train)
        elif validation_count + amount <= 0.1*total:
            validation_count += amount
            products[product].append(validation)
        else:
            test_count += amount
            products[product].append(test)

    print(f"total : {total} | train : {train_count/total} | validation : {validation_count/total} | test : {test_count/total}")


def split_dataset_class(current_class: str, product: dict[list], logos_to_func = list):
    res = {}

    for logo in tqdm.tqdm(logos_to_func):
        if logo["class"] == current_class:
            dataset_id = product[str(get_product_from_source(logo["source_img"]))][1]
            res[str(logo["logo_id"])] = dataset_id
    return res
    
def create_datasets(count_train, count_val, count_test):
    with h5py.File(settings.train_dataset, 'a') as f:
        f.create_dataset(
            "embedding", (count_train, 512), dtype="f", chunks=True
        )
        f.create_dataset(
            "external_id", (count_train,), dtype="i", chunks=True
        )
        f.create_dataset(
            "class", (count_train,), dtype="i", chunks=True
        )
    with h5py.File(settings.val_dataset, 'a') as f:
        f.create_dataset(
            "embedding", (count_val, 512), dtype="f", chunks=True
        )
        f.create_dataset(
            "external_id", (count_val,), dtype="i", chunks=True
        )
        f.create_dataset(
            "class", (count_val,), dtype="i", chunks=True
        )
    with h5py.File(settings.test_dataset, 'a') as f:
        f.create_dataset(
            "embedding", (count_test, 512), dtype="f", chunks=True
        )
        f.create_dataset(
            "external_id", (count_test,), dtype="i", chunks=True
        )
        f.create_dataset(
            "class", (count_test,), dtype="i", chunks=True
        )

def create_splitted_dataset():
    for dict_split in read_jsonl(settings.split_ids_dataset):
        continue
    count_train = len([a for a in dict_split.values() if a == 1])
    count_val = len([a for a in dict_split.values() if a == 2])
    count_test = len([a for a in dict_split.values() if a == 3])

    create_datasets(count_train, count_val, count_test)


    with h5py.File(settings.train_dataset, 'a') as f_training:
        train_embeddings = f_training['embedding']
        train_ids = f_training['external_id']
        train_classes = f_training['class']
        train_offset = 0
        with h5py.File(settings.val_dataset, 'a') as f_val:
            val_embeddings = f_val['embedding']
            val_ids = f_val['external_id']
            val_classes = f_val['class']
            val_offset = 0
            with h5py.File(settings.test_dataset, 'a') as f_test:
                test_embeddings = f_test['embedding']
                test_ids = f_test['external_id']
                test_classes = f_test['class']
                test_offset = 0
                with h5py.File(settings.complete_dataset, 'a') as f_complete:
                    complete_embeddings = f_complete['embedding']
                    complete_ids = f_complete['external_id']
                    complete_classes = f_complete['class']

                    debug = 0
                    for i in tqdm.tqdm(range(len(complete_ids))):
                        id = str(complete_ids[i])
                        if id == '0': break
                        if dict_split[id] == 1:
                            train_embeddings[train_offset] = complete_embeddings[i]
                            train_ids[train_offset] = id
                            train_classes[train_offset] = complete_classes[i]
                            train_offset += 1
                        elif dict_split[id] == 2:
                            val_embeddings[val_offset] = complete_embeddings[i]
                            val_ids[val_offset] = id
                            val_classes[val_offset] = complete_classes[i]
                            val_offset += 1
                        elif dict_split[id] == 3:
                            test_embeddings[test_offset] = complete_embeddings[i]
                            test_ids[test_offset] = id
                            test_classes[test_offset] = complete_classes[i]
                            test_offset += 1


if __name__ == '__main__':
    create_splitted_dataset()
