tables = ['logo_confidence_threshold','logo_annotation','product_insight','image','image_prediction','annotation_vote','prediction']
file_logos_amount = "label_logos.jsonl"
k_logos_classes = "k_logos_in_classes.jsonl"
not_annotated_logos = "random_not_annotated_logos.jsonl"
repo_dataset = "dataset/"
nb_logos_min = 100
base_url = "https://images.openfoodfacts.org/images/products"
hg_url = "https://hunger.openfoodfacts.org/logos"
image_size = (1024,1024)
logo_ids_check = [1044817, 2050955, 2145983, 3179587, 469016, 3781299]
jsonl_dataset = "jsonl_dataset.jsonl"
logos_embeddings = "logos_embeddings_512.hdf5"
not_embedded_logos_ids = "not_embedded_logos_ids.jsonl"
hdf5_dataset = "hdf5_dataset.hdf5"
class_infos = "new_class_infos.jsonl"
error = "error_loading_imgs.jsonl"
train_dataset = "datasets/train_dataset.hdf5"
val_dataset = "datasets/val_dataset.hdf5"
test_dataset = "datasets/test_dataset.hdf5"
complete_dataset = "complete_dataset.hdf5"
split_ids_dataset = "split_ids_dataset.jsonl"
size_logos = 224
num_workers = 5
batch_size = 32
