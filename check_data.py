import settings
from utils import get_list_classes, get_random_dict_from_jsonl, get_image_from_url, crop_image, read_jsonl, append_dict_to_jsonl
import matplotlib.pyplot as plt
import webbrowser
import tqdm

def check_local_1_by_1():
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


    classes, annotation_types, amounts = get_list_classes(settings.nb_logos_min, settings.file_logos_amount)
    for i in range(len(classes)):
        classe = classes[i]
        amount = amounts[i]
        file_name = settings.repo_dataset + classe + ".jsonl"
        for i in range(1): 
            dict = get_random_dict_from_jsonl(file_name, amount)
            image = get_image_from_url(settings.base_url+dict["source_img"])
            logo = crop_image(image, dict["bounding_box"])
            plt.figure(1)
            plt.imshow(logo)
            plt.title(dict["class"])
            plt.show()

def check_HG_annotated_logos():
    classes, annotation_types, amounts = get_list_classes(settings.nb_logos_min, settings.file_logos_amount)
    
    for i in range(0,len(classes)):
        amount = min(200,amounts[i])
        webbrowser.open(settings.hg_url + "/search?type=" + annotation_types[i] + "&value=" + classes[i] + "&count=" + str(amount))
        input("Press enter to continue...")

def add_HG_not_annotated_logos():
    dicts = read_jsonl(settings.not_annotated_logos)
    for dict in dicts:
        webbrowser.open(settings.hg_url + "?logo_id=" + str(dict["logo_id"]) + "&count=50")
        input("Press enter to continue...")

def get_classes_infos():
    res = []
    current_class = None
    class_id = 0
    class_amount = 0
    for dict in tqdm.tqdm(read_jsonl(settings.jsonl_dataset)):
        classe = dict["class"]
        if current_class != classe:
            if current_class != None : res.append({"class":current_class, "id":class_id, "amount": class_amount})
            class_amount = 0
            class_id += 1
            current_class = classe
            if classe == "no_class":
                class_id = 0
        class_amount += 1
    res.append({"class":current_class, "id":class_id, "amount": class_amount})
    append_dict_to_jsonl(res, settings.class_infos)

get_classes_infos()