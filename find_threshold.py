from utils import read_jsonl, count_jsonl_lines, append_dict_to_jsonl
import settings

count = 0
classes = 0
current_amount = 0
result = [{"k_logos":0, "number_classes": count_jsonl_lines(settings.file_logos_amount), "number_logos": 0}]

data_gen = read_jsonl(settings.file_logos_amount)
for row in data_gen:
    amount = row["amount"]
    if amount < current_amount:
        result.append({"k_logos":current_amount, "number_classes": classes, "number_logos": count})
    count += amount
    classes += 1
    current_amount = amount

result.append({"k_logos":current_amount, "number_classes": classes, "number_logos": count})

for dict in result:
    dict["percentage"] = 100*dict["number_logos"]/count

append_dict_to_jsonl(result, settings.k_logos_classes)
    