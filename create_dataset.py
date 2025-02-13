import json

with open("raw_rag_truth_dataset/response.jsonl", "r") as f:
    res_data = [json.loads(l) for l in f.readlines()]
with open("raw_rag_truth_dataset/source_info.jsonl", "r") as f:
    info_data = [json.loads(l) for l in f.readlines()]

id_list = [d["source_id"] for d in info_data]

train_data = []
test_data = []
for id in id_list:
    # reference
    for d in info_data[:]:
        if d["source_id"] == id:
            task_type = d["task_type"]
            source = d["source"]
            ref = d["source_info"]
            info_data.remove(d)
            break
    # text
    num = 0
    for d2 in res_data[:]:
        if num == 6:
            break
        if d2["source_id"] == id:
            num += 1
            model_name = d2["model"]
            text = d2["response"]
            labels = 0 if d2["labels"] == [] else 1  # hallucinationが含まれていたら1
            data = {
                "ref": str(ref),
                "text": text,
                "labels": labels,
                "source": source,
                "model": model_name,
                "task_type": task_type,
                "source_id": id,
            }
            if d2["split"] == "train":
                train_data.append(data)
            else:
                test_data.append(data)
            res_data.remove(d2)


with open("dataset/rag_truth_train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open("dataset/rag_truth_test.json", "w") as f:
    json.dump(test_data, f, indent=4)
