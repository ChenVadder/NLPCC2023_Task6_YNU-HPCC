import os
import json
import pandas as pd
import jsonlines
from tqdm import trange
from elasticsearch import Elasticsearch

start_of_entity = '[SOE]'
end_of_entity = '[EOE]'


def compare_sub1_sub2_dif():
    Sub1_file_path = r"C:\Users\ChenVadder\Desktop\FinalSub\val-term.jsonl"
    Sub2_file_path = r"C:\Users\ChenVadder\Desktop\NL\dataset\hansel-val.json"
    out_path = r"C:\Users\ChenVadder\Desktop\NL\dataset2\dif-val-term1.jsonl"
    num = 9674

    subdata1 = []
    subdata2 = []
    num_diff = 0
    with jsonlines.open(Sub1_file_path) as reader:
        for obj in reader:
            subdata1.append(obj)
    with jsonlines.open(Sub2_file_path) as reader:
        for obj in reader:
            subdata2.append(obj)

    dif =[]
    for i in trange(num):
        if subdata1[i]['gold_id'] != subdata2[i]["gold_id"]:
            num_diff += 1
            dif.append(subdata2[i])

    with jsonlines.open(out_path, mode='w') as writer:

        for item in dif:
            # print(item)
            writer.write(item)

    print("Aimed: ", num-num_diff)


def read_data(file_path, data_array):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_array.append(obj)


def write_jsonl(data_array, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for item in data_array:
            writer.write(item)



def load_test_mention_text(file_name):
    mention_all = []
    mention = []
    text = []
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            start = data['start']
            end = data['end']
            mention_all.append(data['text'][:start] +
                           start_of_entity +
                           data['text'][start:end] +
                           end_of_entity +
                           data['text'][end:])
            mention.append(data['mention'])
            text.append(data['text'])
    return  mention_all, mention, text



if __name__ == "__main__":
    # compare_sub1_sub2_dif()

    print("OK")





