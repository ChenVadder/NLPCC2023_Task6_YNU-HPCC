import jsonlines
import json
from tqdm import trange
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
start_of_entity = '[SOE]'
end_of_entity = '[EOE]'

test_temp = r".\dataset\TempData\test_temp-V2-1-Top8.jsonl"
test_file = r".\dataset\hansel-nlpcc-eval.jsonl"
sub_file = r".\dataset\TempData\FinalSub-Temp-V2-1-Top8.jsonl"
top = 8
num = 8000

TestDatas = []


def read_test_id_wiki_score(file_path):
    candi_id, candi_wiki, candi_score = [], [], []
    with jsonlines.open(file_path) as reader:
        for data in reader:
            candi_id.append(data['candi_id'])
            candi_wiki.append(data['candi_wiki'])
            candi_score.append(data['candi_score'])
    return candi_id, candi_wiki, candi_score


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
    return mention_all, mention, text


def read_TestData(file_path):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            TestDatas.append(obj)


def write_Sub_jsonl(data_array, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for item in data_array:
            writer.write(item)


if __name__ == '__main__':
    # 本部分建议用云算力进行

    candi_id, candi_wiki, candi_score = read_test_id_wiki_score(test_temp)
    mention_all, mention, text = load_test_mention_text(test_file)
    read_TestData(test_file)
    for i in trange(num):
        query_embedding = model.encode(mention_all[i])
        passage_embedding = model.encode(
            [candi_wiki[i * top], candi_wiki[i * top + 1], candi_wiki[i * top + 2], candi_wiki[i * top + 3],
             candi_wiki[i * top + 4], candi_wiki[i * top + 5], candi_wiki[i * top + 6], candi_wiki[i * top + 7]])
        sims = util.cos_sim(query_embedding, passage_embedding)
        sims = torch.tensor(
            [sims[0][0] * candi_score[i * top], sims[0][1] * candi_score[i * top + 1],
             sims[0][2] * candi_score[i * top + 2], sims[0][3] * candi_score[i * top + 3],
             sims[0][4] * candi_score[i * top + 4], sims[0][5] * candi_score[i * top + 5],
             sims[0][6] * candi_score[i * top + 6], sims[0][7] * candi_score[i * top + 7]])
        TestDatas[i].update({'gold_id': candi_id[torch.argmax(sims) + i * top]})

    write_Sub_jsonl(TestDatas, sub_file)
    print("4:OK")
