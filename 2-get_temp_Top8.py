import jsonlines
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import trange
from util import load_test_mention_text

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
num = 8000
test_file = r".\dataset\hansel-nlpcc-eval.jsonl"
test_temp = r".\dataset\TempData\test_temp-V2-1-Top8.jsonl"
index_name_et = "kb_entity_text"


client = Elasticsearch("http://localhost:9200")


def get_candidate_wiki_and_score():
    for i in trange(len(mention)):
        b = client.search(index=index_name_et, query={'match': {'entity': mention[i]}})
        a = b["hits"]["hits"]
        if len(a) >= 8:
            candi_id.append(a[0]["_source"]["entity_id"])
            candi_id.append(a[1]["_source"]["entity_id"])
            candi_id.append(a[2]["_source"]["entity_id"])
            candi_id.append(a[3]["_source"]["entity_id"])
            candi_id.append(a[4]["_source"]["entity_id"])
            candi_id.append(a[5]["_source"]["entity_id"])
            candi_id.append(a[6]["_source"]["entity_id"])
            candi_id.append(a[7]["_source"]["entity_id"])

            candi_wiki.append(a[0]["_source"]['wiki'])
            candi_wiki.append(a[1]["_source"]['wiki'])
            candi_wiki.append(a[2]["_source"]['wiki'])
            candi_wiki.append(a[3]["_source"]['wiki'])
            candi_wiki.append(a[4]["_source"]['wiki'])
            candi_wiki.append(a[5]["_source"]['wiki'])
            candi_wiki.append(a[6]["_source"]['wiki'])
            candi_wiki.append(a[7]["_source"]['wiki'])

            candi_score.append(a[0]["_score"])
            candi_score.append(a[1]["_score"])
            candi_score.append(a[2]["_score"])
            candi_score.append(a[3]["_score"])
            candi_score.append(a[4]["_score"])
            candi_score.append(a[5]["_score"])
            candi_score.append(a[6]["_score"])
            candi_score.append(a[7]["_score"])

        else:
            b = client.search(index=index_name_et, query={'match': {'wiki': mention[i]}})
            a = b["hits"]["hits"]
            if len(a) >= 8:
                candi_id.append(a[0]["_source"]["entity_id"])
                candi_id.append(a[1]["_source"]["entity_id"])
                candi_id.append(a[2]["_source"]["entity_id"])
                candi_id.append(a[3]["_source"]["entity_id"])
                candi_id.append(a[4]["_source"]["entity_id"])
                candi_id.append(a[5]["_source"]["entity_id"])
                candi_id.append(a[6]["_source"]["entity_id"])
                candi_id.append(a[7]["_source"]["entity_id"])

                candi_wiki.append(a[0]["_source"]['wiki'])
                candi_wiki.append(a[1]["_source"]['wiki'])
                candi_wiki.append(a[2]["_source"]['wiki'])
                candi_wiki.append(a[3]["_source"]['wiki'])
                candi_wiki.append(a[4]["_source"]['wiki'])
                candi_wiki.append(a[5]["_source"]['wiki'])
                candi_wiki.append(a[6]["_source"]['wiki'])
                candi_wiki.append(a[7]["_source"]['wiki'])

                candi_score.append(a[0]["_score"])
                candi_score.append(a[1]["_score"])
                candi_score.append(a[2]["_score"])
                candi_score.append(a[3]["_score"])
                candi_score.append(a[4]["_score"])
                candi_score.append(a[5]["_score"])
                candi_score.append(a[6]["_score"])
                candi_score.append(a[7]["_score"])

            else:
                b = client.search(index=index_name_et, query={'match': {'wiki': text[i]}})
                a = b["hits"]["hits"]

                candi_id.append(a[0]["_source"]["entity_id"])
                candi_id.append(a[1]["_source"]["entity_id"])
                candi_id.append(a[2]["_source"]["entity_id"])
                candi_id.append(a[3]["_source"]["entity_id"])
                candi_id.append(a[4]["_source"]["entity_id"])
                candi_id.append(a[5]["_source"]["entity_id"])
                candi_id.append(a[6]["_source"]["entity_id"])
                candi_id.append(a[7]["_source"]["entity_id"])

                candi_wiki.append(a[0]["_source"]['wiki'])
                candi_wiki.append(a[1]["_source"]['wiki'])
                candi_wiki.append(a[2]["_source"]['wiki'])
                candi_wiki.append(a[3]["_source"]['wiki'])
                candi_wiki.append(a[4]["_source"]['wiki'])
                candi_wiki.append(a[5]["_source"]['wiki'])
                candi_wiki.append(a[6]["_source"]['wiki'])
                candi_wiki.append(a[7]["_source"]['wiki'])

                candi_score.append(a[0]["_score"])
                candi_score.append(a[1]["_score"])
                candi_score.append(a[2]["_score"])
                candi_score.append(a[3]["_score"])
                candi_score.append(a[4]["_score"])
                candi_score.append(a[5]["_score"])
                candi_score.append(a[6]["_score"])
                candi_score.append(a[7]["_score"])


def read_TestData(file_path):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            TestDatas.append(obj)


def get_test_temp():
    for i in trange(num * 8):
        data = {'candi_id': candi_id[i], 'candi_wiki': candi_wiki[i], 'candi_score': candi_score[i]}
        test_t.append(data)


def write_test_temp(out_file):
    with jsonlines.open(out_file, mode='w') as writer:
        for data in test_t:
            writer.write(data)


if __name__ == '__main__':
    TestDatas, candi_id, candi_wiki, candi_score = [], [], [], []
    test_t = []
    # 上面三个的长度是下面三个的top倍
    mention_all, mention, text = load_test_mention_text(test_file)
    get_candidate_wiki_and_score()
    read_TestData(test_file)
    get_test_temp()
    write_test_temp(test_temp)

    print("2:OK")
