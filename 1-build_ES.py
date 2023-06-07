import csv
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd

client = Elasticsearch("http://localhost:9200")
index_name_et = "kb_entity_text"
index_name_mg = "mention_goldid"
KBDatas = []  # 知识库
TestDatas = []  # 测试集

KB_file_path = r'.\dataset\wp_title_desc_wd0315.tsv'
train_data_path = r".\dataset\hansel-train.json"


# 创建KB中由entity指向wiki的索引
def create_index_text():
    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "ik_analyzer": {
                        "tokenizer": "ik_max_word"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "entity_id": {
                    "type": "keyword",
                },
                "entity": {
                    "type": "text",
                    "analyzer": "ik_analyzer"
                },
                "wiki": {
                    "type": "text",
                    "analyzer": "ik_analyzer"
                }
            }
        }
    }
    client.indices.create(index=index_name_et, body=body)


def read_KB(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in reader:
            doc = {
                'entity_id': row['entity_id'],
                'entity': row['entity'],
                'wiki': row['wiki'],
            }
            KBDatas.append(doc)


def index_doc_bulk_et():
    bulk_num = 3000
    load_data = []
    i = 0
    for data in tqdm(KBDatas):
        action = {
            "_index": index_name_et,
            "_id": data['entity_id'],
            "_source": data
        }
        load_data.append(action)
        i += 1

        if i == bulk_num:
            bulk(client, load_data)
            load_data = []
            i = 0
    # 最后剩下的零散数据也加入
    bulk(client, load_data)


def create_index_mg():
    body = {
        "mappings": {
            "properties": {
                "mention": {
                    "type": "keyword",
                },
                "gold_id": {
                    "type": "keyword",
                },
                "num": {
                    "type": "keyword"
                }
            }
        }
    }
    client.indices.create(index=index_name_mg, body=body)


def index_doc_bulk_mg():
    chunk_iter = pd.read_json(train_data_path, lines=True, chunksize=40960)
    for chunk in tqdm(chunk_iter):
        data_list = chunk.to_dict('records')
        load_data = []
        for data in data_list:
            mg = {'mention': data['mention'], 'gold_id': [data['gold_id']], 'num': 1}

            b = client.search(index=index_name_mg, query={"match": {"mention": data['mention']}})
            num = b["hits"]["total"]["value"]
            if 0 != num:
                b2 = b["hits"]["hits"][0]["_source"]["gold_id"]
                num2 = b["hits"]["hits"][0]["_source"]["num"]
                num2 += 1
                b2.append(data['gold_id'])
                mg.update({"num": num2})
                mg.update({"gold_id": b2})
            action = {
                "_index": index_name_mg,
                "_id": mg['mention'],
                "_source": mg
            }
            load_data.append(action)

            if action["_id"] == '' or action["_id"] is None:
                action.update({'_id': data['text'][data['start'] + 1]})
                load_data.append(action)
        # print(load_data)
        bulk(client, load_data)
        client.indices.refresh(index=index_name_mg)


if __name__ == "__main__":
    # 建立索引
    #
    # 创建索引
    create_index_text()
    # 本步骤在本地部署，耗时约2.5小时
    create_index_mg()

    # 读取数据库数据
    read_KB(KB_file_path)

    # 将数据写入ES
    index_doc_bulk_et()
    index_doc_bulk_mg()

    print("1:OK")
