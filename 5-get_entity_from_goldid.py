import jsonlines
from tqdm import tqdm
from elasticsearch import Elasticsearch
from string import digits

index_name = "kb_entity_keyword"
client = Elasticsearch("http://localhost:9200")

test_path = r".\dataset\hansel-nlpcc-eval.jsonl"
test_mg_path = r".\dataset\TempData\test-mg.json"
outof_path = r".\dataset\TempData\outof_sentence.json"
iniTest = []
test_mg_datas = []
outof_sentence = []


def read_Data(file_path, data_array):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_array.append(obj)


def get_not_get_entity():
    i = 0
    for data in tqdm(test_mg_datas):
        b = client.search(index=index_name, query={"term": {"entity_id": data['gold_id']}})
        a = b['hits']['hits']
        if len(a) > 0:
            # entity
            action2 = {
                'entity_id': a[0]['_source']['entity_id'],
                'entity': a[0]['_source']['entity'],
                'wiki': a[0]['_source']['wiki']
            }
            table = str.maketrans('', '', digits)

            if ('中央' in action2['entity'] or '中国' in action2['entity'] or '中华人民共和国' in action2['entity']) or \
                    (('_' in action2['entity'] and action2['entity'][
                                                   action2['entity'].index('_') + 2:action2['entity'].index('_') + 4] in
                      data['text']) and '年' not in action2['entity']
                    ) or ('杯' in data['mention']) or \
                    (data['mention'].translate(table) == action2['entity']) or \
                    (data['mention'].translate(table) == action2['entity'][:-1]):
                outof_sentence.append(data)
                i += 1
    # print(i)


def update_ini_mention():
    for data in outof_sentence:
        for inidata in iniTest:
            if data['id'] == inidata['id']:
                data.update({'mention': inidata['mention']})


def write_Sub_jsonl(data_array, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for item in data_array:
            writer.write(item)


if __name__ == '__main__':
    read_Data(test_mg_path, test_mg_datas)
    read_Data(test_path, iniTest)

    # 获得通过mg索引得到的结果，单独存储。
    get_not_get_entity()
    update_ini_mention()
    write_Sub_jsonl(outof_sentence, outof_path)
    print("5:OK")
