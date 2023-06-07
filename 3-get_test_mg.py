from tqdm import tqdm
from elasticsearch import Elasticsearch
from collections import Counter
import jsonlines

data_path = r".\dataset\hansel-nlpcc-eval.jsonl"
out_path = r".\dataset\TempData\test-mg.json"
index_name_mg = "mention_goldid"

client = Elasticsearch("http://localhost:9200")
TestDatas = []


def read_TestData(file_path):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            TestDatas.append(obj)


def convert_to_2d_array(arr):
    # 统计元素出现的次数
    counts = Counter(arr)

    # 转换为二维数组
    result = [[element, counts[element]] for element in counts]

    # 按第二维的元素（出现次数）降序排列
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    return sorted_result


def write_Sub_jsonl(data_array, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for item in data_array:
            writer.write(item)


if __name__ == '__main__':
    candi = []
    read_TestData(data_path)
    i=0
    for data in tqdm(TestDatas):
        b = client.search(index=index_name_mg, query={"term": {"mention": data['mention']}})
        a = b['hits']['hits']
        if len(a) > 0:
            sorted_candi = convert_to_2d_array(a[0]['_source']['gold_id'])
            data.update({'mention':str(data['mention']+str(len(sorted_candi)))})
            data.update({'gold_id': sorted_candi[0][0]})
            data.update({'id': data['id']})

    write_Sub_jsonl(TestDatas, out_path)
    print("3:OK")
