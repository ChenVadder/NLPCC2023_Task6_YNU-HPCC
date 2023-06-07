import jsonlines

sub_temp_path = r".\dataset\TempData\FinalSub-Temp-V2-1-Top8.jsonl"
outof_path = r".\dataset\TempData\outof_sentence.json"
test_path = r".\dataset\hansel-nlpcc-eval.jsonl"
Final_sub_path = r".\dataset\YNU-HPCC-FinalSub.jsonl"
sub_temp, outof, test = [], [], []


def read_data(file_path, data_array):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_array.append(obj)


def write_Sub_jsonl(data_array, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for item in data_array:
            writer.write(item)


def update_sub_temp_by_outof(sub_temp_array, outof_array):
    i = 0
    for data in sub_temp_array:
        for item in outof_array:
            if data['id'] == item['id'] and data['gold_id'] != item['gold_id']:
                data.update({'gold_id': item['gold_id']})
                i += 1

    # print(i)




if __name__ == '__main__':
    read_data(sub_temp_path, sub_temp)
    read_data(outof_path, outof)

    # 把sub_temp的部分结果更新成outof
    update_sub_temp_by_outof(sub_temp, outof)

    write_Sub_jsonl(sub_temp, Final_sub_path)

    print("6:OK")
