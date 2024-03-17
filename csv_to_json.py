import csv
import json
import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="The path to original data file")
parser.add_argument("--file_name", type=str, help="File name")

args = parser.parse_args()


json_filename = args.file_name + ".json"
csv_filename = args.file_name + ".csv"
json_file = open(os.path.join(args.data_dir, json_filename), 'w+', encoding='utf-8')
csv_file = open(os.path.join(args.data_dir, csv_filename), 'r', encoding='utf-8')

keys = ['utterance','sentiment','img_path']
json_file.write('[')
flag = 0

arr1 = []
arr2 = []
arr3 = []
arr = []

while csv_file.readline():
    values = pd.read_csv(csv_file, header=None)
    rows = values.shape[0]
    cols = values.shape[1]
    for i in range(0, 47):
        arr1.append(values[0][i])
        arr2.append(values[1][i])
        arr3.append(values[2][i])

    arr = np.dstack((arr1, arr2, arr3))

    dict_tmp = dict(zip(keys, arr.tolist()))

    #dict_tmp = [[12,3,4],[2,3,4]]

    json_str = json.dumps(dict_tmp, indent=4)

    json_file.write(json_str)


json_file.write(']')
csv_file.close()
json_file.close()

