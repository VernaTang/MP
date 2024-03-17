"""This script samples K examples randomly without replacement from the original data."""
#Add some operations for MELD dataset
import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import re

def get_label(task, line):
    if task == "MELD_MIX":
        return line[-1]
    else:
        return line[0]


def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        dataset = {}
        dirname = os.path.join(data_dir, task)
        splits = ["train", "dev", "test"]
        for split in splits:
            filename = os.path.join(dirname, f"{split}.csv")
            dataset[split] = pd.read_csv(filename, header=None)
        datasets[task] = dataset
    return datasets


def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task == "MELD":
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
                        help="Training examples for each class.")
    parser.add_argument("--task", type=str, nargs="+",
                        default=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B',
                                 'MNLI', 'SNLI', 'QNLI', 'RTE'],
                        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
                        default=[100, 13, 21, 42, 87],
                        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'],
                        help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    k = args.k
    print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                train_header, train_lines = split_header(task, dataset["train"])
                np.random.shuffle(train_lines)
            elif task == "MELD_MIX":
                # For the MELD dataset
                train_header = dataset["train"].loc[0,:].values.tolist()
                train_lines = dataset['train'].loc[1:,:].values.tolist()
                np.random.shuffle(train_lines)
            else:
                # Other datasets
                train_header = dataset["train"][0:1].values.tolist()
                train_lines = dataset['train'][1:].values.tolist()
                np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            # Write test splits
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace('dev', 'test')
                    with open(os.path.join(setting_dir, f"{split}.csv"), "w") as f:
                        for line in lines:
                            f.write(str(line))
            # elif task == "MELD":
            #     new_dataset = dataset['test'].loc[1:,:]
            #     new_dataset = DataFrame(new_dataset)
            #     new_dataset.columns = ["utterance", "sentiment", "img_path"]
            #     new_dataset.to_json(path_or_buf=os.path.join(setting_dir, 'test.json'), force_ascii=False
            #                         , orient='records', lines=True)
            #
            #     new_dataset['caption'] = ''
            #     #print(new_dataset.loc[1][-2])
            #
            #     caption_file = open(os.path.join(os.path.join(args.data_dir, 'MELD'), "all_caption_complete.txt"))
            #     caption_list = caption_file.readlines()
            #     #Since the loc operation above, the index here should start at 1
            #     for i in range(1, new_dataset.shape[0]+1):
            #         img_id = new_dataset.loc[i][-2].split('/')[-1]
            #         for line in caption_list:
            #             line_id = line.split(',')[0]
            #             line_content = line.split(',')[1]
            #             if str(line_id) == str(img_id):
            #                 # print(line_id, "\n", img_id)
            #                 new_dataset.loc[i, 'caption'] = line_content
            #     caption_file.close()
            #     # print(new_train)
            #
            #     new_dataset.to_json(path_or_buf=os.path.join(setting_dir, 'test_add_caption.json'), force_ascii=False
            #                     , orient='records', lines=True)

            else:
                # Other datasets
                # Use the original test sets
                dataset['test'].to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)


                # Get label list for balanced sampling
            label_list = {}
            for line in train_lines:
                label = get_label(task, line)
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:

                        for line in label_list[label][:k]:
                            f.write(line)
                name = "dev.tsv"
                if task == 'MNLI':
                    name = "dev_matched.tsv"
                with open(os.path.join(setting_dir, name), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        dev_rate = 11 if '10x' in args.mode else 2
                        for line in label_list[label][k:k * dev_rate]:
                            f.write(line)
            else:
                new_train = []

                for label in label_list.keys():
                    if label in ['positive','negative','neutral']:
                        for line in label_list[label][:k]:
                            new_train.append(line)
                new_train = DataFrame(new_train)
                print("======================")
                print(len(new_train))

                if task == "MELD":
                    new_train.columns = ["utterance", "sentiment", "img_path"]
                    new_train.to_json(path_or_buf=os.path.join(setting_dir, 'train.json'), force_ascii=False
                                      , orient='records', lines=True)
                    new_train['caption'] = ''

                    caption_file = open(os.path.join(os.path.join(args.data_dir, 'MELD'), "all_caption_complete.txt"))
                    caption_list = caption_file.readlines()
                    for i in range(0, new_train.shape[0]):
                        img_id = new_train.loc[i][-2].split('/')[-1]
                        for line in caption_list:
                            line_id = line.split(',')[0]
                            line_content = line.split(',')[1]
                            if str(line_id) == str(img_id):
                                #print(line_id, "\n", img_id)
                                new_train.loc[i, 'caption'] = line_content
                    caption_file.close()
                    # print(new_train)
                    new_train.to_json(path_or_buf=os.path.join(setting_dir, 'train_add_caption.json'), force_ascii=False
                                      , orient='records', lines=True)
                else:
                    new_train.to_csv(os.path.join(setting_dir, 'train.csv'), header=['name','text','sentiment'], index=False)

                # new_dev = []
                # for label in label_list:
                #     dev_rate = 11 if '10x' in args.mode else 2
                #     for line in label_list[label][k:k * dev_rate]:
                #         new_dev.append(line)
                #
                #
                # if task == "MELD":
                #     new_dev = DataFrame(new_dev)
                #     new_dev.columns = ["utterance", "sentiment", "img_path"]
                #     new_dev.to_json(path_or_buf=os.path.join(setting_dir, 'dev.json'), force_ascii=False
                #                    , orient='records', lines=True)
                #
                #     new_dev['caption'] = ''
                #
                #     caption_file = open(os.path.join(os.path.join(args.data_dir,'MELD'), "all_caption_complete.txt"))
                #     caption_list = caption_file.readlines()
                #     for i in range(0, new_dev.shape[0]):
                #         img_id = new_dev.loc[i][-2].split('/')[-1]
                #         for line in caption_list:
                #             line_id = line.split(',')[0]
                #             line_content = line.split(',')[1]
                #             if str(line_id) == str(img_id):
                #                 #print(line_id, "\n", img_id)
                #                 new_dev.loc[i, 'caption'] = line_content
                #     caption_file.close()
                #     #print(new_dev)
                #     new_dev.to_json(path_or_buf=os.path.join(setting_dir, 'dev_add_caption.json'), force_ascii=False
                #                     , orient='records', lines=True)
                #     # json_file = open(os.path.join(setting_dir, 'dev.json'), 'w', encoding='utf-8')
                #     #
                #     # new_dev_json = new_dev.to_json(force_ascii=False, orient='records',lines=True)
                #     # json_file.write('[')
                #     # json_file.write("\n")
                #     # for line in new_dev_json.splitlines():
                #     #     json_file.write(line)
                #     #     json_file.write(",\n")
                #     # json_file.write(']')
                #
                #
                # else:
                #     new_dev.to_csv(os.path.join(setting_dir, 'dev.csv'), header=False, index=False)



if __name__ == "__main__":
    main()
