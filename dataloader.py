# 构建数据集
import numpy as np
import torch.utils.data as Data
import pandas as pd
import torch
from transformers import BertModel,BertConfig,BertModel,BertTokenizer


PREFIX = 'It is [mask]. '
MASK_POS = 3  # "it was [mask]" 中 [mask] 位置

class MyDataSet(Data.Dataset):
    def __init__(self, sen, mask, typ, label):
        super(MyDataSet, self).__init__()
        self.sen = torch.tensor(sen, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long)
        self.typ = torch.tensor(typ, dtype=torch.long)
        self.labels = torch.tensor(label)

    def __len__(self):
        return self.sen.shape[0]

    def __getitem__(self, idx):
        return self.sen[idx], self.mask[idx], self.typ[idx], self.labels[idx]

class Dataset_bert(Data.Dataset):
    def __init__(self, x_data, y_label, tokenizer):
        self.labels = y_label
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=180,
                                truncation=True,
                                return_tensors="pt")
                      for text in x_data]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# load  data

def load_data(tsvpath):
    data = pd.read_csv(tsvpath, sep="\t", header=None, names=["sn", "polarity", "text"])
    data = data[data["polarity"] != "neutral"]
    yy = data["polarity"].replace({"negative": 0, "positive": 1, "neutral": 2})
    # print(data.loc[0:5,[0,1]])  #
    # print(data.iloc[0:5,[1,1]])  #
    # print(data.iloc[:,1:2])  #
    # print(data.iloc[:,2:3])  #
    return data.values[:, 2:3].tolist(), yy.tolist()  # data.values[:,1:2].tolist()


def load_csvdata(csv_file):
    data = pd.read_csv(csv_file)
    data = data[data["sentiment"] != "nan"]
    yy = data["sentiment"].replace({"negative": 0, "positive": 1, "neutral": 2})
    data["text"] = data["text"].replace('"', '')
    # print(data.loc[0:5,[0,1]])  #
    # print(data.iloc[0:5,[1,1]])  #
    # print(data.iloc[:,1:2])  #
    # print(data.iloc[:,2:3])  #
    # print(data.values[:, 1:2])
    # print(yy)
    return data.values[:, 1:2].tolist(), yy.tolist()


def ProcessData_meld_plus_temps(filepath, tokenizer, template):
    template = eval(template)
    pos_id = tokenizer.convert_tokens_to_ids(template['positive'])
    neg_id = tokenizer.convert_tokens_to_ids(template['negative'])
    neu_id = tokenizer.convert_tokens_to_ids(template['neutral'])
    # print("===============ID for classlabel===================")
    # print(pos_id, neg_id, neu_id)
    x_train, y_train = load_csvdata(filepath)
    # print("==========y_train==========")
    # print(y_train)

    Inputid = []
    Labelid = []
    typeid = []
    attenmask = []

    for i in range(len(x_train)):

        # text_ = x_train[i][0] + PREFIX
        text_ = PREFIX + x_train[i][0]

        encode_dict = tokenizer.encode_plus(text_, max_length=180, padding="max_length", truncation=True)
        input_ids = encode_dict["input_ids"]
        type_ids = encode_dict["token_type_ids"]
        atten_mask = encode_dict["attention_mask"]
        labelid, inputid = input_ids[:], input_ids[:]
        # MASK_POS = max(max(numpy.nonzero(labelid))) - 4 #get the index of the mask word
        # if i == 0:
        #     print("=====labelid_pos======")
        #     print(labelid[MASK_POS])
        # if i == 0:
        #     print("=====Before======")
        #     print(labelid)

        if y_train[i] == 0:
            labelid[MASK_POS] = neg_id
            labelid[:MASK_POS] = [-1] * len(labelid[:MASK_POS])
            labelid[MASK_POS + 1:] = [-1] * len(labelid[MASK_POS + 1:])
            inputid[MASK_POS] = tokenizer.mask_token_id
            # if i == 0:
            #     print("=====After======")
            #     print(labelid)
        elif y_train[i] == 1:
            labelid[MASK_POS] = pos_id
            labelid[:MASK_POS] = [-1] * len(labelid[:MASK_POS])
            labelid[MASK_POS + 1:] = [-1] * len(labelid[MASK_POS + 1:])
            inputid[MASK_POS] = tokenizer.mask_token_id
        else:
            labelid[MASK_POS] = neu_id
            labelid[:MASK_POS] = [-1] * len(labelid[:MASK_POS])
            labelid[MASK_POS + 1:] = [-1] * len(labelid[MASK_POS + 1:])
            inputid[MASK_POS] = tokenizer.mask_token_id

        Labelid.append(labelid)
        Inputid.append(inputid)
        typeid.append(type_ids)
        attenmask.append(atten_mask)

    return Inputid, Labelid, typeid, attenmask


def ProcessData(filepath, tokenizer):
    MASK_POS = 3
    pos_id = tokenizer.convert_tokens_to_ids("good")  # 9005
    neg_id = tokenizer.convert_tokens_to_ids("bad")  # 12139
    x_train, y_train = load_data(filepath)
    # x_train,x_test,y_train,y_test=train_test_split(StrongData,StrongLabel,test_size=0.3, random_state=42)

    Inputid = []
    Labelid = []
    typeid = []
    attenmask = []

    for i in range(len(x_train)):
        # mask sentence concate utterance
        text_ = PREFIX + x_train[i][0]

        encode_dict = tokenizer.encode_plus(text_, max_length=60, padding="max_length", truncation=True)

        input_ids = encode_dict["input_ids"]
        type_ids = encode_dict["token_type_ids"]
        atten_mask = encode_dict["attention_mask"]
        labelid, inputid = input_ids[:], input_ids[:]
        if y_train[i] == 0:
            labelid[MASK_POS] = neg_id
            labelid[:MASK_POS] = [-1] * len(labelid[:MASK_POS])
            labelid[MASK_POS + 1:] = [-1] * len(labelid[MASK_POS + 1:])
            inputid[MASK_POS] = tokenizer.mask_token_id
        else:
            labelid[MASK_POS] = pos_id
            labelid[:MASK_POS] = [-1] * len(labelid[:MASK_POS])
            labelid[MASK_POS + 1:] = [-1] * len(labelid[MASK_POS + 1:])
            inputid[MASK_POS] = tokenizer.mask_token_id

        Labelid.append(labelid)
        Inputid.append(inputid)
        typeid.append(type_ids)
        attenmask.append(atten_mask)

    return Inputid, Labelid, typeid, attenmask


def ProcessData_bert(filepath, tokenizer):
    x_data, y_data = load_csvdata(filepath)

    Inputid = []
    typeid = []
    attenmask = []

    for i in range(len(x_data)):
        text_ = x_data[i][0]
        encode_dict = tokenizer.encode_plus(text_, max_length=180, padding="max_length", truncation=True)

        input_ids = encode_dict["input_ids"]
        type_ids = encode_dict["token_type_ids"]
        atten_mask = encode_dict["attention_mask"]

        Inputid.append(input_ids)
        typeid.append(type_ids)
        attenmask.append(atten_mask)

    return Inputid, typeid, attenmask, y_data
