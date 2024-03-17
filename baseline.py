
import warnings
from datetime import datetime
import time
import torch
import os
from transformers import BertModel,BertConfig,BertModel,BertTokenizer,get_cosine_schedule_with_warmup,BertForMaskedLM,set_seed
import pandas  as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score

from dataloader import load_csvdata,Dataset_bert,ProcessData_bert
import argparse
# hyperparameters

TRAIN_BATCH_SIZE=32  #小批训练， 批大小增大时需要提升学习率
TEST_BATCH_SIZE=96   #大批测试
EVAL_PERIOD=20
MODEL_NAME="bert-large-uncased"
DATA_PATH="./data/"
NUM_WORKERS=10

train_file="train_mix_sentence.csv"
dev_file="dev_mix_sentence.csv"
test_file="test_mix_sentence.csv"


# env variables

os.environ['TOKENIZERS_PARALLELISM']="false"

pd.options.display.max_columns = None
pd.options.display.max_rows = None


tokenizer=BertTokenizer.from_pretrained(MODEL_NAME)

config=BertConfig.from_pretrained(MODEL_NAME)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3

class Bert_Model(nn.Module):
    def __init__(self,  bert_path ,config_file ):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path,config=config_file)  # 加载预训练模型权重


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask) #masked LM 输出的是 mask的值 对应的ids的概率 ，输出 会是词表大小，里面是概率
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]

        return logit

class BertClassifier(nn.Module):
    # def __init__(self, dropout=0.3):
    #     super(BertClassifier, self).__init__()
    #     self.bert = BertModel.from_pretrained('bert-large-uncased')
    #     self.dropout = nn.Dropout(dropout)
    #     self.linear = nn.Linear(768, 5)
    #     self.relu = nn.ReLU()
    def __init__(self,  bert_path ,config_file ):
        super(BertClassifier, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path,config=config_file)  # 加载预训练模型权重
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(768,3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(0.3)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

#model=Bert_Model(bert_path=MODEL_NAME,config_file=config).to(device)
model = BertClassifier(bert_path=MODEL_NAME,config_file=config)

# get the data and label
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--seed', type=int, default=16)


args = parser.parse_args()
EPOCH = args.epoch
SEED = args.seed
set_seed(SEED)

# DATA_PATH+os.sep+filepath
#暂时去掉了num_workers
train_x, train_y = load_csvdata(DATA_PATH+os.sep+train_file)
valid_x, valid_y = load_csvdata(DATA_PATH+os.sep+dev_file)
test_x, test_y = load_csvdata(DATA_PATH+os.sep+test_file)

train_dataset = Data.DataLoader(Dataset_bert(train_x, train_y, tokenizer), TRAIN_BATCH_SIZE, shuffle=True)
valid_dataset = Data.DataLoader(Dataset_bert(valid_x, valid_y, tokenizer), TRAIN_BATCH_SIZE,  shuffle=True)
test_dataset = Data.DataLoader(Dataset_bert(test_x, test_y, tokenizer), TEST_BATCH_SIZE,  shuffle=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

optimizer = optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)  #使用Adam优化器
loss_func = nn.CrossEntropyLoss()
#
if use_cuda:
    model = model.cuda()
    loss_func = loss_func.cuda()

schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_dataset),num_training_steps=EPOCH*len(train_dataset))
print("==============Start traning=============")
totaltime = 0
h_f1score = 0

save_path = "./model_baseline_ckpt"
for epoch in range(EPOCH):

    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label in tqdm(train_dataset):
        train_label = train_label.type(torch.LongTensor)
        train_label = train_label.to(device)
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
        output = model(input_id, mask)
        # 计算损失
        batch_loss = loss_func(output, torch.tensor(train_label))
        total_loss_train += batch_loss.item()
        # 计算精度
        acc = (output.argmax(dim=1) == train_label).sum().item()
        print(acc)
        total_acc_train += acc
        # 模型更新
        model.zero_grad()
        batch_loss.backward()
        optimizer.step()

    total_acc_val = 0
    total_loss_val = 0
    with torch.no_grad():
        # 循环获取数据集，并用训练好的模型进行验证
        for val_input, val_label in valid_dataset:
            # 如果有GPU，则使用GPU，接下来的操作同训练
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = loss_func(output, val_label)
            total_loss_val += batch_loss.item()

            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_acc_val += acc

    print(
        f'''Epochs: {EPOCH + 1} 
        | Train Loss: {total_loss_train / len(train_label): .3f} 
        | Train Accuracy: {total_acc_train / len(train_label): .3f} 
        | Val Loss: {total_loss_val / len(val_label): .3f} 
        | Val Accuracy: {total_acc_val / len(val_label): .3f}''')

    # acc =float(correct /train_data_num)
    # f1score_train = f1_score(truelabel.tolist(), predicted.tolist(), average='weighted')
    # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': EPOCH,
    #          'seed': SEED}
    #
    # #Store the best model with the highest f1score
    # if f1score_train > h_f1score:
    #     h_f1score = f1score_train
    #     h_state = state
    #     torch.save(h_state, os.path.join(save_path, "{}_{}_{}.pth".format(time.time(), EPOCH, SEED)))
    #
    # eval_loss_sum=0.0
    # model.eval()
    # correct_test=0

    #
    # with torch.no_grad():
    #     #Change the test dataset to valid dataset to choose best num of epoch
    #     for ids, att, tpe, y in test_dataset:
    #         # if epoch == 0 and idx == 1:
    #         #     print("=============testdata_1===============")
    #         #     print(ids, att, tpe, y)
    #         ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
    #         out_test = model(ids , att , tpe)
    #         loss_eval = loss_func(out_test.view(-1, tokenizer.vocab_size), y.view(-1))
    #         eval_loss_sum += loss_eval.item()
    #         ttruelabel = y
    #         tout_train_mask = out_test[:, MASK_POS, :]
    #         predicted_test = torch.max(tout_train_mask.data, 1)[1]
    #         correct_test += (predicted_test == ttruelabel).sum()
    #         correct_test = float(correct_test)
    # acc_test = float(correct_test / dev_data_num)
    # # print("=============ttruelabel===============")
    # #
    # # print(ttruelabel)
    # # f2 = open("./output_log_alltrain.txt", 'a', encoding='UTF-8')
    # # out = out + " the highest f1score:" + str(h_f1score) + " mapping_words:" + str(mapping_words)
    # # f2.write(out)
    # # f2.close()
    # end = time.time()
    # print("epoch {} duration:".format(epoch+1),end-start)
    # totaltime+=end-start
    # f1score_test = f1_score(ttruelabel.tolist(), predicted_test.tolist(), average='weighted')
    # if epoch % 1 == 0:
    #     out = ("epoch {}, train_loss {},  train_acc {} , eval_loss {} ,acc_test {} ,f1score_test{}"
    #            .format(epoch + 1, train_loss_sum / (len(train_dataset)), acc, eval_loss_sum / (len(valid_dataset)),
    #             acc_test, f1score_test))
    #     print(out)
    #
    # if f1score_test > h_f1score:
    #     print("The highest f1score is updated in epoch {}".format(EPOCH))
    #     h_f1score = f1score_test
    #     h_state = state
    #     torch.save(h_state, os.path.join(save_path, "{}_{}_{}_{}.pth".format(end, EPOCH, seed, mapping_id)))
    #

print("The training step is finished!!! ")